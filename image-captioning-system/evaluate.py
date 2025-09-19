"""
Evaluation script for image captioning models.
"""

import argparse
import torch
import yaml
from pathlib import Path
import logging
import json
from typing import Dict, Any
import numpy as np
from tqdm import tqdm

# Import our modules
from src.models.utils import ModelConfig, load_model
from src.data.dataset import create_dataset, Tokenizer, get_transforms
from src.evaluation.metrics import CaptionEvaluator, ModelEvaluator
from src.utils.logging_config import setup_logging, metrics_logger

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    config_path: str,
    data_config: Dict[str, Any],
    device: str = "cpu",
    output_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path: Path to the trained model
        config_path: Path to the model configuration
        data_config: Data configuration dictionary
        device: Device to run evaluation on
        output_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info(f"Loading model from {model_path}")
    
    # Load model and config
    model, config = load_model(model_path, config_path, device)
    
    # Initialize tokenizer
    tokenizer = Tokenizer(vocab_size=config.vocab_size)
    
    # Load tokenizer
    tokenizer_path = Path(model_path).parent / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer.load(tokenizer_path)
    else:
        logger.warning("Tokenizer not found, using default")
    
    # Prepare test data
    test_transform = get_transforms("val")
    test_dataset = create_dataset(
        dataset_type=data_config["dataset_type"],
        data_path=data_config["test_path"],
        image_dir=data_config["image_dir"],
        tokenizer=tokenizer,
        transform=test_transform,
        max_length=config.max_len,
        split="test"
    )
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Evaluating on {len(test_dataset)} test samples")
    
    # Initialize evaluator
    evaluator = CaptionEvaluator(device=device)
    model_evaluator = ModelEvaluator(evaluator)
    
    # Evaluate model
    metrics = model_evaluator.evaluate_model(model, test_loader, tokenizer, device)
    
    # Additional evaluation on subsets
    subset_results = evaluate_subsets(model, test_dataset, tokenizer, evaluator, device)
    
    # Combine results
    results = {
        "model_path": model_path,
        "config": config.to_dict(),
        "data_config": data_config,
        "overall_metrics": metrics,
        "subset_results": subset_results,
        "model_info": model.get_model_size()
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Log metrics
    metrics_logger.log_model_metrics(
        model_type=config.model_type,
        metrics=metrics,
        dataset=data_config["dataset_type"],
        split="test"
    )
    
    return results


def evaluate_subsets(
    model,
    test_dataset,
    tokenizer,
    evaluator,
    device: str
) -> Dict[str, Any]:
    """Evaluate model on different subsets of the test data."""
    subset_results = {}
    
    # Evaluate on different caption lengths
    caption_lengths = []
    for item in test_dataset.data:
        caption_lengths.append(len(item["caption"].split()))
    
    caption_lengths = np.array(caption_lengths)
    
    # Short captions (< 10 words)
    short_indices = np.where(caption_lengths < 10)[0]
    if len(short_indices) > 0:
        short_subset = torch.utils.data.Subset(test_dataset, short_indices)
        short_loader = torch.utils.data.DataLoader(short_subset, batch_size=32, shuffle=False)
        short_metrics = evaluator.evaluate_model(model, short_loader, tokenizer, device)
        subset_results["short_captions"] = {
            "count": len(short_indices),
            "metrics": short_metrics
        }
    
    # Medium captions (10-20 words)
    medium_indices = np.where((caption_lengths >= 10) & (caption_lengths <= 20))[0]
    if len(medium_indices) > 0:
        medium_subset = torch.utils.data.Subset(test_dataset, medium_indices)
        medium_loader = torch.utils.data.DataLoader(medium_subset, batch_size=32, shuffle=False)
        medium_metrics = evaluator.evaluate_model(model, medium_loader, tokenizer, device)
        subset_results["medium_captions"] = {
            "count": len(medium_indices),
            "metrics": medium_metrics
        }
    
    # Long captions (> 20 words)
    long_indices = np.where(caption_lengths > 20)[0]
    if len(long_indices) > 0:
        long_subset = torch.utils.data.Subset(test_dataset, long_indices)
        long_loader = torch.utils.data.DataLoader(long_subset, batch_size=32, shuffle=False)
        long_metrics = evaluator.evaluate_model(model, long_loader, tokenizer, device)
        subset_results["long_captions"] = {
            "count": len(long_indices),
            "metrics": long_metrics
        }
    
    return subset_results


def compare_models(
    model_paths: list,
    config_paths: list,
    data_config: Dict[str, Any],
    device: str = "cpu",
    output_dir: str = "comparison_results"
) -> Dict[str, Any]:
    """
    Compare multiple models.
    
    Args:
        model_paths: List of model paths
        config_paths: List of config paths
        data_config: Data configuration
        device: Device to run evaluation on
        output_dir: Directory to save results
        
    Returns:
        Dictionary of comparison results
    """
    logger.info(f"Comparing {len(model_paths)} models")
    
    results = {}
    
    for i, (model_path, config_path) in enumerate(zip(model_paths, config_paths)):
        logger.info(f"Evaluating model {i+1}/{len(model_paths)}: {model_path}")
        
        model_results = evaluate_model(
            model_path=model_path,
            config_path=config_path,
            data_config=data_config,
            device=device,
            output_dir=f"{output_dir}/model_{i+1}"
        )
        
        model_name = Path(model_path).stem
        results[model_name] = model_results
    
    # Create comparison summary
    comparison_summary = create_comparison_summary(results)
    
    # Save comparison results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison_file = output_path / "model_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump({
            "comparison_summary": comparison_summary,
            "detailed_results": results
        }, f, indent=2)
    
    logger.info(f"Comparison results saved to {comparison_file}")
    
    return results


def create_comparison_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary comparing multiple models."""
    summary = {
        "models": list(results.keys()),
        "metrics_comparison": {},
        "best_models": {}
    }
    
    # Get all metrics
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results["overall_metrics"].keys())
    
    # Compare metrics
    for metric in all_metrics:
        metric_values = {}
        for model_name, model_results in results.items():
            if metric in model_results["overall_metrics"]:
                metric_values[model_name] = model_results["overall_metrics"][metric]
        
        if metric_values:
            summary["metrics_comparison"][metric] = metric_values
            
            # Find best model for this metric
            best_model = max(metric_values.items(), key=lambda x: x[1])
            summary["best_models"][metric] = {
                "model": best_model[0],
                "value": best_model[1]
            }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate image captioning model")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--config-path", type=str, help="Path to model config")
    parser.add_argument("--data-config", type=str, required=True, help="Path to data config")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--compare", action="store_true", help="Compare multiple models")
    parser.add_argument("--model-paths", nargs="+", help="List of model paths for comparison")
    parser.add_argument("--config-paths", nargs="+", help="List of config paths for comparison")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load data configuration
    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)
    
    if args.compare:
        # Compare multiple models
        if not args.model_paths or not args.config_paths:
            logger.error("Model paths and config paths required for comparison")
            return
        
        if len(args.model_paths) != len(args.config_paths):
            logger.error("Number of model paths must match number of config paths")
            return
        
        results = compare_models(
            model_paths=args.model_paths,
            config_paths=args.config_paths,
            data_config=data_config,
            device=device,
            output_dir=args.output_dir
        )
        
        # Print comparison summary
        print("\n" + "="*50)
        print("MODEL COMPARISON SUMMARY")
        print("="*50)
        
        for metric, best_info in results["comparison_summary"]["best_models"].items():
            print(f"{metric}: {best_info['model']} ({best_info['value']:.4f})")
        
    else:
        # Evaluate single model
        if not args.model_path or not args.config_path:
            logger.error("Model path and config path required for single model evaluation")
            return
        
        results = evaluate_model(
            model_path=args.model_path,
            config_path=args.config_path,
            data_config=data_config,
            device=device,
            output_dir=args.output_dir
        )
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        for metric, value in results["overall_metrics"].items():
            print(f"{metric}: {value:.4f}")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
