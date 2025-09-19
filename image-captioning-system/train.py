"""
Training script for image captioning models.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import yaml
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Import our modules
from src.models.utils import ModelConfig, create_model, save_model, get_default_configs
from src.data.dataset import create_dataset, Tokenizer, get_transforms
from src.evaluation.metrics import CaptionEvaluator
from src.utils.logging_config import setup_logging, metrics_logger

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class Trainer:
    """Training class for image captioning models."""
    
    def __init__(self, config: ModelConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        
        # Initialize model
        self.model = create_model(config)
        self.model.to(device)
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(vocab_size=config.vocab_size)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # Initialize evaluator
        self.evaluator = CaptionEvaluator(device=device)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_history = []
        
    def prepare_data(self, data_config: Dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare training, validation, and test data loaders."""
        # Get transforms
        train_transform = get_transforms("train")
        val_transform = get_transforms("val")
        
        # Create datasets
        train_dataset = create_dataset(
            dataset_type=data_config["dataset_type"],
            data_path=data_config["train_path"],
            image_dir=data_config["image_dir"],
            tokenizer=self.tokenizer,
            transform=train_transform,
            max_length=self.config.max_len,
            split="train"
        )
        
        val_dataset = create_dataset(
            dataset_type=data_config["dataset_type"],
            data_path=data_config["val_path"],
            image_dir=data_config["image_dir"],
            tokenizer=self.tokenizer,
            transform=val_transform,
            max_length=self.config.max_len,
            split="val"
        )
        
        test_dataset = create_dataset(
            dataset_type=data_config["dataset_type"],
            data_path=data_config["test_path"],
            image_dir=data_config["image_dir"],
            tokenizer=self.tokenizer,
            transform=val_transform,
            max_length=self.config.max_len,
            split="test"
        )
        
        # Fit tokenizer on training data
        train_texts = [item["caption"] for item in train_dataset.data]
        self.tokenizer.fit(train_texts)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch["images"].to(self.device)
            captions = batch["captions"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, captions)
            
            # Compute loss
            loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
                outputs["logits"].view(-1, outputs["logits"].size(-1)),
                outputs["target"].view(-1)
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            current_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log training metrics
            if batch_idx % 100 == 0:
                metrics_logger.log_training_metrics(
                    epoch=self.current_epoch,
                    step=batch_idx,
                    loss=loss.item(),
                    learning_rate=self.optimizer.param_groups[0]['lr']
                )
        
        return {"train_loss": total_loss / num_batches}
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move data to device
                images = batch["images"].to(self.device)
                captions = batch["captions"].to(self.device)
                caption_texts = batch["caption_texts"]
                
                # Forward pass
                outputs = self.model(images, captions)
                
                # Compute loss
                loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
                    outputs["logits"].view(-1, outputs["logits"].size(-1)),
                    outputs["target"].view(-1)
                )
                total_loss += loss.item()
                
                # Generate predictions for evaluation
                pred_outputs = self.model(images)
                predictions = pred_outputs["captions"]
                
                # Decode predictions
                for pred in predictions:
                    pred_text = self.tokenizer.decode(pred.cpu().numpy())
                    all_predictions.append(pred_text)
                
                # Store references
                for ref_text in caption_texts:
                    all_references.append([ref_text])
        
        # Compute validation metrics
        val_loss = total_loss / len(val_loader)
        
        # Evaluate captions
        if all_predictions and all_references:
            eval_metrics = self.evaluator.evaluate_batch(all_predictions, all_references)
        else:
            eval_metrics = {}
        
        return {
            "val_loss": val_loss,
            **eval_metrics
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, save_dir: Path) -> None:
        """Train the model."""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics["epoch"] = epoch
            epoch_metrics["learning_rate"] = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.training_history.append(epoch_metrics)
            metrics_logger.log_model_metrics(
                model_type=self.config.model_type,
                metrics=epoch_metrics,
                dataset="coco",
                split="train"
            )
            
            # Log to wandb if available
            if wandb.run:
                wandb.log(epoch_metrics)
            
            # Save checkpoint if best
            current_metric = val_metrics.get("bleu_4", 0.0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.save_checkpoint(save_dir / "best_model.pt", epoch_metrics)
                logger.info(f"New best model saved with BLEU-4: {current_metric:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(save_dir / f"checkpoint_epoch_{epoch}.pt", epoch_metrics)
            
            logger.info(f"Epoch {epoch}: {epoch_metrics}")
        
        # Save final model
        self.save_checkpoint(save_dir / "final_model.pt", epoch_metrics)
        logger.info("Training completed!")
    
    def save_checkpoint(self, path: Path, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
            "metrics": metrics,
            "epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "training_history": self.training_history
        }
        
        torch.save(checkpoint, path)
        
        # Save tokenizer
        tokenizer_path = path.parent / "tokenizer.json"
        self.tokenizer.save(tokenizer_path)
        
        logger.info(f"Checkpoint saved to {path}")


def load_config(config_path: str) -> ModelConfig:
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return ModelConfig.from_dict(config_dict)


def main():
    parser = argparse.ArgumentParser(description="Train image captioning model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data-config", type=str, required=True, help="Path to data config file")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--wandb-project", type=str, help="Weights & Biases project name")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config: {config.model_type}")
    
    # Load data configuration
    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if specified
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=config.to_dict(),
            name=f"{config.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Initialize trainer
    trainer = Trainer(config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.current_epoch = checkpoint["epoch"]
        trainer.best_metric = checkpoint["best_metric"]
        trainer.training_history = checkpoint["training_history"]
        logger.info(f"Resumed from epoch {trainer.current_epoch}")
    
    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(data_config)
    logger.info(f"Data loaded: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches")
    
    # Train
    trainer.train(train_loader, val_loader, output_dir)
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.validate_epoch(test_loader)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save final results
    results = {
        "config": config.to_dict(),
        "data_config": data_config,
        "test_metrics": test_metrics,
        "training_history": trainer.training_history,
        "best_metric": trainer.best_metric
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    if wandb.run:
        wandb.finish()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
