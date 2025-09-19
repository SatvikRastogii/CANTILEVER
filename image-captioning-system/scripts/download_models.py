"""
Script to download pre-trained models and datasets.
"""

import argparse
import os
import sys
from pathlib import Path
import logging
import requests
import zipfile
import tarfile
from tqdm import tqdm
import torch
import clip

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.utils import get_default_configs, create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, destination: Path, description: str = "Downloading") -> None:
    """Download a file with progress bar."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def download_clip_models() -> None:
    """Download CLIP models."""
    logger.info("Downloading CLIP models...")
    
    models_dir = Path("models/clip")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download CLIP models
    clip_models = ["ViT-B/32", "ViT-B/16", "ViT-L/14"]
    
    for model_name in clip_models:
        try:
            logger.info(f"Downloading CLIP model: {model_name}")
            model, preprocess = clip.load(model_name, device="cpu")
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")


def download_coco_dataset() -> None:
    """Download MS COCO dataset."""
    logger.info("Downloading MS COCO dataset...")
    
    data_dir = Path("data/coco")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO dataset URLs
    urls = {
        "train_images": "http://images.cocodataset.org/zips/train2017.zip",
        "val_images": "http://images.cocodataset.org/zips/val2017.zip",
        "test_images": "http://images.cocodataset.org/zips/test2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    }
    
    for name, url in urls.items():
        filename = url.split("/")[-1]
        filepath = data_dir / filename
        
        if filepath.exists():
            logger.info(f"{name} already exists, skipping...")
            continue
        
        try:
            logger.info(f"Downloading {name}...")
            download_file(url, filepath, f"Downloading {name}")
            
            # Extract if it's a zip file
            if filename.endswith('.zip'):
                logger.info(f"Extracting {filename}...")
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                filepath.unlink()  # Remove zip file after extraction
            
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")


def download_conceptual_captions() -> None:
    """Download Conceptual Captions dataset."""
    logger.info("Downloading Conceptual Captions dataset...")
    
    data_dir = Path("data/conceptual_captions")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Note: Conceptual Captions requires manual download from Google Research
    # This is a placeholder for the download process
    logger.info("Conceptual Captions dataset requires manual download from:")
    logger.info("https://ai.google.com/research/ConceptualCaptions/")
    logger.info("Please download the dataset manually and place it in data/conceptual_captions/")


def create_sample_models() -> None:
    """Create sample models for testing."""
    logger.info("Creating sample models...")
    
    models_dir = Path("models/sample")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample configurations
    configs = get_default_configs()
    
    for config_name, config in configs.items():
        try:
            logger.info(f"Creating sample model: {config_name}")
            
            # Create model
            model = create_model(config)
            
            # Save model
            model_path = models_dir / f"{config_name}.pt"
            torch.save(model.state_dict(), model_path)
            
            # Save config
            config_path = models_dir / f"{config_name}_config.json"
            config.save(config_path)
            
            logger.info(f"Saved {config_name} model to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to create {config_name} model: {e}")


def setup_directories() -> None:
    """Create necessary directories."""
    logger.info("Setting up directories...")
    
    directories = [
        "data",
        "data/coco",
        "data/conceptual_captions",
        "models",
        "models/clip",
        "models/sample",
        "logs",
        "outputs",
        "configs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def main():
    parser = argparse.ArgumentParser(description="Download models and datasets")
    parser.add_argument("--models", action="store_true", help="Download models")
    parser.add_argument("--datasets", action="store_true", help="Download datasets")
    parser.add_argument("--coco", action="store_true", help="Download COCO dataset")
    parser.add_argument("--conceptual-captions", action="store_true", help="Download Conceptual Captions")
    parser.add_argument("--clip", action="store_true", help="Download CLIP models")
    parser.add_argument("--sample-models", action="store_true", help="Create sample models")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--setup-dirs", action="store_true", help="Setup directories only")
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    if args.setup_dirs:
        logger.info("Directory setup completed")
        return
    
    if args.all or args.models:
        if args.all or args.clip:
            download_clip_models()
        
        if args.all or args.sample_models:
            create_sample_models()
    
    if args.all or args.datasets:
        if args.all or args.coco:
            download_coco_dataset()
        
        if args.all or args.conceptual_captions:
            download_conceptual_captions()
    
    logger.info("Download completed!")


if __name__ == "__main__":
    main()
