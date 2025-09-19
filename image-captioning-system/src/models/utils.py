"""
Model utilities for loading, saving, and configuration.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import json
import os
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from .baseline import BaselineCaptionModel
from .production import ProductionCaptionModel

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for image captioning models."""
    
    # Model architecture
    model_type: str = "baseline"  # "baseline" or "production"
    vocab_size: int = 10000
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ff_dim: int = 2048
    max_len: int = 100
    dropout: float = 0.1
    
    # Production model specific
    clip_model_name: str = "ViT-B/32"
    freeze_clip: bool = False
    use_clip_scoring: bool = True
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 100
    warmup_steps: int = 1000
    
    # Inference
    beam_size: int = 1
    temperature: float = 1.0
    max_length: int = 50
    
    # Special tokens
    pad_token_id: int = 0
    start_token_id: int = 1
    end_token_id: int = 2
    unk_token_id: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save config to file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ModelConfig':
        """Load config from file."""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def create_model(config: ModelConfig) -> nn.Module:
    """
    Create a model based on configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized model
    """
    if config.model_type == "baseline":
        model = BaselineCaptionModel(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ff_dim=config.ff_dim,
            max_len=config.max_len,
            dropout=config.dropout
        )
    elif config.model_type == "production":
        model = ProductionCaptionModel(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ff_dim=config.ff_dim,
            max_len=config.max_len,
            dropout=config.dropout,
            model_name=config.clip_model_name,
            freeze_clip=config.freeze_clip,
            use_clip_scoring=config.use_clip_scoring
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    return model


def load_model(
    model_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    device: str = "cpu"
) -> tuple[nn.Module, ModelConfig]:
    """
    Load a trained model and its configuration.
    
    Args:
        model_path: Path to the model checkpoint
        config_path: Path to the config file (optional)
        device: Device to load the model on
        
    Returns:
        Tuple of (model, config)
    """
    model_path = Path(model_path)
    
    # Load config
    if config_path is None:
        config_path = model_path.parent / "config.json"
    
    config = ModelConfig.load(config_path)
    
    # Create model
    model = create_model(config)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Model type: {config.model_type}")
    logger.info(f"Model size: {model.get_model_size()}")
    
    return model, config


def save_model(
    model: nn.Module,
    config: ModelConfig,
    save_path: Union[str, Path],
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a trained model and its configuration.
    
    Args:
        model: The model to save
        config: Model configuration
        save_path: Path to save the model
        additional_info: Additional information to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'model_type': config.model_type
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    # Save model
    torch.save(checkpoint, save_path)
    
    # Save config separately
    config_path = save_path.parent / "config.json"
    config.save(config_path)
    
    logger.info(f"Saved model to {save_path}")
    logger.info(f"Saved config to {config_path}")


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive information about a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with model information
    """
    info = {
        'model_class': model.__class__.__name__,
        'model_size': model.get_model_size(),
        'device': next(model.parameters()).device,
        'dtype': next(model.parameters()).dtype,
        'is_training': model.training
    }
    
    # Add model-specific info
    if hasattr(model, 'vocab_size'):
        info['vocab_size'] = model.vocab_size
    if hasattr(model, 'embed_dim'):
        info['embed_dim'] = model.embed_dim
    if hasattr(model, 'max_len'):
        info['max_len'] = model.max_len
    
    return info


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: The model to count parameters for
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def freeze_parameters(model: nn.Module, freeze_encoder: bool = True, freeze_decoder: bool = False) -> None:
    """
    Freeze model parameters.
    
    Args:
        model: The model to freeze parameters for
        freeze_encoder: Whether to freeze encoder parameters
        freeze_decoder: Whether to freeze decoder parameters
    """
    if freeze_encoder and hasattr(model, 'image_encoder'):
        for param in model.image_encoder.parameters():
            param.requires_grad = False
        logger.info("Frozen encoder parameters")
    
    if freeze_decoder and hasattr(model, 'text_decoder'):
        for param in model.text_decoder.parameters():
            param.requires_grad = False
        logger.info("Frozen decoder parameters")


def unfreeze_parameters(model: nn.Module) -> None:
    """
    Unfreeze all model parameters.
    
    Args:
        model: The model to unfreeze parameters for
    """
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Unfrozen all parameters")


def get_default_configs() -> Dict[str, ModelConfig]:
    """
    Get default configurations for different model types.
    
    Returns:
        Dictionary of default configurations
    """
    configs = {
        'baseline_small': ModelConfig(
            model_type='baseline',
            embed_dim=256,
            num_heads=4,
            num_layers=3,
            ff_dim=1024
        ),
        'baseline_medium': ModelConfig(
            model_type='baseline',
            embed_dim=512,
            num_heads=8,
            num_layers=6,
            ff_dim=2048
        ),
        'baseline_large': ModelConfig(
            model_type='baseline',
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            ff_dim=3072
        ),
        'production_small': ModelConfig(
            model_type='production',
            embed_dim=256,
            num_heads=4,
            num_layers=3,
            ff_dim=1024,
            clip_model_name='ViT-B/16'
        ),
        'production_medium': ModelConfig(
            model_type='production',
            embed_dim=512,
            num_heads=8,
            num_layers=6,
            ff_dim=2048,
            clip_model_name='ViT-B/32'
        ),
        'production_large': ModelConfig(
            model_type='production',
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            ff_dim=3072,
            clip_model_name='ViT-L/14'
        )
    }
    
    return configs


# Predefined model configurations
DEFAULT_CONFIGS = get_default_configs()


if __name__ == "__main__":
    # Test the utilities
    config = ModelConfig()
    print(f"Default config: {config}")
    
    # Test model creation
    model = create_model(config)
    print(f"Created model: {model.__class__.__name__}")
    
    # Test model info
    info = get_model_info(model)
    print(f"Model info: {info}")
    
    # Test parameter counting
    param_counts = count_parameters(model)
    print(f"Parameter counts: {param_counts}")
    
    # Test default configs
    default_configs = get_default_configs()
    print(f"Available default configs: {list(default_configs.keys())}")
