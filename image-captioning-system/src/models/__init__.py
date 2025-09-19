"""
Image Captioning Models Package

This package contains the model architectures for the image captioning system:
- Baseline: CNN + Transformer for fast prototyping
- Production: ViT + Transformer with CLIP integration for high quality
"""

from .baseline import BaselineCaptionModel
from .production import ProductionCaptionModel
from .utils import ModelConfig, load_model, save_model

__all__ = [
    "BaselineCaptionModel",
    "ProductionCaptionModel", 
    "ModelConfig",
    "load_model",
    "save_model"
]
