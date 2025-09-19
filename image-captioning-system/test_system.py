"""
Test script to verify the image captioning system is working correctly.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from src.models.baseline import BaselineCaptionModel
        from src.models.production import ProductionCaptionModel
        from src.models.utils import ModelConfig, create_model
        from src.data.dataset import Tokenizer, ImageCaptionDataset
        from src.utils.safety import SafetyPipeline
        from src.evaluation.metrics import CaptionEvaluator
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    logger.info("Testing model creation...")
    
    try:
        from src.models.utils import ModelConfig, create_model
        
        # Test baseline model
        config = ModelConfig(model_type="baseline", vocab_size=1000, embed_dim=256)
        model = create_model(config)
        logger.info(f"✓ Baseline model created: {model.__class__.__name__}")
        
        # Test production model
        config = ModelConfig(model_type="production", vocab_size=1000, embed_dim=256)
        model = create_model(config)
        logger.info(f"✓ Production model created: {model.__class__.__name__}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        return False


def test_model_forward():
    """Test model forward pass."""
    logger.info("Testing model forward pass...")
    
    try:
        from src.models.utils import ModelConfig, create_model
        
        config = ModelConfig(model_type="baseline", vocab_size=1000, embed_dim=256)
        model = create_model(config)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        captions = torch.randint(0, 1000, (batch_size, 20))
        
        # Test training mode
        with torch.no_grad():
            outputs = model(images, captions)
            assert "logits" in outputs
            assert outputs["logits"].shape[0] == batch_size
            logger.info("✓ Training forward pass successful")
        
        # Test inference mode
        with torch.no_grad():
            outputs = model(images)
            assert "captions" in outputs
            logger.info("✓ Inference forward pass successful")
        
        return True
    except Exception as e:
        logger.error(f"✗ Model forward pass failed: {e}")
        return False


def test_tokenizer():
    """Test tokenizer functionality."""
    logger.info("Testing tokenizer...")
    
    try:
        from src.data.dataset import Tokenizer
        
        tokenizer = Tokenizer(vocab_size=1000)
        
        # Test fitting
        texts = ["A cat sitting on a chair", "A dog running in the park"]
        tokenizer.fit(texts)
        logger.info("✓ Tokenizer fitting successful")
        
        # Test encoding/decoding
        text = "A cat sitting on a chair"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        logger.info(f"✓ Encoding/decoding successful: '{text}' -> '{decoded}'")
        
        return True
    except Exception as e:
        logger.error(f"✗ Tokenizer test failed: {e}")
        return False


def test_safety_pipeline():
    """Test safety pipeline."""
    logger.info("Testing safety pipeline...")
    
    try:
        from src.utils.safety import SafetyPipeline
        
        pipeline = SafetyPipeline()
        
        # Test caption processing
        caption = "A beautiful woman in a red dress"
        result = pipeline.process_caption(caption)
        
        assert "is_safe" in result
        assert "cleaned_caption" in result
        logger.info("✓ Safety pipeline test successful")
        
        return True
    except Exception as e:
        logger.error(f"✗ Safety pipeline test failed: {e}")
        return False


def test_evaluator():
    """Test caption evaluator."""
    logger.info("Testing caption evaluator...")
    
    try:
        from src.evaluation.metrics import CaptionEvaluator
        
        evaluator = CaptionEvaluator()
        
        # Test single evaluation
        pred_caption = "A cat sitting on a chair"
        ref_captions = ["A cat is sitting on a chair", "There is a cat on the chair"]
        
        metrics = evaluator.evaluate_single(pred_caption, ref_captions)
        
        assert "bleu_4" in metrics
        assert "meteor" in metrics
        logger.info("✓ Caption evaluator test successful")
        
        return True
    except Exception as e:
        logger.error(f"✗ Caption evaluator test failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    logger.info("Testing configuration loading...")
    
    try:
        from src.models.utils import ModelConfig
        
        # Test default configs
        configs = ModelConfig()
        logger.info("✓ Default config creation successful")
        
        # Test config serialization
        config_dict = configs.to_dict()
        config_loaded = ModelConfig.from_dict(config_dict)
        logger.info("✓ Config serialization successful")
        
        return True
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting system tests...")
    
    tests = [
        test_imports,
        test_model_creation,
        test_model_forward,
        test_tokenizer,
        test_safety_pipeline,
        test_evaluator,
        test_config_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        logger.info("-" * 50)
    
    logger.info(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready to use.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
