"""
Safety and content moderation utilities for image captioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import re
import json
from pathlib import Path
import cv2
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import requests
from io import BytesIO

logger = logging.getLogger(__name__)


class NSFWDetector:
    """NSFW content detection for images."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load NSFW detection model
        try:
            self.nsfw_classifier = pipeline(
                "image-classification",
                model="microsoft/DialoGPT-medium",  # Placeholder - use actual NSFW model
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            logger.warning(f"Failed to load NSFW classifier: {e}")
            self.nsfw_classifier = None
    
    def detect_nsfw(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Dict[str, Any]:
        """
        Detect NSFW content in image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with NSFW detection results
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert('RGB')
            
            # For now, use a simple heuristic-based approach
            # In production, use a proper NSFW detection model
            nsfw_score = self._heuristic_nsfw_detection(image)
            
            return {
                'is_nsfw': nsfw_score > 0.5,
                'nsfw_score': nsfw_score,
                'confidence': abs(nsfw_score - 0.5) * 2
            }
            
        except Exception as e:
            logger.error(f"NSFW detection failed: {e}")
            return {
                'is_nsfw': False,
                'nsfw_score': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _heuristic_nsfw_detection(self, image: Image.Image) -> float:
        """Simple heuristic-based NSFW detection."""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Check for skin tone pixels (simplified)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Define skin tone range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin tones
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pixel_ratio = np.sum(skin_mask > 0) / (image.width * image.height)
        
        # Simple heuristic: if more than 30% of pixels are skin tone, flag as potentially NSFW
        nsfw_score = min(skin_pixel_ratio * 2, 1.0)
        
        return nsfw_score


class PIIDetector:
    """PII (Personally Identifiable Information) detection in text."""
    
    def __init__(self):
        # PII patterns
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'address': r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b',
            'zip_code': r'\b\d{5}(?:-\d{4})?\b',
            'name': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'url': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
        }
        
        # Compile patterns
        self.compiled_patterns = {name: re.compile(pattern, re.IGNORECASE) 
                                for name, pattern in self.patterns.items()}
    
    def detect_pii(self, text: str) -> Dict[str, Any]:
        """
        Detect PII in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with PII detection results
        """
        detected_pii = {}
        total_matches = 0
        
        for pii_type, pattern in self.compiled_patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected_pii[pii_type] = matches
                total_matches += len(matches)
        
        return {
            'has_pii': total_matches > 0,
            'pii_types': list(detected_pii.keys()),
            'pii_matches': detected_pii,
            'total_matches': total_matches
        }
    
    def remove_pii(self, text: str, replacement: str = "[REDACTED]") -> str:
        """
        Remove PII from text.
        
        Args:
            text: Input text
            replacement: Replacement string for PII
            
        Returns:
            Text with PII removed
        """
        cleaned_text = text
        
        for pii_type, pattern in self.compiled_patterns.items():
            cleaned_text = pattern.sub(replacement, cleaned_text)
        
        return cleaned_text


class ContentModerator:
    """Content moderation for captions."""
    
    def __init__(self):
        # Load toxicity classifier
        try:
            self.toxicity_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Failed to load toxicity classifier: {e}")
            self.toxicity_classifier = None
        
        # Define inappropriate content patterns
        self.inappropriate_patterns = [
            r'\b(?:hate|violence|abuse|harassment)\b',
            r'\b(?:discrimination|racism|sexism)\b',
            r'\b(?:illegal|criminal|fraud)\b',
            r'\b(?:spam|scam|phishing)\b',
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                for pattern in self.inappropriate_patterns]
    
    def moderate_content(self, text: str) -> Dict[str, Any]:
        """
        Moderate content for inappropriate material.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with moderation results
        """
        results = {
            'is_appropriate': True,
            'toxicity_score': 0.0,
            'inappropriate_patterns': [],
            'warnings': []
        }
        
        # Check for inappropriate patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                results['inappropriate_patterns'].append(pattern.pattern)
                results['is_appropriate'] = False
        
        # Use toxicity classifier if available
        if self.toxicity_classifier:
            try:
                toxicity_result = self.toxicity_classifier(text)
                toxicity_score = max([r['score'] for r in toxicity_result 
                                    if r['label'] in ['TOXIC', 'SEVERE_TOXIC']], default=0.0)
                results['toxicity_score'] = toxicity_score
                
                if toxicity_score > 0.5:
                    results['is_appropriate'] = False
                    results['warnings'].append(f"High toxicity score: {toxicity_score:.2f}")
                    
            except Exception as e:
                logger.warning(f"Toxicity classification failed: {e}")
        
        return results


class BiasDetector:
    """Bias detection in captions."""
    
    def __init__(self):
        # Define bias-related keywords
        self.bias_keywords = {
            'gender': ['man', 'woman', 'male', 'female', 'boy', 'girl', 'guy', 'lady'],
            'race': ['white', 'black', 'asian', 'hispanic', 'latino', 'caucasian', 'african'],
            'age': ['young', 'old', 'elderly', 'teenager', 'adult', 'child', 'baby'],
            'appearance': ['beautiful', 'ugly', 'attractive', 'handsome', 'pretty', 'cute'],
            'profession': ['doctor', 'nurse', 'teacher', 'engineer', 'lawyer', 'artist'],
        }
        
        # Compile patterns
        self.bias_patterns = {}
        for category, keywords in self.bias_keywords.items():
            pattern = r'\b(?:' + '|'.join(keywords) + r')\b'
            self.bias_patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def detect_bias(self, text: str) -> Dict[str, Any]:
        """
        Detect potential bias in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with bias detection results
        """
        detected_bias = {}
        
        for category, pattern in self.bias_patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected_bias[category] = matches
        
        return {
            'has_bias': len(detected_bias) > 0,
            'bias_categories': list(detected_bias.keys()),
            'bias_matches': detected_bias
        }


class SafetyPipeline:
    """Complete safety pipeline for image captioning."""
    
    def __init__(self):
        self.nsfw_detector = NSFWDetector()
        self.pii_detector = PIIDetector()
        self.content_moderator = ContentModerator()
        self.bias_detector = BiasDetector()
    
    def process_image(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Dict[str, Any]:
        """
        Process image through safety pipeline.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with safety analysis results
        """
        results = {
            'is_safe': True,
            'warnings': [],
            'errors': []
        }
        
        # NSFW detection
        nsfw_result = self.nsfw_detector.detect_nsfw(image)
        if nsfw_result['is_nsfw']:
            results['is_safe'] = False
            results['warnings'].append(f"NSFW content detected (score: {nsfw_result['nsfw_score']:.2f})")
        
        results['nsfw_analysis'] = nsfw_result
        
        return results
    
    def process_caption(self, caption: str) -> Dict[str, Any]:
        """
        Process caption through safety pipeline.
        
        Args:
            caption: Input caption
            
        Returns:
            Dictionary with safety analysis results
        """
        results = {
            'is_safe': True,
            'warnings': [],
            'errors': [],
            'cleaned_caption': caption
        }
        
        # PII detection
        pii_result = self.pii_detector.detect_pii(caption)
        if pii_result['has_pii']:
            results['warnings'].append(f"PII detected: {pii_result['pii_types']}")
            results['cleaned_caption'] = self.pii_detector.remove_pii(caption)
        
        results['pii_analysis'] = pii_result
        
        # Content moderation
        moderation_result = self.content_moderator.moderate_content(caption)
        if not moderation_result['is_appropriate']:
            results['is_safe'] = False
            results['warnings'].extend(moderation_result['warnings'])
        
        results['moderation_analysis'] = moderation_result
        
        # Bias detection
        bias_result = self.bias_detector.detect_bias(caption)
        if bias_result['has_bias']:
            results['warnings'].append(f"Potential bias detected: {bias_result['bias_categories']}")
        
        results['bias_analysis'] = bias_result
        
        return results
    
    def process_image_caption_pair(
        self, 
        image: Union[Image.Image, np.ndarray, str, Path], 
        caption: str
    ) -> Dict[str, Any]:
        """
        Process image-caption pair through complete safety pipeline.
        
        Args:
            image: Input image
            caption: Input caption
            
        Returns:
            Dictionary with complete safety analysis
        """
        # Process image
        image_results = self.process_image(image)
        
        # Process caption
        caption_results = self.process_caption(caption)
        
        # Combine results
        combined_results = {
            'is_safe': image_results['is_safe'] and caption_results['is_safe'],
            'warnings': image_results['warnings'] + caption_results['warnings'],
            'errors': image_results['errors'] + caption_results['errors'],
            'image_analysis': image_results,
            'caption_analysis': caption_results,
            'cleaned_caption': caption_results['cleaned_caption']
        }
        
        return combined_results


def create_safety_pipeline() -> SafetyPipeline:
    """Create a safety pipeline instance."""
    return SafetyPipeline()


if __name__ == "__main__":
    # Test the safety pipeline
    pipeline = create_safety_pipeline()
    
    # Test PII detection
    test_text = "My name is John Doe and my email is john@example.com"
    pii_result = pipeline.pii_detector.detect_pii(test_text)
    print(f"PII detection: {pii_result}")
    
    # Test content moderation
    moderation_result = pipeline.content_moderator.moderate_content(test_text)
    print(f"Content moderation: {moderation_result}")
    
    # Test bias detection
    bias_result = pipeline.bias_detector.detect_bias("A beautiful woman in a red dress")
    print(f"Bias detection: {bias_result}")
    
    # Test complete pipeline
    complete_result = pipeline.process_caption(test_text)
    print(f"Complete pipeline result: {complete_result}")
