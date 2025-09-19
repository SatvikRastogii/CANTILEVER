"""
Data preprocessing and filtering pipeline for image captioning datasets.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import pandas as pd
import numpy as np
from collections import defaultdict
import re
import hashlib
from tqdm import tqdm
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import clip

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Image preprocessing utilities."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def preprocess_image(self, image_path: Union[str, Path]) -> Optional[Image.Image]:
        """Preprocess a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Resize while maintaining aspect ratio
            image.thumbnail(self.target_size, Image.Resampling.LANCZOS)
            
            # Create a new image with target size and paste the resized image
            new_image = Image.new('RGB', self.target_size, (0, 0, 0))
            new_image.paste(image, ((self.target_size[0] - image.width) // 2,
                                  (self.target_size[1] - image.height) // 2))
            
            return new_image
            
        except Exception as e:
            logger.warning(f"Failed to preprocess image {image_path}: {e}")
            return None
    
    def is_valid_image(self, image_path: Union[str, Path]) -> bool:
        """Check if image is valid."""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    def get_image_hash(self, image_path: Union[str, Path]) -> str:
        """Get hash of image for deduplication."""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""


class CaptionPreprocessor:
    """Caption preprocessing utilities."""
    
    def __init__(self):
        self.pii_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Names
            r'\b\d{3}-\d{3}-\d{4}\b',        # Phone numbers
            r'\b\d{5}\b',                     # ZIP codes
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
        ]
    
    def clean_caption(self, caption: str) -> str:
        """Clean and normalize caption."""
        # Remove HTML tags
        caption = re.sub(r'<[^>]+>', '', caption)
        
        # Remove extra whitespace
        caption = re.sub(r'\s+', ' ', caption).strip()
        
        # Remove PII
        for pattern in self.pii_patterns:
            caption = re.sub(pattern, '[REDACTED]', caption)
        
        return caption
    
    def is_valid_caption(self, caption: str, min_length: int = 5, max_length: int = 200) -> bool:
        """Check if caption is valid."""
        if not caption or len(caption.strip()) < min_length:
            return False
        
        if len(caption) > max_length:
            return False
        
        # Check for too many special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:]', caption)) / len(caption)
        if special_char_ratio > 0.3:
            return False
        
        return True
    
    def detect_language(self, text: str) -> str:
        """Simple language detection (English vs non-English)."""
        # Count English characters
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u00C0-\u017F]', text))
        
        if total_chars == 0:
            return 'unknown'
        
        english_ratio = english_chars / total_chars
        return 'en' if english_ratio > 0.8 else 'non-en'


class CLIPFilter:
    """CLIP-based filtering for image-caption pairs."""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
    
    def compute_similarity(self, image_path: Union[str, Path], caption: str) -> float:
        """Compute CLIP similarity between image and caption."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize caption
            text_input = clip.tokenize([caption], truncate=True).to(self.device)
            
            # Get features
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)
                
                # Normalize features
                image_features = F.normalize(image_features, p=2, dim=1)
                text_features = F.normalize(text_features, p=2, dim=1)
                
                # Compute similarity
                similarity = torch.cosine_similarity(image_features, text_features, dim=1)
                
            return similarity.item()
            
        except Exception as e:
            logger.warning(f"Failed to compute CLIP similarity: {e}")
            return 0.0
    
    def filter_pairs(self, data: List[Dict[str, Any]], threshold: float = 0.2) -> List[Dict[str, Any]]:
        """Filter image-caption pairs based on CLIP similarity."""
        filtered_data = []
        
        for item in tqdm(data, desc="Filtering with CLIP"):
            similarity = self.compute_similarity(item['image_path'], item['caption'])
            if similarity >= threshold:
                item['clip_similarity'] = similarity
                filtered_data.append(item)
        
        logger.info(f"Filtered {len(data)} -> {len(filtered_data)} pairs (threshold: {threshold})")
        return filtered_data


class QualityFilter:
    """Quality-based filtering for captions."""
    
    def __init__(self):
        self.quality_keywords = [
            'beautiful', 'amazing', 'stunning', 'gorgeous', 'wonderful',
            'incredible', 'spectacular', 'breathtaking', 'magnificent'
        ]
        
        self.low_quality_patterns = [
            r'^[a-z\s]*$',  # All lowercase
            r'^[A-Z\s]*$',  # All uppercase
            r'^\d+$',       # Only numbers
            r'^[^\w\s]*$',  # Only special characters
        ]
    
    def compute_quality_score(self, caption: str) -> float:
        """Compute quality score for caption."""
        score = 0.0
        
        # Length score (prefer medium length)
        length = len(caption.split())
        if 5 <= length <= 20:
            score += 0.3
        elif 3 <= length <= 30:
            score += 0.2
        
        # Quality keywords
        caption_lower = caption.lower()
        quality_count = sum(1 for keyword in self.quality_keywords if keyword in caption_lower)
        score += min(quality_count * 0.1, 0.3)
        
        # Punctuation score
        if caption.endswith(('.', '!', '?')):
            score += 0.1
        
        # Capitalization score
        if caption[0].isupper():
            score += 0.1
        
        # Penalize low quality patterns
        for pattern in self.low_quality_patterns:
            if re.match(pattern, caption):
                score -= 0.5
                break
        
        return max(0.0, min(1.0, score))
    
    def filter_by_quality(self, data: List[Dict[str, Any]], threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Filter captions by quality score."""
        filtered_data = []
        
        for item in data:
            quality_score = self.compute_quality_score(item['caption'])
            if quality_score >= threshold:
                item['quality_score'] = quality_score
                filtered_data.append(item)
        
        logger.info(f"Quality filtered {len(data)} -> {len(filtered_data)} captions (threshold: {threshold})")
        return filtered_data


class DatasetProcessor:
    """Main dataset processing pipeline."""
    
    def __init__(
        self,
        image_dir: Union[str, Path],
        output_dir: Union[str, Path],
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_preprocessor = ImagePreprocessor(target_size)
        self.caption_preprocessor = CaptionPreprocessor()
        self.quality_filter = QualityFilter()
        
        # Initialize CLIP filter if available
        try:
            self.clip_filter = CLIPFilter()
        except Exception as e:
            logger.warning(f"Failed to initialize CLIP filter: {e}")
            self.clip_filter = None
    
    def process_dataset(
        self,
        data: List[Dict[str, Any]],
        dataset_name: str,
        use_clip_filtering: bool = True,
        clip_threshold: float = 0.2,
        quality_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """Process a dataset through the full pipeline."""
        logger.info(f"Processing {len(data)} samples for {dataset_name}")
        
        # Step 1: Basic validation
        valid_data = self._validate_data(data)
        logger.info(f"After validation: {len(valid_data)} samples")
        
        # Step 2: Image preprocessing
        processed_data = self._preprocess_images(valid_data)
        logger.info(f"After image preprocessing: {len(processed_data)} samples")
        
        # Step 3: Caption preprocessing
        processed_data = self._preprocess_captions(processed_data)
        logger.info(f"After caption preprocessing: {len(processed_data)} samples")
        
        # Step 4: Quality filtering
        processed_data = self.quality_filter.filter_by_quality(processed_data, quality_threshold)
        
        # Step 5: CLIP filtering
        if use_clip_filtering and self.clip_filter:
            processed_data = self.clip_filter.filter_pairs(processed_data, clip_threshold)
        
        # Step 6: Deduplication
        processed_data = self._deduplicate(processed_data)
        logger.info(f"After deduplication: {len(processed_data)} samples")
        
        # Step 7: Split into train/val/test
        splits = self._create_splits(processed_data)
        
        # Step 8: Save processed data
        self._save_processed_data(splits, dataset_name)
        
        return {
            'total_samples': len(processed_data),
            'splits': {k: len(v) for k, v in splits.items()},
            'dataset_name': dataset_name
        }
    
    def _validate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate data samples."""
        valid_data = []
        
        for item in data:
            # Check required fields
            if 'image_path' not in item or 'caption' not in item:
                continue
            
            # Check if image exists
            image_path = self.image_dir / item['image_path']
            if not image_path.exists():
                continue
            
            # Check if image is valid
            if not self.image_preprocessor.is_valid_image(image_path):
                continue
            
            # Check if caption is valid
            if not self.caption_preprocessor.is_valid_caption(item['caption']):
                continue
            
            valid_data.append(item)
        
        return valid_data
    
    def _preprocess_images(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess images."""
        processed_data = []
        
        for item in data:
            image_path = self.image_dir / item['image_path']
            
            # Get image hash for deduplication
            image_hash = self.image_preprocessor.get_image_hash(image_path)
            item['image_hash'] = image_hash
            
            processed_data.append(item)
        
        return processed_data
    
    def _preprocess_captions(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess captions."""
        processed_data = []
        
        for item in data:
            # Clean caption
            cleaned_caption = self.caption_preprocessor.clean_caption(item['caption'])
            
            # Check if cleaned caption is still valid
            if self.caption_preprocessor.is_valid_caption(cleaned_caption):
                item['caption'] = cleaned_caption
                item['language'] = self.caption_preprocessor.detect_language(cleaned_caption)
                processed_data.append(item)
        
        return processed_data
    
    def _deduplicate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate samples."""
        seen_hashes = set()
        unique_data = []
        
        for item in data:
            image_hash = item.get('image_hash', '')
            caption = item['caption'].lower().strip()
            
            # Create a combined hash
            combined_hash = hashlib.md5(f"{image_hash}_{caption}".encode()).hexdigest()
            
            if combined_hash not in seen_hashes:
                seen_hashes.add(combined_hash)
                unique_data.append(item)
        
        return unique_data
    
    def _create_splits(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Split data into train/val/test."""
        # Shuffle data
        np.random.shuffle(data)
        
        # Split ratios
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
        
        n_samples = len(data)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        splits = {
            'train': data[:n_train],
            'val': data[n_train:n_train + n_val],
            'test': data[n_train + n_val:]
        }
        
        return splits
    
    def _save_processed_data(self, splits: Dict[str, List[Dict[str, Any]]], dataset_name: str) -> None:
        """Save processed data to files."""
        for split_name, split_data in splits.items():
            output_file = self.output_dir / f"{dataset_name}_{split_name}.json"
            
            with open(output_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            
            logger.info(f"Saved {len(split_data)} samples to {output_file}")


def process_coco_dataset(
    coco_path: Union[str, Path],
    image_dir: Union[str, Path],
    output_dir: Union[str, Path]
) -> Dict[str, Any]:
    """Process MS COCO dataset."""
    processor = DatasetProcessor(image_dir, output_dir)
    
    # Load COCO data
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)
    
    # Convert to our format
    images = {img['id']: img for img in coco_data['images']}
    captions_by_image = defaultdict(list)
    
    for ann in coco_data['annotations']:
        captions_by_image[ann['image_id']].append(ann['caption'])
    
    data = []
    for image_id, image_info in images.items():
        if image_id in captions_by_image:
            for caption in captions_by_image[image_id]:
                data.append({
                    'image_path': image_info['file_name'],
                    'caption': caption,
                    'image_id': image_id
                })
    
    return processor.process_dataset(data, "coco")


def process_conceptual_captions_dataset(
    cc_path: Union[str, Path],
    image_dir: Union[str, Path],
    output_dir: Union[str, Path],
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Process Conceptual Captions dataset."""
    processor = DatasetProcessor(image_dir, output_dir)
    
    data = []
    with open(cc_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            
            image_url, caption = parts
            image_filename = image_url.split('/')[-1]
            
            data.append({
                'image_path': image_filename,
                'caption': caption,
                'image_url': image_url
            })
    
    return processor.process_dataset(data, "conceptual_captions")


if __name__ == "__main__":
    # Test the preprocessing pipeline
    processor = DatasetProcessor("data/images", "data/processed")
    
    # Create dummy data
    dummy_data = [
        {"image_path": "test1.jpg", "caption": "A beautiful cat sitting on a chair"},
        {"image_path": "test2.jpg", "caption": "A dog running in the park"},
        {"image_path": "test3.jpg", "caption": "Invalid caption"},  # Too short
    ]
    
    # Process data
    result = processor.process_dataset(dummy_data, "test")
    print(f"Processing result: {result}")
