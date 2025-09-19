"""
Dataset classes for image captioning.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
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

logger = logging.getLogger(__name__)


class ImageCaptionDataset(Dataset):
    """Base dataset class for image captioning."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        image_dir: Union[str, Path],
        tokenizer,
        transform: Optional[transforms.Compose] = None,
        max_length: int = 100,
        split: str = "train"
    ):
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.split = split
        
        # Load data
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file."""
        if self.data_path.suffix == '.json':
            with open(self.data_path, 'r') as f:
                data = json.load(f)
        elif self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.data[idx]
        
        # Load image
        image_path = self.image_dir / sample['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Tokenize caption
        caption = sample['caption']
        caption_tokens = self.tokenizer.encode(caption)
        
        # Truncate or pad to max_length
        if len(caption_tokens) > self.max_length:
            caption_tokens = caption_tokens[:self.max_length]
        else:
            caption_tokens.extend([self.tokenizer.pad_token_id] * (self.max_length - len(caption_tokens)))
        
        return {
            'image': image,
            'caption': torch.tensor(caption_tokens, dtype=torch.long),
            'caption_text': caption,
            'image_path': str(image_path)
        }


class COCODataset(ImageCaptionDataset):
    """MS COCO dataset for image captioning."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        image_dir: Union[str, Path],
        tokenizer,
        transform: Optional[transforms.Compose] = None,
        max_length: int = 100,
        split: str = "train"
    ):
        super().__init__(data_path, image_dir, tokenizer, transform, max_length, split)
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load COCO format data."""
        with open(self.data_path, 'r') as f:
            coco_data = json.load(f)
        
        # Create image id to filename mapping
        images = {img['id']: img for img in coco_data['images']}
        
        # Group captions by image id
        captions_by_image = defaultdict(list)
        for ann in coco_data['annotations']:
            captions_by_image[ann['image_id']].append(ann['caption'])
        
        # Create dataset samples
        data = []
        for image_id, image_info in images.items():
            if image_id in captions_by_image:
                for caption in captions_by_image[image_id]:
                    data.append({
                        'image_path': image_info['file_name'],
                        'caption': caption,
                        'image_id': image_id
                    })
        
        return data


class ConceptualCaptionsDataset(ImageCaptionDataset):
    """Conceptual Captions dataset."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        image_dir: Union[str, Path],
        tokenizer,
        transform: Optional[transforms.Compose] = None,
        max_length: int = 100,
        split: str = "train"
    ):
        super().__init__(data_path, image_dir, tokenizer, transform, max_length, split)
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load Conceptual Captions format data."""
        data = []
        
        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse TSV format: image_url\tcaption
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    continue
                
                image_url, caption = parts
                
                # Extract image filename from URL
                image_filename = image_url.split('/')[-1]
                
                data.append({
                    'image_path': image_filename,
                    'caption': caption,
                    'image_url': image_url
                })
        
        return data


class Tokenizer:
    """Simple tokenizer for captions."""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.unk_token = '<UNK>'
        
        self.pad_token_id = 0
        self.start_token_id = 1
        self.end_token_id = 2
        self.unk_token_id = 3
        
        # Initialize with special tokens
        self.word_to_idx = {
            self.pad_token: self.pad_token_id,
            self.start_token: self.start_token_id,
            self.end_token: self.end_token_id,
            self.unk_token: self.unk_token_id
        }
        
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        
        self._is_fitted = False
    
    def fit(self, texts: List[str]) -> None:
        """Fit tokenizer on texts."""
        word_counts = defaultdict(int)
        
        for text in texts:
            words = self._preprocess_text(text).split()
            for word in words:
                word_counts[word] += 1
        
        # Sort by frequency and take top vocab_size - 4 (for special tokens)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for word, count in sorted_words[:self.vocab_size - 4]:
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        self._is_fitted = True
        logger.info(f"Fitted tokenizer with {len(self.word_to_idx)} tokens")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for tokenization."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not self._is_fitted:
            raise ValueError("Tokenizer must be fitted before encoding")
        
        words = self._preprocess_text(text).split()
        token_ids = [self.start_token_id]
        
        for word in words:
            if word in self.word_to_idx:
                token_ids.append(self.word_to_idx[word])
            else:
                token_ids.append(self.unk_token_id)
        
        token_ids.append(self.end_token_id)
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        words = []
        
        for token_id in token_ids:
            if token_id in self.idx_to_word:
                word = self.idx_to_word[token_id]
                if word not in [self.pad_token, self.start_token, self.end_token]:
                    words.append(word)
        
        return ' '.join(words)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump({
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'vocab_size': self.vocab_size
            }, f, indent=2)
    
    def load(self, path: Union[str, Path]) -> None:
        """Load tokenizer from file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = {int(k): v for k, v in data['idx_to_word'].items()}
        self.vocab_size = data['vocab_size']
        self._is_fitted = True


def get_transforms(split: str = "train", image_size: int = 224) -> transforms.Compose:
    """Get image transforms for training or validation."""
    if split == "train":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Create a DataLoader for the dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for batching samples."""
    images = torch.stack([item['image'] for item in batch])
    captions = torch.stack([item['caption'] for item in batch])
    
    return {
        'images': images,
        'captions': captions,
        'caption_texts': [item['caption_text'] for item in batch],
        'image_paths': [item['image_path'] for item in batch]
    }


def create_dataset(
    dataset_type: str,
    data_path: Union[str, Path],
    image_dir: Union[str, Path],
    tokenizer: Tokenizer,
    split: str = "train",
    **kwargs
) -> Dataset:
    """Create a dataset of the specified type."""
    if dataset_type == "coco":
        return COCODataset(data_path, image_dir, tokenizer, split=split, **kwargs)
    elif dataset_type == "conceptual_captions":
        return ConceptualCaptionsDataset(data_path, image_dir, tokenizer, split=split, **kwargs)
    else:
        return ImageCaptionDataset(data_path, image_dir, tokenizer, split=split, **kwargs)


if __name__ == "__main__":
    # Test the dataset
    tokenizer = Tokenizer()
    
    # Create dummy data for testing
    dummy_data = [
        {"image_path": "test1.jpg", "caption": "A cat sitting on a chair"},
        {"image_path": "test2.jpg", "caption": "A dog running in the park"}
    ]
    
    # Fit tokenizer
    texts = [item["caption"] for item in dummy_data]
    tokenizer.fit(texts)
    
    # Test encoding/decoding
    test_text = "A cat sitting on a chair"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Test transforms
    transform = get_transforms("train")
    print(f"Train transforms: {transform}")
    
    transform = get_transforms("val")
    print(f"Val transforms: {transform}")
