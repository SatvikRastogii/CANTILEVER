"""
Baseline Image Captioning Model

A fast prototyping model using ResNet50 as image encoder and Transformer as text decoder.
This model is designed for quick experiments and low infrastructure requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ImageEncoder(nn.Module):
    """ResNet50-based image encoder."""
    
    def __init__(self, embed_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection to embed_dim
        self.projection = nn.Linear(2048, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)  # [B, 2048, H, W]
        
        # Global pooling
        pooled = self.global_pool(features)  # [B, 2048, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [B, 2048]
        
        # Project to embed_dim
        embedded = self.projection(pooled)  # [B, embed_dim]
        embedded = self.dropout(embedded)
        
        return embedded


class TransformerDecoder(nn.Module):
    """Transformer decoder for caption generation."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 2048,
        max_len: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        tgt: torch.Tensor, 
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Token embedding
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.embed_dim)
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
        tgt_emb = self.dropout(tgt_emb)
        
        # Transformer decoder
        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory.unsqueeze(1),  # Add sequence dimension
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Output projection
        logits = self.output_projection(output)
        
        return logits
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence."""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask


class BaselineCaptionModel(nn.Module):
    """Baseline image captioning model with ResNet50 + Transformer."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 2048,
        max_len: int = 100,
        dropout: float = 0.1,
        pretrained_encoder: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Image encoder
        self.image_encoder = ImageEncoder(embed_dim, pretrained_encoder)
        
        # Text decoder
        self.text_decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            max_len=max_len,
            dropout=dropout
        )
        
        # Special tokens
        self.pad_token_id = 0
        self.start_token_id = 1
        self.end_token_id = 2
        self.unk_token_id = 3
        
    def forward(
        self, 
        images: torch.Tensor, 
        captions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.
        
        Args:
            images: Input images [B, C, H, W]
            captions: Target captions [B, seq_len] (optional, for training)
            
        Returns:
            Dictionary containing logits and other outputs
        """
        batch_size = images.size(0)
        
        # Encode images
        image_features = self.image_encoder(images)  # [B, embed_dim]
        
        if captions is not None:
            # Training mode
            # Remove the last token from captions for input
            decoder_input = captions[:, :-1]
            # Remove the first token from captions for target
            decoder_target = captions[:, 1:]
            
            # Create mask for decoder input
            tgt_mask = self.text_decoder.generate_square_subsequent_mask(
                decoder_input.size(1)
            ).to(images.device)
            
            # Create padding mask
            tgt_key_padding_mask = (decoder_input == self.pad_token_id)
            
            # Decode
            logits = self.text_decoder(
                tgt=decoder_input,
                memory=image_features,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
            return {
                'logits': logits,
                'target': decoder_target,
                'image_features': image_features
            }
        else:
            # Inference mode - generate captions
            return self.generate(images)
    
    def generate(
        self, 
        images: torch.Tensor, 
        max_length: Optional[int] = None,
        beam_size: int = 1,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Generate captions for images.
        
        Args:
            images: Input images [B, C, H, W]
            max_length: Maximum caption length
            beam_size: Beam search size
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing generated captions and scores
        """
        if max_length is None:
            max_length = self.max_len
            
        batch_size = images.size(0)
        device = images.device
        
        # Encode images
        image_features = self.image_encoder(images)  # [B, embed_dim]
        
        if beam_size == 1:
            # Greedy decoding
            return self._greedy_decode(image_features, max_length, temperature)
        else:
            # Beam search
            return self._beam_search(image_features, max_length, beam_size)
    
    def _greedy_decode(
        self, 
        image_features: torch.Tensor, 
        max_length: int,
        temperature: float
    ) -> Dict[str, torch.Tensor]:
        """Greedy decoding for caption generation."""
        batch_size = image_features.size(0)
        device = image_features.device
        
        # Initialize with start token
        generated = torch.full(
            (batch_size, 1), 
            self.start_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            if finished.all():
                break
                
            # Get logits for current sequence
            logits = self.text_decoder(
                tgt=generated,
                memory=image_features
            )  # [B, seq_len, vocab_size]
            
            # Get logits for last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample next token
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Update finished sequences
            finished = finished | (next_token == self.end_token_id)
            
            # Append next token
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        
        return {
            'captions': generated,
            'image_features': image_features
        }
    
    def _beam_search(
        self, 
        image_features: torch.Tensor, 
        max_length: int,
        beam_size: int
    ) -> Dict[str, torch.Tensor]:
        """Beam search for caption generation."""
        batch_size = image_features.size(0)
        device = image_features.device
        
        # Initialize beams
        beams = [(torch.tensor([self.start_token_id], device=device), 0.0)]
        
        for _ in range(max_length - 1):
            new_beams = []
            
            for sequence, score in beams:
                if sequence[-1] == self.end_token_id:
                    new_beams.append((sequence, score))
                    continue
                
                # Get logits
                logits = self.text_decoder(
                    tgt=sequence.unsqueeze(0),
                    memory=image_features[0:1]  # Use first image for simplicity
                )
                
                # Get top-k tokens
                next_token_logits = logits[0, -1, :]
                top_k_logits, top_k_tokens = torch.topk(next_token_logits, beam_size)
                
                for token, logit in zip(top_k_tokens, top_k_logits):
                    new_sequence = torch.cat([sequence, token.unsqueeze(0)])
                    new_score = score + logit.item()
                    new_beams.append((new_sequence, new_score))
            
            # Keep top beam_size beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # Return best beam
        best_sequence, best_score = beams[0]
        
        return {
            'captions': best_sequence.unsqueeze(0),
            'scores': torch.tensor([best_score], device=device),
            'image_features': image_features
        }
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_parameters': sum(p.numel() for p in self.image_encoder.parameters()),
            'decoder_parameters': sum(p.numel() for p in self.text_decoder.parameters())
        }


def create_baseline_model(
    vocab_size: int = 10000,
    embed_dim: int = 512,
    **kwargs
) -> BaselineCaptionModel:
    """Create a baseline captioning model with default parameters."""
    return BaselineCaptionModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        **kwargs
    )


if __name__ == "__main__":
    # Test the model
    model = create_baseline_model()
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, 1000, (batch_size, 20))
    
    # Training mode
    outputs = model(images, captions)
    print(f"Training outputs: {list(outputs.keys())}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # Inference mode
    outputs = model(images)
    print(f"Inference outputs: {list(outputs.keys())}")
    print(f"Generated captions shape: {outputs['captions'].shape}")
    
    # Model size
    size_info = model.get_model_size()
    print(f"Model size: {size_info}")
