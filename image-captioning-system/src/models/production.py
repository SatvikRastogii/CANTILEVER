"""
Production Image Captioning Model

A high-quality model using Vision Transformer (ViT) as image encoder and 
Transformer decoder with CLIP integration for superior performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union
import logging
from transformers import CLIPModel, CLIPProcessor
import clip

logger = logging.getLogger(__name__)


class ViTImageEncoder(nn.Module):
    """Vision Transformer-based image encoder with CLIP integration."""
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        embed_dim: int = 512,
        freeze_clip: bool = False,
        use_clip_features: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_clip_features = use_clip_features
        
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load(model_name, device="cpu")
        
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Get CLIP's vision encoder
        self.vision_encoder = self.clip_model.visual
        
        # Projection layer to match embed_dim
        clip_dim = self.vision_encoder.output_dim
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the vision encoder.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Image features [B, embed_dim]
        """
        # Extract features using CLIP's vision encoder
        if self.use_clip_features:
            # Use CLIP's global features
            features = self.vision_encoder(x)  # [B, clip_dim]
        else:
            # Use patch features (if needed for more detailed features)
            x = self.vision_encoder.conv1(x)  # [B, embed_dim, H', W']
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, embed_dim, H'*W']
            x = x.permute(0, 2, 1)  # [B, H'*W', embed_dim]
            
            # Add class token
            class_token = self.vision_encoder.class_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([class_token, x], dim=1)
            
            # Add positional embeddings
            x = x + self.vision_encoder.positional_embedding
            
            # Apply transformer layers
            x = self.vision_encoder.ln_pre(x)
            x = x.permute(1, 0, 2)  # [seq_len, B, embed_dim]
            x = self.vision_encoder.transformer(x)
            x = x.permute(1, 0, 2)  # [B, seq_len, embed_dim]
            
            # Use class token
            features = x[:, 0, :]  # [B, embed_dim]
        
        # Project to target dimension
        features = self.projection(features)
        features = self.layer_norm(features)
        
        return features


class CrossModalAttention(nn.Module):
    """Cross-modal attention between image and text features."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self, 
        text_features: torch.Tensor, 
        image_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-modal attention.
        
        Args:
            text_features: Text features [B, seq_len, embed_dim]
            image_features: Image features [B, embed_dim]
            attention_mask: Attention mask [B, seq_len]
            
        Returns:
            Attended text features [B, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = text_features.shape
        
        # Project to Q, K, V
        q = self.q_proj(text_features)  # [B, seq_len, embed_dim]
        k = self.k_proj(image_features.unsqueeze(1))  # [B, 1, embed_dim]
        v = self.v_proj(image_features.unsqueeze(1))  # [B, 1, embed_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, seq_len, 1]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, seq_len, 1]
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, seq_len, head_dim]
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(attn_output)
        
        return output


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


class EnhancedTransformerDecoder(nn.Module):
    """Enhanced transformer decoder with cross-modal attention."""
    
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
        
        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(embed_dim, num_heads, dropout)
        
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
        image_features: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Token embedding
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.embed_dim)
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
        tgt_emb = self.dropout(tgt_emb)
        
        # Cross-modal attention
        tgt_emb = self.cross_modal_attention(
            tgt_emb, 
            image_features,
            tgt_key_padding_mask
        )
        
        # Transformer decoder
        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=tgt_emb,  # Self-attention
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


class CLIPScore(nn.Module):
    """CLIP-based scoring for caption quality assessment."""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        super().__init__()
        self.clip_model, _ = clip.load(model_name, device="cpu")
        self.clip_model.eval()
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def forward(self, images: torch.Tensor, captions: List[str]) -> torch.Tensor:
        """
        Compute CLIP scores between images and captions.
        
        Args:
            images: Input images [B, C, H, W]
            captions: List of caption strings
            
        Returns:
            CLIP scores [B]
        """
        # Tokenize captions
        text_tokens = clip.tokenize(captions, truncate=True)
        
        # Move to same device as images
        text_tokens = text_tokens.to(images.device)
        
        # Get features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            text_features = self.clip_model.encode_text(text_tokens)
            
            # Normalize features
            image_features = F.normalize(image_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)
            
            # Compute similarity
            scores = torch.sum(image_features * text_features, dim=1)
            
        return scores


class ProductionCaptionModel(nn.Module):
    """Production image captioning model with ViT + Transformer + CLIP."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 2048,
        max_len: int = 100,
        dropout: float = 0.1,
        model_name: str = "ViT-B/32",
        freeze_clip: bool = False,
        use_clip_scoring: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.use_clip_scoring = use_clip_scoring
        
        # Image encoder
        self.image_encoder = ViTImageEncoder(
            model_name=model_name,
            embed_dim=embed_dim,
            freeze_clip=freeze_clip
        )
        
        # Text decoder
        self.text_decoder = EnhancedTransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            max_len=max_len,
            dropout=dropout
        )
        
        # CLIP scorer
        if use_clip_scoring:
            self.clip_scorer = CLIPScore(model_name)
        
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
                image_features=image_features,
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
        temperature: float = 1.0,
        num_candidates: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Generate captions for images with CLIP re-ranking.
        
        Args:
            images: Input images [B, C, H, W]
            max_length: Maximum caption length
            beam_size: Beam search size
            temperature: Sampling temperature
            num_candidates: Number of candidates to generate
            
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
            # Beam search with CLIP re-ranking
            return self._beam_search_with_clip_rerank(
                images, image_features, max_length, beam_size, num_candidates
            )
    
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
                image_features=image_features
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
    
    def _beam_search_with_clip_rerank(
        self, 
        images: torch.Tensor,
        image_features: torch.Tensor, 
        max_length: int,
        beam_size: int,
        num_candidates: int
    ) -> Dict[str, torch.Tensor]:
        """Beam search with CLIP re-ranking."""
        batch_size = images.size(0)
        device = images.device
        
        all_candidates = []
        all_scores = []
        
        for batch_idx in range(batch_size):
            # Generate multiple candidates using beam search
            candidates = self._single_image_beam_search(
                image_features[batch_idx:batch_idx+1], 
                max_length, 
                beam_size * 2  # Generate more candidates
            )
            
            # Re-rank using CLIP if available
            if self.use_clip_scoring and hasattr(self, 'clip_scorer'):
                # Convert token sequences to text (simplified)
                candidate_texts = [self._tokens_to_text(cand) for cand in candidates]
                
                # Compute CLIP scores
                clip_scores = self.clip_scorer(
                    images[batch_idx:batch_idx+1].repeat(len(candidates), 1, 1, 1),
                    candidate_texts
                )
                
                # Sort by CLIP score
                sorted_indices = torch.argsort(clip_scores, descending=True)
                best_candidates = [candidates[i] for i in sorted_indices[:num_candidates]]
                best_scores = clip_scores[sorted_indices[:num_candidates]]
            else:
                best_candidates = candidates[:num_candidates]
                best_scores = torch.ones(num_candidates, device=device)
            
            all_candidates.append(best_candidates[0])  # Take best candidate
            all_scores.append(best_scores[0])
        
        # Stack results
        final_captions = torch.stack(all_candidates)
        final_scores = torch.stack(all_scores)
        
        return {
            'captions': final_captions,
            'scores': final_scores,
            'image_features': image_features
        }
    
    def _single_image_beam_search(
        self, 
        image_features: torch.Tensor, 
        max_length: int,
        beam_size: int
    ) -> List[torch.Tensor]:
        """Beam search for a single image."""
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
                    image_features=image_features
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
        
        # Return all sequences
        return [beam[0] for beam in beams]
    
    def _tokens_to_text(self, tokens: torch.Tensor) -> str:
        """Convert token sequence to text (simplified)."""
        # This is a simplified version - in practice, you'd use a proper tokenizer
        # Remove special tokens and convert to text
        text_tokens = tokens[(tokens != self.pad_token_id) & 
                           (tokens != self.start_token_id) & 
                           (tokens != self.end_token_id)]
        
        # Convert to text (simplified)
        return f"Generated caption with {len(text_tokens)} tokens"
    
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


def create_production_model(
    vocab_size: int = 10000,
    embed_dim: int = 512,
    **kwargs
) -> ProductionCaptionModel:
    """Create a production captioning model with default parameters."""
    return ProductionCaptionModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        **kwargs
    )


if __name__ == "__main__":
    # Test the model
    model = create_production_model()
    
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
