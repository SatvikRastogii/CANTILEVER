"""
Evaluation metrics for image captioning models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import json
from pathlib import Path
import re
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import clip
from PIL import Image
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")


class CaptionEvaluator:
    """Comprehensive caption evaluation metrics."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        
        # Initialize CLIP model for CLIPScore
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
        except Exception as e:
            logger.warning(f"Failed to load CLIP model: {e}")
            self.clip_model = None
        
        # Smoothing function for BLEU
        self.smoothing = SmoothingFunction().method1
    
    def evaluate_single(
        self,
        predicted_caption: str,
        reference_captions: List[str],
        image: Optional[Union[Image.Image, str, Path]] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single predicted caption against reference captions.
        
        Args:
            predicted_caption: Generated caption
            reference_captions: List of reference captions
            image: Input image (optional, for CLIPScore)
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # Tokenize captions
        pred_tokens = word_tokenize(predicted_caption.lower())
        ref_tokens_list = [word_tokenize(ref.lower()) for ref in reference_captions]
        
        # BLEU scores
        metrics['bleu_1'] = sentence_bleu(ref_tokens_list, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smoothing)
        metrics['bleu_2'] = sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing)
        metrics['bleu_3'] = sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoothing)
        metrics['bleu_4'] = sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing)
        
        # METEOR score
        try:
            metrics['meteor'] = meteor_score(ref_tokens_list, pred_tokens)
        except Exception as e:
            logger.warning(f"METEOR calculation failed: {e}")
            metrics['meteor'] = 0.0
        
        # ROUGE-L (simplified)
        metrics['rouge_l'] = self._rouge_l(pred_tokens, ref_tokens_list[0])
        
        # CIDEr (simplified)
        metrics['cider'] = self._cider(pred_tokens, ref_tokens_list)
        
        # SPICE (simplified)
        metrics['spice'] = self._spice(pred_tokens, ref_tokens_list)
        
        # CLIPScore
        if image is not None and self.clip_model is not None:
            metrics['clip_score'] = self._clip_score(image, predicted_caption)
        else:
            metrics['clip_score'] = 0.0
        
        return metrics
    
    def evaluate_batch(
        self,
        predicted_captions: List[str],
        reference_captions_list: List[List[str]],
        images: Optional[List[Union[Image.Image, str, Path]]] = None
    ) -> Dict[str, float]:
        """
        Evaluate a batch of predicted captions.
        
        Args:
            predicted_captions: List of generated captions
            reference_captions_list: List of reference caption lists
            images: List of input images (optional)
            
        Returns:
            Dictionary of average metric scores
        """
        all_metrics = []
        
        for i, (pred, refs) in enumerate(zip(predicted_captions, reference_captions_list)):
            image = images[i] if images else None
            metrics = self.evaluate_single(pred, refs, image)
            all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def _rouge_l(self, pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """Calculate ROUGE-L score."""
        def lcs_length(seq1, seq2):
            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i-1] == seq2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        lcs = lcs_length(pred_tokens, ref_tokens)
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _cider(self, pred_tokens: List[str], ref_tokens_list: List[List[str]]) -> float:
        """Calculate CIDEr score (simplified)."""
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        def compute_tf_idf(ngrams, all_ngrams):
            tf = Counter(ngrams)
            idf = {}
            for ngram in tf:
                idf[ngram] = sum(1 for doc in all_ngrams if ngram in doc)
            
            tf_idf = {}
            for ngram, count in tf.items():
                tf_idf[ngram] = count * np.log(len(all_ngrams) / idf[ngram])
            
            return tf_idf
        
        # Get n-grams
        pred_ngrams = get_ngrams(pred_tokens, 4)  # 4-grams
        ref_ngrams_list = [get_ngrams(ref, 4) for ref in ref_tokens_list]
        
        # Compute TF-IDF
        all_ngrams = [pred_ngrams] + ref_ngrams_list
        pred_tf_idf = compute_tf_idf(pred_ngrams, all_ngrams)
        
        # Compute similarity with each reference
        similarities = []
        for ref_ngrams in ref_ngrams_list:
            ref_tf_idf = compute_tf_idf(ref_ngrams, all_ngrams)
            
            # Cosine similarity
            common_ngrams = set(pred_tf_idf.keys()) & set(ref_tf_idf.keys())
            if not common_ngrams:
                similarities.append(0.0)
                continue
            
            numerator = sum(pred_tf_idf[ngram] * ref_tf_idf[ngram] for ngram in common_ngrams)
            pred_norm = np.sqrt(sum(v**2 for v in pred_tf_idf.values()))
            ref_norm = np.sqrt(sum(v**2 for v in ref_tf_idf.values()))
            
            if pred_norm == 0 or ref_norm == 0:
                similarities.append(0.0)
            else:
                similarities.append(numerator / (pred_norm * ref_norm))
        
        return np.mean(similarities)
    
    def _spice(self, pred_tokens: List[str], ref_tokens_list: List[List[str]]) -> float:
        """Calculate SPICE score (simplified)."""
        # Simplified SPICE implementation
        # In practice, use the official SPICE implementation
        
        def extract_objects(tokens):
            # Simple object extraction
            objects = set()
            for token in tokens:
                if token in ['person', 'man', 'woman', 'child', 'dog', 'cat', 'car', 'tree', 'house']:
                    objects.add(token)
            return objects
        
        pred_objects = extract_objects(pred_tokens)
        ref_objects_list = [extract_objects(ref) for ref in ref_tokens_list]
        
        # Compute F1 score
        precisions = []
        recalls = []
        
        for ref_objects in ref_objects_list:
            if len(pred_objects) == 0:
                precisions.append(0.0)
            else:
                precisions.append(len(pred_objects & ref_objects) / len(pred_objects))
            
            if len(ref_objects) == 0:
                recalls.append(0.0)
            else:
                recalls.append(len(pred_objects & ref_objects) / len(ref_objects))
        
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        
        if avg_precision + avg_recall == 0:
            return 0.0
        
        return 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    
    def _clip_score(self, image: Union[Image.Image, str, Path], caption: str) -> float:
        """Calculate CLIPScore."""
        try:
            # Load image if needed
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, str) and image.startswith('http'):
                response = requests.get(image)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Preprocess image
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize caption
            text_input = clip.tokenize([caption], truncate=True).to(self.device)
            
            # Get features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
                
                # Normalize features
                image_features = F.normalize(image_features, p=2, dim=1)
                text_features = F.normalize(text_features, p=2, dim=1)
                
                # Compute similarity
                similarity = torch.cosine_similarity(image_features, text_features, dim=1)
                
            return similarity.item()
            
        except Exception as e:
            logger.warning(f"CLIPScore calculation failed: {e}")
            return 0.0


class ModelEvaluator:
    """Model evaluation utilities."""
    
    def __init__(self, evaluator: CaptionEvaluator):
        self.evaluator = evaluator
    
    def evaluate_model(
        self,
        model,
        dataloader,
        tokenizer,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Evaluate a model on a dataset.
        
        Args:
            model: The model to evaluate
            dataloader: DataLoader with test data
            tokenizer: Tokenizer for decoding
            device: Device to run evaluation on
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        all_predictions = []
        all_references = []
        all_images = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(device)
                captions = batch['captions'].to(device)
                caption_texts = batch['caption_texts']
                
                # Generate predictions
                outputs = model.generate(images)
                predictions = outputs['captions']
                
                # Decode predictions
                for pred in predictions:
                    pred_text = tokenizer.decode(pred.cpu().numpy())
                    all_predictions.append(pred_text)
                
                # Store references
                for ref_text in caption_texts:
                    all_references.append([ref_text])  # Single reference per image
                
                # Store images for CLIPScore
                for img in images:
                    all_images.append(img)
        
        # Evaluate
        metrics = self.evaluator.evaluate_batch(all_predictions, all_references, all_images)
        
        return metrics
    
    def compare_models(
        self,
        models: Dict[str, Any],
        dataloader,
        tokenizer,
        device: str = "cpu"
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of model_name -> model
            dataloader: DataLoader with test data
            tokenizer: Tokenizer for decoding
            device: Device to run evaluation on
            
        Returns:
            Dictionary of model_name -> metrics
        """
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating model: {name}")
            metrics = self.evaluate_model(model, dataloader, tokenizer, device)
            results[name] = metrics
        
        return results


class HumanEvaluation:
    """Human evaluation utilities."""
    
    def __init__(self):
        self.evaluation_criteria = [
            'correctness',
            'relevance',
            'fluency',
            'completeness',
            'safety'
        ]
    
    def create_evaluation_form(
        self,
        image_path: str,
        predicted_caption: str,
        reference_captions: List[str],
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a human evaluation form.
        
        Args:
            image_path: Path to the image
            predicted_caption: Generated caption
            reference_captions: Reference captions
            criteria: Evaluation criteria
            
        Returns:
            Evaluation form data
        """
        if criteria is None:
            criteria = self.evaluation_criteria
        
        form = {
            'image_path': image_path,
            'predicted_caption': predicted_caption,
            'reference_captions': reference_captions,
            'criteria': criteria,
            'ratings': {},  # To be filled by human evaluators
            'comments': '',
            'timestamp': None
        }
        
        return form
    
    def aggregate_human_scores(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Aggregate human evaluation scores.
        
        Args:
            evaluations: List of human evaluations
            
        Returns:
            Aggregated scores
        """
        if not evaluations:
            return {}
        
        # Get all criteria
        all_criteria = set()
        for eval_data in evaluations:
            all_criteria.update(eval_data['ratings'].keys())
        
        # Aggregate scores
        aggregated = {}
        for criterion in all_criteria:
            scores = [eval_data['ratings'].get(criterion, 0) for eval_data in evaluations]
            aggregated[criterion] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'count': len(scores)
            }
        
        return aggregated


def create_evaluator(device: str = "cpu") -> CaptionEvaluator:
    """Create a caption evaluator instance."""
    return CaptionEvaluator(device)


if __name__ == "__main__":
    # Test the evaluator
    evaluator = create_evaluator()
    
    # Test single evaluation
    pred_caption = "A cat sitting on a chair"
    ref_captions = ["A cat is sitting on a chair", "There is a cat on the chair"]
    
    metrics = evaluator.evaluate_single(pred_caption, ref_captions)
    print(f"Single evaluation metrics: {metrics}")
    
    # Test batch evaluation
    pred_captions = [pred_caption, "A dog running in the park"]
    ref_captions_list = [ref_captions, ["A dog is running in the park", "There is a dog running"]]
    
    batch_metrics = evaluator.evaluate_batch(pred_captions, ref_captions_list)
    print(f"Batch evaluation metrics: {batch_metrics}")
