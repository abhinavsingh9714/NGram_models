"""
Stupid Backoff algorithm for N-gram models.
Implements probability computation and sentence log probability calculation.
"""
from typing import Tuple, Dict, List
import math

from ngram_model import NGramCounter
from mle import MLEModel


class StupidBackoffModel:
    """Stupid Backoff algorithm for N-gram models."""
    
    def __init__(self, ngram_counter: NGramCounter, alpha: float = 0.4, n: int = 3):
        """
        Initialize Stupid Backoff model.
        
        Args:
            ngram_counter: Pre-computed N-gram counts
            alpha: Backoff factor (default: 0.4)
            n: Maximum N-gram order (default: 3 for trigram)
        """
        self.ngram_counter = ngram_counter
        self.n = n
        self.alpha = alpha
        self.vocab_size = ngram_counter.get_vocabulary_size()
        
        # Create MLE models for each order
        self.mle_models = {}
        for i in range(1, n + 1):
            self.mle_models[i] = MLEModel(ngram_counter, i)
    
    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Get probability using Stupid Backoff.
        
        Formula: 
        - If trigram seen: P(wn | w1, w2)
        - Else if bigram seen: α * P(wn | w2)  
        - Else: α² * P(wn)
        
        Args:
            ngram: N-gram tuple
            
        Returns:
            Backoff probability (not normalized)
        """
        if len(ngram) != self.n:
            raise ValueError(f"Expected {self.n}-gram, got {len(ngram)}-gram")
        
        # Try trigram first
        if self.n >= 3:
            trigram_count = self.ngram_counter.get_ngram_count(ngram)
            if trigram_count > 0:
                return self.mle_models[3].get_probability(ngram)
        
        # Try bigram
        if self.n >= 2:
            bigram = ngram[-2:]
            bigram_count = self.ngram_counter.get_ngram_count(bigram)
            if bigram_count > 0:
                return self.alpha * self.mle_models[2].get_probability(bigram)
        
        # Fall back to unigram
        unigram = (ngram[-1],)
        return (self.alpha ** (self.n - 1)) * self.mle_models[1].get_probability(unigram)
    
    def get_log_probability(self, ngram: Tuple[str, ...]) -> float:
        """Get log probability."""
        prob = self.get_probability(ngram)
        if prob == 0.0:
            return float('-inf')
        return math.log(prob)
    
    def calculate_sentence_log_prob(self, sentence: List[str]) -> Tuple[float, bool]:
        """Calculate log probability of a sentence."""
        if len(sentence) < self.n:
            return 0.0, False
        
        total_log_prob = 0.0
        
        for i in range(len(sentence) - self.n + 1):
            ngram = tuple(sentence[i:i + self.n])
            log_prob = self.get_log_probability(ngram)
            total_log_prob += log_prob
        
        return total_log_prob, False  # Stupid Backoff never has zero probabilities
    
    def get_model_info(self) -> Dict:
        """Get information about the model."""
        return {
            'type': 'Stupid Backoff',
            'n': self.n,
            'alpha': self.alpha,
            'vocab_size': self.vocab_size,
            'total_ngrams': len(self.ngram_counter.ngram_counts[self.n])
        }