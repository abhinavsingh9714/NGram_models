"""
Linear interpolation of multiple N-gram models.
Implements probability computation and sentence log probability calculation.
"""
from typing import List, Dict, Tuple
import math
import itertools

from ngram_model import NGramCounter
from mle import MLEModel


class LinearInterpolationModel:
    """Linear interpolation of multiple N-gram models."""
    
    def __init__(self, ngram_counter: NGramCounter, lambdas: List[float], n: int = 3):
        """
        Initialize linear interpolation model.
        
        Args:
            ngram_counter: Pre-computed N-gram counts
            lambdas: Interpolation weights [λ1, λ2, λ3] for [unigram, bigram, trigram]
            n: Maximum N-gram order (default: 3 for trigram)
        """
        self.ngram_counter = ngram_counter
        self.n = n
        self.lambdas = lambdas
        self.vocab_size = ngram_counter.get_vocabulary_size()
        
        # Validate lambdas
        if len(lambdas) != n:
            raise ValueError(f"Expected {n} lambda values, got {len(lambdas)}")
        if abs(sum(lambdas) - 1.0) > 1e-6:
            raise ValueError(f"Lambdas must sum to 1, got {sum(lambdas)}")
        
        # Create MLE models for each order
        self.mle_models = {}
        for i in range(1, n + 1):
            self.mle_models[i] = MLEModel(ngram_counter, i)
    
    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Get probability using linear interpolation.
        
        Formula: P(wn | w1, ..., wn-1) = Σ λi * Pi(wn | context_i)
        
        Args:
            ngram: N-gram tuple
            
        Returns:
            Interpolated probability
        """
        if len(ngram) != self.n:
            raise ValueError(f"Expected {self.n}-gram, got {len(ngram)}-gram")
        
        total_prob = 0.0
        
        # Interpolate from unigram to n-gram
        for i in range(1, self.n + 1):
            if i == 1:
                # Unigram: P(w)
                unigram = (ngram[-1],)
                prob = self.mle_models[1].get_probability(unigram)
            else:
                # Higher order: P(wn | w1, ..., wn-1)
                context_ngram = ngram[-(i):]
                prob = self.mle_models[i].get_probability(context_ngram)
            
            total_prob += self.lambdas[i-1] * prob
        
        return total_prob
    
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
        has_zero_prob = False
        
        for i in range(len(sentence) - self.n + 1):
            ngram = tuple(sentence[i:i + self.n])
            log_prob = self.get_log_probability(ngram)
            
            if log_prob == float('-inf'):
                has_zero_prob = True
                return float('-inf'), True
            
            total_log_prob += log_prob
        
        return total_log_prob, has_zero_prob
    
    def get_model_info(self) -> Dict:
        """Get information about the model."""
        return {
            'type': 'Linear Interpolation',
            'n': self.n,
            'lambdas': self.lambdas,
            'vocab_size': self.vocab_size,
            'total_ngrams': len(self.ngram_counter.ngram_counts[self.n])
        }
    
    @staticmethod
    def generate_lambda_combinations(n: int = 3, step: float = 0.1) -> List[List[float]]:
        """
        Generate lambda combinations for grid search.
        
        Args:
            n: Number of lambda values
            step: Step size for grid search
            
        Returns:
            List of lambda combinations that sum to 1
        """
        combinations = []
        
        # Generate all combinations with step size
        for combo in itertools.product([round(i * step, 1) for i in range(1, int(1/step))], repeat=n):
            if abs(sum(combo) - 1.0) < 1e-6:
                combinations.append(list(combo))
        
        return combinations