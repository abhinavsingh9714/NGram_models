"""
Smoothing techniques for N-gram language models.
Implements Add-1 smoothing
"""

import math
from typing import List, Dict, Tuple
from ngram_model import NGramCounter
from linear_interpolation import LinearInterpolationModel
from stupid_backoff import StupidBackoffModel


class Add1SmoothingModel:
    """Add-1 (Laplace) smoothing for N-gram models."""
    
    def __init__(self, ngram_counter: NGramCounter, n: int = 3):
        """
        Initialize Add-1 smoothing model.
        
        Args:
            ngram_counter: Pre-computed N-gram counts
            n: N-gram order (default: 3 for trigram)
        """
        self.ngram_counter = ngram_counter
        self.n = n
        self.vocab_size = ngram_counter.get_vocabulary_size()
        
        if n > ngram_counter.max_n:
            raise ValueError(f"N-gram order {n} exceeds maximum order {ngram_counter.max_n}")
    
    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Get probability using Add-1 smoothing.
        
        Formula: P(wn | w1, ..., wn-1) = (C(w1, ..., wn) + 1) / (C(w1, ..., wn-1) + V)
        
        Args:
            ngram: N-gram tuple
            
        Returns:
            Smoothed probability
        """
        if len(ngram) != self.n:
            raise ValueError(f"Expected {self.n}-gram, got {len(ngram)}-gram")
        
        if self.n == 1:
            # Unigram: P(w) = (C(w) + 1) / (total_words + V)
            count = self.ngram_counter.get_ngram_count(ngram)
            total_words = sum(self.ngram_counter.ngram_counts[1].values())
            return (count + 1) / (total_words + self.vocab_size)
        else:
            # Conditional: P(wn | w1, ..., wn-1) = (C(w1, ..., wn) + 1) / (C(w1, ..., wn-1) + V)
            ngram_count = self.ngram_counter.get_ngram_count(ngram)
            context = ngram[:-1]
            context_count = self.ngram_counter.get_context_count(context)
            
            return (ngram_count + 1) / (context_count + self.vocab_size)
    
    def get_log_probability(self, ngram: Tuple[str, ...]) -> float:
        """Get log probability."""
        prob = self.get_probability(ngram)
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
        
        return total_log_prob, False  # Add-1 never has zero probabilities
    
    def get_model_info(self) -> Dict:
        """Get information about the model."""
        return {
            'type': 'Add-1 Smoothing',
            'n': self.n,
            'vocab_size': self.vocab_size,
            'total_ngrams': len(self.ngram_counter.ngram_counts[self.n])
        }

def main():
    """Test the smoothing models."""
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    train_sentences, vocab = loader.load_training_data()
    
    # Count N-grams
    counter = NGramCounter(max_n=3)
    counter.count_ngrams(train_sentences)
    
    # Test Add-1 smoothing
    print("Testing Add-1 Smoothing:")
    add1_model = Add1SmoothingModel(counter, n=3)
    test_sentence = train_sentences[0]
    log_prob, has_zero = add1_model.calculate_sentence_log_prob(test_sentence)
    print(f"  Log probability: {log_prob}")
    print(f"  Model info: {add1_model.get_model_info()}")
    
    # Test Linear Interpolation
    print("\nTesting Linear Interpolation:")
    lambdas = [0.1, 0.3, 0.6]
    interp_model = LinearInterpolationModel(counter, lambdas, n=3)
    log_prob, has_zero = interp_model.calculate_sentence_log_prob(test_sentence)
    print(f"  Log probability: {log_prob}")
    print(f"  Model info: {interp_model.get_model_info()}")
    
    # Test Stupid Backoff
    print("\nTesting Stupid Backoff:")
    backoff_model = StupidBackoffModel(counter, alpha=0.4, n=3)
    log_prob, has_zero = backoff_model.calculate_sentence_log_prob(test_sentence)
    print(f"  Log probability: {log_prob}")
    print(f"  Model info: {backoff_model.get_model_info()}")
    
    # Test lambda combinations
    print("\nLambda combinations for grid search:")
    combinations = LinearInterpolationModel.generate_lambda_combinations(n=3, step=0.1)
    print(f"  Found {len(combinations)} combinations")
    for i, combo in enumerate(combinations[:5]):  # Show first 5
        print(f"  {i+1}: {combo}")


if __name__ == "__main__":
    main()
