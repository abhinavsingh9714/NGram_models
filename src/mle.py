"""
Maximum Likelihood Estimation (MLE) N-gram model implementation.
Implements probability computation and sentence log probability calculation.
"""

from typing import Tuple, Dict, List
import math

from ngram_model import NGramCounter


class MLEModel:
    """Maximum Likelihood Estimation N-gram model."""
    
    def __init__(self, ngram_counter: NGramCounter, n: int):
        """
        Initialize MLE model.
        
        Args:
            ngram_counter: Pre-computed N-gram counts
            n: N-gram order (1 for unigram, 2 for bigram, etc.)
        """
        self.ngram_counter = ngram_counter
        self.n = n
        self.vocab_size = ngram_counter.get_vocabulary_size()
        
        if n > ngram_counter.max_n:
            raise ValueError(f"N-gram order {n} exceeds maximum order {ngram_counter.max_n}")
    
    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Get probability of an N-gram using MLE.
        
        Args:
            ngram: N-gram tuple (w1, w2, ..., wn)
            
        Returns:
            Probability P(wn | w1, w2, ..., wn-1)
        """
        if len(ngram) != self.n:
            raise ValueError(f"Expected {self.n}-gram, got {len(ngram)}-gram")
        
        if self.n == 1:
            # Unigram probability: P(w) = C(w) / total_words
            count = self.ngram_counter.get_ngram_count(ngram)
            total_words = sum(self.ngram_counter.ngram_counts[1].values())
            return count / total_words if total_words > 0 else 0.0
        else:
            # Conditional probability: P(wn | w1, ..., wn-1) = C(w1, ..., wn) / C(w1, ..., wn-1)
            ngram_count = self.ngram_counter.get_ngram_count(ngram)
            context = ngram[:-1]
            context_count = self.ngram_counter.get_context_count(context)
            
            if context_count == 0:
                return 0.0  # Zero probability for unseen contexts
            return ngram_count / context_count
    
    def get_log_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Get log probability of an N-gram.
        
        Args:
            ngram: N-gram tuple
            
        Returns:
            Log probability (returns -inf for zero probabilities)
        """
        prob = self.get_probability(ngram)
        if prob == 0.0:
            return float('-inf')
        return math.log(prob)
    
    def calculate_sentence_log_prob(self, sentence: List[str]) -> Tuple[float, bool]:
        """
        Calculate log probability of a sentence.
        
        Args:
            sentence: List of tokens
            
        Returns:
            Tuple of (log probability, has_zero_prob)
        """
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
            'type': 'MLE',
            'n': self.n,
            'vocab_size': self.vocab_size,
            'total_ngrams': len(self.ngram_counter.ngram_counts[self.n])
        }

def main():
    """Test the N-gram model functionality."""
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    train_sentences, vocab = loader.load_training_data()
    
    # Count N-grams
    counter = NGramCounter(max_n=4)
    counter.count_ngrams(train_sentences)
    
    # Test MLE models
    for n in range(1, 5):
        print(f"\nTesting {n}-gram MLE model:")
        model = MLEModel(counter, n)
        
        # Test on first sentence
        test_sentence = train_sentences[0]
        log_prob, has_zero = model.calculate_sentence_log_prob(test_sentence)
        
        print(f"  Sentence: {' '.join(test_sentence)}")
        print(f"  Log probability: {log_prob}")
        print(f"  Has zero probability: {has_zero}")
        print(f"  Model info: {model.get_model_info()}")


if __name__ == "__main__":
    main()
