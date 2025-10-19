"""
Core N-gram model classes and probability computations.
Implements N-gram counting and Maximum Likelihood Estimation.
"""

import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Union


class NGramCounter:
    """Efficiently counts N-grams from training data."""
    
    def __init__(self, max_n: int = 4):
        """
        Initialize N-gram counter.
        
        Args:
            max_n: Maximum N-gram order to count (default: 4)
        """
        self.max_n = max_n
        self.ngram_counts: Dict[int, Dict[Tuple[str, ...], int]] = {}
        self.context_counts: Dict[int, Dict[Tuple[str, ...], int]] = {}
        
        # Initialize dictionaries for each N-gram order
        for n in range(1, max_n + 1):
            self.ngram_counts[n] = defaultdict(int)
            self.context_counts[n] = defaultdict(int)
    
    def count_ngrams(self, sentences: List[List[str]]) -> None:
        """
        Count all N-grams from sentences.
        
        Args:
            sentences: List of sentences, each sentence is a list of tokens
        """
        print(f"Counting N-grams up to order {self.max_n}...")
        
        for sentence in sentences:
            # Count N-grams for each order
            for n in range(1, self.max_n + 1):
                for i in range(len(sentence) - n + 1):
                    # Extract N-gram
                    ngram = tuple(sentence[i:i + n])
                    self.ngram_counts[n][ngram] += 1
                    
                    # Count context (first n-1 tokens) for conditional probabilities
                    if n > 1:
                        context = ngram[:-1]
                        self.context_counts[n][context] += 1
        
        # Convert defaultdicts to regular dicts for efficiency
        for n in range(1, self.max_n + 1):
            self.ngram_counts[n] = dict(self.ngram_counts[n])
            self.context_counts[n] = dict(self.context_counts[n])
        
        # Print statistics
        for n in range(1, self.max_n + 1):
            print(f"  {n}-grams: {len(self.ngram_counts[n]):,} unique")
    
    def get_ngram_count(self, ngram: Tuple[str, ...]) -> int:
        """Get count for a specific N-gram."""
        n = len(ngram)
        if n > self.max_n:
            return 0
        return self.ngram_counts[n].get(ngram, 0)
    
    def get_context_count(self, context: Tuple[str, ...]) -> int:
        """Get count for a specific context."""
        n = len(context) + 1
        if n > self.max_n:
            return 0
        return self.context_counts[n].get(context, 0)
    
    def get_vocabulary_size(self) -> int:
        """Get vocabulary size from unigram counts."""
        return len(self.ngram_counts[1])

