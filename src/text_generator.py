"""
Text generation utilities for N-gram language models.
Implements sampling-based text generation using probability distributions.
"""

import random
import math
from typing import List, Tuple, Dict
from collections import Counter

from ngram_model import NGramCounter
from stupid_backoff import StupidBackoffModel
from linear_interpolation import LinearInterpolationModel
from mle import MLEModel


class TextGenerator:
    """Text generator for N-gram language models."""
    
    def __init__(self, model, ngram_counter: NGramCounter):
        """
        Initialize text generator.
        
        Args:
            model: Trained N-gram model (MLE, StupidBackoff, or LinearInterpolation)
            ngram_counter: N-gram counter with vocabulary
        """
        self.model = model
        self.ngram_counter = ngram_counter
        # Extract vocabulary from unigram counts
        self.vocabulary = []
        for ngram in ngram_counter.ngram_counts[1].keys():
            if isinstance(ngram, tuple) and len(ngram) == 1:
                self.vocabulary.append(ngram[0])
            else:
                self.vocabulary.append(str(ngram))
        self.n = model.n if hasattr(model, 'n') else 3
        
    def get_next_word_probabilities(self, context: Tuple[str, ...]) -> Dict[str, float]:
        """
        Get probability distribution for next word given context.
        
        Args:
            context: Context tuple (last n-1 words)
            
        Returns:
            Dictionary mapping words to probabilities
        """
        probabilities = {}
        
        for word in self.vocabulary:
            # Create n-gram by combining context and word
            if len(context) == self.n - 1:
                ngram = context + (word,)
            else:
                # Pad context if needed
                if len(context) < self.n - 1:
                    padded_context = ('<s>',) * (self.n - 1 - len(context)) + context
                    ngram = padded_context + (word,)
                else:
                    ngram = context[-(self.n-1):] + (word,)
            
            prob = self.model.get_probability(ngram)
            probabilities[word] = prob
        
        return probabilities
    
    def sample_next_word(self, context: Tuple[str, ...], temperature: float = 1.0) -> str:
        """
        Sample next word from probability distribution.
        
        Args:
            context: Context tuple (last n-1 words)
            temperature: Sampling temperature (1.0 = normal, >1.0 = more random, <1.0 = more deterministic)
            
        Returns:
            Sampled word
        """
        probabilities = self.get_next_word_probabilities(context)
        
        # Apply temperature scaling
        if temperature != 1.0:
            scaled_probs = {}
            for word, prob in probabilities.items():
                if prob > 0:
                    scaled_probs[word] = prob ** (1.0 / temperature)
                else:
                    scaled_probs[word] = 0.0
            probabilities = scaled_probs
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob == 0:
            # Fallback to uniform distribution
            return random.choice(self.vocabulary)
        
        normalized_probs = {word: prob / total_prob for word, prob in probabilities.items()}
        
        # Sample from distribution
        words = list(normalized_probs.keys())
        weights = list(normalized_probs.values())
        
        return random.choices(words, weights=weights, k=1)[0]
    
    def generate_sentence(self, max_length: int = 50, temperature: float = 1.0, 
                         start_tokens: List[str] = None) -> List[str]:
        """
        Generate a complete sentence.
        
        Args:
            max_length: Maximum sentence length
            temperature: Sampling temperature
            start_tokens: Optional starting tokens (default: ['<s>'])
            
        Returns:
            Generated sentence as list of tokens
        """
        if start_tokens is None:
            start_tokens = ['<s>']
        
        sentence = start_tokens.copy()
        
        # Generate until we hit end token or max length
        while len(sentence) < max_length:
            # Get context (last n-1 tokens)
            if len(sentence) >= self.n - 1:
                context = tuple(sentence[-(self.n-1):])
            else:
                # Pad with start tokens if needed
                context = ('<s>',) * (self.n - 1 - len(sentence)) + tuple(sentence)
            
            # Sample next word
            next_word = self.sample_next_word(context, temperature)
            sentence.append(next_word)
            
            # Stop if we hit end token
            if next_word == '</s>':
                break
        
        return sentence
    
    def generate_multiple_sentences(self, num_sentences: int = 5, max_length: int = 50, 
                                  temperature: float = 1.0) -> List[List[str]]:
        """
        Generate multiple sentences.
        
        Args:
            num_sentences: Number of sentences to generate
            max_length: Maximum length per sentence
            temperature: Sampling temperature
            
        Returns:
            List of generated sentences
        """
        sentences = []
        
        for i in range(num_sentences):
            sentence = self.generate_sentence(max_length, temperature)
            sentences.append(sentence)
        
        return sentences


def create_generator_for_best_model(ngram_counter: NGramCounter) -> TextGenerator:
    """
    Create text generator for the best performing model (Stupid Backoff).
    
    Args:
        ngram_counter: N-gram counter with training data
        
    Returns:
        TextGenerator instance
    """
    # Create the best model (Stupid Backoff with alpha=0.6)
    model = StupidBackoffModel(ngram_counter, alpha=0.6, n=3)
    return TextGenerator(model, ngram_counter)


def main():
    """Test text generation functionality."""
    from data_loader import DataLoader
    
    print("Loading data and creating text generator...")
    
    # Load data
    loader = DataLoader()
    train_sentences, vocab = loader.load_training_data()
    
    # Count N-grams
    ngram_counter = NGramCounter(max_n=4)
    ngram_counter.count_ngrams(train_sentences)
    
    # Create generator
    generator = create_generator_for_best_model(ngram_counter)
    
    print("\nGenerating sample sentences...")
    print("="*60)
    
    # Generate sentences with different temperatures
    temperatures = [0.8, 1.0, 1.2]
    
    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        print("-" * 40)
        
        sentences = generator.generate_multiple_sentences(num_sentences=3, temperature=temp)
        
        for i, sentence in enumerate(sentences, 1):
            # Convert to readable format
            readable_sentence = ' '.join(sentence[1:-1])  # Remove <s> and </s>
            print(f"{i}. {readable_sentence}")


if __name__ == "__main__":
    main()
