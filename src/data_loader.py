"""
Data loading and preprocessing utilities for N-gram language models.
Handles PTB data format with proper tokenization and sentence boundaries.
"""

import os
from typing import List, Set, Tuple


class DataLoader:
    """Handles loading and preprocessing of Penn Treebank data."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader with data directory.
        
        Args:
            data_dir: Path to directory containing PTB data files
        """
        self.data_dir = data_dir
        self.vocabulary: Set[str] = set()
        
    def load_file(self, filename: str) -> List[str]:
        """
        Load sentences from a PTB data file.
        
        Args:
            filename: Name of the file to load (e.g., 'ptb.train.txt')
            
        Returns:
            List of sentences, where each sentence is a list of tokens
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        sentences = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    # Split by whitespace and filter out empty tokens
                    tokens = [token for token in line.split() if token]
                    if tokens:  # Only add non-empty token lists
                        sentences.append(tokens)
                        
        return sentences
    
    def add_sentence_boundaries(self, sentences: List[List[str]]) -> List[List[str]]:
        """
        Add sentence boundary markers to sentences.
        
        Args:
            sentences: List of sentences (each sentence is a list of tokens)
            
        Returns:
            List of sentences with <s> and </s> markers added
        """
        bounded_sentences = []
        
        for sentence in sentences:
            # Add start and end markers
            bounded_sentence = ['<s>'] + sentence + ['</s>']
            bounded_sentences.append(bounded_sentence)
            
        return bounded_sentences
    
    def build_vocabulary(self, sentences: List[List[str]]) -> Set[str]:
        """
        Build vocabulary from training sentences.
        
        Args:
            sentences: List of sentences (each sentence is a list of tokens)
            
        Returns:
            Set of unique tokens in the vocabulary
        """
        vocabulary = set()
        
        for sentence in sentences:
            for token in sentence:
                vocabulary.add(token)
                
        self.vocabulary = vocabulary
        return vocabulary
    
    def load_training_data(self) -> Tuple[List[List[str]], Set[str]]:
        """
        Load and preprocess training data.
        
        Returns:
            Tuple of (sentences with boundaries, vocabulary)
        """
        print("Loading training data...")
        sentences = self.load_file('ptb.train.txt')
        sentences = self.add_sentence_boundaries(sentences)
        vocabulary = self.build_vocabulary(sentences)
        
        print(f"Loaded {len(sentences)} training sentences")
        print(f"Vocabulary size: {len(vocabulary)}")
        
        return sentences, vocabulary
    
    def load_validation_data(self) -> List[List[str]]:
        """
        Load and preprocess validation data.
        
        Returns:
            List of sentences with boundaries
        """
        print("Loading validation data...")
        sentences = self.load_file('ptb.valid.txt')
        sentences = self.add_sentence_boundaries(sentences)
        
        print(f"Loaded {len(sentences)} validation sentences")
        return sentences
    
    def load_test_data(self) -> List[List[str]]:
        """
        Load and preprocess test data.
        
        Returns:
            List of sentences with boundaries
        """
        print("Loading test data...")
        sentences = self.load_file('ptb.test.txt')
        sentences = self.add_sentence_boundaries(sentences)
        
        print(f"Loaded {len(sentences)} test sentences")
        return sentences
    
    def get_vocabulary_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.vocabulary)
    
    def is_known_word(self, word: str) -> bool:
        """Check if a word is in the vocabulary."""
        return word in self.vocabulary


def main():
    """Test the DataLoader functionality."""
    loader = DataLoader()
    
    # Test loading training data
    train_sentences, vocab = loader.load_training_data()
    print(f"First training sentence: {train_sentences[0]}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Test loading validation data
    valid_sentences = loader.load_validation_data()
    print(f"First validation sentence: {valid_sentences[0]}")
    
    # Test loading test data
    test_sentences = loader.load_test_data()
    print(f"First test sentence: {test_sentences[0]}")


if __name__ == "__main__":
    main()
