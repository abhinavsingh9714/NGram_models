"""
Evaluation utilities for N-gram language models.
Implements perplexity calculation and model evaluation.
"""

import math
import json
import csv
from typing import List, Dict, Any, Union
from smoothing import Add1SmoothingModel
from mle import MLEModel
from linear_interpolation import LinearInterpolationModel
from stupid_backoff import StupidBackoffModel


def calculate_perplexity(model: Union[MLEModel, Add1SmoothingModel, LinearInterpolationModel, StupidBackoffModel], 
                        sentences: List[List[str]]) -> Dict[str, Any]:
    """
    Calculate perplexity of a model on a set of sentences.
    
    Args:
        model: Trained N-gram model
        sentences: List of sentences to evaluate
        
    Returns:
        Dictionary with perplexity and evaluation details
    """
    total_log_prob = 0.0
    total_tokens = 0
    has_zero_prob = False
    sentences_with_zeros = 0
    
    print(f"Evaluating {model.get_model_info()['type']} model...")
    
    for i, sentence in enumerate(sentences):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(sentences)} sentences")
        
        # Calculate log probability for this sentence
        log_prob, sentence_has_zero = model.calculate_sentence_log_prob(sentence)
        
        if sentence_has_zero:
            has_zero_prob = True
            sentences_with_zeros += 1
            # For MLE models with zero probabilities, return INF
            if model.get_model_info()['type'] == 'MLE':
                return {
                    'perplexity': float('inf'),
                    'log_probability': float('-inf'),
                    'total_tokens': 0,
                    'has_zero_probability': True,
                    'sentences_with_zeros': len(sentences),
                    'model_info': model.get_model_info()
                }
        
        total_log_prob += log_prob
        
        # Count tokens (excluding sentence boundaries for fair comparison)
        sentence_tokens = len([token for token in sentence if token not in ['<s>', '</s>']])
        total_tokens += sentence_tokens
    
    # Calculate perplexity: PP = exp(-1/N * Σ log P(wi))
    if total_tokens == 0:
        perplexity = float('inf')
    elif has_zero_prob:
        perplexity = float('inf')
    else:
        avg_log_prob = total_log_prob / total_tokens
        perplexity = math.exp(-avg_log_prob)
    
    return {
        'perplexity': perplexity,
        'log_probability': total_log_prob,
        'total_tokens': total_tokens,
        'has_zero_probability': has_zero_prob,
        'sentences_with_zeros': sentences_with_zeros,
        'model_info': model.get_model_info()
    }


def evaluate_model(model: Union[MLEModel, Add1SmoothingModel, LinearInterpolationModel, StupidBackoffModel], 
                  test_sentences: List[List[str]], 
                  model_name: str = None) -> Dict[str, Any]:
    """
    Evaluate a model and return comprehensive results.
    
    Args:
        model: Trained N-gram model
        test_sentences: Test sentences for evaluation
        model_name: Optional name for the model
        
    Returns:
        Dictionary with evaluation results
    """
    if model_name is None:
        model_info = model.get_model_info()
        model_name = f"{model_info['type']}_n{model_info['n']}"
    
    print(f"\nEvaluating {model_name}...")
    
    # Calculate perplexity
    results = calculate_perplexity(model, test_sentences)
    
    # Add model name and additional info
    results['model_name'] = model_name
    results['num_test_sentences'] = len(test_sentences)
    
    # Format results for output
    if results['perplexity'] == float('inf'):
        perplexity_str = "INF"
        notes = "Zero probabilities encountered"
    else:
        perplexity_str = f"{results['perplexity']:.2f}"
        notes = "Normal evaluation"
    
    results['perplexity_str'] = perplexity_str
    results['notes'] = notes
    
    print(f"  Perplexity: {perplexity_str}")
    print(f"  Total tokens: {results['total_tokens']:,}")
    print(f"  Notes: {notes}")
    
    return results


def save_results_to_json(results: List[Dict[str, Any]], filename: str = "results.json") -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: List of evaluation result dictionaries
        filename: Output filename
    """
    print(f"\nSaving results to {filename}...")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {filename}")


def save_results_to_csv(results: List[Dict[str, Any]], filename: str = "results.csv") -> None:
    """
    Save evaluation results to CSV file.
    
    Args:
        results: List of evaluation result dictionaries
        filename: Output filename
    """
    print(f"\nSaving results to {filename}...")
    
    if not results:
        print("No results to save")
        return
    
    # Define CSV columns
    fieldnames = [
        'model_name',
        'model_type', 
        'n_gram_order',
        'smoothing_method',
        'parameters',
        'perplexity',
        'perplexity_str',
        'total_tokens',
        'num_test_sentences',
        'has_zero_probability',
        'sentences_with_zeros',
        'notes'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            model_info = result['model_info']
            
            # Extract parameters
            if 'lambdas' in model_info:
                params = f"lambdas={model_info['lambdas']}"
            elif 'alpha' in model_info:
                params = f"alpha={model_info['alpha']}"
            else:
                params = "None"
            
            # Extract smoothing method
            smoothing_method = model_info['type']
            if smoothing_method == 'MLE':
                smoothing_method = "None"
            
            row = {
                'model_name': result['model_name'],
                'model_type': model_info['type'],
                'n_gram_order': model_info['n'],
                'smoothing_method': smoothing_method,
                'parameters': params,
                'perplexity': result['perplexity'],
                'perplexity_str': result['perplexity_str'],
                'total_tokens': result['total_tokens'],
                'num_test_sentences': result['num_test_sentences'],
                'has_zero_probability': result['has_zero_probability'],
                'sentences_with_zeros': result['sentences_with_zeros'],
                'notes': result['notes']
            }
            
            writer.writerow(row)
    
    print(f"Results saved to {filename}")


def print_results_summary(results: List[Dict[str, Any]]) -> None:
    """
    Print a summary of all evaluation results.
    
    Args:
        results: List of evaluation result dictionaries
    """
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)
    
    # Sort results by perplexity (INF last)
    sorted_results = sorted(results, key=lambda x: (
        x['perplexity'] == float('inf'),  # INF values go last
        x['perplexity']
    ))
    
    print(f"{'Model':<25} {'N-gram':<8} {'Method':<20} {'Perplexity':<12} {'Notes'}")
    print("-" * 80)
    
    for result in sorted_results:
        model_info = result['model_info']
        model_name = result['model_name']
        n_gram = f"N={model_info['n']}"
        method = model_info['type']
        perplexity = result['perplexity_str']
        notes = result['notes']
        
        print(f"{model_name:<25} {n_gram:<8} {method:<20} {perplexity:<12} {notes}")
    
    print("="*80)
    
    # Find best model (excluding INF)
    finite_results = [r for r in results if r['perplexity'] != float('inf')]
    if finite_results:
        best_result = min(finite_results, key=lambda x: x['perplexity'])
        print(f"\nBest performing model: {best_result['model_name']}")
        print(f"Best perplexity: {best_result['perplexity']:.2f}")
    else:
        print("\nNo models achieved finite perplexity (all had zero probabilities)")


def main():
    """Test the evaluation functions."""
    from data_loader import DataLoader
    from ngram_model import NGramCounter, MLEModel
    from smoothing import Add1SmoothingModel, LinearInterpolationModel, StupidBackoffModel
    
    # Load data
    loader = DataLoader()
    train_sentences, vocab = loader.load_training_data()
    test_sentences = loader.load_test_data()
    
    # Count N-grams
    counter = NGramCounter(max_n=3)
    counter.count_ngrams(train_sentences)
    
    # Test different models
    results = []
    
    # MLE Unigram
    mle_uni = MLEModel(counter, n=1)
    result = evaluate_model(mle_uni, test_sentences[:100])  # Test on subset
    results.append(result)
    
    # Add-1 Smoothing
    add1_model = Add1SmoothingModel(counter, n=3)
    result = evaluate_model(add1_model, test_sentences[:100])
    results.append(result)
    
    # Linear Interpolation
    interp_model = LinearInterpolationModel(counter, [0.1, 0.3, 0.6], n=3)
    result = evaluate_model(interp_model, test_sentences[:100])
    results.append(result)
    
    # Stupid Backoff
    backoff_model = StupidBackoffModel(counter, alpha=0.4, n=3)
    result = evaluate_model(backoff_model, test_sentences[:100])
    results.append(result)
    
    # Print summary
    print_results_summary(results)
    
    # Save results
    save_results_to_json(results, "test_results.json")
    save_results_to_csv(results, "test_results.csv")


if __name__ == "__main__":
    main()
