"""
Main training script for N-gram language models.
Orchestrates all experiments and saves results to CSV/JSON.
"""

import time
from typing import List, Dict, Any
from data_loader import DataLoader
from ngram_model import NGramCounter
from smoothing import Add1SmoothingModel
from evaluate import evaluate_model, save_results_to_json, save_results_to_csv, print_results_summary
from mle import MLEModel
from linear_interpolation import LinearInterpolationModel
from stupid_backoff import StupidBackoffModel


def train_mle_models(ngram_counter: NGramCounter, test_sentences: List[List[str]]) -> List[Dict[str, Any]]:
    """
    Train and evaluate MLE models for N=1,2,3,4.
    
    Args:
        ngram_counter: Pre-computed N-gram counts
        test_sentences: Test sentences for evaluation
        
    Returns:
        List of evaluation results
    """
    print("\n" + "="*60)
    print("TRAINING MLE MODELS (N=1,2,3,4)")
    print("="*60)
    
    results = []
    
    for n in range(1, 5):
        print(f"\nTraining MLE {n}-gram model...")
        model = MLEModel(ngram_counter, n)
        result = evaluate_model(model, test_sentences, f"MLE_n{n}")
        results.append(result)
    
    return results


def train_add1_smoothing(ngram_counter: NGramCounter, test_sentences: List[List[str]]) -> List[Dict[str, Any]]:
    """
    Train and evaluate Add-1 smoothing model (trigram).
    
    Args:
        ngram_counter: Pre-computed N-gram counts
        test_sentences: Test sentences for evaluation
        
    Returns:
        List of evaluation results
    """
    print("\n" + "="*60)
    print("TRAINING ADD-1 SMOOTHING MODEL (TRIGRAM)")
    print("="*60)
    
    print("Training Add-1 smoothing trigram model...")
    model = Add1SmoothingModel(ngram_counter, n=3)
    result = evaluate_model(model, test_sentences, "Add1_Trigram")
    
    return [result]


def train_linear_interpolation(ngram_counter: NGramCounter, 
                              valid_sentences: List[List[str]], 
                              test_sentences: List[List[str]]) -> List[Dict[str, Any]]:
    """
    Train and evaluate Linear Interpolation model with lambda tuning.
    
    Args:
        ngram_counter: Pre-computed N-gram counts
        valid_sentences: Validation sentences for lambda tuning
        test_sentences: Test sentences for evaluation
        
    Returns:
        List of evaluation results
    """
    print("\n" + "="*60)
    print("TRAINING LINEAR INTERPOLATION MODEL (TRIGRAM)")
    print("="*60)
    
    # Generate lambda combinations for grid search
    print("Generating lambda combinations for grid search...")
    lambda_combinations = LinearInterpolationModel.generate_lambda_combinations(n=3, step=0.1)
    print(f"Found {len(lambda_combinations)} lambda combinations")
    
    # Grid search on validation set
    print("\nPerforming grid search on validation set...")
    best_lambdas = None
    best_perplexity = float('inf')
    validation_results = []
    
    for i, lambdas in enumerate(lambda_combinations):
        if i % 10 == 0:
            print(f"  Testing combination {i+1}/{len(lambda_combinations)}: {lambdas}")
        
        # Create model with current lambdas
        model = LinearInterpolationModel(ngram_counter, lambdas, n=3)
        
        # Evaluate on validation set
        result = evaluate_model(model, valid_sentences, f"Interp_Val_{i}")
        validation_results.append({
            'lambdas': lambdas,
            'perplexity': result['perplexity'],
            'result': result
        })
        
        # Track best combination
        if result['perplexity'] < best_perplexity:
            best_perplexity = result['perplexity']
            best_lambdas = lambdas
    
    print(f"\nBest lambda combination: {best_lambdas}")
    print(f"Best validation perplexity: {best_perplexity:.2f}")
    
    # Train final model with best lambdas on test set
    print("\nEvaluating best model on test set...")
    final_model = LinearInterpolationModel(ngram_counter, best_lambdas, n=3)
    final_result = evaluate_model(final_model, test_sentences, "LinearInterp_Trigram")
    
    # Add lambda information to result
    final_result['best_lambdas'] = best_lambdas
    final_result['validation_perplexity'] = best_perplexity
    
    return [final_result]


def train_stupid_backoff(ngram_counter: NGramCounter, 
                        valid_sentences: List[List[str]], 
                        test_sentences: List[List[str]]) -> List[Dict[str, Any]]:
    """
    Train and evaluate Stupid Backoff model with optional alpha tuning.
    
    Args:
        ngram_counter: Pre-computed N-gram counts
        valid_sentences: Validation sentences for alpha tuning
        test_sentences: Test sentences for evaluation
        
    Returns:
        List of evaluation results
    """
    print("\n" + "="*60)
    print("TRAINING STUPID BACKOFF MODEL (TRIGRAM)")
    print("="*60)
    
    # Alpha values to try
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    print("Tuning alpha parameter on validation set...")
    best_alpha = 0.4  # Default
    best_perplexity = float('inf')
    validation_results = []
    
    for alpha in alpha_values:
        print(f"  Testing alpha = {alpha}")
        
        # Create model with current alpha
        model = StupidBackoffModel(ngram_counter, alpha=alpha, n=3)
        
        # Evaluate on validation set
        result = evaluate_model(model, valid_sentences, f"Backoff_Val_alpha{alpha}")
        validation_results.append({
            'alpha': alpha,
            'perplexity': result['perplexity'],
            'result': result
        })
        
        # Track best alpha
        if result['perplexity'] < best_perplexity:
            best_perplexity = result['perplexity']
            best_alpha = alpha
    
    print(f"\nBest alpha: {best_alpha}")
    print(f"Best validation perplexity: {best_perplexity:.2f}")
    
    # Train final model with best alpha on test set
    print("\nEvaluating best model on test set...")
    final_model = StupidBackoffModel(ngram_counter, alpha=best_alpha, n=3)
    final_result = evaluate_model(final_model, test_sentences, "StupidBackoff_Trigram")
    
    # Add alpha information to result
    final_result['best_alpha'] = best_alpha
    final_result['validation_perplexity'] = best_perplexity
    
    return [final_result]


def main():
    """Main training pipeline."""
    print("N-gram Language Models Training Pipeline")
    print("="*60)
    
    start_time = time.time()
    
    # Load data
    print("Loading data...")
    loader = DataLoader()
    train_sentences, vocab = loader.load_training_data()
    valid_sentences = loader.load_validation_data()
    test_sentences = loader.load_test_data()
    
    print(f"Training sentences: {len(train_sentences):,}")
    print(f"Validation sentences: {len(valid_sentences):,}")
    print(f"Test sentences: {len(test_sentences):,}")
    print(f"Vocabulary size: {len(vocab):,}")
    
    # Count N-grams
    print("\nCounting N-grams...")
    ngram_counter = NGramCounter(max_n=4)
    ngram_counter.count_ngrams(train_sentences)
    
    # Store all results
    all_results = []
    
    # Step 1: Train MLE models (N=1,2,3,4)
    mle_results = train_mle_models(ngram_counter, test_sentences)
    all_results.extend(mle_results)
    
    # Step 2: Train Add-1 Smoothing (Trigram)
    add1_results = train_add1_smoothing(ngram_counter, test_sentences)
    all_results.extend(add1_results)
    
    # Step 3: Train Linear Interpolation (Trigram)
    interp_results = train_linear_interpolation(ngram_counter, valid_sentences, test_sentences)
    all_results.extend(interp_results)
    
    # Step 4: Train Stupid Backoff (Trigram)
    backoff_results = train_stupid_backoff(ngram_counter, valid_sentences, test_sentences)
    all_results.extend(backoff_results)
    
    # Print final summary
    print_results_summary(all_results)
    
    # Save results
    save_results_to_json(all_results, "results.json")
    save_results_to_csv(all_results, "results.csv")
    
    # Print timing information
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds")
    print(f"Results saved to results.json and results.csv")


if __name__ == "__main__":
    main()
