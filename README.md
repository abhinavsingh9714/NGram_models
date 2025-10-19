# N-gram Language Models Implementation

This project implements and evaluates various N-gram Language Models (LMs) to understand the trade-offs between different N-gram orders and the critical role of smoothing and backoff techniques in handling data sparsity.

## Dataset

The implementation uses the Penn Treebank (PTB) dataset with three splits:
- **Training Data**: `ptb.train.txt` (42,068 sentences) - Used for training the language models
- **Validation Data**: `ptb.valid.txt` (3,370 sentences) - Used for tuning hyperparameters
- **Test Data**: `ptb.test.txt` (3,761 sentences) - Used for final, unbiased performance evaluation

## Implementation

### Architecture

The implementation follows a modular design with the following components:

**Core Modules (`src/` directory):**
- `data_loader.py`: Data loading and preprocessing utilities
- `ngram_model.py`: Core N-gram model classes and probability computations
- `mle.py`: Maximum Likelihood Estimation implementation
- `smoothing.py`: Add-1 smoothing technique
- `linear_interpolation.py`: Linear interpolation with grid search
- `stupid_backoff.py`: Stupid backoff algorithm implementation
- `evaluate.py`: Perplexity calculation and model evaluation
- `train.py`: Main training script that orchestrates all experiments
- `text_generator.py`: Text generation utilities for qualitative analysis

**Additional Files:**
- `analytics_report.md`: Comprehensive analysis report with results and insights
- `requirements.txt`: Dependencies

### Models Implemented

#### 1. Maximum Likelihood Estimation (MLE) Models (N=1,2,3,4)
- **Unigram (N=1)**: P(w) = C(w) / total_words
- **Bigram (N=2)**: P(wn | wn-1) = C(wn-1, wn) / C(wn-1)
- **Trigram (N=3)**: P(wn | wn-2, wn-1) = C(wn-2, wn-1, wn) / C(wn-2, wn-1)
- **4-gram (N=4)**: P(wn | wn-3, wn-2, wn-1) = C(wn-3, wn-2, wn-1, wn) / C(wn-3, wn-2, wn-1)

#### 2. Add-1 Smoothing (Laplace)
- **Formula**: P(wn | w1, w2) = (C(w1, w2, wn) + 1) / (C(w1, w2) + V)
- **V**: Vocabulary size (10,001)
- **Applied to**: Trigram model

#### 3. Linear Interpolation
- **Formula**: P(wn | w1, w2) = λ3*P_tri + λ2*P_bi + λ1*P_uni
- **Constraint**: λ1 + λ2 + λ3 = 1
- **Grid Search**: 36 lambda combinations tested on validation set
- **Best Lambdas**: [0.3, 0.5, 0.2] (unigram, bigram, trigram weights)

#### 4. Stupid Backoff
- **Formula**: 
  - If trigram seen: P(wn | w1, w2)
  - Else if bigram seen: α * P(wn | w2)
  - Else: α² * P(wn)
- **Alpha Tuning**: Tested α ∈ [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] on validation set
- **Best Alpha**: 0.6

## Results

### Performance Summary

| Model | N-gram Order | Method | Test Perplexity | Notes |
|-------|-------------|--------|----------------|-------|
| StupidBackoff_Trigram | N=3 | Stupid Backoff | **135.74** | Best performing |
| LinearInterp_Trigram | N=3 | Linear Interpolation | 185.26 | Good performance |
| MLE_n1 | N=1 | MLE | 1,059.67 | Baseline |
| Add1_Trigram | N=3 | Add-1 Smoothing | 3,748.66 | Over-smoothing |
| MLE_n2 | N=2 | MLE | INF | Zero probabilities |
| MLE_n3 | N=3 | MLE | INF | Zero probabilities |
| MLE_n4 | N=4 | MLE | INF | Zero probabilities |

### Key Findings

1. **MLE Models**: Higher-order MLE models (N=2,3,4) fail due to zero probabilities, demonstrating the necessity of smoothing.

2. **Add-1 Smoothing**: While it prevents zero probabilities, it suffers from over-smoothing, resulting in poor perplexity (3,748.66).

3. **Linear Interpolation**: Achieves good performance (185.26) by combining multiple N-gram orders with optimal weights.

4. **Stupid Backoff**: Achieves the best performance (135.74) by using a simple but effective backoff strategy with optimal alpha.

5. **Parameter Tuning**: Both linear interpolation and stupid backoff benefit significantly from hyperparameter tuning on the validation set.

## Analytics Report

A comprehensive analytics report (`analytics_report.md`) provides detailed analysis including:

- **Pre-processing Analysis**: Tokenization strategies and vocabulary decisions
- **N-gram Order Impact**: Analysis of how different N-gram orders affect performance
- **Smoothing Strategy Comparison**: Detailed comparison of all smoothing/backoff methods
- **Qualitative Analysis**: Generated text samples and fluency assessment
- **Technical Insights**: Explanation of data sparsity, Markov assumption trade-offs, and model behavior

## Text Generation

The project includes a text generator (`src/text_generator.py`) that can generate sample sentences using the best-performing model (Stupid Backoff with α=0.6). Features include:

- **Temperature-controlled sampling**: Adjustable randomness in generation
- **Multiple sentence generation**: Generate multiple samples for analysis
- **Best model integration**: Uses the optimal Stupid Backoff model
- **Qualitative evaluation**: Enables assessment of model fluency and coherence

## Usage

### Running the Full Pipeline

```bash
python src/train.py
```

This will:
1. Load and preprocess the PTB data
2. Train all models (MLE, Add-1, Linear Interpolation, Stupid Backoff)
3. Perform hyperparameter tuning on validation data
4. Evaluate all models on test data
5. Save results to `results/results.json` and `results/results.csv`

### Individual Components

```bash
# Test data loading
python src/data_loader.py

# Test N-gram models
python src/ngram_model.py

# Test smoothing techniques
python src/smoothing.py

# Test evaluation
python src/evaluate.py

# Generate sample text using best model
python src/text_generator.py
```

### Viewing Results

```bash
# View comprehensive analytics report
cat analytics_report.md

# View results in JSON format
cat results/results.json

# View results in CSV format
cat results/results.csv
```

## Dependencies

```
numpy>=1.21.0
collections
json
csv
math
```

## File Structure

```
n-gram_models/
├── data/
│   ├── ptb.train.txt
│   ├── ptb.valid.txt
│   └── ptb.test.txt
├── src/
│   ├── data_loader.py
│   ├── ngram_model.py
│   ├── mle.py
│   ├── smoothing.py
│   ├── linear_interpolation.py
│   ├── stupid_backoff.py
│   ├── evaluate.py
│   ├── train.py
│   └── text_generator.py
├── results/
│   ├── results.json
│   ├── results.csv
│   ├── test_results.json
│   └── test_results.csv
├── analytics_report.md
├── requirements.txt
└── README.md
```

## Evaluation Metrics

All models are evaluated using **Perplexity (PP)** on the test data:
- **Formula**: PP = exp(-1/N * Σ log P(wi | context))
- **Lower is better**: Lower perplexity indicates better model performance
- **Zero Probability Handling**: Models with zero probabilities return INF perplexity

## Implementation Details

- **Efficiency**: Uses defaultdict and Counter for efficient N-gram counting
- **Memory**: Pre-computes all counts during training (one pass)
- **Numerical Stability**: Uses log probabilities throughout to prevent underflow
- **Edge Cases**: Properly handles OOV words, sentence boundaries, and empty lines
- **Modularity**: Clean separation of concerns with reusable components
