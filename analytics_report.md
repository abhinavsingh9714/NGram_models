# HW2: N-gram Language Models
Abhinav Singh (121335258)
## Summary

This report presents a comprehensive analysis of N-gram language models trained on the Penn Treebank dataset. I evaluated multiple model architectures including Maximum Likelihood Estimation (MLE), Add-1 Smoothing, Linear Interpolation, and Stupid Backoff across different N-gram orders. The Stupid Backoff model with α=0.6 achieved the best performance with a perplexity of 135.74, significantly outperforming other approaches.

## 4.1 Pre-processing and Vocabulary Decisions

### Tokenization Strategy

The preprocessing pipeline implemented the following strategies:

1. **Whitespace Tokenization**: The Penn Treebank data was already pre-tokenized, so I split sentences on whitespace to extract individual tokens.

2. **Sentence Boundary Markers**: Added explicit sentence boundary markers:
   - `<s>` at the beginning of each sentence
   - `</s>` at the end of each sentence
   - This ensures proper context modeling for N-gram probabilities

3. **Vocabulary Construction**: 
   - Built vocabulary from all unique tokens in the training set
   - Final vocabulary size: **10,001 tokens**
   - No additional preprocessing (lowercasing, stemming, etc.) was applied to maintain the original PTB format

4. **Data Splits**:
   - Training: 42,068 sentences
   - Validation: 3,370 sentences  
   - Test: 3,761 sentences

## 4.2 Impact of N-gram Order

### Perplexity Results by N-gram Order

| Model Type | N-gram Order | Perplexity | Status |
|------------|--------------|------------|---------|
| MLE | 1 (Unigram) | 1,059.67 | Valid |
| MLE | 2 (Bigram) | ∞ | Zero probabilities |
| MLE | 3 (Trigram) | ∞ | Zero probabilities |
| MLE | 4 (4-gram) | ∞ | Zero probabilities |

### Analysis of Trends

**The Markov Assumption and Data Sparsity Problem:**

1. **Unigram Model (N=1)**: 
   - Perplexity: 1,059.67
   - No context dependency, only word frequency
   - No zero probabilities since all words in vocabulary appear in training

2. **Higher-Order Models (N≥2)**:
   - All show infinite perplexity due to zero probabilities
   - As N increases, the number of possible N-grams grows exponentially (V^N)
   - With 10,001 vocabulary words, there are potentially 10,001^2 = 100M bigrams, 10,001^3 = 1T trigrams
   - Only 264,990 bigrams and 586,558 trigrams were observed in training

3. **The Markov Assumption Trade-off**:
   - Higher N-grams capture more context and should theoretically perform better
   - However, data sparsity makes many N-grams unseen, leading to zero probabilities
   - This demonstrates the fundamental challenge: **bias-variance trade-off** in N-gram modeling

## 4.3 Comparison of Smoothing/Backoff Strategies

### Final Perplexity Scores

| Model | Method | Perplexity | Improvement |
|-------|--------|------------|-------------|
| MLE Trigram | No Smoothing | ∞ | Baseline (failed) |
| Add-1 Smoothing | Add-1 | 3,748.66 | Finite |
| Linear Interpolation | λ=[0.3,0.5,0.2] | 185.26 | 95.1% better than Add-1 |
| **Stupid Backoff** | **α=0.6** | **135.74** | **96.4% better than Add-1** |

### Detailed Analysis

#### Why Unsmoothed Models Failed

Test sentences contain many N-grams not seen in training, MLE assigns zero probability to unseen N-grams and log(0) = -∞, making perplexity infinite

#### Smoothing Strategy Comparison

**1. Add-1 Smoothing (Laplace Smoothing)**
- **Performance**: 3,748.66 perplexity
- **Method**: Add 1 to all N-gram counts, normalize
- **Issues**: 
  - Too much probability mass allocated to unseen events
  - Poor performance due to over-smoothing
  - Simple but not optimal for language modeling

**2. Linear Interpolation**
- **Performance**: 185.26 perplexity (validation: 200.23)
- **Method**: Weighted combination of unigram, bigram, and trigram probabilities
- **Optimal λ values**: [0.3, 0.5, 0.2] (found via grid search)
- **Advantages**:
  - Always assigns non-zero probabilities
  - Balances different N-gram orders effectively
  - Validation-based parameter tuning

**3. Stupid Backoff (Best Performer)**
- **Performance**: 135.74 perplexity (validation: 146.28)
- **Method**: Hierarchical backoff with discounting factor α=0.6
- **Algorithm**:
  ```
  If trigram seen: P(w3|w1,w2)
  Else if bigram seen: α × P(w3|w2)
  Else: α² × P(w3)
  ```
- **Why it works best**:
  - Preserves high-order context when available
  - Graceful degradation to lower-order models
  - Optimal α=0.6 balances context vs. generalization
  - No normalization required (simpler computation)

### Performance Ranking

1. **Stupid Backoff** (135.74) - 26.7% better than Linear Interpolation
2. **Linear Interpolation** (185.26) - 95.1% better than Add-1
3. **Add-1 Smoothing** (3,748.66) - Only finite perplexity among basic methods
4. **MLE** (∞) - Complete failure due to zero probabilities

## 4.4 Qualitative Analysis (Generated Text)

### Text Generation Implementation

I implemented a sampling-based text generator using the best-performing Stupid Backoff model (α=0.6). The generator uses temperature-controlled sampling to create diverse text samples.

### Generated Sentences

**Temperature 0.8 (More Deterministic):**
1. "it 's time to time the analyst added employees at the recent natural gas will cause additional disruption of its seven buildings quake that measures the nation 's largest are `<s>` that and other matters mr. mulford responding `<unk>` a place of a government less 's total value"
2. "with interest"
3. "the <unk> <unk> smith state securities firm"

**Temperature 1.0 (Balanced):**
4. "capital stock buy-back program which reduced the cycle consultant the boat too much in line with other cabinet bugs going to be the for of the foreign exchange market analysts said there was such that we 'll raise it through bank loans"
5. "that $ N million"
6. "the government said the market approaches"

**Temperature 1.2 (More Random):**
7. "serial benchmark justice ample the population would help them in inches offensive obstacles earnings before sir of companies within the official john terminal and ual corp. said it organized mill subsidiaries conspired them decline"
8. "father"
9. "in october `<unk>` sales i was interested pretax retail asian economies and trading stem intelligence location management guard on the process of clearing the complaints from $ ways investors thomson new of `<s>` gross ticket see listing protected increased discrimination wall street but vacant actual stocks and fired"

### Analysis of Generated Text Quality

#### Fluency Assessment
**Strengths:**
1. Generated text maintains financial/business vocabulary consistent with PTB training data
2. Some phrases show reasonable grammatical structure ("the analyst added employees", "market analysts said")
3. Business terms appear together appropriately ("capital stock", "buy-back program", "market approaches")

**Limitations:**
1. Sentences lack overall meaning and logical flow
2. Some phrases repeat or loop ("time to time", "`<s>` that")
3. Frequent `<unk>` tokens indicate vocabulary limitations
4. Poor sentence boundaries and incomplete thoughts

#### How the Model Generates Sequences

**Probability-Based Sampling:**
1. **Context Window**: Model uses 2-word context (trigram model)
2. **Probability Distribution**: For each position, model computes probability distribution over entire vocabulary
3. **Sampling Strategy**: 
   - Temperature 0.8: More likely to pick high-probability words
   - Temperature 1.0: Proportional to model probabilities  
   - Temperature 1.2: More exploration of low-probability words

**Generation Process:**
1. Start with `<s>` boundary marker
2. For each position, use last 2 words as context
3. Apply Stupid Backoff to get probability for each possible next word
4. Sample from this distribution based on temperature
5. Continue until `</s>` or maximum length

**Why Generated Text Lacks Coherence:**
1. Only 2-word context insufficient for long-range dependencies
2. PTB contains many short, fragmented sentences
3. Model only captures statistical patterns, not meaning
4. `<unk>` tokens indicate out-of-vocabulary problems

## Conclusions and Insights

### Key Findings
1. Higher-order N-grams fail without smoothing due to unseen events
2. Stupid Backoff significantly outperforms Add-1 smoothing
3. Optimal α=0.6 and λ=[0.3,0.5,0.2] found via validation
4. Even best N-gram models struggle with long-range dependencies

### Model Performance Summary
- Stupid Backoff (α=0.6) with perplexity 135.74
- 96.4% better than basic Add-1 smoothing

This analysis demonstrates the fundamental challenges and solutions in statistical language modeling, providing a foundation for understanding more advanced neural approaches.
