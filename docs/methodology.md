# Enhanced CLMPI Methodology

## Overview

The Enhanced CLMPI (Comprehensive Language Model Performance Index) implements rigorous benchmarking methodology with standardized generation settings, curated expert-validated datasets, and transparent scoring formulas.

## Core Principles

### 1. Standardized Generation Settings

**Problem**: Different models running with different defaults creates unfair comparisons.

**Solution**: Two standardized generation profiles:

#### Deterministic Profile (Accuracy & Contextual Understanding)
```yaml
temperature: 0.0
top_p: 1.0
top_k: 1
max_tokens: 1000
```
- **Purpose**: Factual tasks requiring consistent, deterministic outputs
- **Use Cases**: Accuracy evaluation, contextual understanding
- **Rationale**: Eliminates randomness for fair comparison

#### Creative Profile (Coherence & Fluency)
```yaml
temperature: 0.7
top_p: 0.9
top_k: 40
max_tokens: 1000
```
- **Purpose**: Quality evaluation requiring natural language generation
- **Use Cases**: Coherence evaluation, fluency assessment
- **Rationale**: Allows creativity while maintaining consistency

### 2. Curated Expert-Validated Datasets

#### Accuracy Dataset (`accuracy.json`)
- **Source**: Expert-validated factual questions
- **Structure**: Question + correct answer + acceptable variations
- **Categories**: Geography, science, history, math, general knowledge
- **Validation**: Each question verified by domain experts

#### Contextual Understanding Dataset (`context.json`)
- **Source**: Multi-turn conversations with context
- **Structure**: Context + question + gold answer
- **Types**: Conversation, narrative, instruction
- **Validation**: Expert-validated answers

#### Coherence Dataset (`coherence.json`)
- **Source**: Dedicated coherence evaluation prompts
- **Structure**: Open-ended prompts requiring logical flow
- **Types**: Narrative, argument, explanation, instruction
- **Validation**: No reference text - evaluates internal consistency

#### Fluency Dataset (`fluency.json`)
- **Source**: Surface quality evaluation prompts
- **Structure**: Descriptive, narrative, explanatory tasks
- **Types**: Descriptive, narrative, explanatory, conversational
- **Validation**: No reference text - evaluates grammatical correctness

## Scoring Formulas

### 1. Accuracy Scoring

#### Exact Match (EM)
```
EM = (Number of exact matches) / (Total questions)
```

#### F1 Score
```
Precision = (Response words ∩ Gold words) / (Response words)
Recall = (Response words ∩ Gold words) / (Gold words)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

#### Final Accuracy Score
```
Accuracy = F1_Score  # Already normalized to [0,1]
```

### 2. Contextual Understanding Scoring

#### Base Accuracy
- Uses same EM and F1 calculations as accuracy

#### Context Similarity
```
Context_Similarity = (Response words ∩ Context words) / (Context words)
```

#### Combined Score
```
Contextual_Score = 0.7 × F1_Score + 0.3 × Context_Similarity
```

### 3. Coherence Scoring

#### Sentence-to-Sentence Similarity
```
For each consecutive sentence pair:
Similarity = (Words1 ∩ Words2) / max(Words1, Words2)
Sentence_Similarity = mean(all similarities)
```

#### Repetition Penalty
```
Word_Counts = count_occurrences(all_words)
Repetition_Ratio = sum(counts - 1) / total_words
Repetition_Penalty = min(repetition_ratio, 0.5)
```

#### Final Coherence Score
```
Coherence_Score = Sentence_Similarity × (1 - Repetition_Penalty)
```

### 4. Fluency Scoring

#### Grammar Score
```
If language_tool available:
  Grammar_Score = 1 - (grammar_errors / total_words)
Else (simplified):
  Grammar_Score = 1.0 - penalties
  Penalties: -0.1 for missing capitalization, -0.1 for missing punctuation
```

#### Perplexity Score (Word Diversity)
```
Perplexity_Score = unique_words / total_words
```

#### Final Fluency Score
```
Fluency_Score = 0.6 × Grammar_Score + 0.4 × Perplexity_Score
```

### 5. Efficiency Scoring

#### Raw Efficiency
```
Raw_Efficiency = min(1.0, 3.0 / latency_seconds)
```

#### Normalization
```
Normalized_Efficiency = (Raw_Efficiency - Min_Efficiency) / (Max_Efficiency - Min_Efficiency)
```

## CLMPI Calculation

### Component Weights
```yaml
accuracy: 0.25              # 25% - Factual correctness
contextual_understanding: 0.20  # 20% - Multi-turn conversations
coherence: 0.20             # 20% - Logical flow
fluency: 0.20               # 20% - Language quality
performance_efficiency: 0.15   # 15% - Resource efficiency
```

### Final CLMPI Formula
```
CLMPI_01 = Σ(weight_i × normalized_score_i)
CLMPI_100 = CLMPI_01 × 100
```

## Validation Procedures

### 1. Pre-Evaluation Validation
- [ ] All config files exist and are valid YAML
- [ ] Weights sum to 1.0
- [ ] Generation settings are within valid ranges
- [ ] Datasets are loaded successfully

### 2. Per-Metric Validation
- [ ] Accuracy: 0 ≤ scores ≤ 1
- [ ] Contextual: 0 ≤ scores ≤ 1
- [ ] Coherence: 0 ≤ scores ≤ 1
- [ ] Fluency: 0 ≤ scores ≤ 1
- [ ] Efficiency: 0 ≤ scores ≤ 1

### 3. Post-Evaluation Validation
- [ ] All metrics completed successfully
- [ ] CLMPI score is in [0,1] range
- [ ] Detailed logs saved for each metric
- [ ] Hardware information logged

## Reproducibility

### Hardware Logging
```python
hardware_info = {
    'cpu_model': platform.processor(),
    'cpu_cores': psutil.cpu_count(),
    'memory_gb': psutil.virtual_memory().total / (1024**3),
    'os': platform.system() + ' ' + platform.release(),
    'python_version': platform.python_version()
}
```

### Fixed Random Seed
- **Seed**: 42 (configurable)
- **Purpose**: Deterministic sampling of questions
- **Location**: `summary.json` in results

### Version Control
- **Config Versioning**: All configs have version numbers
- **Formula Documentation**: All formulas documented in this file
- **Dependency Pinning**: Exact versions in `requirements.txt`

## File Structure

```
results/YYYY-MM-DD_HHMMSS_label/
├── summary.json                    # Run summary with hardware info
├── model_results.json             # Individual model results
├── model_detailed/                # Per-metric detailed results
│   ├── accuracy/
│   │   ├── detail.jsonl          # All Q&A pairs with scores
│   │   └── summary.json          # Accuracy summary
│   ├── contextual_understanding/
│   ├── coherence/
│   ├── fluency/
│   └── efficiency/
└── clmpi_scores.json             # Final CLMPI scores
```

## Usage Examples

### Basic Run
```bash
python scripts/enhanced_evaluate_models.py \
    --model-config config/model_config.yaml \
    --generation-config config/generation_config.yaml \
    --device-config config/device_default.yaml
```

### Specific Models
```bash
python scripts/enhanced_evaluate_models.py \
    --model-config config/model_config.yaml \
    --generation-config config/generation_config.yaml \
    --device-config config/device_default.yaml \
    --models phi3:mini mistral \
    --label quick_comparison
```

## Troubleshooting

### Common Issues

1. **Language Tool Not Available**
   - Falls back to simplified grammar checking
   - Install: `pip install language-tool-python`

2. **Model Timeout**
   - Check device config timeout settings
   - Verify Ollama is running and accessible

3. **Insufficient Memory**
   - Reduce number of concurrent evaluations
   - Check device config memory thresholds

### Validation Failures

1. **Weights Don't Sum to 1.0**
   - Check `evaluation_weights` in model config
   - Ensure all weights are positive

2. **Invalid Generation Settings**
   - Check `generation_config.yaml` validation rules
   - Verify temperature, top_p, top_k ranges

3. **Missing Datasets**
   - Verify prompt files exist in `prompts/` directory
   - Check JSON structure matches expected format

## Future Enhancements

1. **Embedding-Based Similarity**
   - Replace word overlap with semantic embeddings
   - Improve context similarity calculation

2. **Advanced Grammar Checking**
   - Integrate multiple grammar checkers
   - Add style and tone analysis

3. **Automated Validation**
   - Add unit tests for all scoring functions
   - Implement automated result validation

4. **Multi-Language Support**
   - Extend to non-English languages
   - Language-specific grammar checking
