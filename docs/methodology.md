# CLMPI Methodology

## Overview

The CLMPI (Comprehensive Language Model Performance Index) implements a benchmarking methodology for evaluating language models across multiple dimensions. This system has been tested with several models and produces CLMPI scores based on standardized generation settings and prompt datasets.

## Current Implementation Status

**Note**: This is a working prototype with limited prompt coverage. Each metric currently uses 1 prompt per evaluation dimension.

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

### 2. Current Prompt Datasets

#### Accuracy Dataset (`accuracy.json`)
- **Source**: Single question from GSM-Hard mathematical reasoning dataset
- **Structure**: Mathematical calculation with expected numeric answer
- **Current Coverage**: 1 question
- **Format**: JSON response with structured answer parsing

#### Contextual Understanding Dataset (`context.json`)
- **Source**: Single question from SQuAD dataset
- **Structure**: Context + question + gold answer
- **Current Coverage**: 1 question
- **Format**: Structured response parsing

#### Coherence Dataset (`coherence.json`)
- **Source**: Single coherence evaluation prompt
- **Structure**: Open-ended prompt requiring logical flow
- **Current Coverage**: 1 question
- **Format**: Creative generation with coherence scoring

#### Fluency Dataset (`fluency.json`)
- **Source**: Single fluency evaluation prompt
- **Structure**: Descriptive task for language quality
- **Current Coverage**: 1 question
- **Format**: Creative generation with fluency scoring

#### Efficiency Dataset (`efficiency_tasks.json`)
- **Source**: Single computational task
- **Structure**: Task requiring consistent processing
- **Current Coverage**: 1 question
- **Format**: Deterministic generation with timing

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

### Weight Configurations

CLMPI supports multiple weight configurations optimized for different deployment scenarios:

#### Current Default (Edge-Focused)
```yaml
accuracy: 0.25              # 25% - Factual correctness
contextual_understanding: 0.20  # 20% - Multi-turn conversations
coherence: 0.20             # 20% - Logical flow
fluency: 0.20               # 20% - Language quality
performance_efficiency: 0.15   # 15% - Resource efficiency
```

#### Edge-First (Strict)
```yaml
accuracy: 0.30              # 30% - Factual correctness (highest quality weight)
contextual_understanding: 0.15  # 15% - Multi-turn conversations
coherence: 0.10             # 10% - Logical flow
fluency: 0.05               # 5% - Language quality
performance_efficiency: 0.40   # 40% - Resource efficiency (highest efficiency weight)
```

#### Edge-Balanced (Practical Default)
```yaml
accuracy: 0.35              # 35% - Factual correctness
contextual_understanding: 0.15  # 15% - Multi-turn conversations
coherence: 0.10             # 10% - Logical flow
fluency: 0.05               # 5% - Language quality
performance_efficiency: 0.35   # 35% - Resource efficiency
```

### Why This Shape Works

The edge-focused weight distributions are designed based on real-world deployment constraints:

1. **Accuracy gets the largest single quality weight (0.30–0.35)**
   - On edge devices, wrong-but-fast is useless
   - Factual correctness is critical for user trust

2. **Context > Coherence > Fluency priority**
   - Instruction-following (context) matters more than perfect prose
   - Logical flow (coherence) is more important than grammatical perfection
   - Surface quality (fluency) is the least critical for edge applications

3. **Efficiency is heavyweight (0.35–0.40)**
   - Battery life, thermal management, and UX latency are hard constraints
   - Resource efficiency directly impacts device usability
   - On-device performance cannot be ignored

### Final CLMPI Formula
```
CLMPI_01 = Σ(weight_i × normalized_score_i)
CLMPI_100 = CLMPI_01 × 100
```

## Current Limitations

### 1. Limited Prompt Coverage
- Each metric currently uses only 1 prompt
- Results may not be statistically representative
- Single-point evaluations can be misleading

### 2. Dataset Sources
- Prompts are sourced from public datasets
- Limited validation of prompt quality
- No custom prompt engineering

### 3. Statistical Significance
- Single prompt evaluations lack statistical power
- Results should be interpreted with caution
- Comparative analysis between models is limited

## Validation Procedures

### 1. Pre-Evaluation Validation
- All config files exist and are valid YAML
- Weights sum to 1.0
- Generation settings are within valid ranges
- Datasets are loaded successfully

### 2. Per-Metric Validation
- Accuracy: 0 ≤ scores ≤ 1
- Contextual: 0 ≤ scores ≤ 1
- Coherence: 0 ≤ scores ≤ 1
- Fluency: 0 ≤ scores ≤ 1
- Efficiency: 0 ≤ scores ≤ 1

### 3. Post-Evaluation Validation
- All metrics completed successfully
- CLMPI score is in [0,1] range
- Detailed logs saved for each metric
- Hardware information logged

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
- **Location**: `clmpi_summary.json` in results

### Version Control
- **Config Versioning**: All configs have version numbers
- **Formula Documentation**: All formulas documented in this file
- **Dependency Pinning**: Exact versions in `requirements.txt`

## File Structure

```
results/YYYY-MM-DD_HHMMSS_label/
├── clmpi_summary.json             # Final CLMPI scores and metadata
├── accuracy/
│   ├── detail.jsonl              # All Q&A pairs with scores
│   └── summary.json              # Accuracy summary
├── context/
│   ├── detail.jsonl              # Context evaluation details
│   └── summary.json              # Context summary
├── coherence/
│   ├── detail.jsonl              # Coherence evaluation details
│   └── summary.json              # Coherence summary
├── fluency/
│   ├── detail.jsonl              # Fluency evaluation details
│   └── summary.json              # Fluency summary
└── efficiency/
    ├── detail.jsonl              # Efficiency evaluation details
    └── summary.json              # Efficiency summary
```

## Usage Examples

### Stepwise Evaluation (Current Implementation)
```bash
# Run each metric individually
python scripts/runners/step_accuracy.py --model "mistral:7b"
python scripts/runners/step_context.py --model "mistral:7b"
python scripts/runners/step_coherence.py --model "mistral:7b"
python scripts/runners/step_fluency.py --model "mistral:7b"
python scripts/runners/step_efficiency.py --model "mistral:7b"

# Combine results
python scripts/combine_clmpi.py --results-dir results/YYYY-MM-DD_HHMMSS_stepwise
```

### Complete Evaluation (Legacy)
```bash
python scripts/evaluate_models.py \
    --model-config config/model_config.yaml \
    --generation-config config/generation_config.yaml \
    --device-config config/device_default.yaml
```

## Tested Results

The system has been tested with several models:

- **Mistral 7B**: CLMPI_100: 64.5 (current run)
- **Phi3 Mini**: CLMPI_100: 70.6 (previous run)
- **Qwen2.5 0.5B**: CLMPI_100: 70.6 (previous run)

**Note**: These scores are based on single-prompt evaluations and should be interpreted with caution. Only Mistral 7B has complete results in the current evaluation run.

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

1. **Expand Prompt Coverage**
   - Add more prompts per metric for statistical significance
   - Implement prompt sampling strategies
   - Create custom prompt engineering

2. **Embedding-Based Similarity**
   - Replace word overlap with semantic embeddings
   - Improve context similarity calculation

3. **Advanced Grammar Checking**
   - Integrate multiple grammar checkers
   - Add style and tone analysis

4. **Automated Validation**
   - Add unit tests for all scoring functions
   - Implement automated result validation

5. **Multi-Language Support**
   - Extend to non-English languages
   - Language-specific grammar checking
