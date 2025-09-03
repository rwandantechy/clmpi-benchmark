# CLMPI Usage Guide

## Overview

The CLMPI (Comprehensive Language Model Performance Index) provides a benchmarking framework for evaluating language models across multiple dimensions. The stepwise evaluation system has been tested with several models and produces CLMPI scores.

## Current Status

**Note**: This is a working prototype. Each metric currently uses 1 prompt per evaluation dimension, which limits statistical significance.

## System Requirements

### Software Dependencies
- Python 3.8+
- Ollama runtime
- Required Python packages (see requirements.txt)

### Hardware Requirements
- Minimum 8GB RAM
- CPU with 4+ cores recommended
- SSD storage for optimal performance

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd clmpi-benchmark
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama Models
```bash
# Pull models as needed for your evaluation
ollama pull mistral:7b
```

## Configuration

### Model Configuration
Edit `config/model_config.yaml` to define:
- Evaluation weights
- Prompt set mappings
- Default model settings (optional)

### Generation Configuration
Edit `config/generation_config.yaml` to set:
- Standardized generation profiles
- Model-specific overrides
- Validation rules

### Device Configuration
Edit `config/device_default.yaml` to specify:
- Hardware specifications
- Performance thresholds
- Runtime parameters

## Usage

### Stepwise Evaluation (Current Implementation)
```bash
# Run each metric individually
python scripts/runners/step_accuracy.py --model "mistral:7b"
python scripts/runners/step_context.py --model "mistral:7b"
python scripts/runners/step_coherence.py --model "mistral:7b"
python scripts/runners/step_fluency.py --model "mistral:7b"
python scripts/runners/step_efficiency.py --model "mistral:7b"

# Combine into final CLMPI score
python scripts/combine_clmpi.py --results-dir results/YYYY-MM-DD_HHMMSS_stepwise
```

### Testing Multiple Models
**Simple Approach**: Use models directly without configuration changes:

```bash
# Pull different models
ollama pull phi3:mini
ollama pull llama3.1:8b

# Test them directly
python scripts/runners/step_accuracy.py --model "phi3:mini"
python scripts/runners/step_accuracy.py --model "llama3.1:8b"
```

### Complete Evaluation (Legacy)
```bash
python scripts/evaluate_models.py \
    --model-config config/model_config.yaml \
    --generation-config config/generation_config.yaml \
    --device-config config/device_default.yaml \
    --models mistral:7b
```

### System Validation
```bash
python scripts/test_system.py
```

## Output Structure

### Results Directory
```
results/YYYY-MM-DD_HHMMSS_stepwise/
├── clmpi_summary.json          # Final CLMPI scores
├── accuracy/                   # Accuracy evaluation results
├── context/                    # Context evaluation results
├── coherence/                  # Coherence evaluation results
├── fluency/                    # Fluency evaluation results
└── efficiency/                 # Efficiency evaluation results
```

### Model Responses
```
results/model_responses/
├── model_name_1/
│   ├── accuracy_responses.md
│   ├── context_responses.md
│   ├── coherence_responses.md
│   ├── fluency_responses.md
│   └── efficiency_responses.md
└── model_name_2/
    └── ... (same structure)
```

### Key Files

#### `clmpi_summary.json`
Contains final CLMPI scores and component breakdown:
```json
{
  "clmpi_scores": {
    "clmpi_01": 0.706,
    "clmpi_100": 70.6
  },
  "component_scores": {
    "accuracy": {"score": 0.0, "contribution": 0.0},
    "context": {"score": 1.0, "contribution": 0.2},
    "coherence": {"score": 0.964, "contribution": 0.181},
    "fluency": {"score": 0.920, "contribution": 0.184},
    "efficiency": {"score": 0.860, "contribution": 0.129}
  }
}
```

#### Individual Metric Results
Each metric directory contains:
- `summary.json` - Aggregated scores and metadata
- `detail.jsonl` - Detailed evaluation results
- `evaluation_report.md` - Human-readable summary

## Evaluation Dimensions

### 1. Accuracy (25% weight)
- **Purpose**: Measure factual correctness
- **Method**: Mathematical reasoning problems
- **Current Coverage**: 1 question from GSM-Hard dataset
- **Scoring**: F1 score based on word overlap

### 2. Contextual Understanding (20% weight)
- **Purpose**: Measure multi-turn conversation ability
- **Method**: Context + question comprehension
- **Current Coverage**: 1 question from SQuAD dataset
- **Scoring**: Combined F1 and context similarity

### 3. Coherence (20% weight)
- **Purpose**: Measure logical flow and consistency
- **Method**: Open-ended narrative generation
- **Current Coverage**: 1 custom prompt
- **Scoring**: Sentence similarity with repetition penalty

### 4. Fluency (20% weight)
- **Purpose**: Measure language quality
- **Method**: Descriptive text generation
- **Current Coverage**: 1 custom prompt
- **Scoring**: Grammar and word diversity

### 5. Efficiency (15% weight)
- **Purpose**: Measure resource usage
- **Method**: Performance timing and memory
- **Current Coverage**: 1 computational task
- **Scoring**: Latency and resource efficiency

## Reproducibility

### Fixed Random Seed
- **Seed**: 42 (configurable in config)
- **Purpose**: Deterministic question sampling
- **Location**: `clmpi_summary.json`

### Hardware Logging
System automatically logs:
- CPU model and cores
- Memory capacity
- Operating system
- Python version

### Configuration Versioning
All configs are version-controlled and documented in the repository.

## Limitations and Considerations

### 1. Limited Statistical Power
- Single prompt per metric limits statistical significance
- Results should be interpreted with caution
- Comparative analysis between models is limited

### 2. Prompt Quality
- Prompts sourced from public datasets
- Limited validation of prompt effectiveness
- No custom prompt engineering

### 3. Model Variability
- Single evaluation run per model
- No confidence intervals
- Results may vary with different prompts

## Troubleshooting

### Common Issues

1. **Model Timeout**
   - Check Ollama is running: `ollama list`
   - Verify model is available: `ollama list | grep model_name`
   - Large models may need longer timeouts

2. **Memory Issues**
   - Close other applications
   - Use smaller models for testing
   - Check available RAM: `free -h`

3. **Configuration Errors**
   - Verify YAML syntax in config files
   - Check file paths exist
   - Ensure weights sum to 1.0

### Validation

1. **Check Results Structure**
   - Verify all metric directories exist
   - Check `clmpi_summary.json` is generated
   - Confirm model responses are saved

2. **Verify Scores**
   - All scores should be in [0,1] range
   - Component contributions should sum to CLMPI score
   - Weights should match configuration

## Version Information

- **Current Version**: 1.1.0
- **Last Updated**: 2025-09-02
- **Tested Models**: Mistral 7B, Llama3.1 8B, Gemma2 2B, Phi3 Mini, Qwen2.5 0.5B
- **Status**: Working prototype with limited prompt coverage

## Next Steps

To improve the system:
1. **Expand prompt coverage** for statistical significance
2. **Add more evaluation metrics** for comprehensive assessment
3. **Implement confidence intervals** for result reliability
4. **Create custom prompt engineering** for better evaluation
