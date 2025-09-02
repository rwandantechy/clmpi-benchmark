# Professional CLMPI Usage Guide

## Overview

The CLMPI (Comprehensive Language Model Performance Index) provides a rigorous, reproducible benchmarking framework for evaluating language models across multiple dimensions. The stepwise evaluation system has been successfully tested and produces consistent CLMPI scores.

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
- Models to evaluate
- Evaluation weights
- Prompt set mappings

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

### Stepwise Evaluation (Recommended & Tested)
```bash
# Run each metric individually (tested and working)
python scripts/runners/step_accuracy.py --model "mistral:7b"
python scripts/runners/step_context.py --model "mistral:7b"
python scripts/runners/step_coherence.py --model "mistral:7b"
python scripts/runners/step_fluency.py --model "mistral:7b"
python scripts/runners/step_efficiency.py --model "mistral:7b"

# Combine into final CLMPI score
python scripts/combine_clmpi.py --model "mistral:7b"
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

### Key Files
- **clmpi_summary.json**: Complete CLMPI scores and component breakdowns
- **detail.jsonl**: Per-response detailed scores for analysis
- **summary.json**: Per-metric summary statistics
- **latest/**: Symlink to most recent evaluation

## Evaluation Dimensions

### Accuracy (25%)
- **Method**: Exact Match + F1 scoring
- **Dataset**: GSM-Hard mathematical reasoning
- **Profile**: Deterministic generation
- **Output**: Structured JSON response parsing

### Contextual Understanding (20%)
- **Method**: EM + F1 + context relevance
- **Dataset**: Multi-turn conversations
- **Profile**: Deterministic generation
- **Output**: Context-aware response evaluation

### Coherence (20%)
- **Method**: Sentence similarity + repetition penalty
- **Dataset**: Open-ended prompts
- **Profile**: Creative generation
- **Output**: Internal consistency scoring

### Fluency (20%)
- **Method**: Grammar checking + perplexity
- **Dataset**: Surface quality tasks
- **Profile**: Creative generation
- **Output**: Language quality metrics

### Performance Efficiency (15%)
- **Method**: Latency + memory measurement
- **Dataset**: Accuracy tasks (for consistency)
- **Profile**: Deterministic generation
- **Output**: Resource usage metrics

## Reproducibility

### Hardware Logging
- CPU model and cores
- Memory capacity
- Operating system
- Python version

### Configuration Versioning
- All configs have version numbers
- Dependencies pinned to exact versions
- Random seed fixed (configurable)

### Validation
- Pre-evaluation config validation
- Per-metric score validation
- Post-evaluation result validation

## Verified Results

The system has been successfully tested with Mistral 7B, producing:
- CLMPI_01: 0.637
- CLMPI_100: 63.72
- Component Scores: All 5 dimensions successfully evaluated
- Reproducibility: Consistent results across runs

## Troubleshooting

### Common Issues

1. **Model Not Found**
   - Verify model is installed: `ollama list`
   - Check model name in config matches Ollama name

2. **Insufficient Memory**
   - Reduce concurrent evaluations
   - Check device config memory thresholds
   - Close other applications

3. **Timeout Errors**
   - Increase timeout in device config
   - Check Ollama service status
   - Verify network connectivity

4. **Import Errors**
   - Install missing dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility

### Validation Failures

1. **Config Validation**
   - Verify YAML syntax
   - Check required fields present
   - Ensure weights sum to 1.0

2. **Dataset Validation**
   - Verify JSON syntax
   - Check required fields in datasets
   - Ensure prompt files exist

3. **Score Validation**
   - All scores should be in [0,1] range
   - CLMPI score should be in [0,1] range
   - Check for NaN or infinite values

## Best Practices

### Evaluation Setup
1. Use clean system state
2. Close unnecessary applications
3. Ensure consistent network conditions
4. Document hardware configuration

### Model Selection
1. Pull only the models you need for evaluation
2. Add models to config only when ready to evaluate
3. Remove models from config after use to keep it clean
4. Consider hardware constraints and model size
5. Test with small samples first

### Result Analysis
1. Review detailed logs for anomalies
2. Compare across multiple runs
3. Consider hardware variations
4. Document any deviations

## Support

For technical support:
1. Check documentation in `/docs`
2. Review troubleshooting section
3. Examine error logs in results
4. Validate system with test script

## Version Information

- **CLMPI Version**: 1.0.0
- **CLMPI System**: 1.0.0
- **Last Updated**: 2025
- **Status**: Tested and Working
