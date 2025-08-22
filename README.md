# CLMPI Benchmark

**Comprehensive Language Model Performance Index**

A practical implementation of Maikel Leon's unified performance ranking metric for evaluating open-source Large Language Models.

## What & Why

CLMPI evaluates language models across 5 dimensions for edge deployment suitability:
- **Accuracy** (25%) - Factual correctness
- **Contextual Understanding** (20%) - Multi-turn conversations  
- **Coherence** (20%) - Logical flow
- **Fluency** (20%) - Language quality
- **Performance Efficiency** (15%) - Resource usage

**Note**: Legacy scripts are deprecated. Use the enhanced pipeline for professional results.

## Quick Start

### Complete Evaluation (All Metrics)
```bash
# 1) Install
pip install -r requirements.txt

# 2) Pull at least one model
ollama pull phi3:mini

# 3) Run the enhanced benchmark (recommended)
python scripts/enhanced_evaluate_models.py \
    --model-config config/model_config.yaml \
    --generation-config config/generation_config.yaml \
    --device-config config/device_default.yaml \
    --models phi3:mini

# 4) Inspect outputs
ls -l results/latest/
```

### Stepwise Evaluation (Individual Metrics)
```bash
# Run each metric individually
python scripts/runners/step_accuracy.py --model phi3:mini
python scripts/runners/step_context.py --model phi3:mini
python scripts/runners/step_coherence.py --model phi3:mini
python scripts/runners/step_fluency.py --model phi3:mini
python scripts/runners/step_efficiency.py --model phi3:mini

# Combine into final CLMPI score
python scripts/combine_clmpi.py --model phi3:mini --detailed
```

Note: Legacy scripts are deprecated; see the deprecation notice at the top of those files.

## Configure

### Weights & Devices

Edit `config/model_config.yaml`:
```yaml
evaluation_weights:
  accuracy: 0.25
  contextual_understanding: 0.20
  coherence: 0.20
  fluency: 0.20
  performance_efficiency: 0.15
```

**Note**: Weights are edge-focused with efficiency at 15% for on-device viability.

Edit `config/device_default.yaml` for your hardware.

### Add Models

1. Pull your model: `ollama pull <your_model_name>`
2. Copy `docs/examples/model_config_example.yaml` to `config/model_config.yaml` and edit for your models:
   ```yaml
   models:
     your_model_name:
       ollama_name: "your_model_name"
       timeout_seconds: 30
   ```
3. Run evaluation
4. Remove model from config after use if desired

## Outputs

- **`results/<timestamp>_<name>/`** - Run results with timestamp
- **`results/latest/`** - Symlink to most recent run
- **`summary.json`** - Machine-readable results
- **`evaluations/visualizations/`** - Radar and bar charts
- **`evaluations/clmpi_scorebook.xlsx`** - Excel comparison

## Reproducibility

- Fixed random seed (42) for consistent results
- Hardware information logged automatically
- Clean runs recommended (restart runtime, clear caches)
- All configurations versioned

## Citation

Based on: "Benchmarking Large Language Models with a Unified Performance Ranking Metric" by Maikel Leon (University of Miami, 2024)

## License

MIT License - see [LICENSE](LICENSE) file.

---

**For detailed documentation, see `/docs`**
