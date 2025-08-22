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

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test system
python scripts/test_enhanced_system.py

# 3. Pull your model via Ollama
ollama pull <your_model_name>

# 4. Add model to config/model_config.yaml
# 5. Run evaluation
python scripts/enhanced_evaluate_models.py \
    --model-config config/model_config.yaml \
    --generation-config config/generation_config.yaml \
    --device-config config/device_default.yaml \
    --models <your_model_name> \
    --label evaluation

# 6. View results
ls results/$(date +%Y-%m-%d)_*_evaluation/
cat results/latest/summary.json
```

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
