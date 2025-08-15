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

# 2. Install Ollama models
ollama pull phi3:mini
ollama pull mistral

# 3. Run benchmark
python scripts/evaluate_models.py \
    --config config/model_config.yaml \
    --device config/device_default.yaml \
    --models phi3:mini mistral \
    --output results/edge_demo

# 4. View results
ls results/edge_demo/
cat results/edge_demo/summary.json
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

Edit `config/device_default.yaml` for your hardware.

### Add Models

1. Pull via Ollama: `ollama pull <model_name>`
2. Add to `config/model_config.yaml`
3. Run benchmark

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
