# CLMPI Benchmark

**Comprehensive Language Model Performance Index**

A practical implementation of Maikel Leon's unified performance ranking metric for evaluating open-source Large Language Models. Successfully tested and producing reproducible CLMPI scores.

## What & Why

CLMPI evaluates language models across 5 dimensions for edge deployment suitability:
- **Accuracy** (25%) - Factual correctness and mathematical reasoning
- **Contextual Understanding** (20%) - Multi-turn conversations and comprehension  
- **Coherence** (20%) - Logical flow and narrative structure
- **Fluency** (20%) - Language quality and grammatical correctness
- **Performance Efficiency** (15%) - Resource usage and inference speed

**Verified Working**: The stepwise evaluation system has been successfully tested with Mistral 7B, producing reproducible CLMPI scores.

## Quick Start

### One Command Evaluation (Recommended)
```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Pull a model
ollama pull mistral:7b

# 3) Run everything with one command
python scripts/run_clmpi.py --model "mistral:7b"
```

That's it! The system will automatically:
- Check if the model is available
- Run all 5 evaluation metrics
- Combine results into final CLMPI score
- Display results and save everything

### Stepwise Evaluation (Advanced)
```bash
# Run each metric individually if needed
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
# Run all metrics at once (legacy approach)
python scripts/evaluate_models.py \
    --model-config config/model_config.yaml \
    --generation-config config/generation_config.yaml \
    --device-config config/device_default.yaml \
    --models mistral:7b
```

## What You Get

### CLMPI Scores
- **CLMPI_01**: Normalized score [0,1] for academic use
- **CLMPI_100**: Scaled score [0,100] for intuitive comparison
- **Component Breakdown**: Individual scores for each dimension

### Detailed Results
- **Per-metric evaluations**: Detailed scoring for each dimension
- **Response analysis**: Raw model outputs with scoring breakdown
- **Hardware logging**: System specifications and performance metrics
- **Reproducibility**: Fixed random seed (42) for consistent results

### Example Output
```json
{
  "clmpi_scores": {
    "clmpi_01": 0.637,
    "clmpi_100": 63.72
  },
  "component_scores": {
    "accuracy": {"score": 0.0, "contribution": 0.0},
    "context": {"score": 1.0, "contribution": 0.2},
    "coherence": {"score": 0.898, "contribution": 0.180},
    "fluency": {"score": 0.943, "contribution": 0.189},
    "efficiency": {"score": 0.460, "contribution": 0.069}
  }
}
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

#### Alternative Weight Configurations

**Edge-First (Strict)** - Maximum efficiency focus:
```yaml
accuracy: 0.30              # 30% - Factual correctness
contextual_understanding: 0.15  # 15% - Multi-turn conversations
coherence: 0.10             # 10% - Logical flow
fluency: 0.05               # 5% - Language quality
performance_efficiency: 0.40   # 40% - Resource efficiency
```

**Edge-Balanced (Practical)** - Balanced quality and efficiency:
```yaml
accuracy: 0.35              # 35% - Factual correctness
contextual_understanding: 0.15  # 15% - Multi-turn conversations
coherence: 0.10             # 10% - Logical flow
fluency: 0.05               # 5% - Language quality
performance_efficiency: 0.35   # 35% - Resource efficiency
```

#### Why This Shape Works

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

Edit `config/device_default.yaml` for your hardware.

### Adding Models

**Super Simple**: Just pull and run - no configuration needed!
```bash
# Pull any model
ollama pull phi3:mini
ollama pull llama3.1:8b

# Run evaluation immediately
python scripts/run_clmpi.py --model "phi3:mini"
python scripts/run_clmpi.py --model "llama3.1:8b"
```

**Advanced Configuration**: For custom settings, edit `config/model_config.yaml`:
```yaml
models:
  your_model_name:
    ollama_name: "your_model_name"
    timeout_seconds: 30
```

## Outputs

- **`results/<timestamp>_<name>/`** - Run results with timestamp
- **`results/latest/`** - Symlink to most recent run
- **`clmpi_summary.json`** - Machine-readable results with CLMPI scores
- **`<metric>/detail.jsonl`** - Detailed per-response scoring
- **`<metric>/summary.json`** - Per-metric summary statistics

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
