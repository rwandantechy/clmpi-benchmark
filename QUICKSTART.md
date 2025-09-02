# CLMPI Quick Start Guide

**Get CLMPI scores in 5 simple steps**

This guide will walk you through evaluating a language model using the tested and working stepwise evaluation system.

## Prerequisites

- Python 3.8+
- Ollama runtime installed and running
- At least 8GB RAM

## Step 1: Setup

```bash
# Clone and setup
git clone <repository-url>
cd clmpi-benchmark
pip install -r requirements.txt
```

## Step 2: Pull a Model

```bash
# Pull a model to test (Mistral 7B is recommended for testing)
ollama pull mistral:7b

# Verify it's available
ollama list
```

## Step 3: Run Individual Metrics

Run each metric evaluation separately:

```bash
# Accuracy (25% weight) - Mathematical reasoning
python scripts/runners/step_accuracy.py --model "mistral:7b"

# Context (20% weight) - Multi-turn conversations  
python scripts/runners/step_context.py --model "mistral:7b"

# Coherence (20% weight) - Logical flow
python scripts/runners/step_coherence.py --model "mistral:7b"

# Fluency (20% weight) - Language quality
python scripts/runners/step_fluency.py --model "mistral:7b"

# Efficiency (15% weight) - Resource usage
python scripts/runners/step_efficiency.py --model "mistral:7b"
```

## Step 4: Generate CLMPI Score

```bash
# Combine all metrics into final CLMPI score
python scripts/combine_clmpi.py --model "mistral:7b"
```

## Step 5: View Results

```bash
# Check the latest results
ls -l results/latest/

# View the CLMPI summary
cat results/latest/clmpi_summary.json

# View individual metric results
cat results/latest/accuracy/summary.json
cat results/latest/context/summary.json
cat results/latest/coherence/summary.json
cat results/latest/fluency/summary.json
cat results/latest/efficiency/summary.json
```

## Expected Output

You should see a `clmpi_summary.json` file with:

```json
{
  "model": "mistral:7b",
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

## What Each Metric Tests

- **Accuracy (25%)**: Mathematical reasoning and factual correctness
- **Context (20%)**: Understanding multi-turn conversations
- **Coherence (20%)**: Logical flow and narrative structure
- **Fluency (20%)**: Language quality and grammatical correctness
- **Efficiency (15%)**: Resource usage and inference speed

## Testing Multiple Models

**Simple**: Just pull and use different models without config changes:

```bash
# Pull additional models
ollama pull phi3:mini
ollama pull llama3.1:8b

# Test them directly
python scripts/runners/step_accuracy.py --model "phi3:mini"
python scripts/runners/step_accuracy.py --model "llama3.1:8b"
```

## Troubleshooting

### Common Issues

1. **"Model not found"**
   - Run `ollama list` to see available models
   - Make sure you pulled the model: `ollama pull mistral:7b`

2. **Memory errors**
   - Close other applications
   - Ensure you have at least 8GB RAM available

3. **Timeout errors**
   - Check that Ollama is running: `ollama serve`
   - Increase timeout in config if needed

### Getting Help

- Check the detailed documentation in `/docs`
- Review error messages in the terminal output
- Verify all dependencies are installed: `pip list`

## Next Steps

Once you've successfully run the evaluation:

1. **Try different models**: Pull and test other models
2. **Customize weights**: Edit `config/model_config.yaml` if needed
3. **Add custom prompts**: Modify files in `prompts/` directory
4. **Batch evaluation**: Use the legacy `evaluate_models.py` for multiple models

## Success Indicators

- All 5 metric runners complete without errors  
- `clmpi_summary.json` is generated  
- CLMPI scores are between 0 and 1 (or 0 and 100)  
- Individual metric scores are between 0 and 1  

**You're ready to benchmark language models!**
