# CLMPI Quick Start Guide

**Get CLMPI scores in 3 simple steps**

This guide will walk you through evaluating a language model using the tested and working automated CLMPI system.

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

## Step 3: Run Everything

```bash
# One command runs everything automatically
python scripts/run_clmpi.py --model "mistral:7b"
```

That's it! The system will automatically:
- Check if the model is available
- Run all 5 evaluation metrics (accuracy, context, coherence, fluency, efficiency)
- Combine results into final CLMPI score
- Display results and save everything

## What Each Metric Tests

- **Accuracy (25%)**: Mathematical reasoning and factual correctness
- **Context (20%)**: Understanding multi-turn conversations
- **Coherence (20%)**: Logical flow and narrative structure
- **Fluency (20%)**: Language quality and grammatical correctness
- **Efficiency (15%)**: Resource usage and inference speed

## Testing Multiple Models

**Super Simple**: Just pull and run different models:

```bash
# Pull additional models
ollama pull phi3:mini
ollama pull llama3.1:8b

# Run evaluation immediately
python scripts/run_clmpi.py --model "phi3:mini"
python scripts/run_clmpi.py --model "llama3.1:8b"
```

## View Results

Results are automatically displayed after each run, but you can also view them separately:

```bash
# Show latest results without running evaluation
python scripts/run_clmpi.py --model "any_model" --show-results

# Check the results directory
ls -l results/latest/
cat results/latest/clmpi_summary.json
```

## Expected Output

You should see output like this:

```
Looking for model 'mistral:7b'...
Great! Found 'mistral:7b', starting evaluation...
============================================================
Running accuracy evaluation...
✓ accuracy completed successfully
Running context evaluation...
✓ context completed successfully
Running coherence evaluation...
✓ coherence completed successfully
Running fluency evaluation...
✓ fluency completed successfully
Running efficiency evaluation...
✓ efficiency completed successfully

Combining results into final CLMPI score...
✓ CLMPI score calculated successfully

All done! Everything completed successfully.

============================================================
CLMPI Results from: 2025-09-01_142730_stepwise
==================================================
CLMPI_01:  0.637
CLMPI_100: 63.72

Component Scores:
  accuracy              0.000 (contributes 0.000)
  context              1.000 (contributes 0.200)
  coherence            0.898 (contributes 0.180)
  fluency              0.943 (contributes 0.189)
  efficiency           0.460 (contributes 0.069)

Total time: 45.2 seconds
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
4. **Advanced usage**: Use individual step runners for specific needs

## Success Indicators

- All 5 metric runners complete without errors  
- `clmpi_summary.json` is generated  
- CLMPI scores are between 0 and 1 (or 0 and 100)  
- Individual metric scores are between 0 and 1  

**You're ready to benchmark language models!**
