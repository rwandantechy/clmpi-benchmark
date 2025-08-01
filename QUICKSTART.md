# CLMPI Benchmark - Quick Start Guide

## 🚀 Getting Started

This guide will help you quickly set up and run the CLMPI (Comprehensive Language Model Performance Index) benchmark framework.

## Prerequisites

- Python 3.8 or higher
- Git
- pip (Python package installer)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rwandantechy/clmpi-benchmark.git
   cd clmpi-benchmark
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

## Quick Test

Run a quick test to verify everything is working:

```bash
python scripts/clmpi_calculator.py
```

You should see output like:
```
CLMPI Score: 17.87/25
Interpretation: Good - Model shows strong performance with minor areas for improvement
Report saved to: models/outputs/example_llm_clmpi_report.json
```

## Running Evaluations

### Basic Evaluation

```bash
python scripts/evaluate_models.py --config config/model_config.yaml --output results/
```

### Custom Configuration

1. **Edit the model configuration**
   ```bash
   # Edit config/model_config.yaml to add your models
   nano config/model_config.yaml
   ```

2. **Add your API keys** (if using cloud models)
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   ```

3. **Run evaluation**
   ```bash
   python scripts/evaluate_models.py --config config/model_config.yaml --output results/
   ```

## Understanding the Results

After running an evaluation, you'll find:

- **`results/model_comparison.csv`** - Comparison table of all models
- **`results/clmpi_scores_comparison.png`** - Bar chart of CLMPI scores
- **`results/component_scores_radar.png`** - Radar chart of component scores
- **`results/component_scores_heatmap.png`** - Heatmap of all scores
- **`results/detailed_results.json`** - Detailed evaluation reports

## CLMPI Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 20-25 | Excellent - Outstanding performance across all dimensions |
| 15-20 | Good - Strong performance with minor areas for improvement |
| 10-15 | Fair - Adequate performance with significant room for improvement |
| 5-10 | Poor - Limited capabilities across multiple dimensions |
| 0-5 | Very Poor - Requires substantial improvement |

## Components Explained

The CLMPI score is calculated from five components:

1. **Accuracy (25%)** - Factual and grammatical correctness
2. **Contextual Understanding (20%)** - Ability to use conversation context
3. **Coherence (20%)** - Logical flow and structural soundness
4. **Fluency (20%)** - Linguistic smoothness and readability
5. **Resource Efficiency (15%)** - Computational resource usage

## Adding New Models

1. **Edit `config/model_config.yaml`**
   ```yaml
   models:
     your-model-name:
       type: "openai"  # or "anthropic", "local"
       model_name: "your-model"
       api_key_env: "YOUR_API_KEY"
       max_tokens: 1000
       temperature: 0.1
   ```

2. **Set your API key**
   ```bash
   export YOUR_API_KEY="your-api-key"
   ```

3. **Run evaluation**
   ```bash
   python scripts/evaluate_models.py --config config/model_config.yaml
   ```

## Customizing Weights

You can customize the importance of each component:

```yaml
evaluation_weights:
  accuracy: 0.30  # Increase accuracy importance
  contextual_understanding: 0.20
  coherence: 0.20
  fluency: 0.20
  resource_efficiency: 0.10  # Decrease efficiency importance
```

## Troubleshooting

### Common Issues

1. **Import errors**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **API key errors**
   - Ensure your API keys are set as environment variables
   - Check that your API keys are valid and have sufficient credits

3. **Memory issues**
   - Reduce `max_tokens` in the configuration
   - Use smaller models for local evaluation

### Getting Help

- Check the logs in `evaluation.log`
- Review the documentation in `docs/`
- Open an issue on GitHub

## Next Steps

1. **Read the full documentation** in `docs/`
2. **Explore the methodology** in `docs/clmpi_definition.md`
3. **Customize the framework** for your specific needs
4. **Contribute** to the project

## Citation

This implementation is based on the research paper:

```bibtex
@article{leon2024benchmarking,
  title={Benchmarking Large Language Models with a Unified Performance Ranking Metric},
  author={Leon, Maikel},
  journal={International Journal on Foundations of Computer Science & Technology},
  volume={4},
  number={4},
  pages={15--25},
  year={2024},
  publisher={University of Miami, Florida, USA}
}
```

If you use this CLMPI implementation in your research, please cite both the original paper and this implementation:

```bibtex
@software{clmpi_benchmark_implementation,
  title={CLMPI Benchmark: Implementation of Comprehensive Language Model Performance Index},
  author={Developer Implementation},
  year={2024},
  url={https://github.com/rwandantechy/clmpi-benchmark},
  note={Based on Leon (2024) unified performance ranking metric}
}
```

---

**Happy benchmarking! 🎯** 