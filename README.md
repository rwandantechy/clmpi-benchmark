# CLMPI Benchmark

**Comprehensive Language Model Performance Index**

A practical implementation of the unified performance ranking metric proposed by Maikel Leon (University of Miami, 2024). This framework provides a **local edge deployment benchmarking system** for evaluating open-source Large Language Models across cognitive tasks and reasoning capabilities.

## Overview

CLMPI (Comprehensive Language Model Performance Index) is a practical implementation of the unified performance ranking metric proposed by Maikel Leon in "Benchmarking Large Language Models with a Unified Performance Ranking Metric" (2024). This framework provides a **local benchmarking system** designed to assess the performance of open-source LLMs (DeepSeek, LLaMA, Mistral, Phi families) for edge deployment scenarios, optimized for CPU-based workflows on macOS.

## Features

- **Local Edge Deployment Focus**: Optimized for running models locally via Ollama runtime
- **Open-Source Model Support**: DeepSeek, LLaMA, Mistral, Phi families
- **Mac-Optimized Workflow**: Designed for MacBook Pro 2019 CPU-based evaluation
- **Four-Core Evaluation**: Accuracy, Contextual Understanding, Fluency & Coherence, Performance Efficiency
- **Automated Metrics Collection**: BERTScore, psutil monitoring, token throughput
- **Fair Comparison Framework**: Consistent prompts and evaluation across all models
- **Visualization Tools**: Radar charts, bar charts, Excel scorebook integration
- **Modular Architecture**: Organized scripts, prompts, configs, logs, and evaluations

## Target Models

This framework is specifically designed for evaluating:

- **Phi Family**: phi3:mini, phi3:small, phi3:medium
- **Mistral Family**: mistral, mixtral, mistral-openorca
- **LLaMA Family**: llama2:7b-chat, llama2:13b-chat, llama2:70b-chat
- **DeepSeek Family**: deepseek-coder, deepseek-llm
- **Gemma Family**: gemma:2b, gemma:7b
- **Other Open-Source Models**: Available through Ollama

## Evaluation Axes

### 1. Accuracy (25%)
- **Classification Tasks**: Sentiment analysis, topic classification, intent detection
- **Question Answering**: Factual accuracy on curated datasets
- **Mathematical Reasoning**: Problem-solving capabilities

### 2. Contextual Understanding (20%)
- **Multi-turn Conversations**: Maintaining coherence across extended dialogues
- **Long Context Processing**: Ability to handle and reference lengthy prompts
- **Logical Flow**: Consistent reasoning throughout responses

### 3. Fluency & Coherence (20%)
- **Grammatical Quality**: Automated assessment using BERTScore
- **Stylistic Consistency**: Human rubric evaluation (optional)
- **Structural Soundness**: Logical organization of responses

### 4. Performance Efficiency (35%)
- **Latency**: Response time measurement
- **CPU Usage**: Resource utilization monitoring via psutil
- **Memory Footprint**: RAM consumption tracking
- **Token Throughput**: Processing speed metrics

## Project Structure

```
clmpi-benchmark/
├── config/                 # Model and device configurations
│   ├── model_config.yaml  # Ollama model specifications
│   └── macbook_pro_2019.yaml # Device-specific settings
├── scripts/                # Core benchmarking scripts
│   ├── clmpi_calculator.py # CLMPI scoring engine
│   ├── evaluate_models.py  # Main evaluation orchestrator
│   └── ollama_runner.py    # Ollama integration
├── prompts/                # Evaluation datasets
│   ├── classification_tasks.json
│   ├── reasoning_tasks.json
│   └── contextual_tasks.json
├── models/                 # Results and logs
│   ├── logs/              # Raw evaluation logs
│   └── outputs/           # Processed results
└── evaluations/           # Final reports and visualizations
    ├── clmpi_scorebook.xlsx
    └── visualizations/
```

## Quick Start

### Prerequisites

- **macOS** (optimized for MacBook Pro 2019)
- **Python 3.8+**
- **Ollama** installed and running
- **Open-source models** pulled via Ollama

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rwandantechy/clmpi-benchmark.git
   cd clmpi-benchmark
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama models**
   ```bash
   ollama pull phi3:mini
   ollama pull mistral
   ollama pull llama2:7b-chat
   ollama pull gemma:2b
   ```

### Running Benchmarks

```bash
# Evaluate all configured models
python scripts/evaluate_models.py --config config/model_config.yaml --output results/

# Evaluate specific model
python scripts/evaluate_models.py --config config/model_config.yaml --models phi3:mini mistral

# Generate visualizations
python scripts/generate_visualizations.py --input results/ --output evaluations/
```

## Results

The framework generates:

- **Excel Scorebook**: Comprehensive results in `evaluations/clmpi_scorebook.xlsx`
- **Radar Charts**: Component-wise comparison visualizations
- **Bar Charts**: Overall CLMPI score rankings
- **Performance Metrics**: Detailed efficiency analysis
- **Raw Logs**: Complete evaluation traces for analysis

## CLMPI Score Interpretation

| Score Range | Interpretation | Edge Deployment Suitability |
|-------------|----------------|------------------------------|
| 8-10 | Excellent | Optimal for edge deployment |
| 6-8 | Good | Suitable with minor optimizations |
| 4-6 | Fair | Requires significant optimization |
| 2-4 | Poor | Not recommended for edge use |
| 0-2 | Very Poor | Unsuitable for edge deployment |

## Configuration

### Model Configuration (`config/model_config.yaml`)
```yaml
models:
  phi3:mini:
    ollama_name: "phi3:mini"
    max_tokens: 1000
    temperature: 0.1
    evaluation_weights:
      accuracy: 0.25
      contextual_understanding: 0.20
      fluency_coherence: 0.20
      performance_efficiency: 0.35
```

### Device Configuration (`config/macbook_pro_2019.yaml`)
```yaml
device:
  name: "MacBook Pro 2019"
  cpu_cores: 8
  memory_gb: 16
  storage_type: "SSD"
  ollama_host: "http://localhost:11434"
```

## Performance Optimization

- **Sequential Evaluation**: Models run one at a time to ensure fair resource allocation
- **Memory Management**: Automatic cleanup between model evaluations
- **CPU Monitoring**: Real-time resource usage tracking
- **Caching**: Efficient prompt and response storage

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Attribution

This implementation is based on the research paper:
- **"Benchmarking Large Language Models with a Unified Performance Ranking Metric"**
- **Author**: Maikel Leon
- **Institution**: Department of Business Technology, Miami Herbert Business School, University of Miami, Florida, USA
- **Journal**: International Journal on Foundations of Computer Science & Technology (IJFCST) Vol.4, No.4, July 2024

This repository provides a practical implementation of the theoretical framework proposed in the original research, specifically adapted for local edge deployment benchmarking.

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
  author={Niyonziima, Innocent},
  year={2024},
  url={https://github.com/rwandantechy/clmpi-benchmark},
  note={Based on Leon (2024) unified performance ranking metric}
}
```

## Contact

For questions and contributions, please open an issue on GitHub or contact:
- **Email**: niyonzima@cua.edu
- **LinkedIn**: [Innocent Niyonziima](https://www.linkedin.com/in/innocent-niyonziima/)

---

**Note**: This is a research tool implementing Maikel Leon's unified performance ranking metric, specifically optimized for local edge deployment benchmarking of open-source LLMs on macOS. Results may vary based on model versions, hardware configurations, and evaluation parameters.
