# CLMPI Benchmark

**Cognitive Language Model Performance Index**

A comprehensive benchmarking framework for evaluating AI language models across cognitive tasks and reasoning capabilities.

## Overview

CLMPI (Cognitive Language Model Performance Index) is a standardized evaluation framework designed to assess the cognitive performance of language models through systematic testing across multiple dimensions including classification, reasoning, and human-aligned evaluation metrics.

## Features

- **Multi-dimensional Evaluation**: Tests models across classification and reasoning tasks
- **Device-specific Configurations**: Optimized benchmarking for different hardware setups
- **Human Evaluation Integration**: Guidelines and frameworks for human assessment
- **Automated Metrics Collection**: Streamlined data collection and analysis
- **Visualization Tools**: Built-in analytics and result visualization
- **Extensible Architecture**: Easy to add new tasks and evaluation criteria

## Project Structure

```
clmpi-benchmark/
├── config/                 # Model and benchmark configurations
├── devices/                # Device-specific configurations
├── docs/                   # Documentation and methodology
├── evaluations/            # Evaluation results and guidelines
├── models/                 # Model outputs and logs
├── prompts/                # Task definitions and instructions
└── scripts/                # Core benchmarking scripts
```

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd clmpi-benchmark
   ```

2. **Configure your environment**
   ```bash
   # Edit device configuration
   cp devices/macbook_pro_2019.yaml devices/your_device.yaml
   # Edit model configuration
   cp config/model_config.yaml config/your_models.yaml
   ```

3. **Run benchmarks**
   ```bash
   python scripts/run_benchmark.py
   ```

4. **Evaluate results**
   ```bash
   python scripts/evaluate_quality.py
   python scripts/collect_metrics.py
   ```

## Configuration

### Model Configuration
Edit `config/model_config.yaml` to specify which models to evaluate and their parameters.

### Device Configuration
Create device-specific configurations in `devices/` to optimize benchmarking for your hardware.

## Documentation

- [Benchmark Plan](docs/benchmark_plan.md) - Detailed testing methodology
- [CLMPI Definition](docs/clmpi_definition.md) - Framework specifications
- [Methodology](docs/methodology.md) - Evaluation approach and metrics

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use CLMPI in your research, please cite:

```bibtex
@software{clmpi_benchmark,
  title={CLMPI: Cognitive Language Model Performance Index},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/clmpi-benchmark}
}
```

## Contact

For questions and contributions, please open an issue on GitHub or contact [your-email@domain.com].

---

**Note**: This is a research tool. Results may vary based on model versions, hardware configurations, and evaluation parameters.
