# CLMPI Benchmark

**Comprehensive Language Model Performance Index**

A practical implementation of the unified performance ranking metric proposed by Maikel Leon (University of Miami, 2024). This framework provides a comprehensive benchmarking system for evaluating AI language models across cognitive tasks and reasoning capabilities.

## Overview

CLMPI (Comprehensive Language Model Performance Index) is a practical implementation of the unified performance ranking metric proposed by Maikel Leon in "Benchmarking Large Language Models with a Unified Performance Ranking Metric" (2024). This framework provides a standardized evaluation system designed to assess the cognitive performance of language models through systematic testing across multiple dimensions including classification, reasoning, and human-aligned evaluation metrics.

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
   git clone https://github.com/rwandantechy/clmpi-benchmark.git
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

## Attribution

This implementation is based on the research paper:
- **"Benchmarking Large Language Models with a Unified Performance Ranking Metric"**
- **Author**: Maikel Leon
- **Institution**: Department of Business Technology, Miami Herbert Business School, University of Miami, Florida, USA
- **Journal**: International Journal on Foundations of Computer Science & Technology (IJFCST) Vol.4, No.4, July 2024

This repository provides a practical implementation of the theoretical framework proposed in the original research.

## Contact

For questions and contributions, please open an issue on GitHub or contact [your-email@domain.com].

---

**Note**: This is a research tool implementing Maikel Leon's unified performance ranking metric. Results may vary based on model versions, hardware configurations, and evaluation parameters.
