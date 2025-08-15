# Contributing to CLMPI Benchmark

## Quick Start

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Test: `python scripts/evaluate_models.py --help`
5. Commit: `git commit -m 'Add your feature'`
6. Push: `git push origin feature/your-feature`
7. Open a Pull Request

## Guidelines

- **Keep it simple**: Favor clarity over cleverness
- **Test changes**: Run benchmarks before submitting
- **Update docs**: Document new features in `/docs`
- **Follow style**: Use existing code patterns
- **Preserve compatibility**: Don't break existing configs

## Development

- Add new models in `config/model_config.yaml`
- Add new prompts in `prompts/`
- Update metrics in `docs/metric.md`
- Test with: `python scripts/evaluate_models.py --models phi3:mini`

## Questions?

Open an issue or contact: niyonzima@cua.edu
