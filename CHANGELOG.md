# Changelog

## [0.2.0] - 2024-12-19

### Refactored
- **Repository Structure**: Moved non-essential files to `archive/`
- **Documentation**: Simplified README, moved detailed docs to `/docs`
- **Configuration**: Standardized config files with inline documentation
- **Scripts**: Added proper CLI arguments and help documentation
- **Results**: Standardized output structure with run timestamps
- **Math**: Implemented proper normalization for CLMPI scoring

### Enhanced
- **Filename Sanitization**: Safe filenames across all filesystems
- **Latest Pointer**: `results/latest/` symlink for easy access
- **Run Naming**: Consistent `YYYY-MM-DD_HHMMSS_<label>` pattern
- **JSON Schema**: `docs/schemas/summary.schema.json` for format validation
- **CLI Ergonomics**: Added `--label` and `--seed` flags
- **End-of-Run Summary**: Per-axis scores, CLMPI (0-1 and 0-100), file paths
- **Testing**: Unit tests for calculator and utilities
- **Code Quality**: Pre-commit hooks for formatting and linting
- **Citation**: CITATION.cff for proper academic attribution

### Added
- `CONTRIBUTING.md` - Contribution guidelines
- `CHANGELOG.md` - This file
- `config/device_default.yaml` - Default device configuration
- `prompts/README.md` - Prompt set documentation
- `docs/` - Detailed documentation directory
- `docs/metric.md` - CLMPI formula documentation
- `docs/protocol.md` - Reproducible run procedures
- `docs/adding-a-model.md` - Model addition guide
- `scripts/generate_visualizations.py` - Visualization generation

### Changed
- **README.md**: Simplified to essential information only
- **Config**: Added inline comments and standardized structure
- **Scripts**: Added argparse, docstrings, and proper error handling
- **Results**: Changed to timestamped run folders
- **CLMPI Calculator**: Implemented proper normalization formulas

### Moved to Archive
- `docs/` → `archive/docs/` (detailed methodology)
- `devices/` → `archive/devices/` (device configs)
- `models/` → `archive/models/` (example outputs)
- `setup.py` → `archive/setup.py` (package config)
- `QUICKSTART.md` → `archive/QUICKSTART.md` (extended guide)
- `evaluation.log` → `archive/evaluation.log` (debug logs)

### Removed
- Hardcoded paths in scripts
- Redundant configuration files
- Non-essential documentation from main directory

## [0.1.0] - 2024-12-18

### Added
- Initial CLMPI benchmark implementation
- Ollama integration for local model evaluation
- Basic scoring system for 5 dimensions
- Configuration system for models and devices
- Example evaluation results for demonstration models
