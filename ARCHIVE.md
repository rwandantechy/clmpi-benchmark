# Archive

This directory contains historical files and deprecated components that are no longer part of the active CLMPI benchmark system.

## Structure

### `archive/docs/`
Historical documentation files that have been superseded by the current documentation in `docs/`.

### `archive/models/`
Old model configurations and examples that are no longer relevant.

### `prompts/archive/`
Deprecated prompt files that have been replaced by curated datasets:
- `accuracy_tasks.json` → `accuracy_tasks_curated.json`
- `contextual_tasks.json` → `contextual_tasks_curated.json`
- `contextual_understanding_tasks.json` → `contextual_tasks_curated.json`
- `fluency_coherence_tasks.json` → `coherence_tasks.json` + `fluency_tasks.json`
- `classification_tasks.json` (deprecated)
- `reasoning_tasks.json` (deprecated)
- `performance_efficiency_tasks.json` (deprecated)

### `results/archive/`
Demo runs and test results that are not part of the current evaluation system.

## Current System

The active CLMPI benchmark system uses:
- **Scripts**: `scripts/enhanced_evaluate_models.py` and `scripts/enhanced_clmpi_calculator.py`
- **Configs**: `config/model_config.yaml`, `config/generation_config.yaml`, `config/device_default.yaml`
- **Prompts**: Curated datasets in `prompts/` (no archive files)
- **Docs**: `docs/` directory with current methodology and usage guides

## Migration Notes

If you were using the old system:
1. Use `scripts/enhanced_evaluate_models.py` instead of `scripts/evaluate_models.py`
2. Use curated prompt files instead of the archived ones
3. Follow the enhanced methodology documented in `docs/enhanced_methodology.md`

The enhanced system provides better accuracy, reproducibility, and professional results.
