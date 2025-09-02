# CLMPI Results Directory

This directory contains the results of CLMPI benchmark evaluations.

## Directory Structure

```
results/
├── latest/                    # Symlink to most recent evaluation run
├── .gitkeep                   # Preserves this directory in git
└── YYYY-MM-DD_HHMMSS_stepwise/  # Individual evaluation runs
    ├── clmpi_summary.json     # Final CLMPI scores and metadata
    ├── accuracy/              # Accuracy evaluation results
    │   ├── detail.jsonl      # Detailed scoring for each response
    │   └── summary.json      # Accuracy summary statistics
    ├── context/               # Context evaluation results
    │   ├── detail.jsonl      # Detailed scoring for each response
    │   └── summary.json      # Context summary statistics
    ├── coherence/             # Coherence evaluation results
    │   ├── detail.jsonl      # Detailed scoring for each response
    │   └── summary.json      # Coherence summary statistics
    ├── fluency/               # Fluency evaluation results
    │   ├── detail.jsonl      # Detailed scoring for each response
    │   └── summary.json      # Fluency summary statistics
    └── efficiency/            # Efficiency evaluation results
        ├── detail.jsonl      # Detailed scoring for each response
        └── summary.json      # Efficiency summary statistics
```

## How It Works

1. **Automatic Creation**: Each evaluation run creates a new timestamped directory
2. **Latest Symlink**: The `latest/` symlink automatically points to the most recent run
3. **Consistent Structure**: All runs follow the same directory structure
4. **Easy Access**: Use `results/latest/` to always access the most recent results

## Accessing Results

### Most Recent Results
```bash
# View latest CLMPI summary
cat results/latest/clmpi_summary.json

# View latest accuracy results
cat results/latest/accuracy/summary.json

# List all files in latest run
ls -la results/latest/
```

### Specific Run Results
```bash
# View results from a specific run
cat results/2025-09-01_142730_stepwise/clmpi_summary.json

# Compare multiple runs
ls -la results/*_stepwise/
```

## File Formats

- **`clmpi_summary.json`**: Complete CLMPI scores and component breakdowns
- **`detail.jsonl`**: Per-response detailed scoring (JSON Lines format)
- **`summary.json`**: Per-metric summary statistics

## Cleanup

The system automatically keeps the 5 most recent evaluation runs. Older runs are automatically cleaned up to save disk space.

## Notes

- The `latest/` symlink is automatically updated after each evaluation
- All evaluation runs are preserved with timestamps for reproducibility
- Results are machine-readable for automated analysis
