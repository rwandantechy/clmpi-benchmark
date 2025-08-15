# CLMPI Benchmark Protocol

## Reproducible Run Procedures

This document outlines the standard procedures for running CLMPI benchmarks to ensure reproducible results.

## Pre-Run Setup

### 1. Environment Preparation

```bash
# Clean environment
conda create -n clmpi python=3.8
conda activate clmpi

# Install dependencies
pip install -r requirements.txt

# Verify Ollama installation
ollama --version
```

### 2. Model Preparation

```bash
# Pull required models
ollama pull phi3:mini
ollama pull mistral
ollama pull llama2:7b-chat

# Verify models are available
ollama list
```

### 3. System Preparation

- **Close unnecessary applications** to free up memory
- **Disable background processes** that might interfere
- **Ensure stable internet connection** for model downloads
- **Note hardware specifications** for reproducibility

## Standard Run Procedure

### 1. Hardware Snapshot

The system automatically logs:
- CPU model and core count
- Total memory (GB)
- Operating system and version
- Python version
- Timestamp

### 2. Configuration Validation

```bash
# Validate configuration files
python -c "import yaml; yaml.safe_load(open('config/model_config.yaml'))"
python -c "import yaml; yaml.safe_load(open('config/device_default.yaml'))"
```

### 3. Run Execution

```bash
# Standard benchmark run
python scripts/evaluate_models.py \
    --config config/model_config.yaml \
    --device config/device_default.yaml \
    --models phi3:mini mistral \
    --output results/edge_demo
```

### 4. Post-Run Validation

- **Check output directory** structure
- **Verify summary.json** contains all expected fields
- **Confirm visualizations** were generated
- **Review logs** for any errors or warnings

## Reproducibility Controls

### 1. Random Seed

- **Fixed seed**: 42 (configured in model_config.yaml)
- **Consistent prompt ordering** across runs
- **Deterministic sampling** when prompts exceed limit

### 2. Resource Management

- **Sequential execution**: Models run one at a time (single-threaded)
- **Memory cleanup**: Caches cleared between model evaluations
- **Timeout controls**: Prevent hanging evaluations
- **Error isolation**: One model failure doesn't stop others

### 3. Data Consistency

- **Prompt sets**: Immutable during production runs
- **Evaluation weights**: Fixed in configuration
- **Scoring algorithms**: Deterministic implementations
- **Hardware logging**: Captured for each run

## Quality Assurance

### 1. Pre-Run Checks

- [ ] All required models available via Ollama
- [ ] Configuration files valid
- [ ] Sufficient disk space for results
- [ ] System resources adequate
- [ ] Network connectivity stable

### 2. During Run Monitoring

- [ ] Models responding within timeouts
- [ ] Memory usage within limits
- [ ] No system errors in logs
- [ ] Results being saved correctly

### 3. Post-Run Validation

- [ ] All requested models evaluated
- [ ] CLMPI scores in expected ranges [0,1]
- [ ] Component scores properly normalized
- [ ] Visualizations generated successfully
- [ ] Summary file contains complete data

## Troubleshooting

### Common Issues

1. **Model not found**
   ```bash
   # Check available models
   ollama list
   
   # Pull missing model
   ollama pull <model_name>
   ```

2. **Timeout errors**
   - Increase timeout in device config
   - Check system resources
   - Verify Ollama service running

3. **Memory errors**
   - Close other applications
   - Reduce concurrent evaluations
   - Check available RAM

4. **Configuration errors**
   - Validate YAML syntax
   - Check file paths
   - Verify required fields

### Recovery Procedures

1. **Partial run failure**
   - Check logs for specific errors
   - Re-run failed models only
   - Merge results if needed

2. **Complete run failure**
   - Restart Ollama service
   - Clear temporary files
   - Re-run from beginning

## Performance Optimization

### 1. System Tuning

- **CPU affinity**: Pin to specific cores if needed
- **Memory allocation**: Reserve adequate RAM
- **Disk I/O**: Use SSD for faster writes
- **Network**: Stable connection for model downloads

### 2. Evaluation Tuning

- **Sample size**: Adjust samples_per_task in config
- **Timeout values**: Balance speed vs. reliability
- **Concurrency**: Sequential execution for fairness
- **Caching**: Enable if supported

## Reporting Standards

### 1. Required Information

- **Run identifier**: Timestamp + name
- **Hardware specifications**: CPU, RAM, OS
- **Model versions**: Ollama model tags
- **Configuration**: Weights and parameters
- **Results**: CLMPI scores and components

### 2. Optional Information

- **Performance metrics**: Timing, memory usage
- **Error logs**: Any issues encountered
- **System load**: During evaluation
- **Network stats**: If relevant

## Version Control

### 1. Configuration Versioning

- **Tag configurations** for major changes
- **Document changes** in CHANGELOG.md
- **Maintain compatibility** when possible
- **Archive old configs** in archive/

### 2. Result Versioning

- **Timestamped directories** for each run
- **Latest symlink** for easy access
- **Backup important results** externally
- **Document run conditions** thoroughly
