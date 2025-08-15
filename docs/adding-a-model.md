# Adding a Model to CLMPI Benchmark

## Quick Start

1. **Pull the model** via Ollama
2. **Add to config** in `config/model_config.yaml`
3. **Test locally** with a single model run
4. **Run full benchmark** to compare with others

## Step-by-Step Guide

### 1. Install Model via Ollama

```bash
# Pull the model (replace with your model name)
ollama pull llama2:13b-chat

# Verify it's available
ollama list | grep llama2:13b-chat

# Check model requirements (VRAM/CPU)
ollama show llama2:13b-chat
```

### 2. Add to Configuration

Edit `config/model_config.yaml` and add your model:

```yaml
models:
  # ... existing models ...
  
  llama2:13b-chat:
    ollama_name: "llama2:13b-chat"
    max_tokens: 1000
    temperature: 0.1
    timeout_seconds: 90  # Larger model needs more time
```

### 3. Test the Model

```bash
# Test with just your new model
python scripts/evaluate_models.py \
    --config config/model_config.yaml \
    --device config/device_default.yaml \
    --models llama2:13b-chat \
    --output results/test_new_model
```

### 4. Run Full Comparison

```bash
# Include your model in full benchmark
python scripts/evaluate_models.py \
    --config config/model_config.yaml \
    --device config/device_default.yaml \
    --models phi3:mini mistral llama2:13b-chat \
    --output results/full_comparison
```

## Configuration Parameters

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `ollama_name` | Exact name in Ollama | `"llama2:13b-chat"` |
| `max_tokens` | Maximum response length | `1000` |
| `temperature` | Response randomness | `0.1` |
| `timeout_seconds` | Response timeout | `60` |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `top_p` | Nucleus sampling | `0.9` |
| `top_k` | Top-k sampling | `40` |
| `repeat_penalty` | Repetition penalty | `1.1` |

## Model-Specific Considerations

### Large Models (>7B parameters)

```yaml
llama2:70b-chat:
  ollama_name: "llama2:70b-chat"
  max_tokens: 1000
  temperature: 0.1
  timeout_seconds: 120  # Longer timeout
```

### Small Models (<3B parameters)

```yaml
phi3:mini:
  ollama_name: "phi3:mini"
  max_tokens: 1000
  temperature: 0.1
  timeout_seconds: 30  # Shorter timeout
```

### Specialized Models

```yaml
deepseek-coder:6.7b:
  ollama_name: "deepseek-coder:6.7b"
  max_tokens: 1000
  temperature: 0.1
  timeout_seconds: 60
  # May need different prompt sets for coding tasks
```

## Troubleshooting

### Model Not Found

```bash
# Check available models
ollama list

# Pull if missing
ollama pull <model_name>

# Verify exact name
ollama show <model_name>
```

### Timeout Errors

1. **Increase timeout** in config
2. **Check system resources** (RAM, CPU)
3. **Reduce max_tokens** if needed
4. **Verify Ollama service** is running

### Memory Issues

1. **Close other applications**
2. **Use smaller models** first
3. **Check available RAM**
4. **Reduce concurrent evaluations**

### Performance Issues

1. **Adjust temperature** (lower = faster)
2. **Reduce max_tokens**
3. **Check CPU usage**
4. **Monitor memory consumption**

## Best Practices

### 1. Model Selection

- **Start with known models** (phi3:mini, mistral)
- **Test locally** before adding to config
- **Consider hardware requirements**
- **Document model characteristics**

### 2. Configuration

- **Use consistent parameters** across models
- **Adjust timeouts** based on model size
- **Test with small sample** first
- **Validate config syntax**

### 3. Testing

- **Run single model** before full benchmark
- **Check response quality** manually
- **Verify scores** are reasonable
- **Compare with known baselines**

### 4. Documentation

- **Note model version** used
- **Record hardware requirements**
- **Document any special settings**
- **Update model list** in README

## Example: Adding Mistral-7B-Instruct

```bash
# 1. Pull model
ollama pull mistral:7b-instruct

# 2. Add to config/model_config.yaml
mistral:7b-instruct:
  ollama_name: "mistral:7b-instruct"
  max_tokens: 1000
  temperature: 0.1
  timeout_seconds: 60

# 3. Test
python scripts/evaluate_models.py \
    --config config/model_config.yaml \
    --device config/device_default.yaml \
    --models mistral:7b-instruct \
    --output results/test_mistral_instruct

# 4. Full benchmark
python scripts/evaluate_models.py \
    --config config/model_config.yaml \
    --device config/device_default.yaml \
    --models phi3:mini mistral:7b-instruct \
    --output results/mistral_comparison
```

## Validation Checklist

- [ ] Model available via `ollama list`
- [ ] Configuration syntax valid
- [ ] Single model test successful
- [ ] CLMPI scores in expected range [0,1]
- [ ] No timeout or memory errors
- [ ] Results saved correctly
- [ ] Visualizations generated
- [ ] Documentation updated
