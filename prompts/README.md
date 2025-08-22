# CLMPI Evaluation Prompts

This directory contains prompt sets used for CLMPI benchmark evaluation.

## Active Prompt Sets

Each JSON file maps to specific CLMPI evaluation dimensions:

| File | Dimension | Description | Target Count |
|------|-----------|-------------|--------------|
| `accuracy.json` | Accuracy | Factual correctness with expert-validated answers | 5-10 prompts |
| `context.json` | Contextual Understanding | Multi-turn conversations with context | 5-10 prompts |
| `coherence.json` | Coherence | Logical flow evaluation | 5-10 prompts |
| `fluency.json` | Fluency | Language quality assessment | 5-10 prompts |

## File Format

### Accuracy & Context Tasks
```json
[
  {
    "id": "acc_0001",
    "question": "What is the capital of France?",
    "correct_answer": "Paris",
    "answers": ["Paris", "paris", "PARIS"],
    "category": "geography",
    "difficulty": "easy"
  }
]
```

### Coherence & Fluency Tasks
```json
[
  {
    "id": "coh_0001",
    "prompt": "Explain the process of photosynthesis in simple terms.",
    "category": "science",
    "difficulty": "medium"
  }
]
```

## Legacy Files

Old prompt files have been moved to `prompts/archive/` with `_legacy.json` suffix:
- `accuracy_legacy.json`
- `context_legacy.json`
- `context_understanding_legacy.json`
- `fluency_coherence_legacy.json`
- `performance_efficiency_legacy.json`
- `classification_legacy.json`
- `reasoning_legacy.json`

## Configuration

Prompt sets are configured in `config/model_config.yaml`:

```yaml
prompt_sets:
  accuracy:
    - "accuracy.json"
  contextual_understanding:
    - "context.json"
  coherence:
    - "coherence.json"
  fluency:
    - "fluency.json"
  performance_efficiency:
    - "coherence.json"  # Reuses coherence prompts for efficiency measurement
```

## Best Practices

- **Keep prompts concise** - avoid very long inputs
- **Use diverse topics** - cover different domains
- **Include edge cases** - test model robustness
- **Maintain consistency** - similar format across files
- **Document sources** - note if prompts are from existing datasets
- **Use stable IDs** - acc_0001, ctx_0001, coh_0001, flu_0001 format
