# CLMPI Evaluation Prompts

This directory contains prompt sets used for CLMPI benchmark evaluation.

## Prompt Set Mapping

Each JSON file maps to specific CLMPI evaluation dimensions:

| File | Dimension | Description | Target Count |
|------|-----------|-------------|--------------|
| `classification_tasks.json` | Accuracy | Factual correctness tasks | 5-10 prompts |
| `reasoning_tasks.json` | Accuracy | Mathematical/logical reasoning | 5-10 prompts |
| `contextual_tasks.json` | Contextual Understanding | Multi-turn conversations | 5-10 prompts |
| `contextual_understanding_tasks.json` | Contextual Understanding | Context-aware responses | 5-10 prompts |
| `fluency_coherence_tasks.json` | Fluency | Language quality assessment | 5-10 prompts |
| `performance_efficiency_tasks.json` | Performance Efficiency | Resource usage measurement | 5-10 prompts |

## File Format

Each prompt file contains an array of objects:

```json
[
  {
    "prompt": "What is 2 + 2?",
    "answer": "4",
    "category": "math",
    "difficulty": "easy"
  }
]
```

## Adding/Removing Prompts

### Adding New Prompts

1. **Choose the appropriate file** based on the evaluation dimension
2. **Add prompt object** with required fields:
   - `prompt`: The input text
   - `answer`: Expected answer (for accuracy tasks)
   - `category`: Optional categorization
   - `difficulty`: Optional difficulty level

3. **Maintain balance** - keep similar numbers across categories
4. **Test locally** before committing

### Removing Prompts

1. **Keep prompt counts balanced** across dimensions
2. **Preserve diversity** - don't remove all prompts from one category
3. **Update documentation** if removing entire categories

### Breaking Changes

To maintain comparability:

- **Don't change existing prompts** in production runs
- **Add new prompts** to new files or append to existing
- **Version prompt sets** if major changes needed
- **Document changes** in CHANGELOG.md

## Configuration

Prompt sets are configured in `config/model_config.yaml`:

```yaml
prompt_sets:
  accuracy:
    - "classification_tasks.json"
    - "reasoning_tasks.json"
  contextual_understanding:
    - "contextual_tasks.json"
```

## Best Practices

- **Keep prompts concise** - avoid very long inputs
- **Use diverse topics** - cover different domains
- **Include edge cases** - test model robustness
- **Maintain consistency** - similar format across files
- **Document sources** - note if prompts are from existing datasets
