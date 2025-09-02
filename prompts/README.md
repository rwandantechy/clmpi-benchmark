# CLMPI Evaluation Prompts

This directory contains prompt sets used for CLMPI benchmark evaluation. All datasets are currently working and producing consistent results.

## Active Prompt Sets

Each JSON file maps to specific CLMPI evaluation dimensions:

| File | Dimension | Description | Target Count | Status |
|------|-----------|-------------|--------------|---------|
| `accuracy.json` | Accuracy | GSM-Hard mathematical reasoning | 1 prompt | Working |
| `context.json` | Contextual Understanding | Multi-turn conversations with context | 1 prompt | Working |
| `coherence.json` | Coherence | Logical flow evaluation | 1 prompt | Working |
| `fluency.json` | Fluency | Language quality assessment | 1 prompt | Working |
| `efficiency_tasks.json` | Performance Efficiency | Resource usage measurement | 1 prompt | Working |

## File Format

### Accuracy Tasks (GSM-Hard)
```json
[
  {
    "id": "acc_gsmhard_001",
    "category": "accuracy",
    "type": "numeric",
    "prompt": "Calculate: 13 × 47 = ?\n\nAnswer in this format: {\"id\":\"acc_gsmhard_001\",\"answer\":\"number\"}",
    "reference": ["611"],
    "source": "https://huggingface.co/datasets/reasoning-machines/gsm-hard"
  }
]
```

### Context Tasks (Multi-turn)
```json
[
  {
    "id": "ctx_001",
    "category": "context",
    "type": "conversation",
    "prompt": "Context: [conversation context]\nQuestion: [specific question]\n\nAnswer:",
    "reference": ["expected answer"],
    "source": "curated"
  }
]
```

### Coherence & Fluency Tasks (Open-ended)
```json
[
  {
    "id": "coh_001",
    "category": "coherence",
    "type": "narrative",
    "prompt": "Write a short story about...",
    "source": "curated"
  }
]
```

### Efficiency Tasks (Performance)
```json
[
  {
    "id": "eff_001",
    "category": "efficiency",
    "type": "computation",
    "prompt": "Task description for performance measurement",
    "source": "curated"
  }
]
```

## Current Implementation Status

### Working Components
- **Accuracy**: GSM-Hard mathematical reasoning with structured JSON responses
- **Context**: Multi-turn conversation understanding with context relevance
- **Coherence**: Open-ended prompts with internal consistency scoring
- **Fluency**: Language quality evaluation with grammar and diversity metrics
- **Efficiency**: Resource usage measurement with timing and memory metrics

### Technical Details
- **Response Parsing**: Structured JSON extraction for accuracy and context
- **Scoring Methods**: F1, exact match, coherence, fluency, and efficiency metrics
- **Generation Profiles**: Deterministic (accuracy/context) and creative (coherence/fluency)
- **Output Format**: Standardized JSON with detailed scoring breakdown

## Configuration

Prompt sets are configured in `config/model_config.yaml`:

```yaml
prompt_sets:
  accuracy:
    - "accuracy.json"  # GSM-Hard mathematical reasoning
  contextual_understanding:
    - "context.json"  # Multi-turn conversations
  coherence:
    - "coherence.json"  # Logical flow evaluation
  fluency:
    - "fluency.json"  # Language quality assessment
  performance_efficiency:
    - "efficiency_tasks.json"  # Resource usage measurement
```

## Dataset Sources

- **Accuracy**: [GSM-Hard](https://huggingface.co/datasets/reasoning-machines/gsm-hard) - Mathematical reasoning
- **Context**: Curated multi-turn conversations
- **Coherence**: Curated open-ended prompts
- **Fluency**: Curated language quality tasks
- **Efficiency**: Curated performance measurement tasks

## Best Practices

- **Keep prompts concise** - avoid very long inputs
- **Use diverse topics** - cover different domains
- **Include edge cases** - test model robustness
- **Maintain consistency** - similar format across files
- **Document sources** - note if prompts are from existing datasets
- **Use stable IDs** - acc_0001, ctx_0001, coh_0001, flu_0001, eff_0001 format

## Testing Status

All prompt sets have been successfully tested with Mistral 7B:
- Accuracy evaluation completed
- Context evaluation completed  
- Coherence evaluation completed
- Fluency evaluation completed
- Efficiency evaluation completed

**Ready for production use!**
