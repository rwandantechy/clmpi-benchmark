# CLMPI Evaluation Prompts

This directory contains prompt sets used for CLMPI benchmark evaluation. Each metric currently uses 1 prompt per evaluation dimension.

## Current Prompt Sets

Each JSON file maps to specific CLMPI evaluation dimensions:

| File | Dimension | Description | Current Count | Status |
|------|-----------|-------------|---------------|---------|
| `accuracy.json` | Accuracy | GSM-Hard mathematical reasoning | 1 prompt | Working |
| `context.json` | Contextual Understanding | SQuAD passage comprehension | 1 prompt | Working |
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

### Context Tasks (SQuAD)
```json
[
  {
    "id": "ctx_squad_001",
    "category": "context",
    "type": "span",
    "prompt": "Passage: The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. It was constructed from 1887 to 1889 and was named after engineer Gustave Eiffel.\n\nQuestion: Who was the Eiffel Tower named after?\n\nAnswer in this format: {\"id\":\"ctx_squad_001\",\"answer\":\"name\"}",
    "reference": ["gustave eiffel"],
    "source": "https://huggingface.co/datasets/squad"
  }
]
```

### Coherence Tasks (Custom)
```json
[
  {
    "id": "coh_001",
    "category": "coherence",
    "type": "narrative",
    "prompt": "Write a short story about...",
    "source": "custom"
  }
]
```

### Fluency Tasks (Custom)
```json
[
  {
    "id": "flu_001",
    "category": "fluency",
    "type": "descriptive",
    "prompt": "Describe the importance of education...",
    "source": "custom"
  }
]
```

### Efficiency Tasks (Custom)
```json
[
  {
    "id": "eff_mmlu_001",
    "category": "resource_efficiency",
    "type": "span",
    "prompt": "What is the chemical symbol for gold?\n\nAnswer in this format: {\"id\":\"eff_mmlu_001\",\"answer\":\"symbol\"}",
    "reference": ["au"],
    "source": "https://huggingface.co/datasets/mmlu"
  }
]
```

## Current Implementation Status

### Working Components
- **Accuracy**: GSM-Hard mathematical reasoning with structured JSON responses
- **Context**: SQuAD passage comprehension with context relevance
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
    - "context.json"  # SQuAD passage comprehension
  coherence:
    - "coherence.json"  # Logical flow evaluation
  fluency:
    - "fluency.json"  # Language quality assessment
  performance_efficiency:
    - "efficiency_tasks.json"  # Resource usage measurement
```

## Current Limitations

### 1. Limited Coverage
- Only 1 prompt per metric
- Results lack statistical significance
- Single-point evaluations can be misleading

### 2. Dataset Sources
- Some prompts from public datasets (GSM-Hard, SQuAD, MMLU)
- Some custom prompts for coherence and fluency
- Limited validation of prompt effectiveness

### 3. Statistical Power
- Single prompt evaluations lack statistical power
- Results should be interpreted with caution
- Comparative analysis between models is limited

## Testing Status

### Verified Working
- All prompt files load correctly
- Response parsing works for all metrics
- Scoring calculations produce valid results
- Integration with evaluation pipeline functional

### Known Issues
- Limited prompt coverage affects result reliability
- Single-prompt evaluations lack statistical significance
- No prompt difficulty classification

## Future Improvements

### 1. Expand Coverage
- Add multiple prompts per metric (5-10 minimum)
- Include different question types and difficulty levels
- Implement prompt sampling strategies

### 2. Dataset Quality
- Create custom prompt engineering
- Validate prompt effectiveness
- Add difficulty level classification

### 3. Statistical Analysis
- Calculate confidence intervals
- Add standard deviations and error margins
- Implement cross-validation approaches

## Usage Guidelines

### 1. Result Interpretation
- Results are based on single-prompt evaluations
- Compare models with caution
- Consider prompt-specific performance

### 2. Model Comparison
- Limited statistical significance
- Focus on relative performance patterns
- Consider model size and architecture differences

### 3. Extensions
- Add more questions from same datasets
- Create custom prompts for specific domains
- Implement prompt difficulty scaling

## Conclusion

The current CLMPI prompt system provides a working framework for model evaluation but is limited by single-prompt coverage. The system demonstrates potential for comprehensive evaluation but requires expansion to provide statistically reliable results.

**Next Steps**:
1. Expand prompt coverage for statistical significance
2. Improve dataset quality and validation
3. Implement statistical analysis tools
4. Create custom prompt engineering capabilities
