# Dataset Information for CLMPI Framework

## Overview

This document describes the current datasets used in the CLMPI framework. The system currently uses single prompts from established public datasets for each evaluation dimension.

## Current Implementation Status

**Note**: Each metric currently uses only 1 prompt per evaluation dimension. This is a limitation that affects statistical significance of results.

## Dataset Sources

### Accuracy: GSM-Hard Mathematical Reasoning
- **Source**: Single question from reasoning-machines/gsm-hard dataset
- **Rationale**: Mathematical reasoning tests factual accuracy
- **Question Type**: Mathematical calculation requiring logical reasoning
- **Current Coverage**: 1 question
- **Example**: "Calculate: 13 × 47 = ?"

### Contextual Understanding: SQuAD
- **Source**: Single question from Stanford Question Answering Dataset
- **Rationale**: Passage comprehension tests contextual understanding
- **Question Type**: Span-based question answering from passages
- **Current Coverage**: 1 question
- **Example**: Context about Eiffel Tower, question about who it was named after

### Coherence: Custom Prompt
- **Source**: Single custom coherence evaluation prompt
- **Rationale**: Tests logical flow and internal consistency
- **Question Type**: Open-ended narrative generation
- **Current Coverage**: 1 question
- **Example**: Generate coherent narrative about a topic

### Fluency: Custom Prompt
- **Source**: Single custom fluency evaluation prompt
- **Rationale**: Tests language quality and grammatical correctness
- **Question Type**: Descriptive text generation
- **Current Coverage**: 1 question
- **Example**: Generate descriptive text about a subject

### Performance Efficiency: Custom Task
- **Source**: Single computational task
- **Rationale**: Measures resource usage and performance
- **Question Type**: Task requiring consistent processing
- **Current Coverage**: 1 question
- **Example**: Chemical symbol identification task

## Current Limitations

### 1. Limited Coverage
- Only 1 prompt per metric
- Results lack statistical significance
- Single-point evaluations can be misleading

### 2. Dataset Quality
- Prompts sourced from public datasets
- Limited validation of prompt effectiveness
- No custom prompt engineering

### 3. Statistical Power
- Single prompt evaluations lack statistical power
- Results should be interpreted with caution
- Comparative analysis between models is limited

## Implementation Details

### Prompt Format
All prompts use JSON format with:
- `id`: Unique identifier
- `category`: Metric category
- `type`: Question type
- `prompt`: The actual question/prompt
- `reference`: Expected answer(s)
- `source`: Dataset source attribution

### Response Parsing
- Accuracy and Context: Structured JSON response parsing
- Coherence and Fluency: Free-form text generation
- Efficiency: Performance metrics collection

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

## Dataset Attribution

When using CLMPI results, acknowledge:
- CLMPI framework methodology
- Original dataset sources where applicable
- Current limitations of single-prompt evaluation

## Conclusion

The current CLMPI implementation provides a working framework for model evaluation but is limited by single-prompt coverage. The system demonstrates potential for comprehensive evaluation but requires expansion to provide statistically reliable results.

**Next Steps**:
1. Expand prompt coverage for statistical significance
2. Improve dataset quality and validation
3. Implement statistical analysis tools
4. Create custom prompt engineering capabilities
