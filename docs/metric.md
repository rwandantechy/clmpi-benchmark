# CLMPI Metric Documentation

## Overview

The Comprehensive Language Model Performance Index (CLMPI) is a unified scoring system that evaluates language models across 5 dimensions with proper normalization.

## Formula

CLMPI = Σ(wᵢ × sᵢ)

Where:
- wᵢ are the evaluation weights (must sum to 1.0)
- sᵢ are normalized scores in [0,1] range

**Final Output**: Both 0-1 and 0-100 scales (CLMPI × 100)

## Component Scoring

### 1. Accuracy (ACC)
**Range**: [0,1]  
**Calculation**: correct_answers / total_answers  
**Normalization**: None (already in [0,1])

### 2. Contextual Understanding (CON)
**Range**: [0,5] → [0,1]  
**Calculation**: Average of human/automated scores  
**Normalization**: score / 5.0

### 3. Coherence (COH)
**Range**: [0,5] → [0,1]  
**Calculation**: Average of logical flow scores  
**Normalization**: score / 5.0

### 4. Fluency (FLU)
**Range**: [0,5] → [0,1]  
**Calculation**: Average of language quality scores  
**Normalization**: score / 5.0

### 5. Performance Efficiency (EFF)
**Range**: Raw → [0,1]  
**Calculation**: EFF_raw = 1 / (latency_seconds + memory_mb/100)  
**Normalization**: Min-max normalization per run: EFF_norm = (EFF_raw - EFF_min) / (EFF_max - EFF_min + 1e-9)

## Normalization Details

### Quality Scores (CON, COH, FLU)
Raw scores in [0,5] are normalized to [0,1]:
```
normalized_score = raw_score / 5.0
```

### Efficiency Score (EFF)
Raw efficiency scores are normalized using min-max normalization:
```
EFF_raw = 1 / (latency_seconds + memory_mb/100)
EFF_norm = (EFF_raw - EFF_min) / (EFF_max - EFF_min + 1e-9)
```

The +1e-9 prevents division by zero when all models have identical efficiency.

## Default Weights

| Component | Weight | Description |
|-----------|--------|-------------|
| Accuracy | 0.25 | 25% - Factual correctness |
| Contextual Understanding | 0.20 | 20% - Multi-turn conversations |
| Coherence | 0.20 | 20% - Logical flow |
| Fluency | 0.20 | 20% - Language quality |
| Performance Efficiency | 0.15 | 15% - Resource usage |

**Total**: 1.0

## Worked Example

Consider two models with the following raw scores:

### Model A
- Accuracy: 0.8 (80% correct)
- Contextual Understanding: 4.0/5
- Coherence: 3.5/5
- Fluency: 4.2/5
- Performance Efficiency: 0.15 (raw)

### Model B
- Accuracy: 0.6 (60% correct)
- Contextual Understanding: 3.0/5
- Coherence: 2.8/5
- Fluency: 3.5/5
- Performance Efficiency: 0.10 (raw)

### Normalization

**Model A**:
- ACC_norm = 0.8
- CON_norm = 4.0/5 = 0.8
- COH_norm = 3.5/5 = 0.7
- FLU_norm = 4.2/5 = 0.84
- EFF_norm = (0.15 - 0.10) / (0.15 - 0.10 + 1e-9) = 1.0

**Model B**:
- ACC_norm = 0.6
- CON_norm = 3.0/5 = 0.6
- COH_norm = 2.8/5 = 0.56
- FLU_norm = 3.5/5 = 0.7
- EFF_norm = (0.10 - 0.10) / (0.15 - 0.10 + 1e-9) = 0.0

### CLMPI Calculation

**Model A**:
```
CLMPI = 0.25×0.8 + 0.20×0.8 + 0.20×0.7 + 0.20×0.84 + 0.15×1.0
      = 0.20 + 0.16 + 0.14 + 0.168 + 0.15
      = 0.818
```

**Model B**:
```
CLMPI = 0.25×0.6 + 0.20×0.6 + 0.20×0.56 + 0.20×0.7 + 0.15×0.0
      = 0.15 + 0.12 + 0.112 + 0.14 + 0.0
      = 0.522
```

### Final Scores
- Model A: CLMPI = 0.818 (81.8/100)
- Model B: CLMPI = 0.522 (52.2/100)

## Score Interpretation

| CLMPI Range | Interpretation | Edge Deployment Suitability |
|-------------|----------------|------------------------------|
| 0.8-1.0 | Excellent | Optimal for edge deployment |
| 0.6-0.8 | Good | Suitable with minor optimizations |
| 0.4-0.6 | Fair | Requires significant optimization |
| 0.2-0.4 | Poor | Not recommended for edge use |
| 0.0-0.2 | Very Poor | Unsuitable for edge deployment |

## Implementation Notes

1. **Reproducibility**: Fixed random seed ensures consistent prompt ordering
2. **Hardware Logging**: System information captured for each run
3. **Error Handling**: Graceful degradation when models fail
4. **Scalability**: Supports any number of models in single run
5. **Extensibility**: Weights configurable via YAML

## References

Based on: "Benchmarking Large Language Models with a Unified Performance Ranking Metric" by Maikel Leon (University of Miami, 2024)
