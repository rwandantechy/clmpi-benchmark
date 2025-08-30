#!/usr/bin/env python3
"""
CLMPI Logger - handles both markdown reports and response logging
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


def save_responses_markdown(model_name: str, metric_name: str, questions: List[Dict], 
                          responses: List[str], gold_answers: List[str] = None, 
                          scores: List[float] = None, audit_trail: List[Dict] = None) -> str:
    """Save model responses in organized Markdown format"""
    
    # Create model-specific directory
    model_dir = Path("results/model_responses") / model_name.replace(":", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create markdown content
    markdown_content = f"""# {model_name} - {metric_name.title()} Responses

**Evaluation Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Metric**: {metric_name.title()}
**Total Questions**: {len(questions)}

---

"""
    
    for i, (question_data, response) in enumerate(zip(questions, responses)):
        question_id = question_data.get("id", f"{metric_name}_{i+1}")
        question = question_data.get("prompt", "")
        
        # Get additional information if available
        gold_answer = gold_answers[i] if gold_answers and i < len(gold_answers) else "N/A"
        score = scores[i] if scores and i < len(scores) else "N/A"
        audit_info = audit_trail[i] if audit_trail and i < len(audit_trail) else {}
        
        markdown_content += f"""## Question {i+1}: {question_id}

**Question/Prompt**: 
```
{question}
```

**Model Response**: 
```
{response}
```

**Expected Answer**: {gold_answer}

**Score**: {score}

"""
        
        # Add audit information if available
        if audit_info:
            markdown_content += f"""**Evaluation Details**:
- **Parsed Answer**: {audit_info.get('parsed_answer', 'N/A')}
- **Match Type**: {audit_info.get('match_type', 'N/A')}
- **Parse Step**: {audit_info.get('parse_step', 'N/A')}
- **Violations**: {', '.join(audit_info.get('violations', [])) if audit_info.get('violations') else 'None'}
- **Exact Match**: {'Yes' if audit_info.get('is_exact_match', False) else 'No'}

"""
        
        markdown_content += "---\n\n"
    
    # Save to file
    filename = f"{metric_name}_responses.md"
    filepath = model_dir / filename
    
    with open(filepath, 'w') as f:
        f.write(markdown_content)
    
    return str(filepath)


def generate_accuracy_markdown(model_name: str, questions: List[Dict], responses: List[str], 
                              gold_answers: List[str], accuracy_result: Any, dataset: Dict) -> str:
    """Generate markdown report for accuracy evaluation"""
    
    markdown_content = f"""# CLMPI Accuracy Evaluation Report

**Model**: {model_name}
**Evaluation Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Dataset**: {dataset.get('name', 'Unknown')}

## Summary

- **Exact Match Score**: {accuracy_result.exact_match:.3f}
- **F1 Score**: {accuracy_result.f1_score:.3f}
- **Total Questions**: {len(questions)}

## Detailed Results

"""
    
    for i, (question_data, response, gold_answer) in enumerate(zip(questions, responses, gold_answers)):
        question_id = question_data.get("id", f"q_{i+1}")
        question = question_data.get("question", "")
        
        markdown_content += f"""### Question {i+1}: {question_id}

**Question**: {question}

**Model Response**: 
```
{response}
```

**Expected Answer**: {gold_answer}

**Score**: {accuracy_result.detailed_scores[i]:.3f}

---
"""
    
    return markdown_content
