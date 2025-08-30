#!/usr/bin/env python3
"""
CLMPI Prompt Validator
Validates that all prompt files follow the standardized schema
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


# Standardized schema for prompt items
REQUIRED_FIELDS = {
    "id": str,
    "category": str,
    "type": str,
    "prompt": str,
    "reference": list,
    "source": str
}

OPTIONAL_FIELDS = {
    "options": dict,
    "correct": str,
    "note": str
}

ALLOWED_CATEGORIES = {
    "accuracy", "context", "coherence", "fluency", "resource_efficiency"
}

ALLOWED_TYPES = {
    "numeric", "span", "mc_single", "string_exact", "regex"
}

TYPE_REQUIREMENTS = {
    "mc_single": ["options", "correct"],
    "numeric": [],
    "span": [],
    "string_exact": [],
    "regex": []
}


def validate_prompt_item(item: Dict[str, Any], file_path: str) -> List[str]:
    """Validate a single prompt item"""
    errors = []
    
    # Check required fields
    for field, field_type in REQUIRED_FIELDS.items():
        if field not in item:
            errors.append(f"Missing required field '{field}' in {file_path}")
        elif not isinstance(item[field], field_type):
            errors.append(f"Field '{field}' must be {field_type.__name__} in {file_path}")
    
    # Check category
    if "category" in item and item["category"] not in ALLOWED_CATEGORIES:
        errors.append(f"Invalid category '{item['category']}' in {file_path}. Must be one of: {ALLOWED_CATEGORIES}")
    
    # Check type
    if "type" in item and item["type"] not in ALLOWED_TYPES:
        errors.append(f"Invalid type '{item['type']}' in {file_path}. Must be one of: {ALLOWED_TYPES}")
    
    # Check type-specific requirements
    if "type" in item:
        required_fields = TYPE_REQUIREMENTS.get(item["type"], [])
        for field in required_fields:
            if field not in item:
                errors.append(f"Type '{item['type']}' requires field '{field}' in {file_path}")
    
    # Check prompt format
    if "prompt" in item and "id" in item:
        expected_format = f'Return ONLY: {{"id":"{item["id"]}","answer":"<value>"}} Do not explain.'
        if expected_format not in item["prompt"]:
            errors.append(f"Prompt must contain standardized return format in {file_path}")
    
    # Check reference format
    if "reference" in item and not isinstance(item["reference"], list):
        errors.append(f"Reference must be a list in {file_path}")
    
    # Check source URL format
    if "source" in item and not item["source"].startswith(("http://", "https://")):
        errors.append(f"Source must be a valid URL in {file_path}")
    
    return errors


def validate_prompt_file(file_path: Path) -> List[str]:
    """Validate a prompt file"""
    errors = []
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            errors.append(f"{file_path}: Root must be a list of prompt items")
            return errors
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                errors.append(f"{file_path}: Item {i} must be a dictionary")
                continue
            
            item_errors = validate_prompt_item(item, str(file_path))
            errors.extend(item_errors)
    
    except json.JSONDecodeError as e:
        errors.append(f"{file_path}: Invalid JSON - {e}")
    except Exception as e:
        errors.append(f"{file_path}: Error reading file - {e}")
    
    return errors


def main():
    """Main validation function"""
    prompts_dir = Path("prompts")
    if not prompts_dir.exists():
        print("Error: prompts directory not found")
        sys.exit(1)
    
    all_errors = []
    prompt_files = [
        "accuracy.json",
        "context.json", 
        "coherence.json",
        "fluency.json",
        "efficiency_tasks.json"
    ]
    
    for filename in prompt_files:
        file_path = prompts_dir / filename
        if file_path.exists():
            errors = validate_prompt_file(file_path)
            all_errors.extend(errors)
        else:
            all_errors.append(f"Missing required file: {file_path}")
    
    if all_errors:
        print("❌ Validation failed!")
        for error in all_errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("✅ All prompt files validated successfully!")
        print(f"  - Validated {len(prompt_files)} files")
        print("  - All files follow standardized schema")
        print("  - All required fields present")
        print("  - All types and categories valid")


if __name__ == "__main__":
    main()
