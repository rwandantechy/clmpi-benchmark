#!/usr/bin/env python3
"""
Test Enhanced CLMPI System

Quick validation script to test the enhanced evaluation system
with a small sample to ensure everything works correctly.
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_clmpi_calculator import EnhancedCLMPICalculator
import json
import yaml


def test_enhanced_calculator():
    """Test the enhanced CLMPI calculator with sample data"""
    print("Testing Enhanced CLMPI Calculator...")
    
    # Initialize calculator
    calculator = EnhancedCLMPICalculator()
    
    # Test accuracy evaluation
    print("  Testing accuracy evaluation...")
    responses = ["Paris", "Au", "42", "1945", "Jupiter"]
    gold_answers = ["Paris", "Au", "42", "1945", "Jupiter"]
    acceptable_answers = [["Paris", "paris"], ["Au", "au"], ["42"], ["1945"], ["Jupiter", "jupiter"]]
    
    accuracy_result = calculator.evaluate_accuracy(responses, gold_answers, acceptable_answers)
    print(f"    Accuracy - EM: {accuracy_result.exact_match:.3f}, F1: {accuracy_result.f1_score:.3f}")
    
    # Test contextual understanding
    print("  Testing contextual understanding...")
    contexts = ["Alice has 5 apples. Bob gives her 3 more.", "John is a doctor who works at City Hospital."]
    contextual_responses = ["8", "John is a doctor"]
    contextual_gold = ["8", "John is a doctor"]
    
    contextual_result = calculator.evaluate_contextual_understanding(
        contextual_responses, contexts, contextual_gold
    )
    print(f"    Contextual - Combined: {contextual_result.combined_score:.3f}")
    
    # Test coherence
    print("  Testing coherence evaluation...")
    coherence_responses = [
        "The cat walked to the pond. It looked at the water. Then it jumped in and swam.",
        "Renewable energy is important. Solar power reduces emissions. Wind energy is clean. We need sustainable solutions."
    ]
    
    coherence_result = calculator.evaluate_coherence(coherence_responses)
    print(f"    Coherence Score: {coherence_result.coherence_score:.3f}")
    
    # Test fluency
    print("  Testing fluency evaluation...")
    fluency_responses = [
        "The sunset was beautiful. The sky turned orange and red. Birds flew overhead.",
        "This is a test sentence with good grammar and proper punctuation."
    ]
    
    fluency_result = calculator.evaluate_fluency(fluency_responses)
    print(f"    Fluency Score: {fluency_result.fluency_score:.3f}")
    
    # Test efficiency measurement
    print("  Testing efficiency measurement...")
    def dummy_function():
        import time
        time.sleep(0.1)
        return "test response"
    
    efficiency_result = calculator.measure_efficiency(dummy_function)
    print(f"    Efficiency - Latency: {efficiency_result.latency_seconds:.3f}s, Raw: {efficiency_result.raw_efficiency:.3f}")
    
    # Test CLMPI calculation
    print("  Testing CLMPI calculation...")
    clmpi_results = calculator.calculate_clmpi(
        accuracy_result,
        contextual_result,
        coherence_result,
        fluency_result,
        0.8  # Mock efficiency score
    )
    
    print(f"    CLMPI Score: {clmpi_results['clmpi_01']:.3f} (0-1) / {clmpi_results['clmpi_100']:.1f} (0-100)")
    
    print("[OK] Enhanced calculator tests passed!")
    return True


def test_config_files():
    """Test that all config files are valid"""
    print("Testing configuration files...")
    
    config_files = [
        "config/model_config.yaml",
        "config/generation_config.yaml", 
        "config/device_default.yaml"
    ]
    
    for config_file in config_files:
        if not Path(config_file).exists():
            print(f"  [ERROR] Config file not found: {config_file}")
            return False
        
        try:
            with open(config_file, 'r') as f:
                yaml.safe_load(f)
            print(f"  [OK] {config_file} is valid YAML")
        except Exception as e:
            print(f"  [ERROR] {config_file} is invalid: {e}")
            return False
    
    print("[OK] All config files are valid!")
    return True


def test_datasets():
    """Test that all curated datasets are valid JSON"""
    print("Testing curated datasets...")
    
    dataset_files = [
        "prompts/accuracy_tasks_curated.json",
        "prompts/contextual_tasks_curated.json",
        "prompts/coherence_tasks.json",
        "prompts/fluency_tasks.json"
    ]
    
    for dataset_file in dataset_files:
        if not Path(dataset_file).exists():
            print(f"  [ERROR] Dataset file not found: {dataset_file}")
            return False
        
        try:
            with open(dataset_file, 'r') as f:
                data = json.load(f)
            
            # Check basic structure
            if 'version' not in data:
                print(f"  [WARNING] {dataset_file} missing version field")
            
            print(f"  [OK] {dataset_file} is valid JSON")
        except Exception as e:
            print(f"  [ERROR] {dataset_file} is invalid: {e}")
            return False
    
    print("[OK] All datasets are valid!")
    return True


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing module imports...")
    
    try:
        import numpy as np
        print("  [OK] numpy imported successfully")
    except ImportError as e:
        print(f"  [ERROR] numpy import failed: {e}")
        return False
    
    try:
        import psutil
        print("  [OK] psutil imported successfully")
    except ImportError as e:
        print(f"  [ERROR] psutil import failed: {e}")
        return False
    
    try:
        import yaml
        print("  [OK] pyyaml imported successfully")
    except ImportError as e:
        print(f"  [ERROR] pyyaml import failed: {e}")
        return False
    
    try:
        import language_tool_python
        print("  [OK] language_tool_python imported successfully")
    except ImportError as e:
        print(f"  [WARNING] language_tool_python not available (will use simplified grammar checking): {e}")
    
    print("[OK] All required imports successful!")
    return True


def main():
    """Run all tests"""
    print("Starting Enhanced CLMPI System Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_files,
        test_datasets,
        test_enhanced_calculator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"[ERROR] Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! Enhanced system is ready to use.")
        print("\nNext steps:")
        print("1. Pull your model: ollama pull <your_model_name>")
        print("2. Add model to config/model_config.yaml")
        print("3. Run enhanced evaluation:")
        print("   python scripts/enhanced_evaluate_models.py \\")
        print("     --model-config config/model_config.yaml \\")
        print("     --generation-config config/generation_config.yaml \\")
        print("     --device-config config/device_default.yaml \\")
        print("     --models <your_model_name> --label test_run")
        return 0
    else:
        print("[ERROR] Some tests failed. Please fix the issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())
