#!/usr/bin/env python3
"""
Demo script showing the stepwise CLMPI evaluation workflow
This script demonstrates how to run individual metrics and combine them
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and show its output"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    """Run stepwise evaluation demo"""
    print("CLMPI STEPWISE EVALUATION DEMO")
    print("This demo shows how to run individual metrics and combine them")
    print("Note: This requires a model to be available via Ollama")
    
    model_name = "test_model"  # You can change this to an actual model name
    
    # List of stepwise commands
    steps = [
        ([sys.executable, "scripts/step_accuracy.py", "--model", model_name, "--verbose"], 
         "Run Accuracy Evaluation"),
        ([sys.executable, "scripts/step_context.py", "--model", model_name, "--verbose"], 
         "Run Contextual Understanding Evaluation"),
        ([sys.executable, "scripts/step_coherence.py", "--model", model_name, "--verbose"], 
         "Run Coherence Evaluation"),
        ([sys.executable, "scripts/step_fluency.py", "--model", model_name, "--verbose"], 
         "Run Fluency Evaluation"),
        ([sys.executable, "scripts/step_efficiency.py", "--model", model_name, "--verbose"], 
         "Run Efficiency Evaluation"),
        ([sys.executable, "scripts/combine_clmpi.py", "--model", model_name, "--detailed"], 
         "Combine All Metrics into CLMPI Score")
    ]
    
    print(f"\nUsing model: {model_name}")
    print("To run with a real model, edit this script and change 'test_model' to your model name")
    print("\nThis demo will FAIL because 'test_model' doesn't exist, but shows the workflow structure")
    
    # Run each step
    for i, (cmd, description) in enumerate(steps, 1):
        success = run_command(cmd, f"{i}. {description}")
        if not success:
            print(f"\nStep {i} failed - this is expected for demo purposes")
            break
        time.sleep(1)  # Brief pause between steps
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print(f"{'='*60}")
    print("\nTo run with a real model:")
    print("1. Make sure Ollama is running")
    print("2. Pull a model: ollama pull phi3:mini")
    print("3. Run each step script with --model phi3:mini")
    print("4. Combine results with combine_clmpi.py")
    
    print("\nStepwise evaluation allows you to:")
    print("- Run metrics independently")
    print("- Debug individual metric issues")
    print("- Resume from failed steps")
    print("- Parallelize metric evaluation")
    print("- Analyze per-metric performance")


if __name__ == "__main__":
    main()
