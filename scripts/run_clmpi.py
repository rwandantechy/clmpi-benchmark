#!/usr/bin/env python3
"""
CLMPI Automated Runner
One command to run the complete CLMPI evaluation for any model
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# Import utility functions
sys.path.append(str(Path(__file__).parent))
from utils import create_results_structure, create_run_directory, update_latest_symlink, get_latest_run_directory

def check_ollama_model(model_name: str) -> bool:
    """Check if a model is available via Ollama"""
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return model_name in result.stdout
    except subprocess.CalledProcessError:
        return False

def run_step(step_name: str, model_name: str, verbose: bool = False) -> bool:
    """Run a single evaluation step"""
    script_path = Path(f"scripts/runners/step_{step_name}.py")
    
    if not script_path.exists():
        print(f"Error: Step script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path), "--model", model_name]
    if verbose:
        cmd.append("--verbose")
    
    print(f"Running {step_name} evaluation...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {step_name} failed: {e}")
        if verbose:
            print(f"Error output: {e.stderr}")
        return False

def run_combine_clmpi(model_name: str, verbose: bool = False) -> bool:
    """Run the CLMPI combination step"""
    script_path = Path("scripts/combine_clmpi.py")
    
    if not script_path.exists():
        print(f"Error: Combine script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path), "--model", model_name]
    if verbose:
        cmd.append("--verbose")
    
    print("Combining results into final CLMPI score...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ CLMPI score calculated successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ CLMPI combination failed: {e}")
        if verbose:
            print(f"Error output: {e.stderr}")
        return False

def show_results():
    """Show the latest results"""
    latest_run = get_latest_run_directory()
    if not latest_run:
        print("No evaluation runs found")
        return
    
    clmpi_file = latest_run / "clmpi_summary.json"
    
    if clmpi_file.exists():
        print(f"\nCLMPI Results from: {latest_run.name}")
        print("=" * 50)
        
        try:
            with open(clmpi_file, 'r') as f:
                import json
                data = json.load(f)
                
            if 'clmpi_scores' in data:
                scores = data['clmpi_scores']
                print(f"CLMPI_01:  {scores.get('clmpi_01', 'N/A'):.3f}")
                print(f"CLMPI_100: {scores.get('clmpi_100', 'N/A'):.3f}")
            
            if 'component_scores' in data:
                print("\nComponent Scores:")
                for metric, info in data['component_scores'].items():
                    score = info.get('score', 0)
                    contribution = info.get('contribution', 0)
                    print(f"  {metric:20} {score:.3f} (contributes {contribution:.3f})")
                    
        except Exception as e:
            print(f"Error reading results: {e}")
    else:
        print(f"CLMPI summary not found in {latest_run.name}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='CLMPI Automated Runner - One command to run everything',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_clmpi.py --model mistral:7b
  python scripts/run_clmpi.py --model phi3:mini --verbose
  python scripts/run_clmpi.py --model llama3.1:8b --show-results
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Model name to evaluate (must be available via Ollama)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--show-results', action='store_true',
                       help='Show latest results without running evaluation')
    
    args = parser.parse_args()
    
    if args.show_results:
        show_results()
        return 0
    
    # Ensure results structure exists
    create_results_structure()
    
    # Check if model is available
    print(f"Looking for model '{args.model}'...")
    if not check_ollama_model(args.model):
        print(f"Model '{args.model}' not found via Ollama")
        print(f"Pull it first with: ollama pull {args.model}")
        return 1
    
    print(f"Great! Found '{args.model}', starting evaluation...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create run directory
    run_dir = create_run_directory("stepwise")
    print(f"Created run directory: {run_dir.name}")
    
    # Run all evaluation steps
    steps = ['accuracy', 'context', 'coherence', 'fluency', 'efficiency']
    success_count = 0
    
    for step in steps:
        if run_step(step, args.model, args.verbose):
            success_count += 1
        else:
            print(f"Continuing with other steps...")
    
    # Combine results
    if success_count == len(steps):
        if run_combine_clmpi(args.model, args.verbose):
            print("\nAll done! Everything completed successfully.")
            
            # Update latest symlink
            if update_latest_symlink(run_dir):
                print(f"Updated latest symlink to: {run_dir.name}")
            else:
                print("Warning: Could not create latest symlink")
        else:
            print("\nEvaluation finished but had trouble combining results")
    else:
        print(f"\n{success_count}/{len(steps)} steps completed successfully")
    
    # Show results
    print("\n" + "=" * 60)
    show_results()
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time:.1f} seconds")
    
    return 0 if success_count == len(steps) else 1

if __name__ == "__main__":
    exit(main())
