#!/usr/bin/env python3
"""
CLMPI Efficiency Step Evaluation Script
Simple performance measurement: time, CPU, memory
"""

import argparse
import json
import yaml
import time
import logging
import psutil
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from clmpi_calculator import CLMPICalculator
from ollama_runner import OllamaRunner
from generation import load_generation_profile
from logger import save_responses_markdown

def load_dataset(dataset_path: str) -> dict:
    """Load dataset from path"""
    import json
    from pathlib import Path
    
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def load_metric_config(metric_name: str) -> dict:
    """Load metric configuration from config/metrics/"""
    config_path = Path(f"config/metrics/{metric_name}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Metric config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_efficiency_tasks() -> dict:
    """Load efficiency tasks dataset"""
    dataset_path = "prompts/efficiency_tasks.json"
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Efficiency tasks dataset not found: {dataset_path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def find_latest_run_directory() -> Path:
    """Find the latest run directory, or create one if none exists"""
    results_dir = Path("results")
    if not results_dir.exists():
        results_dir.mkdir()
    
    # Look for existing stepwise runs
    stepwise_runs = list(results_dir.glob("*_stepwise"))
    if stepwise_runs:
        # Use the most recent one
        latest = max(stepwise_runs, key=lambda p: p.stat().st_mtime)
        return latest
    else:
        # Create new timestamped run directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = results_dir / f"{timestamp}_stepwise"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


def run_efficiency_evaluation(model_name: str, verbose: bool = False) -> dict:
    """Run simple efficiency evaluation - just measure time, CPU, memory"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load efficiency task
    dataset = load_dataset("prompts/efficiency_tasks.json")
    task = dataset[0]  # Just one task
    prompt = task.get("prompt", "What is 15 + 27?")
    
    # Load generation profile
    profile = load_generation_profile("deterministic")
    
    if verbose:
        logger.info(f"Testing efficiency with prompt: {prompt}")
        logger.info(f"Using profile: deterministic")
    
    # Initialize components
    ollama_runner = OllamaRunner("http://localhost:11434")
    
    try:
        # Extract generation parameters
        max_tokens = profile.get("max_tokens", 1000)
        temperature = profile.get("temperature", 0.0)
        
        # Measure performance
        process = psutil.Process()
        initial_cpu = process.cpu_percent()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        start_time = time.time()
        response, _ = ollama_runner.generate_response(model_name, prompt, max_tokens, temperature)
        end_time = time.time()
        
        final_cpu = process.cpu_percent()
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Calculate metrics
        latency = end_time - start_time
        cpu_usage = (initial_cpu + final_cpu) / 2
        memory_used = final_memory - initial_memory
        
        success = True
        
        if verbose:
            logger.info(f"Response: {response}")
            logger.info(f"Time: {latency:.3f}s")
            logger.info(f"CPU: {cpu_usage:.1f}%")
            logger.info(f"Memory: {memory_used:.1f}MB")
        
    except Exception as e:
        logger.error(f"Error measuring efficiency: {e}")
        latency = 30.0
        cpu_usage = 0.0
        memory_used = 0.0
        response = ""
        success = False
    
    # Simple efficiency score based on time
    if latency <= 1.0:
        efficiency = 1.0
    elif latency <= 3.0:
        efficiency = 0.7
    elif latency <= 5.0:
        efficiency = 0.4
    else:
        efficiency = 0.1
    
    # Find or create run directory
    run_dir = find_latest_run_directory()
    metric_dir = run_dir / "efficiency"
    metric_dir.mkdir(exist_ok=True)
    
    # Save responses in organized Markdown format
    task_data = [{"id": "eff_001", "prompt": prompt}]
    response_file = save_responses_markdown(
        model_name, "efficiency", task_data, [response], 
        None, [efficiency]
    )
    
    # Save detailed results
    with open(metric_dir / "detail.jsonl", "w") as f:
        detail = {
            "task_id": "eff_001",
            "prompt": prompt,
            "response": response,
            "latency_seconds": latency,
            "cpu_usage_percent": cpu_usage,
            "memory_used_mb": max(0, memory_used),
            "efficiency": efficiency,
            "success": success
        }
        f.write(json.dumps(detail) + "\n")
    
    # Save summary
    summary = {
        "metric": "efficiency",
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "efficiency": efficiency,
        "latency_seconds": latency,
        "cpu_usage_percent": cpu_usage,
        "memory_used_mb": max(0, memory_used),
        "success": success,
        "generation_profile": "deterministic",
        "dataset_path": "prompts/efficiency_tasks.json"
    }
    
    with open(metric_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print single line summary
    print(f"[EFF] {model_name} efficiency={efficiency:.3f}")
    
    if verbose:
        logger.info(f"Results saved to: {metric_dir}")
        logger.info(f"Response file: {response_file}")
        logger.info(f"Efficiency: {efficiency:.3f}")
        logger.info(f"Latency: {latency:.3f}s")
        logger.info(f"CPU: {cpu_usage:.1f}%")
        logger.info(f"Memory: {memory_used:.1f}MB")
    
    return {
        "metric": "efficiency",
        "score": efficiency,
        "run_dir": str(run_dir),
        "metric_dir": str(metric_dir)
    }


def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run CLMPI efficiency evaluation step',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/step_efficiency.py --model phi3:mini
  python scripts/step_efficiency.py --model phi3:mini --verbose
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Model name to evaluate (must be available via Ollama)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        result = run_efficiency_evaluation(args.model, args.verbose)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
