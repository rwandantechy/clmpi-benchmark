#!/usr/bin/env python3
"""
CLMPI Efficiency Step Evaluation Script
Evaluates efficiency metric independently for stepwise CLMPI evaluation
"""

import argparse
import json
import yaml
import time
import logging
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from clmpi_calculator import CLMPICalculator
from ollama_runner import OllamaRunner
from generation import load_generation_profile


def load_metric_config(metric_name: str) -> dict:
    """Load metric configuration from config/metrics/"""
    config_path = Path(f"config/metrics/{metric_name}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Metric config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path: str) -> dict:
    """Load dataset from path"""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
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
    """Run efficiency evaluation for a single model"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load metric config
    metric_config = load_metric_config("efficiency")
    dataset = load_dataset(metric_config["dataset"])
    
    # Load generation profile
    profile = load_generation_profile(metric_config["profile"])
    
    if verbose:
        logger.info(f"Loaded dataset: {metric_config['dataset']}")
        logger.info(f"Using profile: {metric_config['profile']} - {profile}")
    
    # Initialize components
    calculator = CLMPICalculator()
    ollama_runner = OllamaRunner("http://localhost:11434")
    
    # Generate responses and measure efficiency
    questions = dataset.get("questions", [])[:5]  # Limit for demo
    efficiency_results = []
    
    for question_data in questions:
        question = question_data["question"]
        
        try:
            # Extract generation parameters from profile
            max_tokens = profile.get("max_tokens", 1000)
            temperature = profile.get("temperature", 0.1)
            
            # Create a wrapper function for efficiency measurement
            def generate_with_ollama():
                return ollama_runner.generate_response(model_name, question, max_tokens, temperature)
            
            # Measure efficiency
            efficiency_result = calculator.measure_efficiency(generate_with_ollama)
            efficiency_results.append(efficiency_result)
            
            if verbose:
                logger.info(f"Question: {question}")
                logger.info(f"Latency: {efficiency_result.latency_seconds:.4f}s")
                logger.info(f"CPU: {efficiency_result.cpu_usage_percent:.1f}%")
                logger.info(f"Memory: {efficiency_result.memory_used_mb:.1f}MB")
                logger.info(f"Raw Efficiency: {efficiency_result.raw_efficiency:.3f}")
        
        except Exception as e:
            logger.error(f"Error measuring efficiency for question: {e}")
            # Create a dummy result to maintain consistency
            from clmpi_calculator import EfficiencyResult
            dummy_result = EfficiencyResult(
                latency_seconds=1.0,
                cpu_usage_percent=0.0,
                memory_used_mb=0.0,
                raw_efficiency=0.0,
                normalized_efficiency=0.0
            )
            efficiency_results.append(dummy_result)
    
    # Normalize efficiency scores
    normalized_scores = calculator.normalize_efficiency_scores(efficiency_results)
    for i, norm_score in enumerate(normalized_scores):
        efficiency_results[i].normalized_efficiency = norm_score
    
    # Calculate average efficiency
    avg_efficiency = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0
    
    # Find or create run directory
    run_dir = find_latest_run_directory()
    metric_dir = run_dir / "efficiency"
    metric_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(metric_dir / "detail.jsonl", "w") as f:
        for i, (result, question_data) in enumerate(zip(efficiency_results, questions)):
            detail = {
                "question_id": question_data.get("id", f"eff_{i+1}"),
                "question": question_data["question"],
                "latency_seconds": result.latency_seconds,
                "cpu_usage_percent": result.cpu_usage_percent,
                "memory_used_mb": result.memory_used_mb,
                "raw_efficiency": result.raw_efficiency,
                "normalized_efficiency": result.normalized_efficiency
            }
            f.write(json.dumps(detail) + "\n")
    
    # Calculate averages with error handling
    try:
        avg_latency = sum(r.latency_seconds for r in efficiency_results) / len(efficiency_results)
        avg_cpu = sum(r.cpu_usage_percent for r in efficiency_results) / len(efficiency_results)
        avg_memory = sum(r.memory_used_mb for r in efficiency_results) / len(efficiency_results)
    except ZeroDivisionError:
        avg_latency = 0.0
        avg_cpu = 0.0
        avg_memory = 0.0
    
    # Save summary
    summary = {
        "metric": "efficiency",
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "avg_efficiency": avg_efficiency,
        "avg_latency_seconds": avg_latency,
        "avg_cpu_percent": avg_cpu,
        "avg_memory_mb": avg_memory,
        "total_questions": len(questions),
        "generation_profile": metric_config["profile"],
        "dataset_path": metric_config["dataset"]
    }
    
    with open(metric_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print single line summary
    print(f"[EFF] {model_name} efficiency={avg_efficiency:.3f}")
    
    if verbose:
        logger.info(f"Results saved to: {metric_dir}")
        logger.info(f"Average Efficiency: {avg_efficiency:.3f}")
        logger.info(f"Average Latency: {summary['avg_latency_seconds']:.3f}s")
        logger.info(f"Average CPU: {summary['avg_cpu_percent']:.1f}%")
        logger.info(f"Average Memory: {summary['avg_memory_mb']:.1f}MB")
    
    return {
        "metric": "efficiency",
        "score": avg_efficiency,
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
