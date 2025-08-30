#!/usr/bin/env python3
"""
CLMPI Fluency Step Evaluation Script
Evaluates fluency metric independently for stepwise CLMPI evaluation
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
from logger import save_responses_markdown


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


def run_fluency_evaluation(model_name: str, verbose: bool = False) -> dict:
    """Run fluency evaluation for a single model"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load metric config
    metric_config = load_metric_config("fluency")
    dataset = load_dataset(metric_config["dataset"])
    
    # Load generation profile
    profile = load_generation_profile(metric_config["profile"])
    
    if verbose:
        logger.info(f"Loaded dataset: {metric_config['dataset']}")
        logger.info(f"Using profile: {metric_config['profile']} - {profile}")
    
    # Initialize components
    calculator = CLMPICalculator()
    ollama_runner = OllamaRunner("http://localhost:11434")
    
    # Generate responses
    prompts = dataset.get("prompts", [])[:5]  # Limit for demo
    responses = []
    
    for prompt_data in prompts:
        prompt = prompt_data["prompt"]
        
        try:
            # Extract generation parameters from profile
            max_tokens = profile.get("max_tokens", 1000)
            temperature = profile.get("temperature", 0.1)
            
            response, metrics = ollama_runner.generate_response(
                model_name, prompt, max_tokens, temperature
            )
            responses.append(response)
            
            if verbose:
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Response: {response}")
        
        except Exception as e:
            logger.error(f"Error generating response for prompt: {e}")
            responses.append("")
    
    # Calculate fluency
    fluency_result = calculator.evaluate_fluency(responses)
    
    # Find or create run directory
    run_dir = find_latest_run_directory()
    metric_dir = run_dir / "fluency"
    metric_dir.mkdir(exist_ok=True)
    
    # Save responses in organized Markdown format
    response_file = save_responses_markdown(
        model_name, "fluency", prompts, responses, 
        None, fluency_result.detailed_scores
    )
    
    # Save detailed results
    with open(metric_dir / "detail.jsonl", "w") as f:
        for i, (response, prompt_data) in enumerate(zip(responses, prompts)):
            detail = {
                "prompt_id": prompt_data.get("id", f"flu_{i+1}"),
                "prompt": prompt_data["prompt"],
                "response": response,
                "fluency_score": fluency_result.detailed_scores[i] if i < len(fluency_result.detailed_scores) else 0.0,
                "grammar_score": fluency_result.grammar_score,
                "perplexity_score": fluency_result.perplexity_score
            }
            f.write(json.dumps(detail) + "\n")
    
    # Save summary
    summary = {
        "metric": "fluency",
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fluency_score": fluency_result.fluency_score,
        "grammar_score": fluency_result.grammar_score,
        "perplexity_score": fluency_result.perplexity_score,
        "total_prompts": len(prompts),
        "generation_profile": metric_config["profile"],
        "dataset_path": metric_config["dataset"]
    }
    
    with open(metric_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print single line summary
    print(f"[FLU] {model_name} fluency={fluency_result.fluency_score:.3f}")
    
    if verbose:
        logger.info(f"Results saved to: {metric_dir}")
        logger.info(f"Response file: {response_file}")
        logger.info(f"Fluency Score: {fluency_result.fluency_score:.3f}")
        logger.info(f"Grammar Score: {fluency_result.grammar_score:.3f}")
        logger.info(f"Perplexity Score: {fluency_result.perplexity_score:.3f}")
    
    return {
        "metric": "fluency",
        "score": fluency_result.fluency_score,
        "run_dir": str(run_dir),
        "metric_dir": str(metric_dir)
    }


def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run CLMPI fluency evaluation step',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/step_fluency.py --model phi3:mini
  python scripts/step_fluency.py --model phi3:mini --verbose
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Model name to evaluate (must be available via Ollama)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        result = run_fluency_evaluation(args.model, args.verbose)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
