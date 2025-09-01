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
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List

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


def extract_generated_text(response: str) -> str:
    """Extract generated text from response"""
    try:
        response_json = json.loads(response)
        generated = response_json.get("answer", "")
    except:
        generated = response
    
    return generated


def simple_perplexity_estimate(text: str) -> float:
    """Simple perplexity estimation using word frequency"""
    words = text.lower().split()
    if len(words) < 2:
        return 1.0
    
    # Simple bigram model
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append((words[i], words[i + 1]))
    
    # Count bigram frequencies
    bigram_counts = {}
    for bigram in bigrams:
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
    
    # Calculate simple perplexity
    total_bigrams = len(bigrams)
    if total_bigrams == 0:
        return 1.0
    
    log_prob = 0
    for bigram in bigrams:
        prob = bigram_counts[bigram] / total_bigrams
        log_prob += np.log(prob + 1e-10)  # Add small epsilon to avoid log(0)
    
    perplexity = np.exp(-log_prob / total_bigrams)
    return perplexity


def simple_grammar_check(text: str) -> int:
    """Simple grammar error detection using basic rules"""
    errors = 0
    sentences = re.split(r'[.!?]+', text.strip())
    
    for sentence in sentences:
        words = sentence.split()
        if len(words) == 0:
            continue
        
        # Basic checks
        # 1. Check for capitalization at start
        if words[0] and not words[0][0].isupper():
            errors += 1
        
        # 2. Check for basic punctuation
        if not sentence.endswith(('.', '!', '?')):
            errors += 1
        
        # 3. Check for very short sentences (likely incomplete)
        if len(words) < 3:
            errors += 1
    
    return errors


def calculate_fluency_score(text: str) -> dict:
    """Calculate fluency score using perplexity and grammar error detection"""
    violations = []
    
    # Guards
    if len(text.split()) < 3:
        violations.append("too_short")
        return {
            "perplexity": 1.0,
            "perplexity_score": 0.0,
            "grammar_errors": 0,
            "grammar_score": 0.0,
            "score": 0.0,
            "violations": violations
        }
    
    # 1. Perplexity (60% weight)
    perplexity = simple_perplexity_estimate(text)
    perplexity_cap = 100
    perplexity_capped = min(perplexity, perplexity_cap)
    perplexity_score = 1 / (1 + perplexity_capped)
    
    # 2. Grammar errors (40% weight)
    grammar_errors = simple_grammar_check(text)
    total_tokens = len(text.split())
    grammar_error_rate = grammar_errors / max(total_tokens, 1)
    grammar_error_cap = 0.2
    grammar_error_capped = min(grammar_error_rate, grammar_error_cap)
    grammar_score = 1 - grammar_error_capped
    
    # Combined score
    fluency_score = 0.6 * perplexity_score + 0.4 * grammar_score
    
    return {
        "perplexity": perplexity,
        "perplexity_score": perplexity_score,
        "grammar_errors": grammar_errors,
        "grammar_score": grammar_score,
        "score": fluency_score,
        "violations": violations
    }


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
    dataset = load_dataset("prompts/fluency.json")
    
    # Load generation profile
    profile = load_generation_profile("creative")
    
    if verbose:
        logger.info(f"Loaded dataset: prompts/fluency.json")
        logger.info(f"Using profile: creative - {profile}")
    
    # Initialize components
    calculator = CLMPICalculator()
    ollama_runner = OllamaRunner("http://localhost:11434")
    
    # Generate responses
    prompts = dataset[:5]  # Limit for demo
    responses = []
    fluency_scores = []
    detailed_results = []
    
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
            
            # Extract generated text
            generated_text = extract_generated_text(response)
            
            # Calculate fluency score
            fluency_result = calculate_fluency_score(generated_text)
            fluency_scores.append(fluency_result["score"])
            detailed_results.append(fluency_result)
            
            if verbose:
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Response: {response}")
                logger.info(f"Generated: {generated_text}")
                logger.info(f"Fluency Score: {fluency_result['score']:.3f}")
        
        except Exception as e:
            logger.error(f"Error generating response for prompt: {e}")
            responses.append("")
            fluency_scores.append(0.0)
            detailed_results.append({
                "perplexity": 1.0,
                "perplexity_score": 0.0,
                "grammar_errors": 0,
                "grammar_score": 0.0,
                "score": 0.0,
                "violations": ["error"]
            })
    
    # Calculate overall fluency score
    overall_fluency = np.mean(fluency_scores)
    
    # Find or create run directory
    run_dir = find_latest_run_directory()
    metric_dir = run_dir / "fluency"
    metric_dir.mkdir(exist_ok=True)
    
    # Save responses in organized Markdown format
    response_file = save_responses_markdown(
        model_name, "fluency", prompts, responses, 
        [f"Score: {score:.3f}" for score in fluency_scores], fluency_scores
    )
    
    # Save detailed results
    with open(metric_dir / "detail.jsonl", "w") as f:
        for i, (response, prompt_data, detail_result) in enumerate(zip(responses, prompts, detailed_results)):
            generated_text = extract_generated_text(response)
            detail = {
                "prompt_id": prompt_data.get("id", f"flu_{i+1}"),
                "prompt": prompt_data["prompt"],
                "response": response,
                "generated": generated_text,
                "fluency": detail_result,
                "exact_match": detail_result["score"],  # For compatibility
                "f1_score": detail_result["score"]      # For compatibility
            }
            f.write(json.dumps(detail) + "\n")
    
    # Save summary
    summary = {
        "metric": "fluency",
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exact_match": overall_fluency,  # Use fluency score as exact_match for compatibility
        "f1_score": overall_fluency,    # Use fluency score as f1_score for compatibility
        "fluency_score": overall_fluency,
        "total_prompts": len(prompts),
        "generation_profile": "creative",
        "dataset_path": "prompts/fluency.json"
    }
    
    with open(metric_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print single line summary
    print(f"[FLU] {model_name} fluency={overall_fluency:.3f}")
    
    if verbose:
        logger.info(f"Results saved to: {metric_dir}")
        logger.info(f"Response file: {response_file}")
        logger.info(f"Overall Fluency Score: {overall_fluency:.3f}")
    
    return {
        "metric": "fluency",
        "score": overall_fluency,
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
