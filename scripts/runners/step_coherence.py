#!/usr/bin/env python3
"""
CLMPI Coherence Step Evaluation Script
Evaluates coherence metric independently for stepwise CLMPI evaluation
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
from typing import List, Tuple

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


def extract_context_and_generated(prompt: str, response: str) -> Tuple[str, str]:
    """Extract context and generated sentence from prompt and response"""
    # Extract context from prompt
    context_match = re.search(r'Context:\n"([^"]+)"', prompt)
    if not context_match:
        return "", response
    
    context = context_match.group(1)
    
    # Extract generated sentence from response
    try:
        response_json = json.loads(response)
        generated = response_json.get("answer", "")
    except:
        generated = response
    
    return context, generated


def simple_cosine_similarity(sent1: str, sent2: str) -> float:
    """Simple cosine similarity between two sentences using word overlap"""
    words1 = set(sent1.lower().split())
    words2 = set(sent2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


def calculate_repetition_penalty(text: str, ngram_size: int = 3) -> float:
    """Calculate repetition penalty using n-grams"""
    words = text.lower().split()
    if len(words) < ngram_size:
        return 1.0
    
    # Generate n-grams
    ngrams = []
    for i in range(len(words) - ngram_size + 1):
        ngrams.append(tuple(words[i:i + ngram_size]))
    
    if not ngrams:
        return 1.0
    
    # Count unique vs total n-grams
    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    
    repetition_rate = 1 - (unique_ngrams / total_ngrams)
    return 1 - repetition_rate


def calculate_coherence_score(context: str, generated: str) -> dict:
    """Calculate coherence score using sentence similarity and repetition penalty"""
    violations = []
    
    # Guards
    if len(generated.split()) < 3:
        violations.append("too_short")
        return {
            "sentence_similarity": 0.0,
            "repetition_penalty": 0.0,
            "score": 0.0,
            "violations": violations
        }
    
    # Split context into sentences
    context_sentences = re.split(r'[.!?]+', context.strip())
    context_sentences = [s.strip() for s in context_sentences if s.strip()]
    
    if not context_sentences:
        violations.append("no_context")
        return {
            "sentence_similarity": 0.0,
            "repetition_penalty": 0.0,
            "score": 0.0,
            "violations": violations
        }
    
    # Get last context sentence
    last_context_sentence = context_sentences[-1]
    
    # 1. Sentence similarity (70% weight)
    sentence_similarity = simple_cosine_similarity(last_context_sentence, generated)
    
    # 2. Repetition penalty (30% weight)
    full_text = context + " " + generated
    repetition_penalty = calculate_repetition_penalty(full_text, ngram_size=3)
    
    # Combined score
    coherence_score = 0.7 * sentence_similarity + 0.3 * repetition_penalty
    
    return {
        "sentence_similarity": sentence_similarity,
        "repetition_penalty": repetition_penalty,
        "score": coherence_score,
        "violations": violations
    }


def run_coherence_evaluation(model_name: str, verbose: bool = False) -> dict:
    """Run coherence evaluation for a single model"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load metric config
    metric_config = load_metric_config("coherence")
    dataset = load_dataset("prompts/coherence.json")
    
    # Load generation profile
    profile = load_generation_profile("creative")
    
    if verbose:
        logger.info(f"Loaded dataset: prompts/coherence.json")
        logger.info(f"Using profile: creative - {profile}")
    
    # Initialize components
    calculator = CLMPICalculator()
    ollama_runner = OllamaRunner("http://localhost:11434")
    
    # Generate responses
    prompts = dataset[:5]  # Limit for demo
    responses = []
    coherence_scores = []
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
            
            # Extract context and generated sentence
            context, generated = extract_context_and_generated(prompt, response)
            
            # Calculate coherence score
            coherence_result = calculate_coherence_score(context, generated)
            coherence_scores.append(coherence_result["score"])
            detailed_results.append(coherence_result)
            
            if verbose:
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Response: {response}")
                logger.info(f"Context: {context}")
                logger.info(f"Generated: {generated}")
                logger.info(f"Coherence Score: {coherence_result['score']:.3f}")
        
        except Exception as e:
            logger.error(f"Error generating response for prompt: {e}")
            responses.append("")
            coherence_scores.append(0.0)
            detailed_results.append({
                "sentence_similarity": 0.0,
                "repetition_penalty": 0.0,
                "score": 0.0,
                "violations": ["error"]
            })
    
    # Calculate overall coherence score
    overall_coherence = np.mean(coherence_scores)
    
    # Find or create run directory
    run_dir = find_latest_run_directory()
    metric_dir = run_dir / "coherence"
    metric_dir.mkdir(exist_ok=True)
    
    # Save responses in organized Markdown format
    response_file = save_responses_markdown(
        model_name, "coherence", prompts, responses, 
        [f"Score: {score:.3f}" for score in coherence_scores], coherence_scores
    )
    
    # Save detailed results
    with open(metric_dir / "detail.jsonl", "w") as f:
        for i, (response, prompt_data, detail_result) in enumerate(zip(responses, prompts, detailed_results)):
            context, generated = extract_context_and_generated(prompt_data["prompt"], response)
            detail = {
                "prompt_id": prompt_data.get("id", f"coh_{i+1}"),
                "prompt": prompt_data["prompt"],
                "response": response,
                "context": context,
                "generated": generated,
                "coherence": detail_result,
                "exact_match": detail_result["score"],  # For compatibility
                "f1_score": detail_result["score"]      # For compatibility
            }
            f.write(json.dumps(detail) + "\n")
    
    # Save summary
    summary = {
        "metric": "coherence",
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exact_match": overall_coherence,  # Use coherence score as exact_match for compatibility
        "f1_score": overall_coherence,     # Use coherence score as f1_score for compatibility
        "coherence_score": overall_coherence,
        "total_prompts": len(prompts),
        "generation_profile": "creative",
        "dataset_path": "prompts/coherence.json"
    }
    
    with open(metric_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print single line summary
    print(f"[COH] {model_name} coherence={overall_coherence:.3f}")
    
    if verbose:
        logger.info(f"Results saved to: {metric_dir}")
        logger.info(f"Response file: {response_file}")
        logger.info(f"Overall Coherence Score: {overall_coherence:.3f}")
    
    return {
        "metric": "coherence",
        "score": overall_coherence,
        "run_dir": str(run_dir),
        "metric_dir": str(metric_dir)
    }


def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run CLMPI coherence evaluation step',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/step_coherence.py --model phi3:mini
  python scripts/step_coherence.py --model phi3:mini --verbose
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Model name to evaluate (must be available via Ollama)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        result = run_coherence_evaluation(args.model, args.verbose)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
