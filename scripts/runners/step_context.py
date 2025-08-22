#!/usr/bin/env python3
"""
CLMPI Contextual Understanding Step Evaluation Script
Evaluates contextual understanding metric independently for stepwise CLMPI evaluation
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

from enhanced_clmpi_calculator import EnhancedCLMPICalculator
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


def run_context_evaluation(model_name: str, verbose: bool = False) -> dict:
    """Run contextual understanding evaluation for a single model"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load metric config
    metric_config = load_metric_config("context")
    dataset = load_dataset(metric_config["dataset"])
    
    # Load generation profile
    profile = load_generation_profile(metric_config["profile"])
    
    if verbose:
        logger.info(f"Loaded dataset: {metric_config['dataset']}")
        logger.info(f"Using profile: {metric_config['profile']} - {profile}")
    
    # Initialize components
    calculator = EnhancedCLMPICalculator()
    ollama_runner = OllamaRunner("http://localhost:11434")
    
    # Generate responses
    conversations = dataset.get("conversations", [])[:5]  # Limit for demo
    responses = []
    contexts = []
    gold_answers = []
    
    for conv_data in conversations:
        context = conv_data["context"]
        question = conv_data["question"]
        correct_answer = conv_data["correct_answer"]
        
        prompt = f"Context: {context}\n\nQuestion: {question}"
        
        try:
            response = ollama_runner.generate_response(model_name, prompt, profile)
            responses.append(response)
            contexts.append(context)
            gold_answers.append(correct_answer)
            
            if verbose:
                logger.info(f"Context: {context}")
                logger.info(f"Q: {question}")
                logger.info(f"A: {response}")
                logger.info(f"Expected: {correct_answer}")
        
        except Exception as e:
            logger.error(f"Error generating response for conversation: {e}")
            responses.append("")
            contexts.append(context)
            gold_answers.append(correct_answer)
    
    # Calculate contextual understanding
    context_result = calculator.evaluate_contextual_understanding(responses, contexts, gold_answers)
    
    # Find or create run directory
    run_dir = find_latest_run_directory()
    metric_dir = run_dir / "context"
    metric_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(metric_dir / "detail.jsonl", "w") as f:
        for i, (response, context, gold, conv_data) in enumerate(zip(responses, contexts, gold_answers, conversations)):
            detail = {
                "conversation_id": conv_data.get("id", f"ctx_{i+1}"),
                "context": context,
                "question": conv_data["question"],
                "response": response,
                "gold_answer": gold,
                "exact_match": 1 if response.strip().lower() == gold.strip().lower() else 0,
                "context_similarity": context_result.context_similarity
            }
            f.write(json.dumps(detail) + "\n")
    
    # Save summary
    summary = {
        "metric": "contextual_understanding",
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exact_match": context_result.exact_match,
        "f1_score": context_result.f1_score,
        "context_similarity": context_result.context_similarity,
        "combined_score": context_result.combined_score,
        "total_conversations": len(conversations),
        "generation_profile": metric_config["profile"],
        "dataset_path": metric_config["dataset"]
    }
    
    with open(metric_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print single line summary
    print(f"[CTX] {model_name} context={context_result.combined_score:.3f}")
    
    if verbose:
        logger.info(f"Results saved to: {metric_dir}")
        logger.info(f"Exact Match: {context_result.exact_match:.3f}")
        logger.info(f"F1 Score: {context_result.f1_score:.3f}")
        logger.info(f"Context Similarity: {context_result.context_similarity:.3f}")
        logger.info(f"Combined Score: {context_result.combined_score:.3f}")
    
    return {
        "metric": "contextual_understanding",
        "score": context_result.combined_score,
        "run_dir": str(run_dir),
        "metric_dir": str(metric_dir)
    }


def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run CLMPI contextual understanding evaluation step',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/step_context.py --model phi3:mini
  python scripts/step_context.py --model phi3:mini --verbose
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Model name to evaluate (must be available via Ollama)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        result = run_context_evaluation(args.model, args.verbose)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
