#!/usr/bin/env python3
"""
CLMPI Accuracy Step Evaluation Script
Evaluates accuracy metric independently for stepwise CLMPI evaluation
"""

import argparse
import json
import yaml
import time
import logging
from pathlib import Path
from datetime import datetime

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


def create_run_directory() -> Path:
    """Create timestamped run directory"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path("results") / f"{timestamp}_stepwise"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_accuracy_evaluation(model_name: str, verbose: bool = False) -> dict:
    """Run accuracy evaluation for a single model"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load metric config
    metric_config = load_metric_config("accuracy")
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
    questions = dataset.get("questions", [])[:5]  # Limit for demo
    responses = []
    gold_answers = []
    
    for question_data in questions:
        question = question_data["question"]
        correct_answer = question_data["correct_answer"]
        
        try:
            response = ollama_runner.generate_response(model_name, question, profile)
            responses.append(response)
            gold_answers.append(correct_answer)
            
            if verbose:
                logger.info(f"Q: {question}")
                logger.info(f"A: {response}")
                logger.info(f"Expected: {correct_answer}")
        
        except Exception as e:
            logger.error(f"Error generating response for question: {e}")
            responses.append("")
            gold_answers.append(correct_answer)
    
    # Calculate accuracy
    accuracy_result = calculator.evaluate_accuracy(responses, gold_answers)
    
    # Create run directory
    run_dir = create_run_directory()
    metric_dir = run_dir / "accuracy"
    metric_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(metric_dir / "detail.jsonl", "w") as f:
        for i, (response, gold, question_data) in enumerate(zip(responses, gold_answers, questions)):
            detail = {
                "question_id": question_data.get("id", f"acc_{i+1}"),
                "question": question_data["question"],
                "response": response,
                "gold_answer": gold,
                "score": accuracy_result.detailed_scores[i],
                "exact_match": 1 if response.strip().lower() == gold.strip().lower() else 0
            }
            f.write(json.dumps(detail) + "\n")
    
    # Save summary
    summary = {
        "metric": "accuracy",
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exact_match": accuracy_result.exact_match,
        "f1_score": accuracy_result.f1_score,
        "total_questions": len(questions),
        "generation_profile": metric_config["profile"],
        "dataset_path": metric_config["dataset"]
    }
    
    with open(metric_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print single line summary
    print(f"[ACC] {model_name} accuracy={accuracy_result.f1_score:.3f}")
    
    if verbose:
        logger.info(f"Results saved to: {metric_dir}")
        logger.info(f"Exact Match: {accuracy_result.exact_match:.3f}")
        logger.info(f"F1 Score: {accuracy_result.f1_score:.3f}")
    
    return {
        "metric": "accuracy",
        "score": accuracy_result.f1_score,
        "run_dir": str(run_dir),
        "metric_dir": str(metric_dir)
    }


def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run CLMPI accuracy evaluation step',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/step_accuracy.py --model phi3:mini
  python scripts/step_accuracy.py --model phi3:mini --verbose
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Model name to evaluate (must be available via Ollama)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        result = run_accuracy_evaluation(args.model, args.verbose)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
