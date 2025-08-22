#!/usr/bin/env python3
"""
CLMPI Combiner Script
Combines individual metric results into final CLMPI score for stepwise evaluation
"""

import argparse
import json
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, Optional


def load_model_config() -> dict:
    """Load model configuration for weights"""
    config_path = Path("config/model_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_latest_run_directory() -> Path:
    """Find the latest stepwise run directory"""
    results_dir = Path("results")
    if not results_dir.exists():
        raise FileNotFoundError("No results directory found. Run individual metrics first.")
    
    # Look for existing stepwise runs
    stepwise_runs = list(results_dir.glob("*_stepwise"))
    if not stepwise_runs:
        raise FileNotFoundError("No stepwise runs found. Run individual metrics first.")
    
    # Use the most recent one
    latest = max(stepwise_runs, key=lambda p: p.stat().st_mtime)
    return latest


def load_metric_summary(run_dir: Path, metric: str) -> Optional[Dict]:
    """Load summary for a specific metric"""
    summary_path = run_dir / metric / "summary.json"
    if not summary_path.exists():
        return None
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def calculate_clmpi(metric_scores: Dict[str, float], weights: Dict[str, float]) -> Dict[str, float]:
    """Calculate CLMPI score from individual metric scores and weights"""
    
    # Validate weights sum to 1.0
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
    
    # Calculate weighted CLMPI score
    clmpi_01 = sum(weights[metric] * score for metric, score in metric_scores.items())
    clmpi_100 = clmpi_01 * 100
    
    return {
        "clmpi_01": clmpi_01,
        "clmpi_100": clmpi_100,
        "component_scores": metric_scores.copy(),
        "weights_used": weights.copy()
    }


def combine_clmpi_scores(model_name: str, verbose: bool = False) -> dict:
    """Combine individual metric results into CLMPI score"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load model config for weights
    model_config = load_model_config()
    weights = model_config.get('evaluation_weights', {})
    
    if verbose:
        logger.info(f"Loaded weights: {weights}")
    
    # Find latest run directory
    run_dir = find_latest_run_directory()
    
    if verbose:
        logger.info(f"Using run directory: {run_dir}")
    
    # Load individual metric summaries
    metrics = {
        "accuracy": "accuracy",
        "contextual_understanding": "context", 
        "coherence": "coherence",
        "fluency": "fluency",
        "performance_efficiency": "efficiency"
    }
    
    metric_scores = {}
    metric_summaries = {}
    missing_metrics = []
    
    for weight_key, folder_name in metrics.items():
        summary = load_metric_summary(run_dir, folder_name)
        if summary is None:
            missing_metrics.append(folder_name)
            continue
        
        metric_summaries[weight_key] = summary
        
        # Extract score based on metric type
        if folder_name == "accuracy":
            metric_scores[weight_key] = summary.get("f1_score", 0.0)
        elif folder_name == "context":
            metric_scores[weight_key] = summary.get("combined_score", 0.0)
        elif folder_name == "coherence":
            metric_scores[weight_key] = summary.get("coherence_score", 0.0)
        elif folder_name == "fluency":
            metric_scores[weight_key] = summary.get("fluency_score", 0.0)
        elif folder_name == "efficiency":
            metric_scores[weight_key] = summary.get("avg_efficiency", 0.0)
    
    if missing_metrics:
        logger.warning(f"Missing metrics: {missing_metrics}")
        if verbose:
            logger.info(f"Available metrics: {list(metric_scores.keys())}")
    
    # Validate we have all required metrics
    required_metrics = set(weights.keys())
    available_metrics = set(metric_scores.keys())
    
    if required_metrics != available_metrics:
        missing = required_metrics - available_metrics
        if missing:
            raise ValueError(f"Missing required metrics: {missing}. Run these first: {[metrics.get(m, m) for m in missing]}")
    
    # Calculate CLMPI
    clmpi_result = calculate_clmpi(metric_scores, weights)
    
    # Create combined summary
    combined_summary = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_directory": str(run_dir),
        "clmpi_scores": clmpi_result,
        "individual_metrics": metric_summaries,
        "generation_info": {
            "stepwise_evaluation": True,
            "combination_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Save combined results
    clmpi_summary_path = run_dir / "clmpi_summary.json"
    with open(clmpi_summary_path, "w") as f:
        json.dump(combined_summary, f, indent=2)
    
    # Print results
    print(f"[CLMPI] {model_name} combined_score={clmpi_result['clmpi_01']:.3f} (0-1) / {clmpi_result['clmpi_100']:.1f} (0-100)")
    
    if verbose:
        logger.info(f"Combined results saved to: {clmpi_summary_path}")
        logger.info("Component scores:")
        for metric, score in clmpi_result['component_scores'].items():
            logger.info(f"  {metric}: {score:.3f}")
        logger.info(f"Final CLMPI: {clmpi_result['clmpi_01']:.3f}")
    
    return {
        "clmpi_01": clmpi_result['clmpi_01'],
        "clmpi_100": clmpi_result['clmpi_100'],
        "run_dir": str(run_dir),
        "summary_path": str(clmpi_summary_path)
    }


def print_detailed_summary(run_dir: Path, model_name: str):
    """Print a detailed summary of all metrics and final CLMPI"""
    print("\n" + "="*60)
    print("STEPWISE CLMPI EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Run Directory: {run_dir}")
    print("")
    
    # Load CLMPI summary
    clmpi_path = run_dir / "clmpi_summary.json"
    if clmpi_path.exists():
        with open(clmpi_path, 'r') as f:
            clmpi_data = json.load(f)
        
        scores = clmpi_data['clmpi_scores']['component_scores']
        weights = clmpi_data['clmpi_scores']['weights_used']
        
        print("INDIVIDUAL METRICS:")
        print("-" * 40)
        for metric, score in scores.items():
            weight = weights.get(metric, 0.0)
            weighted = score * weight
            print(f"{metric.replace('_', ' ').title():25s}: {score:.3f} (weight: {weight:.2f}, contrib: {weighted:.3f})")
        
        print("")
        print("FINAL CLMPI SCORE:")
        print("-" * 40)
        print(f"CLMPI (0-1):  {clmpi_data['clmpi_scores']['clmpi_01']:.3f}")
        print(f"CLMPI (0-100): {clmpi_data['clmpi_scores']['clmpi_100']:.1f}")
    
    print("="*60)


def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Combine individual CLMPI metric results into final score',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/combine_clmpi.py --model phi3:mini
  python scripts/combine_clmpi.py --model phi3:mini --verbose
  python scripts/combine_clmpi.py --model phi3:mini --detailed
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (for labeling, must match individual metric runs)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--detailed', action='store_true',
                       help='Print detailed summary at the end')
    
    args = parser.parse_args()
    
    try:
        result = combine_clmpi_scores(args.model, args.verbose)
        
        if args.detailed:
            run_dir = Path(result['run_dir'])
            print_detailed_summary(run_dir, args.model)
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
