#!/usr/bin/env python3
"""
Master script to run full CLMPI benchmark pipeline:
1. Pull Ollama models (if needed)
2. Run batch generation on prompts
3. Evaluate CLMPI components
4. Save detailed and comparative reports
"""

import argparse
import os
from scripts.ollama_runner import OllamaRunner
from scripts.evaluate_models import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run full CLMPI benchmark pipeline")
    parser.add_argument('--config', required=True, help='Path to model_config.yaml')
    parser.add_argument('--output', default='results/', help='Output directory')
    parser.add_argument('--pull', action='store_true', help='Pull models before evaluation')
    parser.add_argument('--cleanup', action='store_true', help='Remove models after evaluation')
    parser.add_argument('--models', nargs='+', help='Evaluate only these model(s)')
    args = parser.parse_args()

    # Step 1: Pull models 
    runner = OllamaRunner()
    from yaml import safe_load
    with open(args.config, 'r') as f:
        model_config = safe_load(f)

    selected_models = args.models if args.models else list(model_config.get('models', {}).keys())

    if args.pull:
        for model_name in selected_models:
            runner.pull_model(model_name)

    # Step 2: Run evaluation
    evaluator = ModelEvaluator(args.config, selected_models=selected_models)
    results = evaluator.evaluate_all_models()

    # Step 3: Generate reports
    if results:
        evaluator.generate_comparison_report(results, args.output)
        print("\nBenchmark complete. Results saved to:", args.output)
    else:
        print("No models evaluated.")

    # Step 4: Cleanup models 
    if args.cleanup:
        for model_name in selected_models:
            runner.cleanup_model(model_name)


if __name__ == "__main__":
    main()
