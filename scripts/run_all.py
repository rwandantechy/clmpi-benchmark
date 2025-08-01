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
import yaml
from scripts.ollama_runner import OllamaRunner
from scripts.evaluate_models import ModelEvaluator

def build_dynamic_config(base_config_path, selected_models):
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    dynamic_config = config.copy()
    dynamic_config['models'] = {}

    # Create temporary config entries with default structure
    for model_name in selected_models:
        dynamic_config['models'][model_name] = {
            'type': 'ollama',
            'ollama_name': model_name,
            'max_tokens': 1000,
            'temperature': 0.1,
            'evaluation_weights': {
                'accuracy': 0.25,
                'contextual_understanding': 0.20,
                'fluency_coherence': 0.20,
                'performance_efficiency': 0.35
            },
            'expected_performance': {
                'latency_ms': '< 3000',
                'memory_mb': '< 6000',
                'cpu_usage': '< 80%'
            }
        }
    return dynamic_config

def main():
    parser = argparse.ArgumentParser(description="Run full CLMPI benchmark pipeline")
    parser.add_argument('--config', required=True, help='Path to model_config.yaml')
    parser.add_argument('--output', default='results/', help='Output directory')
    parser.add_argument('--models', nargs='+', help='Model names to evaluate (Ollama names)')
    parser.add_argument('--pull', action='store_true', help='Pull models before evaluation')
    parser.add_argument('--cleanup', action='store_true', help='Remove models after evaluation')
    args = parser.parse_args()

    # If specific models are passed, override config
    if args.models:
        config_dict = build_dynamic_config(args.config, args.models)
    else:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)

    # Step 1: Pull models
    runner = OllamaRunner()
    if args.pull:
        for model_name in config_dict.get('models', {}).keys():
            runner.pull_model(model_name)

    # Step 2: Run evaluation
    evaluator = ModelEvaluator(config_path=args.config, selected_models=args.models)
    evaluator.config = config_dict  # override with dynamic config
    results = evaluator.evaluate_all_models()

    # Step 3: Generate reports
    if results:
        evaluator.generate_comparison_report(results, args.output)
        print("\nBenchmark complete. Results saved to:", args.output)
    else:
        print("No models evaluated.")

    # Step 4: Cleanup
    if args.cleanup:
        for model_name in config_dict.get('models', {}).keys():
            runner.cleanup_model(model_name)

if __name__ == "__main__":
    main()
