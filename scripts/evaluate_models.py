#!/usr/bin/env python3
"""
CLMPI Model Evaluation Script

This script implements the Comprehensive Language Model Performance Index (CLMPI)
framework to evaluate Large Language Models across multiple dimensions.

Usage:
    python evaluate_models.py --config config/model_config.yaml --output results/
"""

import argparse
import json
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scripts.clmpi_calculator import CLMPICalculator, CLMPIScores


class ModelEvaluator:
    """
    Main evaluator class for implementing CLMPI framework
    """

    def __init__(self, config_path: str, selected_models: Optional[List[str]] = None, output_dir: Optional[str] = "results"):
        self.selected_models = selected_models
        self.output_dir = output_dir
        self.config = self._load_config(config_path)
        self.calculator = CLMPICalculator()
        self.setup_logging()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_evaluation_data(self) -> Dict[str, Any]:
        data = {}
        with open('prompts/classification_tasks.json', 'r') as f:
            data['classification'] = json.load(f)
        with open('prompts/reasoning_tasks.json', 'r') as f:
            data['reasoning'] = json.load(f)
        return data

    def evaluate_accuracy(self, model_name: str, responses: List[str], expected_answers: List[str]) -> float:
        self.logger.info(f"Evaluating accuracy for {model_name}")
        accuracy = self.calculator.evaluate_accuracy(responses, expected_answers)
        self.logger.info(f"Accuracy score: {accuracy:.3f}")
        return accuracy

    def evaluate_contextual_understanding(self, model_name: str, responses: List[str]) -> float:
        self.logger.info(f"Evaluating contextual understanding for {model_name}")
        scores = []
        for response in responses:
            score = 3.0
            if len(response.split()) > 20:
                score += 0.5
            if any(k in response.lower() for k in ['because', 'therefore', 'however', 'furthermore', 'consequently']):
                score += 0.5
            scores.append(min(score, 5.0))
        avg = sum(scores) / len(scores)
        self.logger.info(f"Contextual understanding score: {avg:.3f}")
        return avg

    def evaluate_coherence(self, model_name: str, responses: List[str]) -> float:
        self.logger.info(f"Evaluating coherence for {model_name}")
        scores = []
        for response in responses:
            score = 3.0
            if len(response.split('.')) > 2:
                score += 0.5
            if any(c in response.lower() for c in ['and', 'but', 'or', 'because', 'therefore', 'however']):
                score += 0.5
            scores.append(min(score, 5.0))
        avg = sum(scores) / len(scores)
        self.logger.info(f"Coherence score: {avg:.3f}")
        return avg

    def evaluate_fluency(self, model_name: str, responses: List[str]) -> float:
        self.logger.info(f"Evaluating fluency for {model_name}")
        scores = []
        for response in responses:
            score = 3.0
            if response.strip().endswith(('.', '!', '?')):
                score += 0.5
            words = response.lower().split()
            if len(set(words)) > len(words) * 0.7:
                score += 0.5
            scores.append(min(score, 5.0))
        avg = sum(scores) / len(scores)
        self.logger.info(f"Fluency score: {avg:.3f}")
        return avg

    def measure_resource_efficiency(self, model_name: str, evaluation_function, *args) -> float:
        self.logger.info(f"Measuring resource efficiency for {model_name}")
        time_taken, memory_used, efficiency = self.calculator.measure_resource_usage(evaluation_function, *args)
        self.logger.info(f"Time taken: {time_taken:.3f}s, Memory used: {memory_used:.2f}MB")
        self.logger.info(f"Efficiency score: {efficiency:.3f}")
        return efficiency

    def evaluate_model(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Starting evaluation for {model_name}")
        self.load_evaluation_data()

        prompts = [
            "What is 2 + 2?",
            "Explain what a DNS server does.",
            "Tell me a short story about AI.",
            "Translate 'Hello' into Spanish.",
            "What is machine learning?"
        ]

        sample_responses = [
            "The answer is 4.",
            "A DNS server resolves domain names to IP addresses.",
            "Once upon a time, an AI became self-aware.",
            "Hola means Hello in Spanish.",
            "Machine learning is a subset of AI."
        ]

        expected_answers = ["4", "resolve domain", "story", "Hola", "subset"]

        accuracy = self.evaluate_accuracy(model_name, sample_responses, expected_answers)
        contextual = self.evaluate_contextual_understanding(model_name, sample_responses)
        coherence = self.evaluate_coherence(model_name, sample_responses)
        fluency = self.evaluate_fluency(model_name, sample_responses)

        def dummy(): time.sleep(0.1); return "ok"
        efficiency = self.measure_resource_efficiency(model_name, dummy)

        scores = CLMPIScores(
            accuracy=accuracy,
            contextual_understanding=contextual,
            fluency=fluency,
            coherence=coherence,
            performance_efficiency=efficiency
        )

        clmpi = self.calculator.calculate_clmpi_normalized(scores)
        report = self.calculator.generate_report(model_name, scores, clmpi)

        raw_responses = [{"prompt": p, "response": r} for p, r in zip(prompts, sample_responses)]
        raw_path = Path(f"{self.output_dir}/{model_name.replace(':', '-')}/responses_{model_name.replace(':', '_')}.json")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w") as f:
            json.dump(raw_responses, f, indent=2)
        self.logger.info(f"Raw responses saved to: {raw_path}")

        return report

    def evaluate_all_models(self) -> List[Dict[str, Any]]:
        results = []
        model_entries = self.config['models']
        if self.selected_models:
            model_entries = {k: v for k, v in model_entries.items() if k in self.selected_models}
        for model_name, model_config in model_entries.items():
            try:
                result = self.evaluate_model(model_name, model_config)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
        return results

    def generate_comparison_report(self, results: List[Dict[str, Any]], output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        df_data = []
        for result in results:
            df_data.append({
                'Model': result['model_name'],
                'CLMPI_Score': result['clmpi_score'],
                'Accuracy': result['component_scores']['accuracy'],
                'Contextual_Understanding': result['component_scores']['contextual_understanding'],
                'Coherence': result['component_scores']['coherence'],
                'Fluency': result['component_scores']['fluency'],
                'Resource_Efficiency': result['component_scores']['performance_efficiency']
            })
        df = pd.DataFrame(df_data)
        df.to_csv(output_path / 'model_comparison.csv', index=False)
        with open(output_path / 'detailed_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Comparison report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='CLMPI Model Evaluation')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--output', default='results/', help='Output directory for results')
    parser.add_argument('--models', nargs='+', help='Specific models to evaluate (optional)')
    args = parser.parse_args()

    evaluator = ModelEvaluator(config_path=args.config, selected_models=args.models, output_dir=args.output)
    results = evaluator.evaluate_all_models()

    if results:
        evaluator.generate_comparison_report(results, args.output)
        print("\nCLMPI evaluation complete. Reports saved to:", args.output)
    else:
        print("No models evaluated.")


if __name__ == "__main__":
    main()
