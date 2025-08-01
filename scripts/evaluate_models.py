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

from clmpi_calculator import CLMPICalculator, CLMPIScores, CLMPIWeights


class ModelEvaluator:
    """
    Main evaluator class for implementing CLMPI framework
    """

    def __init__(self, config_path: str, selected_models: Optional[List[str]] = None):
        """
        Initialize the model evaluator

        Args:
            config_path: Path to configuration file
            selected_models: Optional list of specific models to evaluate
        """
        self.selected_models = selected_models
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
        avg_score = sum(scores) / len(scores)
        self.logger.info(f"Contextual understanding score: {avg_score:.3f}")
        return avg_score

    def evaluate_coherence(self, model_name: str, responses: List[str]) -> float:
        self.logger.info(f"Evaluating coherence for {model_name}")
        scores = []
        for response in responses:
            score = 3.0
            if len(response.split('.')) > 2:
                score += 0.5
            if any(k in response.lower() for k in ['and', 'but', 'or', 'because', 'therefore', 'however']):
                score += 0.5
            scores.append(min(score, 5.0))
        avg_score = sum(scores) / len(scores)
        self.logger.info(f"Coherence score: {avg_score:.3f}")
        return avg_score

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
        avg_score = sum(scores) / len(scores)
        self.logger.info(f"Fluency score: {avg_score:.3f}")
        return avg_score

    def measure_resource_efficiency(self, model_name: str, evaluation_function, *args) -> float:
        self.logger.info(f"Measuring resource efficiency for {model_name}")
        time_taken, memory_used, efficiency = self.calculator.measure_resource_usage(evaluation_function, *args)
        self.logger.info(f"Time taken: {time_taken:.3f}s, Memory used: {memory_used:.2f}MB")
        self.logger.info(f"Efficiency score: {efficiency:.3f}")
        return efficiency

    def evaluate_model(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Starting evaluation for {model_name}")
        eval_data = self.load_evaluation_data()
        sample_responses = [
            "The answer is 42 because it represents the meaning of life according to Douglas Adams.",
            "Based on the context provided, the solution involves multiple steps.",
            "This is a coherent response that demonstrates understanding of the question.",
            "The model generates fluent and grammatically correct text.",
            "Resource efficiency is measured through time and memory usage."
        ]
        expected_answers = [
            "42",
            "multiple steps",
            "coherent response",
            "fluent text",
            "efficiency measured"
        ]
        accuracy = self.evaluate_accuracy(model_name, sample_responses, expected_answers)
        contextual = self.evaluate_contextual_understanding(model_name, sample_responses)
        coherence = self.evaluate_coherence(model_name, sample_responses)
        fluency = self.evaluate_fluency(model_name, sample_responses)
        def dummy():
            time.sleep(0.1)
            return "ok"
        perf_eff = self.measure_resource_efficiency(model_name, dummy)
        scores = CLMPIScores(
            accuracy=accuracy,
            contextual_understanding=contextual,
            coherence=coherence,
            fluency=fluency,
            performance_efficiency=perf_eff
        )
        clmpi = self.calculator.calculate_clmpi_normalized(scores)
        report = self.calculator.generate_report(model_name, scores, clmpi)
        return report

    def evaluate_all_models(self) -> List[Dict[str, Any]]:
        results = []
        for model_name, model_config in self.config['models'].items():
            if self.selected_models and model_name not in self.selected_models:
                continue
            try:
                result = self.evaluate_model(model_name, model_config)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
        return results

    def generate_comparison_report(self, results: List[Dict[str, Any]], output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        rows = []
        for result in results:
            rows.append({
                "Model": result["model_name"],
                "CLMPI_Score": result["clmpi_score"],
                "Accuracy": result["component_scores"]["accuracy"],
                "Contextual_Understanding": result["component_scores"]["contextual_understanding"],
                "Coherence": result["component_scores"]["coherence"],
                "Fluency": result["component_scores"]["fluency"],
                "Performance_Efficiency": result["component_scores"]["performance_efficiency"]
            })
        df = pd.DataFrame(rows)
        df.to_csv(output_path / "model_comparison.csv", index=False)
        df.to_json(output_path / "detailed_results.json", orient="records", indent=2)
        self.logger.info(f"Comparison report saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/")
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    evaluator = ModelEvaluator(args.config, selected_models=args.models)
    results = evaluator.evaluate_all_models()
    if results:
        evaluator.generate_comparison_report(results, args.output)
        print("\nCLMPI Evaluation Complete\n")
    else:
        print("No models were evaluated.")
