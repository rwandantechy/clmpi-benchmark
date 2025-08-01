"""
CLMPI Calculator - Core engine for Comprehensive Language Model Performance Index

This module implements the CLMPI formula and evaluation logic as described in:
"Benchmarking Large Language Models with a Unified Performance Ranking Metric"
by Maikel Leon (University of Miami, 2024)

This is a practical implementation of the theoretical framework proposed in the research paper.
"""

import json
import time
import psutil
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CLMPIScores:
    accuracy: float  # 0-1 (percentage)
    contextual_understanding: float  # 0-5 scale
    fluency: float  # 0-5 scale
    coherence: float  # 0-5 scale
    performance_efficiency: float  # calculated efficiency score


@dataclass
class CLMPIWeights:
    accuracy: float = 0.20
    contextual_understanding: float = 0.20
    fluency: float = 0.20
    coherence: float = 0.20
    performance_efficiency: float = 0.20


class CLMPICalculator:
    def __init__(self, weights: Optional[CLMPIWeights] = None):
        self.weights = weights or CLMPIWeights()
        self._validate_weights()

    def _validate_weights(self):
        total_weight = sum([
            self.weights.accuracy,
            self.weights.contextual_understanding,
            self.weights.fluency,
            self.weights.coherence,
            self.weights.performance_efficiency
        ])
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def calculate_resource_efficiency(self, time_taken: float, memory_used_mb: float) -> float:
        if time_taken <= 0 or memory_used_mb <= 0:
            raise ValueError("Time and memory values must be positive")
        return 1 / (time_taken + memory_used_mb / 100)

    def calculate_clmpi(self, scores: CLMPIScores) -> float:
        return (
            self.weights.accuracy * scores.accuracy +
            self.weights.contextual_understanding * scores.contextual_understanding +
            self.weights.fluency * scores.fluency +
            self.weights.coherence * scores.coherence +
            self.weights.performance_efficiency * scores.performance_efficiency
        )

    def calculate_clmpi_normalized(self, scores: CLMPIScores) -> float:
        return self.calculate_clmpi(scores) * 25

    def evaluate_accuracy(self, model_responses: List[str], correct_answers: List[str]) -> float:
        if len(model_responses) != len(correct_answers):
            raise ValueError("Number of responses must match number of answers")
        correct_count = sum(
            1 for r, a in zip(model_responses, correct_answers)
            if r.strip().lower() == a.strip().lower()
        )
        return correct_count / len(model_responses)

    def evaluate_generic_score(self, responses: List[str], scores: List[float], label: str) -> float:
        if len(responses) != len(scores):
            raise ValueError(f"Number of responses must match number of {label} scores")
        if not all(0 <= s <= 5 for s in scores):
            raise ValueError(f"All {label} scores must be between 0 and 5")
        return np.mean(scores)

    def evaluate_contextual_understanding(self, responses: List[str], context_relevance_scores: List[float]) -> float:
        return self.evaluate_generic_score(responses, context_relevance_scores, "context relevance")

    def evaluate_coherence(self, responses: List[str], coherence_scores: List[float]) -> float:
        return self.evaluate_generic_score(responses, coherence_scores, "coherence")

    def evaluate_fluency(self, responses: List[str], fluency_scores: List[float]) -> float:
        return self.evaluate_generic_score(responses, fluency_scores, "fluency")

    def measure_resource_usage(self, func, *args, **kwargs) -> Tuple[float, float, float]:
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        _ = func(*args, **kwargs)
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = max(0.01, final_memory - initial_memory)
        time_taken = max(0.01, end_time - start_time)
        efficiency = self.calculate_resource_efficiency(time_taken, memory_used)
        return time_taken, memory_used, efficiency

    def generate_report(self, model_name: str, scores: CLMPIScores, clmpi_score: float) -> Dict:
        return {
            "model_name": model_name,
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "clmpi_score": clmpi_score,
            "component_scores": {
                "accuracy": scores.accuracy,
                "contextual_understanding": scores.contextual_understanding,
                "fluency": scores.fluency,
                "coherence": scores.coherence,
                "performance_efficiency": scores.performance_efficiency
            },
            "weights_used": vars(self.weights),
            "interpretation": self._interpret_score(clmpi_score)
        }

    def _interpret_score(self, clmpi_score: float) -> str:
        if clmpi_score >= 20:
            return "Excellent - Model demonstrates outstanding performance across all dimensions"
        elif clmpi_score >= 15:
            return "Good - Model shows strong performance with minor areas for improvement"
        elif clmpi_score >= 10:
            return "Fair - Model performs adequately but has significant room for improvement"
        elif clmpi_score >= 5:
            return "Poor - Model shows limited capabilities across multiple dimensions"
        else:
            return "Very Poor - Model requires substantial improvement across all dimensions"

    def save_report(self, report: Dict, output_path: str):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_file}")


def example_usage():
    calculator = CLMPICalculator()
    scores = CLMPIScores(
        accuracy=0.85,
        contextual_understanding=4.2,
        fluency=4.0,
        coherence=4.1,
        performance_efficiency=0.32
    )
    clmpi_score = calculator.calculate_clmpi_normalized(scores)
    report = calculator.generate_report("Example-LLM", scores, clmpi_score)
    calculator.save_report(report, "models/outputs/example_llm_clmpi_report.json")
    print(f"CLMPI Score: {clmpi_score:.2f}/25")
    print(f"Interpretation: {report['interpretation']}")


if __name__ == "__main__":
    example_usage()
