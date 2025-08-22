# DEPRECATED: Use scripts/enhanced_clmpi_calculator.py instead. Not maintained.
"""
DEPRECATED: Use scripts/enhanced_clmpi_calculator.py

This script is deprecated and will be removed in a future version.
Use the enhanced calculator for better scoring methods and reproducibility.

CLMPI Calculator - Core scoring engine for Comprehensive Language Model Performance Index

This module implements the CLMPI formula with proper normalization as described in:
"Benchmarking Large Language Models with a Unified Performance Ranking Metric"
by Maikel Leon (University of Miami, 2024)

Example:
    calculator = CLMPICalculator()
    scores = CLMPIScores(
        accuracy=0.8,
        contextual_understanding=4.0,
        coherence=3.5,
        fluency=4.2,
        performance_efficiency=0.15
    )
    clmpi_score = calculator.calculate_clmpi(scores)
    print(f"CLMPI Score: {clmpi_score:.2f}")
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
    """Raw scores for each CLMPI dimension"""
    accuracy: float  # [0,1] - correct / total
    contextual_understanding: float  # [0,5] - will be normalized to [0,1]
    coherence: float  # [0,5] - will be normalized to [0,1]
    fluency: float  # [0,5] - will be normalized to [0,1]
    performance_efficiency: float  # raw efficiency score - will be normalized


@dataclass
class CLMPIWeights:
    """Evaluation weights (must sum to 1.0)"""
    accuracy: float = 0.25
    contextual_understanding: float = 0.20
    coherence: float = 0.20
    fluency: float = 0.20
    performance_efficiency: float = 0.15


class CLMPICalculator:
    """
    CLMPI scoring engine with proper normalization
    
    Implements the CLMPI formula:
    CLMPI = Σ(weight_i * normalized_score_i)
    
    Where normalized scores are in [0,1] range.
    """
    
    def __init__(self, weights: Optional[CLMPIWeights] = None):
        self.weights = weights or CLMPIWeights()
        self._validate_weights()

    def _validate_weights(self):
        """Ensure weights sum to 1.0"""
        total_weight = sum([
            self.weights.accuracy,
            self.weights.contextual_understanding,
            self.weights.coherence,
            self.weights.fluency,
            self.weights.performance_efficiency
        ])
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def normalize_quality_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize quality scores from [0,5] to [0,1]
        
        Args:
            scores: List of raw scores in [0,5] range
            
        Returns:
            List of normalized scores in [0,1] range
        """
        return [score / 5.0 for score in scores]

    def normalize_efficiency_scores(self, efficiency_scores: List[float]) -> List[float]:
        """
        Normalize efficiency scores using min-max normalization
        
        Args:
            efficiency_scores: List of raw efficiency scores
            
        Returns:
            List of normalized scores in [0,1] range
        """
        if not efficiency_scores:
            return []
        
        min_score = min(efficiency_scores)
        max_score = max(efficiency_scores)
        range_score = (max_score - min_score) or 1e-9  # Avoid division by zero
        
        return [(score - min_score) / range_score for score in efficiency_scores]

    def calculate_raw_efficiency(self, time_taken: float, memory_used_mb: float) -> float:
        """
        Calculate raw efficiency score
        
        Args:
            time_taken: Response time in seconds
            memory_used_mb: Memory usage in MB
            
        Returns:
            Raw efficiency score (higher is better)
        """
        if time_taken <= 0 or memory_used_mb <= 0:
            raise ValueError("Time and memory values must be positive")
        return 1.0 / (time_taken + memory_used_mb / 100.0)

    def calculate_clmpi(self, scores: CLMPIScores) -> float:
        """
        Calculate CLMPI score with proper normalization
        
        Args:
            scores: Raw scores for each dimension
            
        Returns:
            CLMPI score in [0,1] range
        """
        # Normalize quality scores from [0,5] to [0,1]
        norm_contextual = scores.contextual_understanding / 5.0
        norm_coherence = scores.coherence / 5.0
        norm_fluency = scores.fluency / 5.0
        
        # Efficiency is already normalized
        norm_efficiency = scores.performance_efficiency
        
        # Calculate weighted sum
        clmpi = (
            self.weights.accuracy * scores.accuracy +
            self.weights.contextual_understanding * norm_contextual +
            self.weights.coherence * norm_coherence +
            self.weights.fluency * norm_fluency +
            self.weights.performance_efficiency * norm_efficiency
        )
        
        return clmpi

    def calculate_clmpi_100(self, scores: CLMPIScores) -> float:
        """
        Calculate CLMPI score on 0-100 scale
        
        Args:
            scores: Raw scores for each dimension
            
        Returns:
            CLMPI score in [0,100] range
        """
        return self.calculate_clmpi(scores) * 100.0

    def evaluate_accuracy(self, model_responses: List[str], correct_answers: List[str]) -> float:
        """
        Calculate accuracy as correct / total
        
        Args:
            model_responses: List of model responses
            correct_answers: List of correct answers
            
        Returns:
            Accuracy score in [0,1] range
        """
        if len(model_responses) != len(correct_answers):
            raise ValueError("Number of responses must match number of answers")
        
        correct_count = sum(
            1 for r, a in zip(model_responses, correct_answers)
            if r.strip().lower() == a.strip().lower()
        )
        return correct_count / len(model_responses)

    def evaluate_quality_dimension(self, responses: List[str], scores: List[float], dimension: str) -> float:
        """
        Evaluate a quality dimension (contextual understanding, coherence, fluency)
        
        Args:
            responses: List of model responses
            scores: List of scores in [0,5] range
            dimension: Name of dimension for error messages
            
        Returns:
            Average score in [0,5] range
        """
        if len(responses) != len(scores):
            raise ValueError(f"Number of responses must match number of {dimension} scores")
        
        if not all(0 <= s <= 5 for s in scores):
            raise ValueError(f"All {dimension} scores must be between 0 and 5")
        
        return np.mean(scores)

    def measure_resource_usage(self, func, *args, **kwargs) -> Tuple[float, float, float]:
        """
        Measure resource usage of a function
        
        Args:
            func: Function to measure
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Tuple of (time_taken, memory_used_mb, efficiency_score)
        """
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        result = func(*args, **kwargs)
        time_taken = time.time() - start_time
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory
        
        efficiency = self.calculate_raw_efficiency(time_taken, memory_used)
        
        return time_taken, memory_used, efficiency

    def save_results(self, results: Dict, output_path: Path):
        """Save evaluation results to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)


def example_usage():
    """Example usage of CLMPI calculator"""
    calculator = CLMPICalculator()
    
    # Example scores for a model
    scores = CLMPIScores(
        accuracy=0.8,  # 80% correct answers
        contextual_understanding=4.0,  # Good contextual understanding
        coherence=3.5,  # Decent coherence
        fluency=4.2,  # Good fluency
        performance_efficiency=0.15  # Raw efficiency score
    )
    
    # Calculate CLMPI scores
    clmpi_01 = calculator.calculate_clmpi(scores)
    clmpi_100 = calculator.calculate_clmpi_100(scores)
    
    print(f"CLMPI Score (0-1): {clmpi_01:.3f}")
    print(f"CLMPI Score (0-100): {clmpi_100:.1f}")
    
    # Save results
    results = {
        "model_name": "example_model",
        "clmpi_score_01": clmpi_01,
        "clmpi_score_100": clmpi_100,
        "component_scores": {
            "accuracy": scores.accuracy,
            "contextual_understanding": scores.contextual_understanding,
            "coherence": scores.coherence,
            "fluency": scores.fluency,
            "performance_efficiency": scores.performance_efficiency
        }
    }
    
    calculator.save_results(results, Path("example_results.json"))
    print("Results saved to example_results.json")


if __name__ == "__main__":
    example_usage()
