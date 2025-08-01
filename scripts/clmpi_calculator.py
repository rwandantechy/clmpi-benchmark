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
    """Data class to store individual CLMPI component scores"""
    accuracy: float  # 0-1 (percentage)
    contextual_understanding: float  # 0-5 scale
    coherence: float  # 0-5 scale
    fluency: float  # 0-5 scale
    resource_efficiency: float  # Calculated efficiency score


@dataclass
class CLMPIWeights:
    """Data class to store CLMPI component weights"""
    accuracy: float = 0.25
    contextual_understanding: float = 0.20
    coherence: float = 0.20
    fluency: float = 0.20
    resource_efficiency: float = 0.15


class CLMPICalculator:
    """
    Main calculator for the Comprehensive Language Model Performance Index
    
    Implements the CLMPI formula:
    CLMPI = (w1 × ACC) + (w2 × CON) + (w3 × COH) + (w4 × FLU) + (w5 × EFF)
    """
    
    def __init__(self, weights: Optional[CLMPIWeights] = None):
        """
        Initialize CLMPI calculator with custom or default weights
        
        Args:
            weights: Custom weights for CLMPI components. If None, uses default weights.
        """
        self.weights = weights or CLMPIWeights()
        self._validate_weights()
    
    def _validate_weights(self):
        """Validate that weights sum to 1.0"""
        total_weight = sum([
            self.weights.accuracy,
            self.weights.contextual_understanding,
            self.weights.coherence,
            self.weights.fluency,
            self.weights.resource_efficiency
        ])
        
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def calculate_resource_efficiency(self, time_taken: float, memory_used_mb: float) -> float:
        """
        Calculate resource efficiency score
        
        Formula: EFF = 1 / (Time Taken (seconds) + Memory Used (MB)/100)
        
        Args:
            time_taken: Time taken for response generation in seconds
            memory_used_mb: Memory used in MB
            
        Returns:
            Efficiency score (higher is better)
        """
        if time_taken <= 0 or memory_used_mb <= 0:
            raise ValueError("Time and memory values must be positive")
        
        efficiency = 1 / (time_taken + memory_used_mb / 100)
        return efficiency
    
    def calculate_clmpi(self, scores: CLMPIScores) -> float:
        """
        Calculate the overall CLMPI score
        
        Args:
            scores: Individual component scores
            
        Returns:
            Overall CLMPI score
        """
        clmpi = (
            self.weights.accuracy * scores.accuracy +
            self.weights.contextual_understanding * scores.contextual_understanding +
            self.weights.coherence * scores.coherence +
            self.weights.fluency * scores.fluency +
            self.weights.resource_efficiency * scores.resource_efficiency
        )
        
        return clmpi
    
    def calculate_clmpi_normalized(self, scores: CLMPIScores) -> float:
        """
        Calculate normalized CLMPI score (0-25 scale as in the paper)
        
        Args:
            scores: Individual component scores
            
        Returns:
            Normalized CLMPI score (0-25 scale)
        """
        base_clmpi = self.calculate_clmpi(scores)
        # Convert to 0-25 scale (assuming accuracy is 0-1, others are 0-5)
        normalized_clmpi = base_clmpi * 25
        return normalized_clmpi
    
    def evaluate_accuracy(self, model_responses: List[str], 
                         correct_answers: List[str]) -> float:
        """
        Evaluate accuracy by comparing model responses to correct answers
        
        Args:
            model_responses: List of model-generated responses
            correct_answers: List of correct/expected answers
            
        Returns:
            Accuracy score (0-1)
        """
        if len(model_responses) != len(correct_answers):
            raise ValueError("Number of responses must match number of answers")
        
        correct_count = 0
        total_count = len(model_responses)
        
        for response, answer in zip(model_responses, correct_answers):
            # Simple exact match - can be enhanced with semantic similarity
            if response.strip().lower() == answer.strip().lower():
                correct_count += 1
        
        accuracy = correct_count / total_count
        return accuracy
    
    def evaluate_contextual_understanding(self, 
                                        responses: List[str],
                                        context_relevance_scores: List[float]) -> float:
        """
        Evaluate contextual understanding based on relevance scores
        
        Args:
            responses: List of model responses
            context_relevance_scores: Human or automated relevance scores (0-5)
            
        Returns:
            Average contextual understanding score (0-5)
        """
        if len(responses) != len(context_relevance_scores):
            raise ValueError("Number of responses must match number of relevance scores")
        
        # Validate scores are in 0-5 range
        for score in context_relevance_scores:
            if not 0 <= score <= 5:
                raise ValueError("Relevance scores must be between 0 and 5")
        
        avg_score = np.mean(context_relevance_scores)
        return avg_score
    
    def evaluate_coherence(self, 
                          responses: List[str],
                          coherence_scores: List[float]) -> float:
        """
        Evaluate coherence based on logical flow scores
        
        Args:
            responses: List of model responses
            coherence_scores: Human or automated coherence scores (0-5)
            
        Returns:
            Average coherence score (0-5)
        """
        if len(responses) != len(coherence_scores):
            raise ValueError("Number of responses must match number of coherence scores")
        
        # Validate scores are in 0-5 range
        for score in coherence_scores:
            if not 0 <= score <= 5:
                raise ValueError("Coherence scores must be between 0 and 5")
        
        avg_score = np.mean(coherence_scores)
        return avg_score
    
    def evaluate_fluency(self, 
                        responses: List[str],
                        fluency_scores: List[float]) -> float:
        """
        Evaluate fluency based on linguistic quality scores
        
        Args:
            responses: List of model responses
            fluency_scores: Human or automated fluency scores (0-5)
            
        Returns:
            Average fluency score (0-5)
        """
        if len(responses) != len(fluency_scores):
            raise ValueError("Number of responses must match number of fluency scores")
        
        # Validate scores are in 0-5 range
        for score in fluency_scores:
            if not 0 <= score <= 5:
                raise ValueError("Fluency scores must be between 0 and 5")
        
        avg_score = np.mean(fluency_scores)
        return avg_score
    
    def measure_resource_usage(self, func, *args, **kwargs) -> Tuple[float, float, float]:
        """
        Measure resource usage of a function execution
        
        Args:
            func: Function to measure
            *args, **kwargs: Arguments for the function
            
        Returns:
            Tuple of (time_taken, memory_used_mb, efficiency_score)
        """
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        # Measure execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        memory_used = final_memory - initial_memory
        
        time_taken = end_time - start_time
        efficiency = self.calculate_resource_efficiency(time_taken, memory_used)
        
        return time_taken, memory_used, efficiency
    
    def generate_report(self, model_name: str, scores: CLMPIScores, 
                       clmpi_score: float) -> Dict:
        """
        Generate a comprehensive evaluation report
        
        Args:
            model_name: Name of the evaluated model
            scores: Individual component scores
            clmpi_score: Overall CLMPI score
            
        Returns:
            Dictionary containing the evaluation report
        """
        report = {
            "model_name": model_name,
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "clmpi_score": clmpi_score,
            "component_scores": {
                "accuracy": scores.accuracy,
                "contextual_understanding": scores.contextual_understanding,
                "coherence": scores.coherence,
                "fluency": scores.fluency,
                "resource_efficiency": scores.resource_efficiency
            },
            "weights_used": {
                "accuracy": self.weights.accuracy,
                "contextual_understanding": self.weights.contextual_understanding,
                "coherence": self.weights.coherence,
                "fluency": self.weights.fluency,
                "resource_efficiency": self.weights.resource_efficiency
            },
            "interpretation": self._interpret_score(clmpi_score)
        }
        
        return report
    
    def _interpret_score(self, clmpi_score: float) -> str:
        """
        Interpret CLMPI score and provide qualitative assessment
        
        Args:
            clmpi_score: CLMPI score (0-25 scale)
            
        Returns:
            Qualitative interpretation of the score
        """
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
        """
        Save evaluation report to JSON file
        
        Args:
            report: Evaluation report dictionary
            output_path: Path to save the report
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {output_file}")


def example_usage():
    """Example usage of the CLMPI calculator"""
    
    # Initialize calculator with default weights
    calculator = CLMPICalculator()
    
    # Example scores for a fictional LLM (as in the paper)
    scores = CLMPIScores(
        accuracy=0.85,  # 85% correct answers
        contextual_understanding=4.2,  # Good context integration
        coherence=4.0,  # Well-structured responses
        fluency=4.5,  # High linguistic quality
        resource_efficiency=0.32  # Calculated efficiency score
    )
    
    # Calculate CLMPI score
    clmpi_score = calculator.calculate_clmpi_normalized(scores)
    
    # Generate and save report
    report = calculator.generate_report("Example-LLM", scores, clmpi_score)
    calculator.save_report(report, "models/outputs/example_llm_clmpi_report.json")
    
    print(f"CLMPI Score: {clmpi_score:.2f}/25")
    print(f"Interpretation: {report['interpretation']}")


if __name__ == "__main__":
    example_usage() 