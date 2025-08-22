#!/usr/bin/env python3
"""
Unit tests for enhanced CLMPI calculator
"""

import sys
import os
from pathlib import Path
import pytest
import numpy as np

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from enhanced_clmpi_calculator import (
    EnhancedCLMPICalculator, 
    AccuracyResult, 
    ContextualResult, 
    CoherenceResult, 
    FluencyResult,
    EfficiencyResult
)


class TestEnhancedCLMPICalculator:
    """Test enhanced CLMPI calculator functionality"""
    
    def test_weights_validation(self):
        """Test that weights must sum to 1.0"""
        # Valid weights
        valid_weights = {
            'accuracy': 0.25,
            'contextual_understanding': 0.20,
            'coherence': 0.20,
            'fluency': 0.20,
            'performance_efficiency': 0.15
        }
        calculator = EnhancedCLMPICalculator(valid_weights)
        assert sum(calculator.weights.values()) == 1.0
        
        # Invalid weights
        invalid_weights = {
            'accuracy': 0.5,
            'contextual_understanding': 0.5,
            'coherence': 0.5,
            'fluency': 0.5,
            'performance_efficiency': 0.5
        }
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            EnhancedCLMPICalculator(invalid_weights)
    
    def test_accuracy_evaluation(self):
        """Test accuracy evaluation with exact match and F1"""
        calculator = EnhancedCLMPICalculator()
        
        responses = ["Paris", "Au", "42"]
        gold_answers = ["Paris", "Au", "42"]
        acceptable_answers = [["Paris", "paris"], ["Au", "au"], ["42"]]
        
        result = calculator.evaluate_accuracy(responses, gold_answers, acceptable_answers)
        
        assert result.exact_match == 1.0
        assert result.f1_score == 1.0
        assert len(result.detailed_scores) == 3
        assert all(score == 1.0 for score in result.detailed_scores)
    
    def test_contextual_understanding_evaluation(self):
        """Test contextual understanding evaluation"""
        calculator = EnhancedCLMPICalculator()
        
        responses = ["8", "John is a doctor"]
        contexts = ["Alice has 5 apples. Bob gives her 3 more.", "John is a doctor who works at City Hospital."]
        gold_answers = ["8", "John is a doctor"]
        
        result = calculator.evaluate_contextual_understanding(responses, contexts, gold_answers)
        
        assert result.combined_score > 0.5  # Should be reasonably high
        assert result.context_similarity > 0.0
        assert result.f1_score > 0.0
    
    def test_coherence_evaluation(self):
        """Test coherence evaluation with sentence similarity"""
        calculator = EnhancedCLMPICalculator()
        
        responses = [
            "The cat walked to the pond. It looked at the water. Then it jumped in and swam.",
            "Renewable energy is important. Solar power reduces emissions. Wind energy is clean."
        ]
        
        result = calculator.evaluate_coherence(responses)
        
        assert 0.0 <= result.coherence_score <= 1.0
        assert len(result.detailed_scores) == 2
        assert all(0.0 <= score <= 1.0 for score in result.detailed_scores)
    
    def test_fluency_evaluation(self):
        """Test fluency evaluation with grammar and perplexity"""
        calculator = EnhancedCLMPICalculator()
        
        responses = [
            "The sunset was beautiful. The sky turned orange and red. Birds flew overhead.",
            "This is a test sentence with good grammar and proper punctuation."
        ]
        
        result = calculator.evaluate_fluency(responses)
        
        assert 0.0 <= result.fluency_score <= 1.0
        assert 0.0 <= result.grammar_score <= 1.0
        assert 0.0 <= result.perplexity_score <= 1.0
        assert len(result.detailed_scores) == 2
    
    def test_efficiency_normalization(self):
        """Test efficiency score normalization"""
        calculator = EnhancedCLMPICalculator()
        
        # Create mock efficiency results
        efficiency_results = [
            EfficiencyResult(latency_seconds=1.0, cpu_usage_percent=10, memory_used_mb=100, raw_efficiency=1.0, normalized_efficiency=0.0),
            EfficiencyResult(latency_seconds=2.0, cpu_usage_percent=20, memory_used_mb=200, raw_efficiency=0.5, normalized_efficiency=0.0),
            EfficiencyResult(latency_seconds=3.0, cpu_usage_percent=30, memory_used_mb=300, raw_efficiency=0.33, normalized_efficiency=0.0)
        ]
        
        normalized = calculator.normalize_efficiency_scores(efficiency_results)
        
        assert len(normalized) == 3
        assert normalized[0] == 1.0  # Best efficiency
        assert normalized[2] == 0.0  # Worst efficiency
        assert 0.0 <= normalized[1] <= 1.0  # Middle efficiency
    
    def test_clmpi_calculation(self):
        """Test full CLMPI calculation"""
        calculator = EnhancedCLMPICalculator()
        
        # Create mock results
        accuracy_result = AccuracyResult(
            exact_match=1.0, f1_score=0.9, detailed_scores=[0.9], 
            responses=["test"], gold_answers=["test"]
        )
        
        contextual_result = ContextualResult(
            exact_match=1.0, f1_score=0.8, context_similarity=0.7, combined_score=0.77,
            responses=["test"], contexts=["test"], gold_answers=["test"]
        )
        
        coherence_result = CoherenceResult(
            sentence_similarity=0.6, repetition_penalty=0.1, coherence_score=0.54,
            responses=["test"], detailed_scores=[0.54]
        )
        
        fluency_result = FluencyResult(
            grammar_score=0.9, perplexity_score=0.8, fluency_score=0.86,
            responses=["test"], detailed_scores=[0.86]
        )
        
        efficiency_score = 0.7
        
        clmpi_results = calculator.calculate_clmpi(
            accuracy_result, contextual_result, coherence_result, fluency_result, efficiency_score
        )
        
        assert 0.0 <= clmpi_results['clmpi_01'] <= 1.0
        assert 0.0 <= clmpi_results['clmpi_100'] <= 100.0
        assert clmpi_results['clmpi_100'] == clmpi_results['clmpi_01'] * 100
        
        component_scores = clmpi_results['component_scores']
        assert all(0.0 <= score <= 1.0 for score in component_scores.values())
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        calculator = EnhancedCLMPICalculator()
        
        # Empty responses
        with pytest.raises(ValueError, match="Number of responses must match"):
            calculator.evaluate_accuracy([], ["test"], [])
        
        # Empty efficiency results
        normalized = calculator.normalize_efficiency_scores([])
        assert normalized == []
        
        # All equal efficiency scores
        equal_results = [
            EfficiencyResult(latency_seconds=1.0, cpu_usage_percent=10, memory_used_mb=100, raw_efficiency=1.0, normalized_efficiency=0.0),
            EfficiencyResult(latency_seconds=1.0, cpu_usage_percent=10, memory_used_mb=100, raw_efficiency=1.0, normalized_efficiency=0.0)
        ]
        normalized = calculator.normalize_efficiency_scores(equal_results)
        assert normalized == [0.5, 0.5]  # Neutral scores when all equal


if __name__ == "__main__":
    pytest.main([__file__])
