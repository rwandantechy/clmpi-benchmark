"""
Unit tests for CLMPI calculator
"""

import pytest
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from clmpi_calculator import CLMPICalculator, CLMPIScores, CLMPIWeights
from utils import sanitize_filename


class TestCLMPICalculator:
    """Test CLMPI calculator functionality"""
    
    def test_weights_validation(self):
        """Test that weights must sum to 1.0"""
        # Valid weights
        valid_weights = CLMPIWeights(
            accuracy=0.25,
            contextual_understanding=0.20,
            coherence=0.20,
            fluency=0.20,
            performance_efficiency=0.15
        )
        calculator = CLMPICalculator(valid_weights)
        assert calculator.weights == valid_weights
        
        # Invalid weights (don't sum to 1.0)
        invalid_weights = CLMPIWeights(
            accuracy=0.5,
            contextual_understanding=0.5,
            coherence=0.5,
            fluency=0.5,
            performance_efficiency=0.5
        )
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            CLMPICalculator(invalid_weights)
    
    def test_normalization(self):
        """Test score normalization"""
        calculator = CLMPICalculator()
        
        # Test quality score normalization
        raw_scores = [3.0, 4.0, 5.0]
        normalized = calculator.normalize_quality_scores(raw_scores)
        assert normalized == [0.6, 0.8, 1.0]
        
        # Test efficiency score normalization
        eff_scores = [0.1, 0.2, 0.3]
        normalized_eff = calculator.normalize_efficiency_scores(eff_scores)
        assert len(normalized_eff) == 3
        assert normalized_eff[0] == 0.0  # min becomes 0
        assert normalized_eff[2] == 1.0  # max becomes 1
    
    def test_clmpi_calculation(self):
        """Test CLMPI score calculation"""
        calculator = CLMPICalculator()
        
        scores = CLMPIScores(
            accuracy=0.8,
            contextual_understanding=4.0,
            coherence=3.5,
            fluency=4.2,
            performance_efficiency=0.15
        )
        
        clmpi_01 = calculator.calculate_clmpi(scores)
        clmpi_100 = calculator.calculate_clmpi_100(scores)
        
        assert 0 <= clmpi_01 <= 1
        assert 0 <= clmpi_100 <= 100
        assert clmpi_100 == clmpi_01 * 100
    
    def test_accuracy_evaluation(self):
        """Test accuracy calculation"""
        calculator = CLMPICalculator()
        
        responses = ["4", "positive", "finance"]
        answers = ["4", "positive", "finance"]
        accuracy = calculator.evaluate_accuracy(responses, answers)
        assert accuracy == 1.0
        
        responses = ["4", "negative", "finance"]
        answers = ["4", "positive", "finance"]
        accuracy = calculator.evaluate_accuracy(responses, answers)
        assert accuracy == 2/3


class TestUtils:
    """Test utility functions"""
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        # Test colon replacement
        assert sanitize_filename("phi3:mini") == "phi3-mini"
        
        # Test space replacement
        assert sanitize_filename("llama2 7b chat") == "llama2-7b-chat"
        
        # Test multiple special characters
        assert sanitize_filename("model@name#123") == "model-name-123"
        
        # Test leading/trailing hyphens
        assert sanitize_filename("-model-name-") == "model-name"
        
        # Test empty string
        assert sanitize_filename("") == "unnamed"
        
        # Test already clean name
        assert sanitize_filename("clean_name") == "clean_name"
