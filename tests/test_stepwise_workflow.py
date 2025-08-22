#!/usr/bin/env python3
"""
Test for the stepwise CLMPI evaluation workflow
"""

import subprocess
import sys
import tempfile
import json
import yaml
from pathlib import Path
import pytest


def test_stepwise_help_commands():
    """Test that all stepwise scripts show help correctly"""
    scripts = [
        "scripts/step_accuracy.py",
        "scripts/step_context.py", 
        "scripts/step_coherence.py",
        "scripts/step_fluency.py",
        "scripts/step_efficiency.py",
        "scripts/combine_clmpi.py"
    ]
    
    for script in scripts:
        result = subprocess.run([sys.executable, script, "--help"], 
                              capture_output=True, text=True)
        assert result.returncode == 0, f"{script} --help failed"
        assert "usage:" in result.stdout, f"{script} help output malformed"


def test_metric_configs_exist():
    """Test that all metric configuration files exist and are valid"""
    config_dir = Path("config/metrics")
    assert config_dir.exists(), "config/metrics directory not found"
    
    expected_configs = [
        "accuracy.yaml",
        "context.yaml", 
        "coherence.yaml",
        "fluency.yaml",
        "efficiency.yaml"
    ]
    
    for config_file in expected_configs:
        config_path = config_dir / config_file
        assert config_path.exists(), f"Config file not found: {config_file}"
        
        # Validate YAML structure
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "dataset" in config, f"Missing 'dataset' in {config_file}"
        assert "profile" in config, f"Missing 'profile' in {config_file}"
        assert config["profile"] in ["deterministic", "creative"], f"Invalid profile in {config_file}"
        
        # Verify dataset file exists
        dataset_path = Path(config["dataset"])
        assert dataset_path.exists(), f"Dataset not found: {config['dataset']} for {config_file}"


def test_generation_config_profiles():
    """Test that generation config contains required profiles"""
    config_path = Path("config/generation_config.yaml")
    assert config_path.exists(), "generation_config.yaml not found"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert "generation_profiles" in config, "Missing generation_profiles"
    profiles = config["generation_profiles"]
    
    assert "deterministic" in profiles, "Missing deterministic profile"
    assert "creative" in profiles, "Missing creative profile"
    
    # Validate profile structure
    for profile_name, profile in profiles.items():
        assert "temperature" in profile, f"Missing temperature in {profile_name}"
        assert "top_p" in profile, f"Missing top_p in {profile_name}"
        assert "max_tokens" in profile, f"Missing max_tokens in {profile_name}"


def test_model_config_weights():
    """Test that model config contains evaluation weights"""
    config_path = Path("config/model_config.yaml")
    assert config_path.exists(), "model_config.yaml not found"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert "evaluation_weights" in config, "Missing evaluation_weights"
    weights = config["evaluation_weights"]
    
    expected_weights = [
        "accuracy",
        "contextual_understanding", 
        "coherence",
        "fluency",
        "performance_efficiency"
    ]
    
    for weight in expected_weights:
        assert weight in weights, f"Missing weight: {weight}"
        assert isinstance(weights[weight], (int, float)), f"Weight {weight} is not numeric"
        assert 0 <= weights[weight] <= 1, f"Weight {weight} out of range [0,1]"
    
    # Weights should sum to approximately 1.0
    weight_sum = sum(weights.values())
    assert abs(weight_sum - 1.0) < 1e-6, f"Weights sum to {weight_sum}, expected 1.0"


def test_stepwise_import_structure():
    """Test that stepwise scripts can import required modules"""
    # Test imports without running full evaluation
    try:
        from pathlib import Path
        import yaml
        
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        
        config_path = Path("config/metrics/accuracy.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config is not None
        
        # Test generation profile loading
        from generation import load_generation_profile
        
        det_profile = load_generation_profile("deterministic")
        assert det_profile is not None
        assert "temperature" in det_profile
        
        cre_profile = load_generation_profile("creative") 
        assert cre_profile is not None
        assert "temperature" in cre_profile
        
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
