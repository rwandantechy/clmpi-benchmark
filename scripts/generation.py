#!/usr/bin/env python3
"""
Generation profile loader for CLMPI benchmark
Centralizes generation settings to prevent hardcoded parameters
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_generation_profile(mode: str = "deterministic", path: str = "config/generation_config.yaml") -> Dict[str, Any]:
    """
    Load generation profile from config file
    
    Args:
        mode: Profile name ("deterministic" or "creative")
        path: Path to generation config file
        
    Returns:
        Dictionary with generation parameters
        
    Raises:
        ValueError: If mode not found in config
        FileNotFoundError: If config file not found
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Generation config not found: {path}")
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    profiles = cfg.get("generation_profiles", {})
    if mode not in profiles:
        available = list(profiles.keys())
        raise ValueError(f"Unknown generation mode '{mode}'. Available: {available}")
    
    return profiles[mode]


def get_generation_settings_for_metric(metric: str, path: str = "config/generation_config.yaml") -> Dict[str, Any]:
    """
    Get appropriate generation settings for a specific metric
    
    Args:
        metric: Metric name ("accuracy", "contextual_understanding", "coherence", "fluency")
        path: Path to generation config file
        
    Returns:
        Dictionary with generation parameters for the metric
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Generation config not found: {path}")
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    profiles = cfg.get("generation_profiles", {})
    
    # Map metrics to profiles
    metric_to_profile = {
        "accuracy": "deterministic",
        "contextual_understanding": "deterministic", 
        "coherence": "creative",
        "fluency": "creative"
    }
    
    profile_name = metric_to_profile.get(metric, "deterministic")
    if profile_name not in profiles:
        raise ValueError(f"Profile '{profile_name}' not found for metric '{metric}'")
    
    return profiles[profile_name]


if __name__ == "__main__":
    # Test the functions
    try:
        det_profile = load_generation_profile("deterministic")
        print("Deterministic profile:", det_profile)
        
        cre_profile = load_generation_profile("creative")
        print("Creative profile:", cre_profile)
        
        acc_settings = get_generation_settings_for_metric("accuracy")
        print("Accuracy settings:", acc_settings)
        
    except Exception as e:
        print(f"Error: {e}")
