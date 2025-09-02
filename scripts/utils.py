#!/usr/bin/env python3
"""
Utility functions for CLMPI benchmark
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


def create_results_structure():
    """Create the basic results directory structure"""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Create .gitkeep to preserve the directory
    gitkeep_file = results_dir / ".gitkeep"
    if not gitkeep_file.exists():
        gitkeep_file.touch()
    
    return results_dir


def create_run_directory(label: str = "stepwise") -> Path:
    """Create a new timestamped run directory"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = f"{timestamp}_{label}"
    run_dir = Path("results") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metric subdirectories
    metrics = ["accuracy", "context", "coherence", "fluency", "efficiency"]
    for metric in metrics:
        (run_dir / metric).mkdir(exist_ok=True)
    
    return run_dir


def update_latest_symlink(run_dir: Path):
    """Create or update the latest symlink to point to the most recent run"""
    results_dir = Path("results")
    latest_link = results_dir / "latest"
    
    # Remove existing symlink if it exists
    if latest_link.exists() or latest_link.is_symlink():
        if latest_link.is_symlink():
            latest_link.unlink()
        else:
            shutil.rmtree(latest_link)
    
    # Create new symlink
    try:
        latest_link.symlink_to(run_dir.name)
        return True
    except OSError:
        # Fallback for systems that don't support symlinks
        try:
            shutil.copytree(run_dir, latest_link)
            return True
        except Exception:
            return False


def get_latest_run_directory() -> Path:
    """Get the latest run directory, or None if none exists"""
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    
    # Look for stepwise runs
    stepwise_runs = list(results_dir.glob("*_stepwise"))
    if not stepwise_runs:
        return None
    
    # Return the most recent one
    return max(stepwise_runs, key=lambda p: p.stat().st_mtime)


def cleanup_old_runs(keep_count: int = 5):
    """Clean up old run directories, keeping only the most recent ones"""
    results_dir = Path("results")
    if not results_dir.exists():
        return
    
    # Get all stepwise runs
    stepwise_runs = list(results_dir.glob("*_stepwise"))
    if len(stepwise_runs) <= keep_count:
        return
    
    # Sort by modification time and remove old ones
    sorted_runs = sorted(stepwise_runs, key=lambda p: p.stat().st_mtime, reverse=True)
    
    for old_run in sorted_runs[keep_count:]:
        try:
            shutil.rmtree(old_run)
        except Exception:
            pass  # Ignore cleanup errors
