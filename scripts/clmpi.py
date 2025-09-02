#!/usr/bin/env python3
"""
Simple CLMPI wrapper - just run: python scripts/clmpi.py
"""

import sys
from pathlib import Path
from combine_clmpi import main

if __name__ == "__main__":
    # Find the most recent results directory
    results_dir = Path("results")
    if not results_dir.exists():
        print("Error: No results directory found")
        sys.exit(1)
    
    # Look for stepwise directories
    stepwise_dirs = list(results_dir.glob("*_stepwise"))
    if not stepwise_dirs:
        print("Error: No stepwise evaluation results found")
        sys.exit(1)
    
    # Use the most recent one
    latest_dir = max(stepwise_dirs, key=lambda p: p.stat().st_mtime)
    print(f"Using results from: {latest_dir}")
    
    # Set sys.argv to simulate command line arguments
    sys.argv = ["combine_clmpi.py", "--results-dir", str(latest_dir)]
    
    # Run the combine script
    sys.exit(main())
