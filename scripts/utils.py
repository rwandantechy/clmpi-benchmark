"""
Utility functions for CLMPI benchmark
"""

import re
from pathlib import Path
from typing import Union


def sanitize_filename(name: str) -> str:
    """
    Sanitize filename by replacing problematic characters
    
    Args:
        name: Original filename
        
    Returns:
        Sanitized filename safe for all filesystems
    """
    # Replace colons, spaces, and other problematic chars with hyphens
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    # Remove leading/trailing hyphens
    sanitized = sanitized.strip("-")
    # Ensure it's not empty
    return sanitized or "unnamed"


def ensure_path(path: Union[str, Path]) -> Path:
    """Ensure path is a Path object"""
    return Path(path) if isinstance(path, str) else path
