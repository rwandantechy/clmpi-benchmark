#!/usr/bin/env python3
"""
Setup script for CLMPI Benchmark

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="clmpi-benchmark",
    version="0.1.0",
    author="CLMPI Benchmark Contributors",
    author_email="your-email@domain.com",
    description="Comprehensive Language Model Performance Index - A unified benchmarking framework for LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/clmpi-benchmark",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "clmpi-evaluate=scripts.evaluate_models:main",
            "clmpi-calculate=scripts.clmpi_calculator:example_usage",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
) 