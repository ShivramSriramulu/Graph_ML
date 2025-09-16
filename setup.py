#!/usr/bin/env python3
"""
Setup script for CORD-19 GraphRAG Pipeline
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cord19-graphrag",
    version="1.0.0",
    author="Shivram Sriramulu",
    author_email="shivram@example.com",
    description="A comprehensive GraphRAG pipeline for CORD-19 COVID-19 research papers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShivramSriramulu/Graph_ML",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.3.0",
        "scispacy>=0.5.0",
        "spacy>=3.7.0",
        "transformers>=4.56.0",
        "faiss-cpu>=1.12.0",
        "networkx>=3.5.0",
        "openai>=1.107.0",
        "torch>=2.8.0",
        "matplotlib>=3.10.0",
        "seaborn>=0.13.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.7.0",
        "tqdm>=4.67.0",
        "streamlit>=1.49.0",
        "plotly>=6.3.0",
        "pyvis>=0.3.2",
        "python-dotenv>=1.1.0",
        "jupyter>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "cord19-graphrag=src.cord19_graphrag:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
)
