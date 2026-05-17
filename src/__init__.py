"""
1D array signal simulation and parameter identification.

Primary API: src.pipeline
CLI: python -m src --help
"""

from src.pipeline import (
    analyze_csv,
    analyze_one,
    compare_models,
    evaluate_csv,
    generate_datasets,
    run_full,
)

__all__ = [
    "analyze_one",
    "analyze_csv",
    "evaluate_csv",
    "compare_models",
    "generate_datasets",
    "run_full",
]
