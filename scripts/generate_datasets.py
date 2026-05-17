#!/usr/bin/env python
"""Generate labeled datasets — delegates to unified pipeline."""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from src.pipeline import generate_datasets


def main() -> None:
    train_p, test_p = generate_datasets()
    print(f"Train: {train_p}")
    print(f"Test:  {test_p}")
    print("Done.")


if __name__ == "__main__":
    main()
