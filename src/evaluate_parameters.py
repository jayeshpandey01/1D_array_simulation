"""
Batch evaluation plots and metrics.

Implementation lives in src.pipeline — this module exposes plotting helpers
and a thin CLI wrapper for backward compatibility.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline import (
    add_frequency_errors,
    analyze_csv,
    analyze_dataframe,
    evaluate_csv,
)


def frequency_error_df(
    df: pd.DataFrame,
    detected_col: str = "freq_detected",
    actual_col: str = "frequency",
) -> pd.DataFrame:
    """Add frequency error columns (alias for pipeline.add_frequency_errors)."""
    out = df.copy()
    if detected_col != "freq_detected" or actual_col != "frequency":
        mask = out[actual_col].notna() & out[detected_col].notna()
        out.loc[mask, "freq_error"] = out.loc[mask, detected_col] - out.loc[mask, actual_col]
        out.loc[mask, "freq_error_pct"] = (
            100.0 * out.loc[mask, "freq_error"] / out.loc[mask, actual_col].replace(0, np.nan)
        )
        return out
    return add_frequency_errors(df)


def plot_freq_detected_vs_actual(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    mask = df["frequency"].notna() & df["freq_detected"].notna()
    sub = df.loc[mask]
    if sub.empty:
        print("No rows with both actual and detected frequency for plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(sub["frequency"], sub["freq_detected"], alpha=0.5, s=20, edgecolors="none")
    lims = [
        min(sub["frequency"].min(), sub["freq_detected"].min()),
        max(sub["frequency"].max(), sub["freq_detected"].max()),
    ]
    ax.plot(lims, lims, "r--", label="Ideal (y = x)")
    ax.set_xlabel("Actual frequency (Hz)")
    ax.set_ylabel("Detected frequency (Hz)")
    ax.set_title("Detected vs. Actual Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_freq_error_vs_actual(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    sub = df.dropna(subset=["frequency", "freq_error_hz"])
    if sub.empty and "freq_error" in df.columns:
        sub = df.dropna(subset=["frequency", "freq_error"])
    if sub.empty:
        print("No frequency error data to plot.")
        return

    err_col = "freq_error_hz" if "freq_error_hz" in sub.columns else "freq_error"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sub["frequency"], sub[err_col], alpha=0.5, s=20)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual frequency (Hz)")
    ax.set_ylabel("Frequency error (Hz)")
    ax.set_title("Frequency Error vs. Actual Frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_freq_error_hist(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    errs = df["freq_error_hz"].dropna() if "freq_error_hz" in df.columns else df.get("freq_error", pd.Series()).dropna()
    if errs.empty:
        print("No frequency errors for histogram.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errs, bins=40, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Frequency error (Hz)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Frequency Errors")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def print_metrics(df: pd.DataFrame) -> None:
    col = "freq_error_hz" if "freq_error_hz" in df.columns else "freq_error"
    sub = df.dropna(subset=[col])
    if sub.empty:
        print("No frequency ground truth available for metrics.")
        return

    err = sub[col].to_numpy()
    print("\n--- Frequency error metrics ---")
    print(f"  Samples:   {len(sub)}")
    print(f"  MAE:       {np.mean(np.abs(err)):.4f} Hz")
    print(f"  RMSE:      {np.sqrt(np.mean(err**2)):.4f} Hz")
    print(f"  Max |err|: {np.max(np.abs(err)):.4f} Hz")


def run_evaluation(
    csv_path: str,
    output_dir: str = "outputs/evaluation",
    max_rows: Optional[int] = None,
    show_plots: bool = False,
) -> pd.DataFrame:
    """Backward-compatible alias → pipeline.evaluate_csv."""
    return evaluate_csv(csv_path, output_dir=output_dir, max_rows=max_rows, show_plots=show_plots)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate parameter identification on a labeled CSV.")
    parser.add_argument("--csv", default="datasets/test_parameters.csv")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.csv):
        print(f"CSV not found: {args.csv}")
        print("Run: python -m src generate")
        return

    evaluate_csv(args.csv, output_dir=args.output_dir, max_rows=args.max_rows, show_plots=args.show)


if __name__ == "__main__":
    main()
