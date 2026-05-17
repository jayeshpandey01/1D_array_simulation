"""
Central pipeline — all endpoints connect through this module.

Flow:
  generate_datasets → CSV
  analyze_csv       → per-row SignalAnalysisResult columns
  evaluate_csv      → metrics + plots (uses analyze_csv)
  compare_models    → ML vs FFT baseline (uses features + CSV)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.constants import SAMPLE_RATE
from src.identify_amplitude_frequency import (
    SignalAnalysisResult,
    analyze_signal_full,
)
from src.signal_io import (
    extract_signals_from_df,
    ground_truth_from_row,
    load_signal_csv,
    signals_to_dataframe,
)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def result_to_row(result: SignalAnalysisResult) -> Dict[str, Any]:
    """Flatten SignalAnalysisResult for DataFrame columns."""
    return {
        "wave_type_detected": result.wave_type,
        "freq_detected": result.frequency_hz,
        "amp_detected": result.amplitude,
        "phase_detected": result.phase_rad,
        "freq_error_hz": result.frequency_error_hz,
        "freq_error_pct": result.frequency_error_pct,
        "amp_error": result.amplitude_error,
        "phase_error_rad": result.phase_error_rad,
    }


def analyze_one(
    signal: np.ndarray,
    ground_truth: Optional[Dict[str, Any]] = None,
    sampling_rate: float = SAMPLE_RATE,
) -> SignalAnalysisResult:
    """Analyze a single 1D signal."""
    return analyze_signal_full(signal, sampling_rate=sampling_rate, ground_truth=ground_truth)


def analyze_csv(
    csv_path: str,
    max_rows: Optional[int] = None,
    sampling_rate: float = SAMPLE_RATE,
) -> pd.DataFrame:
    """Load CSV and analyze every row; returns metadata + detection columns."""
    df = pd.read_csv(csv_path)
    return analyze_dataframe(df, max_rows=max_rows, sampling_rate=sampling_rate)


def analyze_dataframe(
    df: pd.DataFrame,
    max_rows: Optional[int] = None,
    sampling_rate: float = SAMPLE_RATE,
) -> pd.DataFrame:
    """Analyze all rows in an already-loaded DataFrame."""
    signals, meta = extract_signals_from_df(df)
    n = signals.shape[0] if max_rows is None else min(max_rows, signals.shape[0])

    rows = []
    for i in range(n):
        gt = ground_truth_from_row(meta.iloc[i]) if meta is not None and i < len(meta) else None
        res = analyze_one(signals[i], ground_truth=gt, sampling_rate=sampling_rate)
        rows.append(result_to_row(res))

    det_df = pd.DataFrame(rows)
    if meta is not None:
        return pd.concat([meta.iloc[:n].reset_index(drop=True), det_df], axis=1)
    return det_df


# ---------------------------------------------------------------------------
# Evaluation (plots + metrics)
# ---------------------------------------------------------------------------

def add_frequency_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Add freq_error columns when frequency and freq_detected exist."""
    out = df.copy()
    if "frequency" not in out.columns or "freq_detected" not in out.columns:
        return out
    mask = out["frequency"].notna() & out["freq_detected"].notna()
    out.loc[mask, "freq_error"] = out.loc[mask, "freq_detected"] - out.loc[mask, "frequency"]
    out.loc[mask, "freq_error_pct"] = (
        100.0 * out.loc[mask, "freq_error"] / out.loc[mask, "frequency"].replace(0, np.nan)
    )
    return out


def evaluate_csv(
    csv_path: str,
    output_dir: str = "outputs/evaluation",
    max_rows: Optional[int] = None,
    show_plots: bool = False,
) -> pd.DataFrame:
    """Full evaluation: analyze, metrics, save CSV and plots."""
    from src.evaluate_parameters import (
        plot_freq_detected_vs_actual,
        plot_freq_error_hist,
        plot_freq_error_vs_actual,
        print_metrics,
    )

    os.makedirs(output_dir, exist_ok=True)
    result_df = analyze_csv(csv_path, max_rows=max_rows)
    result_df = add_frequency_errors(result_df)

    out_csv = os.path.join(output_dir, "evaluation_results.csv")
    result_df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

    print_metrics(result_df)

    save = None if show_plots else output_dir
    plot_freq_detected_vs_actual(
        result_df,
        save_path=None if show_plots else os.path.join(output_dir, "freq_detected_vs_actual.png"),
    )
    plot_freq_error_vs_actual(
        result_df,
        save_path=None if show_plots else os.path.join(output_dir, "freq_error_vs_actual.png"),
    )
    plot_freq_error_hist(
        result_df,
        save_path=None if show_plots else os.path.join(output_dir, "freq_error_hist.png"),
    )

    if show_plots:
        import matplotlib.pyplot as plt
        plt.show()

    return result_df


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_datasets(output_dir: str = "datasets") -> Tuple[str, str]:
    """Generate train/test parameter CSVs and classification CSVs."""
    from src.precise_dataset import save_splits

    os.makedirs(output_dir, exist_ok=True)
    train_p, test_p = save_splits(
        output_dir=output_dir,
        train_size=2000,
        test_size=500,
        mode="grid",
    )

    from src.dataset import SignalDataset

    try:
        ds = SignalDataset(num_samples=3000, signal_length=256, seed=42)
        ds.save_to_csv(os.path.join(output_dir, "waveform_dataset.csv"))
    except ImportError:
        from src.precise_dataset import generate_random_labeled

        df = generate_random_labeled(3000, seed=42)
        df.to_csv(os.path.join(output_dir, "waveform_dataset.csv"), index=False)

    return train_p, test_p


# ---------------------------------------------------------------------------
# ML comparison
# ---------------------------------------------------------------------------

def compare_models(
    csv_path: str,
    target_col: str = "frequency",
    output_dir: str = "outputs/comparison",
    train_cnn: bool = False,
    cnn_epochs: int = 30,
) -> pd.DataFrame:
    """
    Run classical ML + FFT baseline (+ optional CNN) on a labeled CSV.
    """
    from src.comparative_analysis import SignalModelComparator

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[target_col])

    comparator = SignalModelComparator(df, target_col=target_col)
    comparator.evaluate_fft_baseline()
    comparator.train_classical_models()

    if train_cnn:
        try:
            comparator.train_pytorch_cnn(epochs=cnn_epochs)
        except ImportError:
            print("PyTorch not installed — skipping CNN.")

    summary = comparator.print_summary()
    summary.to_csv(os.path.join(output_dir, f"comparison_{target_col}.csv"), index=False)

    fft_key = "FFT_Pipeline" if "FFT_Pipeline" in comparator.results else "FFT_Baseline"
    if fft_key in comparator.results:
        preds = comparator.results[fft_key]["predictions"]
        SignalModelComparator.plot_prediction_errors(
            comparator.y_test,
            preds,
            target_name=f"FFT {target_col}",
            save_path=os.path.join(output_dir, f"fft_errors_{target_col}.png"),
        )

    return summary


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full(
    csv_path: str = "datasets/test_parameters.csv",
    skip_ml: bool = False,
) -> None:
    """Generate (if missing) → evaluate → compare models."""
    if not os.path.isfile(csv_path):
        print("Datasets not found — generating…")
        generate_datasets()

    print("\n=== Parameter evaluation ===")
    evaluate_csv(csv_path)

    if not skip_ml and os.path.isfile(csv_path):
        print("\n=== Model comparison (frequency) ===")
        compare_models(csv_path, target_col="frequency", train_cnn=False)
        print("\n=== Model comparison (amplitude) ===")
        compare_models(csv_path, target_col="amplitude", train_cnn=False)
