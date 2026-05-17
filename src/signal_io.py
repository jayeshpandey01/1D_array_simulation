"""Load, validate, and resample 1D signal arrays from CSV files."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import interpolate

from src.constants import SAMPLE_RATE, SIGNAL_LENGTH

METADATA_COLUMNS = {
    "wave_type",
    "label_name",
    "frequency",
    "amplitude",
    "phase",
    "snr_db",
    "label",
}


def validate_signal(
    signal: np.ndarray,
    expected_len: int = SIGNAL_LENGTH,
    allow_resample: bool = True,
) -> np.ndarray:
    """Ensure signal is a finite 1D float array of expected length."""
    x = np.asarray(signal, dtype=np.float64).flatten()
    if x.size == 0:
        raise ValueError("Signal is empty.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains NaN or Inf values.")

    if x.size != expected_len:
        if not allow_resample:
            raise ValueError(
                f"Signal length {x.size} != expected {expected_len}. "
                "Set allow_resample=True to interpolate."
            )
        x = resample_signal(x, expected_len)

    return x


def resample_signal(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a 1D signal to target_len samples."""
    x = np.asarray(signal, dtype=np.float64).flatten()
    if x.size == target_len:
        return x
    if x.size < 2:
        raise ValueError("Need at least 2 samples to resample.")
    src_t = np.linspace(0.0, 1.0, x.size)
    dst_t = np.linspace(0.0, 1.0, target_len)
    f = interpolate.interp1d(src_t, x, kind="linear", fill_value="extrapolate")
    return f(dst_t).astype(np.float64)


def _signal_column_names(n: int) -> List[str]:
    return [f"signal_{i}" for i in range(n)]


def ground_truth_from_row(row: pd.Series) -> Optional[Dict[str, Any]]:
    """Build ground-truth dict from a metadata row (or None if empty)."""
    gt: Dict[str, Any] = {}
    for col in ("wave_type", "label_name", "frequency", "amplitude", "phase", "snr_db"):
        if col in row.index and pd.notna(row[col]):
            gt[col] = row[col]
    if "label_name" in gt and "wave_type" not in gt:
        gt["wave_type"] = gt["label_name"]
    return gt if gt else None


def signal_column_names(df: pd.DataFrame) -> List[str]:
    """Return sorted signal_* column names present in df."""
    cols = [c for c in df.columns if c.startswith("signal_")]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def extract_signals_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Return (signals NxL, metadata_df)."""
    signal_cols = [c for c in df.columns if c.startswith("signal_")]
    if signal_cols:
        signal_cols = sorted(signal_cols, key=lambda c: int(c.split("_")[1]))
        signals = df[signal_cols].to_numpy(dtype=np.float64)
        meta = df.drop(columns=signal_cols, errors="ignore")
        return signals, meta

    numeric_cols = [
        c
        for c in df.columns
        if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(numeric_cols) >= 8:
        signals = df[numeric_cols].to_numpy(dtype=np.float64)
        meta = df.drop(columns=numeric_cols, errors="ignore")
        return signals, meta

    if "signal" in df.columns:
        rows = []
        for val in df["signal"]:
            if isinstance(val, str):
                row = np.fromstring(val.replace("[", "").replace("]", ""), sep=",")
            else:
                row = np.asarray(val, dtype=np.float64).flatten()
            rows.append(row)
        signals = np.vstack(rows)
        meta = df.drop(columns=["signal"], errors="ignore")
        return signals, meta

    raise ValueError(
        "CSV must contain signal_0..signal_N columns, numeric waveform columns, or a 'signal' column."
    )


def load_signal_csv(
    path: str,
    expected_len: int = SIGNAL_LENGTH,
    allow_resample: bool = True,
) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
    """
    Load signals from CSV.

    Returns
    -------
    signals : ndarray, shape (n_samples, expected_len)
    metadata : DataFrame or None
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    signals, meta = extract_signals_from_df(df)

    validated = np.vstack(
        [validate_signal(s, expected_len=expected_len, allow_resample=allow_resample) for s in signals]
    )
    meta_out = meta if not meta.empty else None
    return validated, meta_out


def load_single_signal(
    path: str,
    row_index: int = 0,
    expected_len: int = SIGNAL_LENGTH,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Load one row from a CSV as a 1D signal plus optional metadata dict."""
    signals, meta = load_signal_csv(path, expected_len=expected_len)
    if row_index < 0 or row_index >= signals.shape[0]:
        raise IndexError(f"row_index {row_index} out of range [0, {signals.shape[0]})")

    gt = ground_truth_from_row(meta.iloc[row_index]) if meta is not None else None
    return signals[row_index], gt


def load_config_json(path: str) -> Dict[str, Any]:
    """Load a single-wave generation config from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = {"wave_type"}
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(f"Config missing keys: {missing}")
    return cfg


def signals_to_dataframe(
    signals: np.ndarray,
    metadata: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
) -> pd.DataFrame:
    """Build a standard CSV-ready DataFrame from signals and metadata."""
    n, length = signals.shape
    cols = _signal_column_names(length)
    df = pd.DataFrame(signals, columns=cols)

    if metadata is not None:
        if isinstance(metadata, list):
            meta_df = pd.DataFrame(metadata)
        else:
            meta_df = metadata.reset_index(drop=True)
        if len(meta_df) != n:
            raise ValueError("Metadata row count must match number of signals.")
        df = pd.concat([meta_df, df], axis=1)

    return df
