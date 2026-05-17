"""
Shared feature extraction for ML models (single source — no duplicate FFT logic).
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.stats import kurtosis, skew

from src.constants import SAMPLE_RATE
from src.identify_amplitude_frequency import (
    estimate_frequency,
    get_amplitude_stats,
)

FEATURE_NAMES: List[str] = [
    "mean",
    "std",
    "rms",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "dominant_frequency",
    "spectral_energy",
]


def extract_signal_features(
    signal: np.ndarray,
    sampling_rate: float = SAMPLE_RATE,
) -> np.ndarray:
    """8-D feature vector for one signal (used by comparative_analysis)."""
    x = np.asarray(signal, dtype=np.float64).flatten()
    stats = get_amplitude_stats(x)
    freq, _, yf_pos, _ = estimate_frequency(x, sampling_rate)
    spec_energy = float(np.sum(yf_pos**2)) if len(yf_pos) else 0.0

    return np.array(
        [
            stats["Mean"],
            stats["Std Dev"],
            stats["RMS"],
            stats["Peak-to-Peak"],
            float(skew(x)),
            float(kurtosis(x)),
            float(freq),
            spec_energy,
        ],
        dtype=np.float64,
    )


def extract_batch_features(
    signals: np.ndarray,
    sampling_rate: float = SAMPLE_RATE,
) -> np.ndarray:
    """Feature matrix shape (n_samples, 8)."""
    return np.vstack(
        [extract_signal_features(s, sampling_rate) for s in signals]
    )
