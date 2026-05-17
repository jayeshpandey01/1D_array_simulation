"""
Precise labeled synthetic datasets with stored ground-truth parameters.
"""

from __future__ import annotations

import json
import os
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from sklearn.model_selection import train_test_split

from src.constants import PERIODIC_WAVE_TYPES, SAMPLE_RATE, SIGNAL_LENGTH
from src.dataset import WaveformGenerator
from src.signal_io import signals_to_dataframe

DEFAULT_FREQS = [1.0, 2.0, 3.0, 4.0, 5.0]
DEFAULT_AMPS = [0.5, 1.0, 1.5]
DEFAULT_PHASES = [0.0, np.pi / 4, np.pi / 2, np.pi]
DEFAULT_SNR_DB = [None, 40.0, 30.0, 20.0]  # None = clean


def add_noise_at_snr(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to reach target SNR in dB (relative to signal power)."""
    power_sig = np.mean(signal**2) + 1e-12
    power_noise = power_sig / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(power_noise), size=signal.shape)
    return signal + noise


def generate_wave_with_params(
    wave_type: str,
    length: int = SIGNAL_LENGTH,
    frequency: Optional[float] = None,
    amplitude: Optional[float] = None,
    phase: Optional[float] = None,
    snr_db: Optional[float] = None,
    seed: Optional[int] = None,
    duty: float = 0.3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate one waveform with explicit parameters; returns signal and metadata."""
    rng = np.random.default_rng(seed)
    gen = WaveformGenerator()

    meta: Dict[str, Any] = {
        "wave_type": wave_type,
        "frequency": np.nan,
        "amplitude": np.nan,
        "phase": np.nan,
        "snr_db": snr_db if snr_db is not None else np.nan,
    }

    if wave_type in PERIODIC_WAVE_TYPES:
        frequency = float(frequency if frequency is not None else rng.uniform(1, 5))
        amplitude = float(amplitude if amplitude is not None else rng.uniform(0.5, 1.5))
        phase = float(phase if phase is not None else rng.uniform(0, 2 * np.pi))
        meta.update(frequency=frequency, amplitude=amplitude, phase=phase)

        if wave_type == "sine":
            sig = gen.sine_wave(length, freq=frequency, amp=amplitude, phase=phase)
        elif wave_type == "cosine":
            sig = gen.cosine_wave(length, freq=frequency, amp=amplitude, phase=phase)
        elif wave_type == "square":
            sig = gen.square_wave(length, freq=frequency, amp=amplitude, phase=phase)
        elif wave_type == "triangle":
            sig = gen.triangle_wave(length, freq=frequency, amp=amplitude, phase=phase)
        elif wave_type == "sawtooth":
            sig = gen.sawtooth_wave(length, freq=frequency, amp=amplitude, phase=phase)
        elif wave_type == "pulse_triangle":
            t = np.linspace(0, 1, length)
            sig = amplitude * scipy_signal.sawtooth(
                2 * np.pi * frequency * t + phase, width=duty
            )
        else:
            raise ValueError(f"Unsupported periodic type: {wave_type}")
    else:
        amplitude = float(amplitude if amplitude is not None else rng.uniform(0.5, 1.5))
        meta["amplitude"] = amplitude
        if wave_type == "gaussian":
            freq = float(frequency if frequency is not None else rng.uniform(3, 8))
            meta["frequency"] = freq
            sig = gen.gaussian_wave(length, freq=freq, amp=amplitude)
        elif wave_type == "white_noise":
            sig = gen.white_noise(length, amp=amplitude)
        elif wave_type == "pink_noise":
            sig = gen.pink_noise(length, amp=amplitude)
        elif wave_type == "brownian_noise":
            sig = gen.brownian_noise(length, amp=amplitude)
        elif wave_type == "exponential_decay":
            sig = gen.exponential_decay(length, amp=amplitude)
        elif wave_type == "logarithmic_decay":
            sig = gen.logarithmic_decay(length, amp=amplitude)
        elif wave_type == "step_function":
            sig = gen.step_function(length, amp=amplitude)
        elif wave_type == "hyperbolic_tangent":
            freq = float(frequency if frequency is not None else rng.uniform(1, 5))
            meta["frequency"] = freq
            sig = gen.hyperbolic_tangent_wave(length, freq=freq, amp=amplitude)
        elif wave_type == "sigmoid":
            freq = float(frequency if frequency is not None else rng.uniform(1, 5))
            meta["frequency"] = freq
            sig = gen.sigmoid_wave(length, freq=freq, amp=amplitude)
        else:
            raise ValueError(f"Unknown wave type: {wave_type}")

    if snr_db is not None:
        sig = add_noise_at_snr(sig, float(snr_db), rng)

    return sig.astype(np.float64), meta


def param_grid(
    wave_types: Sequence[str] = PERIODIC_WAVE_TYPES,
    freqs: Sequence[float] = DEFAULT_FREQS,
    amps: Sequence[float] = DEFAULT_AMPS,
    phases: Sequence[float] = DEFAULT_PHASES,
    snr_values: Sequence[Optional[float]] = (None,),
) -> List[Dict[str, Any]]:
    """Full Cartesian grid of periodic-wave parameters."""
    combos = []
    for wt, f, a, p, snr in product(wave_types, freqs, amps, phases, snr_values):
        combos.append(
            {
                "wave_type": wt,
                "frequency": f,
                "amplitude": a,
                "phase": p,
                "snr_db": snr,
            }
        )
    return combos


def generate_labeled_batch(
    configs: Sequence[Dict[str, Any]],
    length: int = SIGNAL_LENGTH,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate signals for a list of parameter dicts."""
    signals = []
    meta_rows = []
    for i, cfg in enumerate(configs):
        sig, meta = generate_wave_with_params(
            wave_type=cfg["wave_type"],
            length=length,
            frequency=cfg.get("frequency"),
            amplitude=cfg.get("amplitude"),
            phase=cfg.get("phase"),
            snr_db=cfg.get("snr_db"),
            seed=seed + i if seed is not None else None,
            duty=cfg.get("duty", 0.3),
        )
        signals.append(sig)
        meta_rows.append(meta)

    signals_arr = np.vstack(signals)
    return signals_to_dataframe(signals_arr, meta_rows)


def generate_random_labeled(
    n: int,
    wave_types: Sequence[str] = PERIODIC_WAVE_TYPES,
    length: int = SIGNAL_LENGTH,
    seed: int = 42,
    snr_choices: Sequence[Optional[float]] = DEFAULT_SNR_DB,
) -> pd.DataFrame:
    """Random parameters with stored ground truth (reproducible via seed)."""
    rng = np.random.default_rng(seed)
    configs = []
    for i in range(n):
        configs.append(
            {
                "wave_type": rng.choice(list(wave_types)),
                "frequency": float(rng.uniform(1, 5)),
                "amplitude": float(rng.uniform(0.5, 1.5)),
                "phase": float(rng.uniform(0, 2 * np.pi)),
                "snr_db": rng.choice(list(snr_choices)),
            }
        )
    return generate_labeled_batch(configs, length=length, seed=seed)


def save_splits(
    output_dir: str = "datasets",
    train_size: int = 2000,
    test_size: int = 500,
    mode: str = "grid",
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Write train/test CSV files.

    mode='grid' : periodic full grid (subsampled if large)
    mode='random' : random labeled periodic waves
    """
    os.makedirs(output_dir, exist_ok=True)

    if mode == "grid":
        configs = param_grid(snr_values=DEFAULT_SNR_DB)
        df = generate_labeled_batch(configs, seed=seed)
        # Also classification-style waveforms (all types, random params)
        df_cls = generate_random_labeled(
            train_size + test_size,
            wave_types=list(PERIODIC_WAVE_TYPES) + ["white_noise", "square"],
            seed=seed,
        )
        df = pd.concat([df, df_cls], ignore_index=True)
    else:
        df = generate_random_labeled(train_size + test_size, seed=seed)

    strat = df["wave_type"].astype(str)
    train_df, test_df = train_test_split(
        df, test_size=test_size / len(df), random_state=seed, stratify=strat
    )

    train_path = os.path.join(output_dir, "train_parameters.csv")
    test_path = os.path.join(output_dir, "test_parameters.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Classification CSVs for check_short_samples.py
    train_cls = os.path.join(output_dir, "train_waveforms.csv")
    test_cls = os.path.join(output_dir, "test_waveforms.csv")
    _save_classification_csv(train_df, train_cls)
    _save_classification_csv(test_df, test_cls)

    return train_path, test_path


def _save_classification_csv(df: pd.DataFrame, path: str) -> None:
    """Format compatible with check_short_samples (label, label_name, signal columns)."""
    out = df.copy()
    types = sorted(out["wave_type"].unique())
    type_to_id = {t: i for i, t in enumerate(types)}
    out["label"] = out["wave_type"].map(type_to_id)
    out["label_name"] = out["wave_type"]
    out.to_csv(path, index=False)


def generate_from_config_file(config_path: str, output_path: str) -> str:
    """Generate a single or batch JSON config into CSV."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    configs = raw if isinstance(raw, list) else [raw]
    df = generate_labeled_batch(configs)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    train_p, test_p = save_splits()
    print(f"Wrote {train_p}")
    print(f"Wrote {test_p}")
