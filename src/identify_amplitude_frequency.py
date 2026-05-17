"""
Identify amplitude, frequency, phase, and wave type from 1D signal arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from scipy.stats import kurtosis

from src.constants import NON_PERIODIC_WAVE_TYPES, PERIODIC_WAVE_TYPES, SAMPLE_RATE

NOISE_TYPES = {"white_noise", "pink_noise", "brownian_noise"}
DECAY_TYPES = {"exponential_decay", "logarithmic_decay", "step_function"}


@dataclass
class SignalAnalysisResult:
    wave_type: str
    frequency_hz: Optional[float]
    amplitude: Optional[float]
    phase_rad: Optional[float]
    peak_to_peak: float
    rms: float
    mean: float
    std_dev: float
    spectral_flatness: float
    warnings: List[str] = field(default_factory=list)
    ground_truth: Optional[Dict[str, Any]] = None
    frequency_error_hz: Optional[float] = None
    frequency_error_pct: Optional[float] = None
    amplitude_error: Optional[float] = None
    phase_error_rad: Optional[float] = None
    method_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wave_type": self.wave_type,
            "frequency_hz": self.frequency_hz,
            "amplitude": self.amplitude,
            "phase_rad": self.phase_rad,
            "peak_to_peak": self.peak_to_peak,
            "rms": self.rms,
            "mean": self.mean,
            "std_dev": self.std_dev,
            "spectral_flatness": self.spectral_flatness,
            "warnings": self.warnings,
            "ground_truth": self.ground_truth,
            "frequency_error_hz": self.frequency_error_hz,
            "frequency_error_pct": self.frequency_error_pct,
            "amplitude_error": self.amplitude_error,
            "phase_error_rad": self.phase_error_rad,
            "method_notes": self.method_notes,
        }


def get_amplitude_stats(signal: np.ndarray) -> Dict[str, float]:
    """Calculate amplitude statistics for a 1D signal."""
    x = np.asarray(signal, dtype=np.float64).flatten()
    return {
        "Peak-to-Peak": float(np.max(x) - np.min(x)),
        "RMS": float(np.sqrt(np.mean(np.square(x)))),
        "Mean": float(np.mean(x)),
        "Std Dev": float(np.std(x)),
    }


def _parabolic_peak_interp(magnitudes: np.ndarray, peak_idx: int) -> float:
    """Sub-bin peak index refinement via parabolic interpolation."""
    if peak_idx <= 0 or peak_idx >= len(magnitudes) - 1:
        return float(peak_idx)
    alpha = magnitudes[peak_idx - 1]
    beta = magnitudes[peak_idx]
    gamma = magnitudes[peak_idx + 1]
    denom = alpha - 2 * beta + gamma
    if abs(denom) < 1e-12:
        return float(peak_idx)
    delta = 0.5 * (alpha - gamma) / denom
    return float(peak_idx) + float(np.clip(delta, -0.5, 0.5))


def estimate_frequency(
    signal: np.ndarray,
    sampling_rate: float = SAMPLE_RATE,
) -> Tuple[float, np.ndarray, np.ndarray, int]:
    """
    Dominant frequency via FFT with parabolic interpolation.

    Returns (freq_hz, xf_pos, magnitude_pos, peak_bin_index).
    """
    x = np.asarray(signal, dtype=np.float64).flatten()
    n = len(x)
    yf = fft(x)
    xf = fftfreq(n, 1.0 / sampling_rate)

    pos_mask = xf > 0
    xf_pos = xf[pos_mask]
    yf_pos = np.abs(yf[pos_mask])

    if len(yf_pos) < 2:
        return 0.0, xf_pos, yf_pos, 0

    peak_idx = int(np.argmax(yf_pos))
    refined_idx = _parabolic_peak_interp(yf_pos, peak_idx)

    if peak_idx < len(xf_pos) - 1:
        df = xf_pos[1] - xf_pos[0]
        dominant_freq = float(xf_pos[peak_idx] + (refined_idx - peak_idx) * df)
    else:
        dominant_freq = float(xf_pos[peak_idx])

    return dominant_freq, xf_pos, yf_pos, peak_idx


def estimate_amplitude_from_fft(
    signal: np.ndarray,
    peak_bin: int,
    sampling_rate: float = SAMPLE_RATE,
) -> float:
    """Amplitude estimate from FFT peak magnitude: A ≈ 2|X[k]|/N."""
    x = np.asarray(signal, dtype=np.float64).flatten()
    n = len(x)
    yf = np.abs(fft(x))
    xf = fftfreq(n, 1.0 / sampling_rate)
    pos_mask = xf > 0
    yf_pos = yf[pos_mask]
    if peak_bin < len(yf_pos):
        return float(2.0 * yf_pos[peak_bin] / n)
    return float((np.max(x) - np.min(x)) / 2.0)


def estimate_phase(
    signal: np.ndarray,
    frequency_hz: float,
    sampling_rate: float = SAMPLE_RATE,
) -> float:
    """Phase (rad) from complex FFT bin at detected frequency."""
    x = np.asarray(signal, dtype=np.float64).flatten()
    n = len(x)
    yf = fft(x)
    xf = fftfreq(n, 1.0 / sampling_rate)
    idx = int(np.argmin(np.abs(xf - frequency_hz)))
    return float(np.angle(yf[idx]))


def _spectral_flatness(signal: np.ndarray, sampling_rate: float = SAMPLE_RATE) -> float:
    """Wiener entropy on positive FFT bins (excludes DC). Low = tonal, high = noise-like."""
    x = np.asarray(signal, dtype=np.float64).flatten()
    n = len(x)
    psd = np.abs(fft(x)) ** 2
    freqs = fftfreq(n, 1.0 / sampling_rate)
    psd = psd[(freqs > 0) & (psd > 1e-12)]
    if len(psd) < 2:
        return 1.0
    geo = np.exp(np.mean(np.log(psd + 1e-12)))
    arith = np.mean(psd)
    return float(geo / (arith + 1e-12))


def _fit_sine_phase_amp(signal: np.ndarray, freq: float) -> Tuple[float, float]:
    """Least-squares fit A*sin(2πft + φ) for amplitude and phase."""
    x = np.asarray(signal, dtype=np.float64).flatten()
    n = len(x)
    t = np.linspace(0, 1, n, endpoint=False)

    def model(t_arr, amp, phase):
        return amp * np.sin(2 * np.pi * freq * t_arr + phase)

    amp0 = (np.max(x) - np.min(x)) / 2.0
    try:
        popt, _ = curve_fit(
            model,
            t,
            x,
            p0=[max(amp0, 1e-3), 0.0],
            maxfev=5000,
        )
        return float(abs(popt[0])), float(popt[1])
    except (RuntimeError, ValueError):
        return float(amp0), 0.0


def classify_wave_type(
    signal: np.ndarray,
    sampling_rate: float = SAMPLE_RATE,
) -> Tuple[str, List[str]]:
    """
    Rule-based wave type classification (no ML model required).
    """
    x = np.asarray(signal, dtype=np.float64).flatten()
    warnings: List[str] = []
    flatness = _spectral_flatness(x, sampling_rate)
    stats = get_amplitude_stats(x)
    freq, _, yf_pos, _ = estimate_frequency(x, sampling_rate)

    n = len(x)
    t = np.linspace(0, 1, n, endpoint=False)
    templates: Dict[str, np.ndarray] = {}

    if freq > 0.5:
        templates["sine"] = np.sin(2 * np.pi * freq * t)
        templates["cosine"] = np.cos(2 * np.pi * freq * t)
        templates["square"] = scipy_signal.square(2 * np.pi * freq * t)
        templates["triangle"] = scipy_signal.sawtooth(2 * np.pi * freq * t, width=0.5)
        templates["sawtooth"] = scipy_signal.sawtooth(2 * np.pi * freq * t)

    scores: Dict[str, float] = {}
    x_norm = x - np.mean(x)
    x_std = np.std(x_norm) + 1e-10
    x_norm = x_norm / x_std

    for name, tmpl in templates.items():
        tmpl_norm = (tmpl - np.mean(tmpl)) / (np.std(tmpl) + 1e-10)
        scores[name] = float(np.abs(np.corrcoef(x_norm, tmpl_norm)[0, 1]))

    peak_ratio = 0.0
    if len(yf_pos) > 0:
        peak_ratio = float(np.max(yf_pos) / (np.mean(yf_pos) + 1e-12))

    # Strong periodic peak → prefer template match over noise heuristics
    if scores and max(scores.values()) >= 0.75 and peak_ratio > 4.0:
        return max(scores, key=scores.get), warnings

    if _is_decay_like(x):
        slope = np.polyfit(np.arange(len(x)), x, 1)[0]
        if slope < -0.01:
            return "exponential_decay", warnings
        return "logarithmic_decay", warnings

    if flatness > 0.5 and peak_ratio < 3.0 and stats["Std Dev"] > 1e-6:
        if _is_pink_like(x):
            return "pink_noise", warnings
        if _is_brownian_like(x):
            return "brownian_noise", warnings
        return "white_noise", warnings

    if not scores:
        warnings.append("Could not match periodic template.")
        return "unknown", warnings

    best = max(scores, key=scores.get)
    if scores[best] < 0.45:
        warnings.append("Low template correlation; classification uncertain.")
        if kurtosis(x) > 1.0:
            return "gaussian", warnings
        return "hyperbolic_tangent", warnings

    return best, warnings


def _is_pink_like(x: np.ndarray) -> bool:
    """Heuristic: low-frequency energy dominates."""
    psd = np.abs(fft(x)) ** 2
    half = len(psd) // 2
    if half < 4:
        return False
    low = np.mean(psd[1 : half // 4])
    high = np.mean(psd[half // 4 : half])
    return low > high * 0.5


def _is_brownian_like(x: np.ndarray) -> bool:
    """Heuristic: very smooth, integrated-noise appearance."""
    diff1 = np.diff(x)
    diff2 = np.diff(diff1)
    return float(np.std(diff2)) < float(np.std(diff1)) * 0.3


def _is_decay_like(x: np.ndarray) -> bool:
    r = np.max(x) - np.min(x)
    if r < 1e-6:
        return False
    mono_frac = np.mean(np.diff(x) <= 0.05 * r / len(x))
    return mono_frac > 0.85


def analyze_signal(signal, sampling_rate: float = SAMPLE_RATE):
    """Legacy API: stats dict + FFT arrays."""
    stats = get_amplitude_stats(signal)
    dominant_freq, xf, yf = get_dominant_frequency(signal, sampling_rate)
    stats["Dominant Frequency"] = dominant_freq
    return stats, xf, yf


def get_dominant_frequency(signal, sampling_rate: float = SAMPLE_RATE):
    """Backward-compatible FFT helper."""
    freq, xf_pos, yf_pos, _ = estimate_frequency(signal, sampling_rate)
    return freq, xf_pos, yf_pos


def analyze_signal_full(
    signal: np.ndarray,
    sampling_rate: float = SAMPLE_RATE,
    ground_truth: Optional[Dict[str, Any]] = None,
    wave_type_hint: Optional[str] = None,
) -> SignalAnalysisResult:
    """
    Full parameter identification with optional ground-truth comparison.
    """
    x = np.asarray(signal, dtype=np.float64).flatten()
    stats = get_amplitude_stats(x)
    flatness = _spectral_flatness(x, sampling_rate)
    warnings: List[str] = []

    if wave_type_hint:
        wave_type = wave_type_hint
    else:
        wave_type, cls_warnings = classify_wave_type(x, sampling_rate)
        warnings.extend(cls_warnings)

    freq_hz: Optional[float] = None
    amplitude: Optional[float] = None
    phase_rad: Optional[float] = None
    method_notes = ""

    if wave_type in NOISE_TYPES:
        amplitude = stats["RMS"]
        method_notes = "Broadband/noise signal: frequency and phase not reported."
    elif wave_type in DECAY_TYPES or wave_type in ("gaussian", "step_function"):
        amplitude = stats["RMS"]
        method_notes = "Non-periodic waveform: only amplitude statistics reported."
    elif wave_type in ("hyperbolic_tangent", "sigmoid"):
        freq_hz, _, _, peak_idx = estimate_frequency(x, sampling_rate)
        amplitude = stats["Peak-to-Peak"] / 2.0
        method_notes = "Modulated waveform: fundamental frequency only."
        if freq_hz < 0.5 or freq_hz > sampling_rate / 2 - 1:
            warnings.append("Estimated frequency near DC or Nyquist limit.")
    else:
        freq_hz, _, _, peak_idx = estimate_frequency(x, sampling_rate)
        amplitude_fft = estimate_amplitude_from_fft(x, peak_idx, sampling_rate)
        amplitude_fit, phase_fit = _fit_sine_phase_amp(x, freq_hz)
        amplitude = float(0.5 * amplitude_fft + 0.5 * amplitude_fit)
        phase_rad = phase_fit
        method_notes = "Periodic signal: FFT + sinusoidal least-squares fit."

        if wave_type in ("square", "triangle", "sawtooth", "pulse_triangle"):
            warnings.append(
                "Harmonic content present; phase refers to fundamental fit."
            )
            amplitude = stats["Peak-to-Peak"] / 2.0

        if freq_hz is not None and (freq_hz < 0.5 or freq_hz > sampling_rate / 2 - 1):
            warnings.append("Estimated frequency near DC or Nyquist limit.")

    result = SignalAnalysisResult(
        wave_type=wave_type,
        frequency_hz=freq_hz,
        amplitude=amplitude,
        phase_rad=phase_rad,
        peak_to_peak=stats["Peak-to-Peak"],
        rms=stats["RMS"],
        mean=stats["Mean"],
        std_dev=stats["Std Dev"],
        spectral_flatness=flatness,
        warnings=warnings,
        ground_truth=ground_truth,
        method_notes=method_notes,
    )

    if ground_truth:
        _apply_ground_truth_errors(result, ground_truth)

    return result


def _apply_ground_truth_errors(result: SignalAnalysisResult, gt: Dict[str, Any]) -> None:
    gt_type = gt.get("wave_type") or gt.get("label_name")
    if gt_type and result.wave_type != gt_type:
        result.warnings.append(
            f"Detected wave type '{result.wave_type}' != ground truth '{gt_type}'."
        )

    gt_f = gt.get("frequency")
    if gt_f is not None and not (isinstance(gt_f, float) and np.isnan(gt_f)):
        if result.frequency_hz is not None:
            result.frequency_error_hz = result.frequency_hz - float(gt_f)
            if float(gt_f) != 0:
                result.frequency_error_pct = 100.0 * result.frequency_error_hz / float(gt_f)

    gt_a = gt.get("amplitude")
    if gt_a is not None and not (isinstance(gt_a, float) and np.isnan(gt_a)):
        if result.amplitude is not None:
            result.amplitude_error = result.amplitude - float(gt_a)

    gt_p = gt.get("phase")
    if gt_p is not None and not (isinstance(gt_p, float) and np.isnan(gt_p)):
        if result.phase_rad is not None:
            err = result.phase_rad - float(gt_p)
            result.phase_error_rad = float((err + np.pi) % (2 * np.pi) - np.pi)
