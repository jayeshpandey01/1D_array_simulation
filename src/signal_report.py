"""Format and export signal analysis results."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.identify_amplitude_frequency import SignalAnalysisResult


def _fmt_optional(value: Optional[float], unit: str = "", precision: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and (value != value):  # NaN
        return "N/A"
    return f"{value:.{precision}f}{unit}"


def format_report(result: SignalAnalysisResult) -> str:
    """Human-readable multi-line report."""
    lines = [
        f"Wave type:      {result.wave_type}",
        f"Amplitude:      {_fmt_optional(result.amplitude)}  "
        f"(RMS: {_fmt_optional(result.rms)}, peak-to-peak: {_fmt_optional(result.peak_to_peak)})",
        f"Phase (rad):    {_fmt_optional(result.phase_rad)}",
        f"Frequency (Hz): {_fmt_optional(result.frequency_hz)}  (FFT + refinement)",
        f"Mean / Std:     {_fmt_optional(result.mean)} / {_fmt_optional(result.std_dev)}",
        f"Spectral flatness: {_fmt_optional(result.spectral_flatness, precision=4)}",
    ]

    if result.method_notes:
        lines.append(f"Notes:          {result.method_notes}")

    if result.ground_truth:
        gt = result.ground_truth
        wt = gt.get("wave_type") or gt.get("label_name", "?")
        lines.append(
            f"Ground truth:   {wt}, A={_fmt_optional(gt.get('amplitude'))}, "
            f"phase={_fmt_optional(gt.get('phase'))}, f={_fmt_optional(gt.get('frequency'))} Hz"
        )

    if result.frequency_error_hz is not None:
        pct = (
            f" ({result.frequency_error_pct:+.2f}%)"
            if result.frequency_error_pct is not None
            else ""
        )
        lines.append(f"Frequency error: {result.frequency_error_hz:+.4f} Hz{pct}")

    if result.amplitude_error is not None:
        lines.append(f"Amplitude error: {result.amplitude_error:+.4f}")

    if result.phase_error_rad is not None:
        lines.append(f"Phase error:     {result.phase_error_rad:+.4f} rad")

    if result.warnings:
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def to_json(result: SignalAnalysisResult, path: Optional[str] = None) -> str:
    """Serialize result to JSON; optionally write to file."""
    payload: Dict[str, Any] = result.to_dict()
    text = json.dumps(payload, indent=2, default=str)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text
