"""
DEPRECATED — use src.dataset.WaveformGenerator and src.precise_dataset instead.

This module remains as a thin compatibility shim for old notebooks/scripts.
"""

from __future__ import annotations

import warnings

import numpy as np

from src.dataset import SignalDataset, WaveformGenerator

warnings.warn(
    "multiple_signal is deprecated. Use src.dataset.WaveformGenerator or src.precise_dataset.",
    DeprecationWarning,
    stacklevel=2,
)

_gen = WaveformGenerator()


class MultipleSignal:
    """Backward-compatible wrapper around WaveformGenerator."""

    def sin_wave(self, freq, length=256, amp=1, phase=0):
        return _gen.sine_wave(length, freq=freq, amp=amp, phase=phase)

    def cos_wave(self, freq, length=256, amp=1, phase=0):
        return _gen.cosine_wave(length, freq=freq, amp=amp, phase=phase)

    def triangle_wave(self, freq, length=256, amp=1, phase=0):
        return _gen.triangle_wave(length, freq=freq, amp=amp, phase=phase)

    def square_wave(self, freq, length=256, amp=1, phase=0):
        return _gen.square_wave(length, freq=freq, amp=amp, phase=phase)

    def sawtooth_wave(self, freq, length=256, amp=1, phase=0):
        return _gen.sawtooth_wave(length, freq=freq, amp=amp, phase=phase)

    def white_noise(self, length=256, amp=1):
        return _gen.white_noise(length, amp=amp)

    @staticmethod
    def plot_signal(signal, title="Signal"):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(16, 4))
        plt.plot(signal)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()


class CreateDataSet_1D_Array:
    """Backward-compatible dataset builder using SignalDataset."""

    def __init__(self, length=256):
        self.length = length
        self._ds = SignalDataset(num_samples=100, signal_length=length, seed=0)

    def generate_wave(self, label, length=256):
        name = "cosine" if label == "cosine" else label
        sig, _ = self._ds._generate_single_signal_with_meta(name)
        return sig

    def create_dataset_tranning_model(self, num_samples_per_class=1000):
        ds = SignalDataset(
            num_samples=num_samples_per_class * len(SignalDataset.WAVEFORM_TYPES),
            signal_length=self.length,
            seed=42,
        )
        return ds.signals, ds.labels
