"""
Create a comprehensive Dataset of 17 waveform types for Model training and testing.
Uses 1D array of the signal as input and the label as output.
"""

import numpy as np
from scipy import signal as scipy_signal
import pandas as pd
import os


class WaveformGenerator:
    """Generator for 17 different waveform types with randomized parameters"""
    
    @staticmethod
    def sine_wave(length, freq=None, amp=None, phase=None):
        freq = freq or np.random.uniform(1, 5)
        amp = amp or np.random.uniform(0.5, 1.5)
        phase = phase or np.random.uniform(0, 2 * np.pi)
        t = np.linspace(0, 1, length)
        return amp * np.sin(2 * np.pi * freq * t + phase)
    
    @staticmethod
    def square_wave(length, freq=None, amp=None, phase=None):
        freq = freq or np.random.uniform(1, 5)
        amp = amp or np.random.uniform(0.5, 1.5)
        phase = phase or np.random.uniform(0, 2 * np.pi)
        t = np.linspace(0, 1, length)
        return amp * scipy_signal.square(2 * np.pi * freq * t + phase)
    
    @staticmethod
    def triangle_wave(length, freq=None, amp=None, phase=None):
        freq = freq or np.random.uniform(1, 5)
        amp = amp or np.random.uniform(0.5, 1.5)
        phase = phase or np.random.uniform(0, 2 * np.pi)
        t = np.linspace(0, 1, length)
        return amp * scipy_signal.sawtooth(2 * np.pi * freq * t + phase, width=0.5)
    
    @staticmethod
    def sawtooth_wave(length, freq=None, amp=None, phase=None):
        freq = freq or np.random.uniform(1, 5)
        amp = amp or np.random.uniform(0.5, 1.5)
        phase = phase or np.random.uniform(0, 2 * np.pi)
        t = np.linspace(0, 1, length)
        return amp * scipy_signal.sawtooth(2 * np.pi * freq * t + phase)
    
    @staticmethod
    def pulse_triangle(length, freq=None, amp=None, phase=None):
        freq = freq or np.random.uniform(1, 5)
        amp = amp or np.random.uniform(0.5, 1.5)
        phase = phase or np.random.uniform(0, 2 * np.pi)
        duty = np.random.uniform(0.2, 0.4)
        t = np.linspace(0, 1, length)
        return amp * scipy_signal.sawtooth(2 * np.pi * freq * t + phase, width=duty)
    
    @staticmethod
    def gaussian_wave(length, freq=None, amp=None):
        freq = freq or np.random.uniform(3, 8)
        amp = amp or np.random.uniform(0.5, 1.5)
        bw = np.random.uniform(0.3, 0.7)
        t = np.linspace(-0.5, 0.5, length)
        return amp * scipy_signal.gausspulse(t, fc=freq, bw=bw)
    
    @staticmethod
    def white_noise(length, amp=None):
        amp = amp or np.random.uniform(0.5, 1.5)
        return amp * np.random.normal(0, 1, length)
    
    @staticmethod
    def pink_noise(length, amp=None):
        amp = amp or np.random.uniform(0.5, 1.5)
        white = np.random.randn(length)
        # Simple pink noise approximation using cumulative sum
        pink = np.cumsum(white)
        pink = pink - np.mean(pink)
        pink = pink / (np.std(pink) + 1e-10)
        return amp * pink
    
    @staticmethod
    def brownian_noise(length, amp=None):
        amp = amp or np.random.uniform(0.5, 1.5)
        white = np.random.randn(length)
        brownian = np.cumsum(np.cumsum(white))
        brownian = brownian - np.mean(brownian)
        brownian = brownian / (np.std(brownian) + 1e-10)
        return amp * brownian
    
    @staticmethod
    def exponential_decay(length, amp=None):
        amp = amp or np.random.uniform(0.5, 1.5)
        decay_rate = np.random.uniform(2, 8)
        t = np.linspace(0, 1, length)
        return amp * np.exp(-decay_rate * t)
    
    @staticmethod
    def logarithmic_decay(length, amp=None):
        amp = amp or np.random.uniform(0.5, 1.5)
        t = np.linspace(1, 10, length)
        log_signal = np.log(t)
        log_signal = log_signal / np.max(np.abs(log_signal))
        return amp * (1 - log_signal)
    
    @staticmethod
    def step_function(length, amp=None):
        amp = amp or np.random.uniform(0.5, 1.5)
        num_steps = np.random.randint(2, 6)
        step_positions = np.sort(np.random.choice(length, num_steps - 1, replace=False))
        signal = np.zeros(length)
        step_values = np.random.uniform(-1, 1, num_steps)
        
        prev_pos = 0
        for i, pos in enumerate(step_positions):
            signal[prev_pos:pos] = step_values[i]
            prev_pos = pos
        signal[prev_pos:] = step_values[-1]
        return amp * signal
    
    @staticmethod
    def cosine_wave(length, freq=None, amp=None, phase=None):
        freq = freq or np.random.uniform(1, 5)
        amp = amp or np.random.uniform(0.5, 1.5)
        phase = phase or np.random.uniform(0, 2 * np.pi)
        t = np.linspace(0, 1, length)
        return amp * np.cos(2 * np.pi * freq * t + phase)
    
    @staticmethod
    def hyperbolic_tangent_wave(length, freq=None, amp=None):
        freq = freq or np.random.uniform(1, 5)
        amp = amp or np.random.uniform(0.5, 1.5)
        t = np.linspace(-3, 3, length)
        return amp * np.tanh(freq * np.sin(t))
    
    @staticmethod
    def sigmoid_wave(length, freq=None, amp=None):
        freq = freq or np.random.uniform(1, 5)
        amp = amp or np.random.uniform(0.5, 1.5)
        t = np.linspace(-6, 6, length)
        return amp * (2 / (1 + np.exp(-freq * np.sin(t))) - 1)


class SignalDataset:
    """
    PyTorch Dataset for 17 waveform types with configurable size and length.
    """
    
    WAVEFORM_TYPES = [
        'sine', 'square', 'triangle', 'sawtooth', 'pulse_triangle',
        'gaussian', 'white_noise', 'pink_noise', 'brownian_noise',
        'exponential_decay', 'logarithmic_decay', 'step_function',
        'cosine', 'hyperbolic_tangent', 'sigmoid'
    ]
    
    def __init__(self, num_samples=10000, signal_length=256, seed=None):
        """
        Waveform dataset; __getitem__ returns torch tensors when torch is installed.

        Args:
            num_samples: Total number of samples to generate
            signal_length: Length of each signal (number of time steps)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.num_samples = num_samples
        self.signal_length = signal_length
        self.generator = WaveformGenerator()
        self.metadata = []
        self.signals, self.labels = self.generate_signals()
    
    def generate_signals(self):
        """Generate all signals with balanced classes"""
        signals = []
        labels = []
        metadata = []

        samples_per_class = self.num_samples // len(self.WAVEFORM_TYPES)
        remainder = self.num_samples % len(self.WAVEFORM_TYPES)

        for label_idx, waveform_type in enumerate(self.WAVEFORM_TYPES):
            num_samples = samples_per_class + (1 if label_idx < remainder else 0)

            for _ in range(num_samples):
                signal, meta = self._generate_single_signal_with_meta(waveform_type)
                signals.append(signal)
                labels.append(label_idx)
                meta["label"] = label_idx
                meta["label_name"] = waveform_type
                metadata.append(meta)
        
        # Shuffle the dataset
        indices = np.random.permutation(len(signals))
        signals = np.array(signals)[indices]
        labels = np.array(labels)[indices]
        metadata = [metadata[i] for i in indices]
        self.metadata = metadata

        return signals, labels

    def _draw_periodic_params(self):
        return {
            "frequency": float(np.random.uniform(1, 5)),
            "amplitude": float(np.random.uniform(0.5, 1.5)),
            "phase": float(np.random.uniform(0, 2 * np.pi)),
        }

    def _generate_single_signal_with_meta(self, waveform_type):
        """Generate a single signal and metadata dict with ground-truth parameters."""
        length = self.signal_length
        meta = {
            "wave_type": waveform_type,
            "frequency": np.nan,
            "amplitude": np.nan,
            "phase": np.nan,
            "snr_db": np.nan,
        }

        if waveform_type in ("sine", "cosine", "square", "triangle", "sawtooth", "pulse_triangle"):
            p = self._draw_periodic_params()
            meta.update(p)
            f, a, ph = p["frequency"], p["amplitude"], p["phase"]
            if waveform_type == "sine":
                sig = self.generator.sine_wave(length, freq=f, amp=a, phase=ph)
            elif waveform_type == "cosine":
                sig = self.generator.cosine_wave(length, freq=f, amp=a, phase=ph)
            elif waveform_type == "square":
                sig = self.generator.square_wave(length, freq=f, amp=a, phase=ph)
            elif waveform_type == "triangle":
                sig = self.generator.triangle_wave(length, freq=f, amp=a, phase=ph)
            elif waveform_type == "sawtooth":
                sig = self.generator.sawtooth_wave(length, freq=f, amp=a, phase=ph)
            else:
                sig = self.generator.pulse_triangle(length, freq=f, amp=a, phase=ph)
        elif waveform_type == "gaussian":
            a = float(np.random.uniform(0.5, 1.5))
            f = float(np.random.uniform(3, 8))
            meta["amplitude"], meta["frequency"] = a, f
            sig = self.generator.gaussian_wave(length, freq=f, amp=a)
        elif waveform_type == "white_noise":
            a = float(np.random.uniform(0.5, 1.5))
            meta["amplitude"] = a
            sig = self.generator.white_noise(length, amp=a)
        elif waveform_type == "pink_noise":
            a = float(np.random.uniform(0.5, 1.5))
            meta["amplitude"] = a
            sig = self.generator.pink_noise(length, amp=a)
        elif waveform_type == "brownian_noise":
            a = float(np.random.uniform(0.5, 1.5))
            meta["amplitude"] = a
            sig = self.generator.brownian_noise(length, amp=a)
        elif waveform_type == "exponential_decay":
            a = float(np.random.uniform(0.5, 1.5))
            meta["amplitude"] = a
            sig = self.generator.exponential_decay(length, amp=a)
        elif waveform_type == "logarithmic_decay":
            a = float(np.random.uniform(0.5, 1.5))
            meta["amplitude"] = a
            sig = self.generator.logarithmic_decay(length, amp=a)
        elif waveform_type == "step_function":
            a = float(np.random.uniform(0.5, 1.5))
            meta["amplitude"] = a
            sig = self.generator.step_function(length, amp=a)
        elif waveform_type == "hyperbolic_tangent":
            a = float(np.random.uniform(0.5, 1.5))
            f = float(np.random.uniform(1, 5))
            meta["amplitude"], meta["frequency"] = a, f
            sig = self.generator.hyperbolic_tangent_wave(length, freq=f, amp=a)
        elif waveform_type == "sigmoid":
            a = float(np.random.uniform(0.5, 1.5))
            f = float(np.random.uniform(1, 5))
            meta["amplitude"], meta["frequency"] = a, f
            sig = self.generator.sigmoid_wave(length, freq=f, amp=a)
        else:
            raise ValueError(f"Unknown waveform type: {waveform_type}")

        return sig, meta

    def _generate_single_signal(self, waveform_type):
        """Generate a single signal of specified type (backward compatible)."""
        signal, _ = self._generate_single_signal_with_meta(waveform_type)
        return signal
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        import torch

        signal = torch.tensor(self.signals[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal, label
    
    def save_to_csv(self, filepath='waveform_dataset.csv'):
        """Save dataset to CSV with signal columns and ground-truth metadata."""
        from src.signal_io import signals_to_dataframe

        meta_df = pd.DataFrame(self.metadata) if self.metadata else pd.DataFrame({
            "label": self.labels,
            "label_name": [self.WAVEFORM_TYPES[i] for i in self.labels],
        })
        df = signals_to_dataframe(self.signals, meta_df)
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
    
    def get_class_distribution(self):
        """Return the distribution of classes in the dataset"""
        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = {self.WAVEFORM_TYPES[i]: count for i, count in zip(unique, counts)}
        return distribution
    