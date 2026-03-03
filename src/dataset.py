"""
Create a comprehensive Dataset of 17 waveform types for Model training and testing.
Uses 1D array of the signal as input and the label as output.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
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


class SignalDataset(Dataset):
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
        self.signals, self.labels = self.generate_signals()
    
    def generate_signals(self):
        """Generate all signals with balanced classes"""
        signals = []
        labels = []
        
        samples_per_class = self.num_samples // len(self.WAVEFORM_TYPES)
        remainder = self.num_samples % len(self.WAVEFORM_TYPES)
        
        for label_idx, waveform_type in enumerate(self.WAVEFORM_TYPES):
            num_samples = samples_per_class + (1 if label_idx < remainder else 0)
            
            for _ in range(num_samples):
                signal = self._generate_single_signal(waveform_type)
                signals.append(signal)
                labels.append(label_idx)
        
        # Shuffle the dataset
        indices = np.random.permutation(len(signals))
        signals = np.array(signals)[indices]
        labels = np.array(labels)[indices]
        
        return signals, labels
    
    def _generate_single_signal(self, waveform_type):
        """Generate a single signal of specified type"""
        if waveform_type == 'sine':
            return self.generator.sine_wave(self.signal_length)
        elif waveform_type == 'square':
            return self.generator.square_wave(self.signal_length)
        elif waveform_type == 'triangle':
            return self.generator.triangle_wave(self.signal_length)
        elif waveform_type == 'sawtooth':
            return self.generator.sawtooth_wave(self.signal_length)
        elif waveform_type == 'pulse_triangle':
            return self.generator.pulse_triangle(self.signal_length)
        elif waveform_type == 'gaussian':
            return self.generator.gaussian_wave(self.signal_length)
        elif waveform_type == 'white_noise':
            return self.generator.white_noise(self.signal_length)
        elif waveform_type == 'pink_noise':
            return self.generator.pink_noise(self.signal_length)
        elif waveform_type == 'brownian_noise':
            return self.generator.brownian_noise(self.signal_length)
        elif waveform_type == 'exponential_decay':
            return self.generator.exponential_decay(self.signal_length)
        elif waveform_type == 'logarithmic_decay':
            return self.generator.logarithmic_decay(self.signal_length)
        elif waveform_type == 'step_function':
            return self.generator.step_function(self.signal_length)
        elif waveform_type == 'cosine':
            return self.generator.cosine_wave(self.signal_length)
        elif waveform_type == 'hyperbolic_tangent':
            return self.generator.hyperbolic_tangent_wave(self.signal_length)
        elif waveform_type == 'sigmoid':
            return self.generator.sigmoid_wave(self.signal_length)
        else:
            raise ValueError(f"Unknown waveform type: {waveform_type}")
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = torch.tensor(self.signals[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal, label
    
    def save_to_csv(self, filepath='waveform_dataset.csv'):
        """Save dataset to CSV file"""
        df = pd.DataFrame(self.signals)
        df['label'] = self.labels
        df['label_name'] = df['label'].apply(lambda x: self.WAVEFORM_TYPES[x])
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
    
    def get_class_distribution(self):
        """Return the distribution of classes in the dataset"""
        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = {self.WAVEFORM_TYPES[i]: count for i, count in zip(unique, counts)}
        return distribution
    