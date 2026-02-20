import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import tensorflow as tf

amp = 1

class MultipleSignal:
    def __init__(self):
        pass

    def sin_wave(self, freq, length=256, amp=1, phase=0):
        t = np.linspace(0, 1, length)
        return amp * np.sin(2 * np.pi * freq * t + phase)
    
    def cos_wave(self, freq, length=256, amp=1, phase=0):
        t = np.linspace(0, 1, length)
        return amp * np.cos(2 * np.pi * freq * t + phase)
    
    def triangle_wave(self, freq, length=256, amp=1, phase=0):
        t = np.linspace(0, 1, length)
        return amp * signal.sawtooth(2 * np.pi * freq * t + phase, width=0.5)

    def square_wave(self, freq, length=256, amp=1, phase=0):
        t = np.linspace(0, 1, length)
        return amp * signal.square(2 * np.pi * freq * t + phase)
    
    def pulse_rectangle_wave(self, freq, length=256, amp=1, phase=0):
        t = np.linspace(0, 1, length)
        return amp * signal.square(2 * np.pi * freq * t + phase, duty=0.3)
    
    def pulse_triangle_wave(self, freq, length=256, amp=1, phase=0):
        t = np.linspace(0, 1, length)
        return amp * signal.sawtooth(2 * np.pi * freq * t + phase, width=0.3)
    
    def gaussian_wave(self, freq, length=256, amp=1, phase=0):
        t = np.linspace(0, 1, length)
        return amp * signal.gausspulse(t - 0.5, fc=freq, bw=0.5) * np.cos(2 * np.pi * freq * t + phase)
    
    def recker_wave(self, freq, length=256, amp=1, phase=0):
        t = np.linspace(0, 1, length)
        return amp * signal.reckoner(t - 0.5, fc=freq, bw=0.5) * np.cos(2 * np.pi * freq * t + phase)
    
    def white_noise(self, length=256, amp=1):
        return amp * np.random.normal(0, 1, length)
    
    def sawtooth_wave(self, freq, length=256, amp=1, phase=0):
        t = np.linspace(0, 1, length)
        return amp * signal.sawtooth(2 * np.pi * freq * t + phase)
    
    
    def plot_signal(self, signal, title='Signal'):
        plt.figure(figsize=(16, 4))
        plt.plot(signal)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.show()

# create a large dataset of 1D arrays with different waveforms for model tranning
class CreateDataSet_1D_Array(MultipleSignal):
    def __init__(self, length=256):
        self.length = length

    def generate_wave(self, label, length=256):
        t = np.linspace(0, 1, length)
        if label == 'sine':   return self.sin_wave(freq=3, length=length)
        elif label == 'cosine': return self.cos_wave(freq=3, length=length)
        elif label == 'triangle': return self.triangle_wave(freq=3, length=length)
        elif label == 'square': return self.square_wave(freq=3, length=length)
        elif label == 'pulse_rectangle': return self.pulse_rectangle_wave(freq=3, length=length)
        elif label == 'pulse_triangle': return self.pulse_triangle_wave(freq=3, length=length)
        elif label == 'gaussian': return self.gaussian_wave(freq=3, length=length)
        elif label == 'recker': return self.recker_wave(freq=3, length=length)
        elif label == 'white_noise': return self.white_noise(length=length)
        elif label == 'sawtooth': return self.sawtooth_wave(freq=3, length=length)
        else: raise ValueError("Unsupported label")


    def plot_samples(self, num_samples=5):
        labels = ['sine', 'cosine', 'triangle', 'square', 'pulse_rectangle',
                  'pulse_triangle', 'gaussian', 'sawtooth', 'white_noise']

        fig, axes = plt.subplots(len(labels), num_samples, figsize=(16, len(labels) * 2))

        for i, label in enumerate(labels):
            for j in range(num_samples):
                signal = self.generate_wave(label=label, length=self.length)
                if len(labels) == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                ax.plot(signal)
                ax.set_title(f'{label} - Sample {j+1}')
                ax.grid(True)

        plt.tight_layout()
        plt.show()


    def create_dataset_tranning_model(self, num_samples_per_class=1000):
        labels = ['sine', 'cosine', 'triangle', 'square', 'pulse_rectangle',
                  'pulse_triangle', 'gaussian', 'sawtooth', 'white_noise']
        X = []
        y = []
        for i, label in enumerate(labels):
            for _ in range(num_samples_per_class):
                signal = self.generate_wave(label=label, length=self.length)
                X.append(signal)
                y.append(i)  # Use index as label
        return np.array(X), np.array(y)


if __name__ == "__main__":
    # Example usage
    # signal = MultipleSignal.sin_wave(freq=5, length=256, amp=1)
    # MultipleSignal.plot_signal(signal, title='Sine Wave')
    dataset = CreateDataSet_1D_Array(length=256)
    signal = dataset.generate_wave(label='sine', length=256)
    dataset.plot_signal(signal, title='Sine Wave')
