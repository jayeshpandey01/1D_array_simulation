import numpy as np
from scipy.fft import fft, fftfreq

def get_amplitude_stats(signal):
    """
    Calculate amplitude statistics for a 1D signal.
    """
    pk_pk = np.max(signal) - np.min(signal)
    rms = np.sqrt(np.mean(np.square(signal)))
    mean = np.mean(signal)
    std = np.std(signal)
    
    return {
        "Peak-to-Peak": pk_pk,
        "RMS": rms,
        "Mean": mean,
        "Std Dev": std
    }

def get_dominant_frequency(signal, sampling_rate=256):
    """
    Identify the dominant frequency using FFT.
    """
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1 / sampling_rate)
    
    # Use only positive frequencies
    idx = np.where(xf >= 0)
    xf_pos = xf[idx]
    yf_pos = np.abs(yf[idx])
    
    # Exclude DC component (0 Hz) for dominant frequency identification
    if len(xf_pos) > 1:
        dominant_idx = np.argmax(yf_pos[1:]) + 1
        dominant_freq = xf_pos[dominant_idx]
        magnitude = yf_pos[dominant_idx]
    else:
        dominant_freq = 0
        magnitude = 0
        
    return dominant_freq, xf_pos, yf_pos

def analyze_signal(signal, sampling_rate=256):
    """
    Comprehensive analysis of a signal.
    """
    stats = get_amplitude_stats(signal)
    dominant_freq, xf, yf = get_dominant_frequency(signal, sampling_rate)
    
    stats["Dominant Frequency"] = dominant_freq
    return stats, xf, yf
