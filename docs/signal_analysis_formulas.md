# Signal Analysis Algorithms and Formula Reference

This document provides the mathematical formulas and algorithmic steps used for identifying frequency and amplitude from 1D array signal datasets.

## 1. Amplitude Statistics

### Peak-to-Peak Amplitude
The total range of the signal from its lowest to highest points. Useful for understanding the maximum swing.
$$A_{pk-pk} = \max(x) - \min(x)$$

### RMS (Root Mean Square) Amplitude
A measure of the signal's effective power, representing the square root of the arithmetic mean of the squares of the values.
$$A_{rms} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}$$

### Standard Deviation ($\sigma$)
Quantifies the amount of variation or dispersion of the signal values around the mean.
$$\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2}$$

---

## 2. Frequency Identification (FFT)

### Discrete Fourier Transform (DFT)
The algorithm transforms the signal from the time domain to the frequency domain to identify periodic components.
$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi}{N} kn}$$

### Dominant Frequency Identification
The frequency associated with each FFT index $k$ is calculated as:
$$f_k = \frac{k \cdot F_s}{N}$$
where:
- **$F_s$**: Sampling rate (e.g., 256 Hz)
- **$N$**: Total number of samples (e.g., 256)
- **$k$**: Bin index (frequency index)

**Algorithm Steps**:
1. Compute the FFT of the 1D signal array.
2. Calculate the magnitude of the FFT output ($|X[k]|$).
3. Focus on positive frequencies (indices $1$ to $N/2$).
4. Identify the index $k_{max}$ with the largest magnitude.
5. Convert $k_{max}$ to Hz using the formula above to get the **Dominant Frequency**.

---

## 3. Waveform Generation Formulas

The following formulas are used in `src/dataset.py` to generate the 1D signal arrays for training and testing. Signals are typically generated over the interval $t \in [0, 1]$ with $n=256$ samples.

### Periodic Waveforms
- **Sine Wave**: $x(t) = A \sin(2\pi f t + \phi)$
- **Cosine Wave**: $x(t) = A \cos(2\pi f t + \phi)$
- **Square Wave**: $x(t) = A \cdot \operatorname{sgn}(\sin(2\pi f t + \phi))$
- **Sawtooth Wave**: $x(t) = A \cdot (2(\frac{t}{T} - \lfloor \frac{t}{T} + \frac{1}{2} \rfloor))$
- **Triangle Wave**: $x(t) = A \cdot |2(\frac{t}{T} - \lfloor \frac{t}{T} + \frac{1}{2} \rfloor)|$

### Noise and Stochastic Signals
- **White Noise**: $x(t) = A \cdot \mathcal{N}(0, 1)$ (Gaussian distribution)
- **Pink Noise**: $x(t) = A \cdot \operatorname{normalize}(\sum \mathcal{N}(0, 1))$ (1/$f$ approximation)
- **Brownian Noise**: $x(t) = A \cdot \operatorname{normalize}(\sum \sum \mathcal{N}(0, 1))$ (1/$f^2$ approximation)

### Mathematical Funstions
- **Exponential Decay**: $x(t) = A \cdot e^{-rt}$ (where $r$ is the decay rate)
- **Logarithmic Decay**: $x(t) = A \cdot (1 - \operatorname{normalize}(\ln(t)))$
- **Hyperbolic Tangent**: $x(t) = A \cdot \tanh(f \sin(t))$
- **Sigmoid Wave**: $x(t) = A \cdot \left(\frac{2}{1+e^{-f\sin(t)}} - 1\right)$

---

## 4. General Statistics

### Mean ($\mu$)
The arithmetic average of all signal values, also known as the DC level or offset.
$$\mu = \frac{1}{n} \sum_{i=1}^{n} x_i$$

### Median
The middle value of the signal when sorted in ascending order.
$$\operatorname{Median}(x) = \begin{cases} 
x_{(n+1)/2} & \text{if } n \text{ is odd} \\
\frac{x_{n/2} + x_{n/2 + 1}}{2} & \text{if } n \text{ is even}
\end{cases}$$

### Variance ($\sigma^2$)
The average of the squared differences from the Mean.
$$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2$$

### Skewness
Measures the asymmetry of the probability distribution of the signal values about its mean.
$$\operatorname{Skewness} = \frac{\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^3}{\left(\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2\right)^{3/2}}$$

### Kurtosis
Measures the "tailedness" of the probability distribution of the signal values.
$$\operatorname{Kurtosis} = \frac{\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^4}{\left(\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2\right)^2}$$


