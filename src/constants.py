"""Shared constants for 1D signal simulation and analysis."""

SAMPLE_RATE = 256  # Hz — matches 256 samples over t in [0, 1]
SIGNAL_LENGTH = 256
DURATION_SEC = 1.0

PERIODIC_WAVE_TYPES = (
    "sine",
    "cosine",
    "square",
    "triangle",
    "sawtooth",
    "pulse_triangle",
)

NON_PERIODIC_WAVE_TYPES = (
    "gaussian",
    "white_noise",
    "pink_noise",
    "brownian_noise",
    "exponential_decay",
    "logarithmic_decay",
    "step_function",
    "hyperbolic_tangent",
    "sigmoid",
)

ALL_WAVE_TYPES = PERIODIC_WAVE_TYPES + NON_PERIODIC_WAVE_TYPES
