"""
=============================================================================
MODULE 1: Sampling and Quantization
=============================================================================
Functions to:
  - Generate a continuous-time analog source signal
  - Sample it at a given rate
  - Quantize uniformly into L levels
  - Convert quantized samples → PCM bits
  - Compute SQNR
=============================================================================
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1A. Source signal generator
# ---------------------------------------------------------------------------
def generate_source_signal(duration=1.0, fs_analog=10000, freqs=(1, 3),
                            amplitudes=(1.0, 1.0)):
    """
    Generate a continuous-time multi-tone sinusoidal signal.

    Parameters
    ----------
    duration    : float  – signal duration in seconds
    fs_analog   : int    – high-resolution sampling rate (acts as 'analog')
    freqs       : tuple  – frequency components in Hz
    amplitudes  : tuple  – amplitude of each component

    Returns
    -------
    t      : ndarray – time axis
    signal : ndarray – signal values
    """
    t = np.arange(0, duration, 1.0 / fs_analog)
    signal = np.zeros(len(t))
    for f, A in zip(freqs, amplitudes):
        signal += A * np.sin(2 * np.pi * f * t)
    return t, signal


# ---------------------------------------------------------------------------
# 1B. Uniform quantizer  (mirrors uniquan.m from the textbook)
# ---------------------------------------------------------------------------
def uniform_quantize(signal, L):
    """
    Uniformly quantize a signal into L levels.

    Parameters
    ----------
    signal : ndarray – input signal
    L      : int     – number of quantization levels (power of 2 recommended)

    Returns
    -------
    q_out  : ndarray – quantized signal
    delta  : float   – quantization step size
    sqnr   : float   – signal-to-quantization-noise ratio in dB
    """
    sig_max = np.max(signal)
    sig_min = np.min(signal)
    delta = (sig_max - sig_min) / L

    # Quantization levels (mid-rise)
    q_levels = np.linspace(sig_min + delta / 2,
                           sig_max - delta / 2, L)

    # Map each sample to nearest level
    sigp = (signal - sig_min) / delta + 0.5     # shift to [0.5, L+0.5]
    qindex = np.round(sigp).astype(int)
    qindex = np.clip(qindex, 1, L) - 1          # 0-based index

    q_out = q_levels[qindex]

    # SQNR in dB
    noise = signal - q_out
    if np.linalg.norm(noise) == 0:
        sqnr = np.inf
    else:
        sqnr = 20 * np.log10(np.linalg.norm(signal) /
                             np.linalg.norm(noise))
    return q_out, delta, sqnr


# ---------------------------------------------------------------------------
# 1C. Sampler
# ---------------------------------------------------------------------------
def sample_signal(t, signal, fs_sample):
    """
    Sample the continuous signal at rate fs_sample.

    Parameters
    ----------
    t         : ndarray – time axis of the 'analog' signal
    signal    : ndarray – analog signal values
    fs_sample : float   – desired sampling frequency (Hz)

    Returns
    -------
    t_s    : ndarray – sample time instants
    s_out  : ndarray – sampled values
    """
    fs_analog = 1.0 / (t[1] - t[0])
    n_factor = int(round(fs_analog / fs_sample))
    s_out = signal[::n_factor]
    t_s = t[::n_factor]
    return t_s, s_out


# ---------------------------------------------------------------------------
# 1D. PCM encoder: quantized samples → bit stream
# ---------------------------------------------------------------------------
def pcm_encode(q_out, L):
    """
    Convert quantized samples to a binary PCM bit stream.

    Parameters
    ----------
    q_out : ndarray – quantized signal samples
    L     : int     – number of quantization levels

    Returns
    -------
    bits        : ndarray (int, 0/1) – PCM bit stream
    bits_per_sample : int
    """
    bits_per_sample = int(np.ceil(np.log2(L)))
    sig_min = np.min(q_out)
    sig_max = np.max(q_out)
    delta = (sig_max - sig_min) / L if L > 1 else 1.0

    # Convert each sample to an integer index [0, L-1]
    indices = np.round((q_out - sig_min) / delta).astype(int)
    indices = np.clip(indices, 0, L - 1)

    bits = []
    for idx in indices:
        b = [(idx >> (bits_per_sample - 1 - k)) & 1
             for k in range(bits_per_sample)]
        bits.extend(b)
    return np.array(bits, dtype=int), bits_per_sample


# ---------------------------------------------------------------------------
# 1E. PCM decoder: bit stream → quantized samples
# ---------------------------------------------------------------------------
def pcm_decode(bits, bits_per_sample, L, sig_min, sig_max):
    """
    Decode a PCM bit stream back to quantized amplitude values.

    Parameters
    ----------
    bits            : ndarray – received bit stream (0/1)
    bits_per_sample : int
    L               : int     – quantization levels
    sig_min         : float   – original signal minimum
    sig_max         : float   – original signal maximum

    Returns
    -------
    q_recv : ndarray – reconstructed amplitude samples
    """
    delta = (sig_max - sig_min) / L
    n_samples = len(bits) // bits_per_sample
    q_recv = np.zeros(n_samples)

    for i in range(n_samples):
        b = bits[i * bits_per_sample:(i + 1) * bits_per_sample]
        idx = sum(int(b[k]) << (bits_per_sample - 1 - k)
                  for k in range(bits_per_sample))
        idx = min(idx, L - 1)
        q_recv[i] = sig_min + (idx + 0.5) * delta

    return q_recv


# ---------------------------------------------------------------------------
# 1F. High-level pipeline: analog signal → bits
# ---------------------------------------------------------------------------
def analog_to_bits(duration=1.0, fs_analog=10000, fs_sample=50,
                   L=16, freqs=(1, 3), amplitudes=(1.0, 1.0)):
    """
    Full Analog-to-Digital conversion pipeline.

    Returns a dictionary with all intermediate results so the test
    code can display / analyse each stage.
    """
    t, signal = generate_source_signal(duration, fs_analog, freqs, amplitudes)
    t_s, s_out = sample_signal(t, signal, fs_sample)
    q_out, delta, sqnr = uniform_quantize(s_out, L)
    bits, bps = pcm_encode(q_out, L)

    return {
        "t": t,
        "signal": signal,
        "t_s": t_s,
        "sampled": s_out,
        "quantized": q_out,
        "delta": delta,
        "sqnr_dB": sqnr,
        "bits": bits,
        "bits_per_sample": bps,
        "L": L,
        "fs_analog": fs_analog,
        "fs_sample": fs_sample,
        "sig_min": float(np.min(s_out)),
        "sig_max": float(np.max(s_out)),
    }
