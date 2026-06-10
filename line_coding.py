"""
=============================================================================
MODULE 2: Line Coding
=============================================================================
Generates waveforms for three binary line codes:
  • Polar NRZ      : bit 1 → +A,   bit 0 → –A
  • On-Off (OOK)   : bit 1 → +A,   bit 0 →  0
  • Bipolar (AMI)  : bit 1 alternates ±A, bit 0 → 0

Each function returns sample-per-bit upsampled waveforms ready for pulse
shaping and eye diagram generation.
=============================================================================
"""

import numpy as np


# ---------------------------------------------------------------------------
# 2A. Polar NRZ
# ---------------------------------------------------------------------------
def polar_nrz(bits, samples_per_bit=64, amplitude=1.0):
    """
    Polar NRZ line code.
      1 → +amplitude
      0 → -amplitude

    Parameters
    ----------
    bits            : array-like of 0/1
    samples_per_bit : int  (oversampling rate / symbol period in samples)
    amplitude       : float

    Returns
    -------
    waveform : ndarray – continuous waveform (upsampled)
    symbols  : ndarray – symbol sequence (±1)
    """
    bits = np.asarray(bits)
    symbols = np.where(bits == 1, amplitude, -amplitude)
    waveform = np.repeat(symbols, samples_per_bit)
    return waveform.astype(float), symbols


# ---------------------------------------------------------------------------
# 2B. On-Off Keying (OOK / Unipolar NRZ)
# ---------------------------------------------------------------------------
def on_off_keying(bits, samples_per_bit=64, amplitude=1.0):
    """
    On-Off (unipolar) line code.
      1 → +amplitude
      0 →  0

    Returns
    -------
    waveform : ndarray
    symbols  : ndarray
    """
    bits = np.asarray(bits)
    symbols = np.where(bits == 1, amplitude, 0.0)
    waveform = np.repeat(symbols, samples_per_bit)
    return waveform.astype(float), symbols


# ---------------------------------------------------------------------------
# 2C. Bipolar / AMI (Alternate Mark Inversion)
# ---------------------------------------------------------------------------
def bipolar_ami(bits, samples_per_bit=64, amplitude=1.0):
    """
    Bipolar AMI line code.
      0 →  0
      1 → alternating +amplitude / -amplitude

    Returns
    -------
    waveform : ndarray
    symbols  : ndarray
    """
    bits = np.asarray(bits)
    symbols = np.zeros(len(bits))
    polarity = 1
    for i, b in enumerate(bits):
        if b == 1:
            symbols[i] = polarity * amplitude
            polarity *= -1          # flip for next 1
    waveform = np.repeat(symbols, samples_per_bit)
    return waveform.astype(float), symbols


# ---------------------------------------------------------------------------
# 2D. Convenience: encode bits with a chosen line code
# ---------------------------------------------------------------------------
def line_encode(bits, code='polar', samples_per_bit=64, amplitude=1.0):
    """
    Unified interface to all line codes.

    Parameters
    ----------
    code : str – 'polar' | 'onoff' | 'bipolar'

    Returns
    -------
    waveform : ndarray
    symbols  : ndarray
    """
    code = code.lower().strip()
    if code in ('polar', 'nrz', 'polar_nrz'):
        return polar_nrz(bits, samples_per_bit, amplitude)
    elif code in ('onoff', 'ook', 'on_off', 'unipolar'):
        return on_off_keying(bits, samples_per_bit, amplitude)
    elif code in ('bipolar', 'ami'):
        return bipolar_ami(bits, samples_per_bit, amplitude)
    else:
        raise ValueError(f"Unknown line code: '{code}'. "
                         "Choose 'polar', 'onoff', or 'bipolar'.")
