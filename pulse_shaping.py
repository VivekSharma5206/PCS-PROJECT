"""
=============================================================================
MODULE 3: Pulse Shaping
=============================================================================
Implements pulse shapes:
  • NRZ  rectangular pulse (width = T)
  • RZ   rectangular pulse (width = T/2)
  • Sinc pulse  (ideal, band-limited)
  • Raised-Cosine (RC) pulse – Eq. 7.36 from Lathi & Ding textbook

      p(t) = Rb * cos(π Rb t) / (1 – 4 Rb² t²)  · sinc(π Rb t)

  Convolution of symbol impulse train with chosen pulse gives shaped waveform.
=============================================================================
"""

import numpy as np
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# 3A. Rectangular NRZ pulse  (width = T samples)
# ---------------------------------------------------------------------------
def pulse_nrz(samples_per_bit):
    """Full-width rectangular pulse (NRZ)."""
    return np.ones(samples_per_bit)


# ---------------------------------------------------------------------------
# 3B. Rectangular RZ pulse  (width = T/2 samples)
# ---------------------------------------------------------------------------
def pulse_rz(samples_per_bit):
    """Half-width rectangular pulse (RZ). Centred in symbol period."""
    p = np.zeros(samples_per_bit)
    half = samples_per_bit // 2
    quarter = samples_per_bit // 4
    p[quarter: quarter + half] = 1.0
    return p


# ---------------------------------------------------------------------------
# 3C. Sinc pulse  (ideal low-pass pulse)
# ---------------------------------------------------------------------------
def pulse_sinc(samples_per_bit, n_periods=8):
    """
    Ideal sinc pulse  p(t) = sinc(t / T)  truncated to ±n_periods symbols.

    Parameters
    ----------
    samples_per_bit : int  – oversampling rate (T in samples)
    n_periods       : int  – one-sided truncation length in symbol periods
    """
    total = 2 * n_periods * samples_per_bit + 1
    t = np.arange(-n_periods * samples_per_bit,
                  n_periods * samples_per_bit + 1)
    # sinc(x) = sin(pi x)/(pi x), NumPy sinc is normalised: sinc(x)=sin(pi x)/(pi x)
    p = np.sinc(t / samples_per_bit)
    return p


# ---------------------------------------------------------------------------
# 3D. Raised-Cosine pulse  (Eq. 7.36 in Lathi & Ding)
#
#   p(t) = Rb * cos(π Rb t) / (1 – 4 Rb² t²) · sinc(π Rb t)
#
#   where Rb = 1/T (bit rate), so Rb·t = t/T (normalised time).
#   With roll-off factor β = 1 this corresponds to full raised cosine.
# ---------------------------------------------------------------------------
def pulse_rc(samples_per_bit, rolloff=0.5, n_periods=8):
    """
    Raised-Cosine pulse as in Eq. 7.36 of Lathi & Ding.

      p(t) = Rb · cos(π Rb t) / (1 – 4 Rb² t²) · sinc(π Rb t)

    For numerical stability the formula is evaluated carefully at t = 0
    and t = ±T/(2β) (where the denominator is zero but the limit is finite).

    Parameters
    ----------
    samples_per_bit : int   – oversampling / symbol period in samples
    rolloff         : float – roll-off factor β  ∈ [0, 1]
    n_periods       : int   – one-sided truncation length

    Returns
    -------
    p : ndarray – pulse samples, length = 2*n_periods*samples_per_bit + 1
    """
    T = samples_per_bit                         # symbol period in samples
    t = np.arange(-n_periods * T, n_periods * T + 1, dtype=float)
    t_norm = t / T                              # t/T  (dimensionless)

    # --- numerator: sinc(t/T) part ---
    sinc_part = np.sinc(t_norm)                 # np.sinc is normalised

    # --- RC shaping factor: cos(π β t/T) / (1 – 4β²(t/T)²) ---
    beta = rolloff
    numerator = np.cos(np.pi * beta * t_norm)
    denominator = 1.0 - (2.0 * beta * t_norm) ** 2

    # Handle singularities where denominator ≈ 0
    # Limit at |t/T| = 1/(2β):  factor → (π/4) · sinc(1/(2β))
    tol = 1e-6
    singular = np.abs(np.abs(t_norm) - 1.0 / (2.0 * beta + 1e-12)) < tol

    rc_factor = np.where(singular,
                         np.pi / 4.0,
                         numerator / (denominator + tol * singular))

    p = sinc_part * rc_factor
    # Normalise so that peak = 1
    if np.max(np.abs(p)) > 0:
        p /= np.max(np.abs(p))
    return p


# ---------------------------------------------------------------------------
# 3E. Frequency-domain representation of a pulse
# ---------------------------------------------------------------------------
def pulse_spectrum(pulse, samples_per_bit, fs=None):
    """
    Compute the two-sided spectrum of a pulse.

    Returns
    -------
    freqs : ndarray – frequency axis (normalised or in Hz if fs given)
    mag   : ndarray – magnitude spectrum
    """
    N = len(pulse)
    P = np.fft.fftshift(np.fft.fft(pulse, n=max(N, 4096)))
    mag = np.abs(P)
    if fs is None:
        fs = samples_per_bit          # normalise to symbol rate
    freqs = np.fft.fftshift(np.fft.fftfreq(len(P), d=1.0 / fs))
    return freqs, mag


# ---------------------------------------------------------------------------
# 3F. Apply pulse shaping to a symbol stream
# ---------------------------------------------------------------------------
def apply_pulse_shaping(symbols, pulse, samples_per_bit):
    """
    Convolve an impulse train of symbols with the chosen pulse.

    Parameters
    ----------
    symbols         : ndarray – symbol values (one per bit)
    pulse           : ndarray – pulse shape (in samples)
    samples_per_bit : int

    Returns
    -------
    shaped : ndarray – pulse-shaped waveform (trimmed to symbol duration)
    """
    # Build impulse train
    impulse_train = np.zeros(len(symbols) * samples_per_bit)
    for i, s in enumerate(symbols):
        impulse_train[i * samples_per_bit] = s

    shaped = fftconvolve(impulse_train, pulse, mode='full')

    # Trim to align with symbol start (pulse delay = half pulse length)
    delay = len(pulse) // 2
    shaped = shaped[delay: delay + len(impulse_train)]
    return shaped


# ---------------------------------------------------------------------------
# 3G. Convenience: get pulse by name
# ---------------------------------------------------------------------------
def get_pulse(name, samples_per_bit, rolloff=0.5, n_periods=8):
    """
    Return a pulse array given its name.

    name choices: 'nrz', 'rz', 'sinc', 'rc'
    """
    name = name.lower().strip()
    if name == 'nrz':
        return pulse_nrz(samples_per_bit)
    elif name == 'rz':
        return pulse_rz(samples_per_bit)
    elif name == 'sinc':
        return pulse_sinc(samples_per_bit, n_periods)
    elif name in ('rc', 'raised_cosine', 'raised-cosine'):
        return pulse_rc(samples_per_bit, rolloff, n_periods)
    else:
        raise ValueError(f"Unknown pulse: '{name}'. "
                         "Choose 'nrz', 'rz', 'sinc', or 'rc'.")
