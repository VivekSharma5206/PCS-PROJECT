"""
=============================================================================
MODULE 4: AWGN Channel + Matched Filter
=============================================================================
  • add_awgn_noise  – adds Gaussian noise at a target SNR (dB)
  • matched_filter  – correlates received signal with pulse template
  • sample_and_decide – samples at symbol instants and threshold-compares
  • compute_ber     – counts bit errors
  • ber_vs_snr      – sweeps SNR and returns BER curve
=============================================================================
"""

import numpy as np
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# 4A. AWGN channel
# ---------------------------------------------------------------------------
def add_awgn_noise(signal, snr_dB):
    """
    Add white Gaussian noise to a signal at a specified SNR.

    Parameters
    ----------
    signal : ndarray – transmitted signal
    snr_dB : float   – desired signal-to-noise ratio in dB

    Returns
    -------
    noisy  : ndarray – received signal with AWGN
    noise  : ndarray – noise vector that was added
    """
    signal = np.asarray(signal, dtype=float)
    P_signal = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_dB / 10.0)
    P_noise = P_signal / snr_linear
    noise = np.sqrt(P_noise) * np.random.randn(len(signal))
    noisy = signal + noise
    return noisy, noise


# ---------------------------------------------------------------------------
# 4B. Full transmitter: symbols → shaped signal using FULL convolution
# ---------------------------------------------------------------------------
def shape_signal_full(symbols, pulse, samples_per_bit):
    """
    Build impulse train and convolve with pulse ('full' mode).
    Returns the full-length convolution so alignment is preserved.
    """
    impulse_train = np.zeros(len(symbols) * samples_per_bit)
    for i, s in enumerate(symbols):
        impulse_train[i * samples_per_bit] = s
    return fftconvolve(impulse_train, pulse, mode='full')


# ---------------------------------------------------------------------------
# 4C. Matched filter (full convolution)
# ---------------------------------------------------------------------------
def matched_filter(received, pulse):
    """
    Apply a matched filter: convolve received with time-reversed pulse.

    For 'full' convolutions the MF output peak for symbol k appears at:
        index = (len(pulse) - 1) + k * samples_per_bit

    Parameters
    ----------
    received : ndarray – noisy signal
    pulse    : ndarray – TX pulse shape

    Returns
    -------
    mf_out        : ndarray – MF output (full conv length)
    symbol_offset : int     – sample index of the 1st symbol peak in mf_out
    """
    h = pulse[::-1]
    mf_out = fftconvolve(received, h, mode='full')
    symbol_offset = len(pulse) - 1
    return mf_out, symbol_offset


# ---------------------------------------------------------------------------
# 4D. Sampler and decision device
# ---------------------------------------------------------------------------
def sample_and_decide(mf_out, symbols_tx, samples_per_bit,
                      symbol_offset, code='polar', threshold=None):
    """
    Sample the MF output at symbol peaks and decide bits.

    Parameters
    ----------
    mf_out          : ndarray – matched filter full output
    symbols_tx      : ndarray – transmitted symbols
    samples_per_bit : int
    symbol_offset   : int     – first symbol peak index (from matched_filter)
    code            : str     – 'polar' | 'onoff' | 'bipolar'
    threshold       : float or None

    Returns
    -------
    bits_detected : ndarray (int 0/1)
    samples_mf    : ndarray – sampled values before threshold
    """
    n_bits = len(symbols_tx)
    sample_idx = symbol_offset + np.arange(n_bits) * samples_per_bit
    sample_idx = np.clip(sample_idx.astype(int), 0, len(mf_out) - 1)
    samples_mf = mf_out[sample_idx]

    if threshold is None:
        if code.lower() in ('polar', 'nrz'):
            threshold = 0.0
        else:
            threshold = np.median(np.abs(samples_mf)) * 0.5

    if code.lower() in ('polar', 'nrz'):
        bits_detected = (samples_mf >= threshold).astype(int)
    elif code.lower() in ('onoff', 'ook'):
        bits_detected = (samples_mf >= threshold).astype(int)
    else:
        bits_detected = (np.abs(samples_mf) >= threshold).astype(int)

    return bits_detected, samples_mf


# ---------------------------------------------------------------------------
# 4E. BER computation
# ---------------------------------------------------------------------------
def compute_ber(bits_tx, bits_rx):
    """
    Compute the Bit Error Rate.

    Returns (ber, n_errors, n_bits).
    """
    bits_tx = np.asarray(bits_tx, dtype=int)
    bits_rx = np.asarray(bits_rx, dtype=int)
    n = min(len(bits_tx), len(bits_rx))
    errors = int(np.sum(bits_tx[:n] != bits_rx[:n]))
    ber = errors / n if n > 0 else 1.0
    return ber, errors, n


# ---------------------------------------------------------------------------
# 4F. BER vs SNR sweep (waterfall curve)
# ---------------------------------------------------------------------------
def ber_vs_snr(bits_tx, pulse, samples_per_bit, snr_range_dB=None,
               code='polar', amplitude=1.0):
    """
    Sweep SNR and return BER at each point.

    Parameters
    ----------
    bits_tx         : ndarray of 0/1
    pulse           : ndarray – TX/RX pulse
    samples_per_bit : int
    snr_range_dB    : list of floats (dB)
    code            : str – line code
    amplitude       : float

    Returns
    -------
    snr_dB_list, ber_list
    """
    from line_coding import line_encode

    if snr_range_dB is None:
        snr_range_dB = list(range(0, 13, 2))

    bits_tx = np.asarray(bits_tx, dtype=int)
    _, symbols = line_encode(bits_tx, code, samples_per_bit, amplitude)
    shaped = shape_signal_full(symbols, pulse, samples_per_bit)

    ber_list = []
    for snr_dB in snr_range_dB:
        noisy, _ = add_awgn_noise(shaped, snr_dB)
        mf_out, sym_offset = matched_filter(noisy, pulse)
        bits_rx, _ = sample_and_decide(
            mf_out, symbols, samples_per_bit, sym_offset, code)
        ber, _, _ = compute_ber(bits_tx, bits_rx)
        ber_list.append(max(ber, 1e-7))

    return list(snr_range_dB), ber_list
