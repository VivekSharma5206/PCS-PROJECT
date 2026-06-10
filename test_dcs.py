"""
=============================================================================
  END-TO-END DIGITAL COMMUNICATION SYSTEM — TEST / DEMO FILE
=============================================================================
  This is the ONLY file you need to open and run for the demo.
 
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  HOW TO RUN                                                             │
  │  1. Install dependencies once:                                          │
  │       pip install numpy scipy matplotlib                                │
  │  2. Make sure all .py files are in the SAME FOLDER as this file.       │
  │  3. Run:   python test_dcs.py                                           │
  │  4. Plots save as PNG files in the same folder.                         │
  └─────────────────────────────────────────────────────────────────────────┘
 
  Change the parameters in the "PARAMETERS" section below.
  Do NOT touch the function modules (sampling_quantization.py etc.).
 
  Blocks demonstrated:
    A. Sampling & Quantization  → PCM bits
    B. Line Coding              → waveform (polar / onoff / bipolar) + eye diagram
    C. Pulse Shaping            → NRZ / RZ / Sinc / RC + frequency + eye diagrams
    D. AWGN Channel             → noisy waveform eye diagram
    E. Matched Filter           → filtered signal eye diagram
    F. Bit Detection & BER
    G. BER vs SNR (waterfall curve)
=============================================================================
"""
 
import numpy as np
import matplotlib
matplotlib.use('TKAgg')           # ← Change to 'TkAgg' or 'Qt5Agg' for popup windows
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
 
# ── Import function modules (DO NOT EDIT THESE MODULES) ───────────────────
from sampling_quantization import analog_to_bits
from line_coding            import line_encode
from pulse_shaping          import get_pulse, apply_pulse_shaping, pulse_spectrum
from channel_noise          import (add_awgn_noise, matched_filter,
                                    sample_and_decide, compute_ber,
                                    ber_vs_snr, shape_signal_full)
from eye_diagram            import plot_eye_diagram, plot_all_eye_diagrams
 
 
# =============================================================================
#  ██████████  CHANGE PARAMETERS HERE — THIS IS WHERE DEMO HAPPENS  ██████████
# =============================================================================
 
# ── Source signal ──────────────────────────────────────────────────────────
DURATION        = 1.0           # seconds
FS_ANALOG       = 10_000        # high-res "analog" sampling rate (Hz)
SIGNAL_FREQS    = (1, 3)        # frequency components in Hz
SIGNAL_AMPS     = (1.0, 1.0)   # amplitudes of each component
 
# ── Sampling & Quantization ────────────────────────────────────────────────
FS_SAMPLE       = 50            # sampling rate (Hz)
L_LEVELS        = 16            # quantization levels: try 4, 8, 16, 32, 64
 
# ── Line Coding  (change to show different codes) ──────────────────────────
LINE_CODE       = 'polar'       # 'polar'  |  'onoff'  |  'bipolar'
AMPLITUDE       = 1.0           # symbol amplitude
 
# ── Pulse Shaping ──────────────────────────────────────────────────────────
SAMPLES_PER_BIT = 64            # oversampling factor (symbol period in samples)
ROLLOFF         = 0.5           # RC roll-off factor β  (0 < β ≤ 1)
N_PERIODS       = 8             # truncation length for sinc / RC (symbol periods)
 
# ── Channel ────────────────────────────────────────────────────────────────
SNR_DB          = 10.0          # SNR for single-point demo (dB)
 
# ── BER vs SNR waterfall curve ────────────────────────────────────────────
N_BITS_BER      = 100_000       # bits for BER sweep (more = smoother curve)
SNR_RANGE_DB    = list(range(0, 13, 2))   # [0, 2, 4, 6, 8, 10, 12] dB
 
# ── Eye diagram display ────────────────────────────────────────────────────
N_EYE_TRACES    = 200           # number of overlaid traces per eye diagram
 
# =============================================================================
#  DO NOT EDIT BELOW THIS LINE
# =============================================================================
 
plt.style.use('dark_background')
NEON = ['#00C8FF', '#FF6B6B', '#6BFF9E', '#FFD700', '#FF69B4', '#DA70D6']
 
def hdr(title):
    w = 65
    print("\n" + "═" * w)
    print(f"  {title}")
    print("═" * w)
 
def style_ax(ax, facecolor='#0d1117'):
    ax.set_facecolor(facecolor)
    ax.tick_params(colors='#888888')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333333')
 
def save_show(fig, filename):
    fig.savefig(filename, dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close(fig)
    print(f"  [Saved] {filename}")
 
 
# =============================================================================
# BLOCK A — Sampling & Quantization
# =============================================================================
hdr("BLOCK A — Sampling & Quantization")
 
adc = analog_to_bits(
    duration=DURATION, fs_analog=FS_ANALOG, fs_sample=FS_SAMPLE,
    L=L_LEVELS, freqs=SIGNAL_FREQS, amplitudes=SIGNAL_AMPS,
)
 
print(f"  Source             : {SIGNAL_FREQS} Hz components, "
      f"amplitudes {SIGNAL_AMPS}")
print(f"  Sampling rate      : {FS_SAMPLE} Hz  "
      f"(Nyquist min = {2*max(SIGNAL_FREQS)} Hz)")
print(f"  Quantization       : L = {L_LEVELS} levels  →  "
      f"{adc['bits_per_sample']} bits/sample")
print(f"  Step size Δ        : {adc['delta']:.5f}")
print(f"  SQNR               : {adc['sqnr_dB']:.2f} dB")
print(f"  Total PCM bits     : {len(adc['bits'])}")
print(f"  First 32 bits      : {list(map(int, adc['bits'][:32]))}")
 
fig, axes = plt.subplots(3, 1, figsize=(13, 8), facecolor='#0d1117')
fig.suptitle('Block A — Sampling & Quantization', color='white',
             fontsize=13, y=0.99)
 
axes[0].plot(adc['t'], adc['signal'], color=NEON[0], lw=1.2)
axes[0].set_title('Original analog signal', color='white')
axes[0].set_ylabel('Amplitude', color='#aaa')
 
axes[1].stem(adc['t_s'], adc['sampled'],
             linefmt=NEON[1], markerfmt='o', basefmt='grey')
axes[1].set_title(f'Sampled  (fs = {FS_SAMPLE} Hz)', color='white')
axes[1].set_ylabel('Amplitude', color='#aaa')
 
axes[2].step(adc['t_s'], adc['quantized'], color=NEON[2],
             lw=1.4, where='post', label='Quantized')
axes[2].plot(adc['t_s'], adc['sampled'], color=NEON[0],
             lw=0.7, ls='--', alpha=0.5, label='Sampled')
axes[2].set_title(f'Quantized  (L = {L_LEVELS},  SQNR = '
                  f'{adc["sqnr_dB"]:.1f} dB)', color='white')
axes[2].set_ylabel('Amplitude', color='#aaa')
axes[2].set_xlabel('Time (s)', color='#aaa')
axes[2].legend(fontsize=8)
 
for ax in axes:
    style_ax(ax)
plt.tight_layout()
save_show(fig, 'block_A_sampling_quantization.png')
 
 
# =============================================================================
# BLOCK B — Line Coding
# =============================================================================
hdr("BLOCK B — Line Coding")
 
bits = adc['bits']
SHOW = 40
t_lc = np.arange(SHOW * SAMPLES_PER_BIT) / SAMPLES_PER_BIT
 
codes_demo = [('polar', NEON[0], 'Polar NRZ  (1→+A, 0→-A)'),
              ('onoff', NEON[1], 'On-Off OOK  (1→+A, 0→0)'),
              ('bipolar', NEON[2], 'Bipolar AMI  (1→±A alternating, 0→0)')]
 
fig, axes = plt.subplots(3, 1, figsize=(13, 7), facecolor='#0d1117')
fig.suptitle('Block B — Line Coding Waveforms (first 40 bits)',
             color='white', fontsize=13, y=0.99)
 
for ax, (code, col, label) in zip(axes, codes_demo):
    wfm, _ = line_encode(bits[:SHOW], code, SAMPLES_PER_BIT, AMPLITUDE)
    ax.plot(t_lc, wfm, color=col, lw=1.1)
    ax.set_title(label, color='white')
    ax.set_ylabel('Amplitude', color='#aaa')
    style_ax(ax)
axes[-1].set_xlabel('Time (bit periods)', color='#aaa')
plt.tight_layout()
save_show(fig, 'block_B_line_coding.png')
print(f"  Active line code : {LINE_CODE.upper()}")
 
# Eye diagram for active line code
wfm_lc, sym_lc = line_encode(bits, LINE_CODE, SAMPLES_PER_BIT, AMPLITUDE)
fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0d1117')
fig.patch.set_facecolor('#0d1117')
plot_eye_diagram(wfm_lc, SAMPLES_PER_BIT,
                 title=f'Eye Diagram — {LINE_CODE.upper()} (rectangular NRZ)',
                 ax=ax, n_traces=N_EYE_TRACES)
plt.tight_layout()
save_show(fig, 'block_B_eye_line_code.png')
 
 
# =============================================================================
# BLOCK C — Pulse Shaping
# =============================================================================
hdr("BLOCK C — Pulse Shaping")
 
pulse_names = ['nrz', 'rz', 'sinc', 'rc']
pulses = {n: get_pulse(n, SAMPLES_PER_BIT, ROLLOFF, N_PERIODS)
          for n in pulse_names}
 
# Time-domain plot
fig, axes = plt.subplots(2, 2, figsize=(13, 7), facecolor='#0d1117')
fig.suptitle('Block C — Pulse Shapes (Time Domain)', color='white', fontsize=13)
for ax, (name, p), col in zip(axes.flatten(), pulses.items(), NEON):
    t_p = (np.arange(len(p)) - len(p) // 2) / SAMPLES_PER_BIT
    ax.plot(t_p, p, color=col, lw=1.4)
    ax.axhline(0, color='#444', lw=0.5)
    ax.axvline(0, color='#444', lw=0.5)
    label = name.upper() + (f'  (β={ROLLOFF})' if name == 'rc' else '')
    ax.set_title(label, color='white')
    ax.set_xlabel('Time (symbol periods)', color='#aaa', fontsize=8)
    ax.set_ylabel('Amplitude', color='#aaa', fontsize=8)
    style_ax(ax)
plt.tight_layout()
save_show(fig, 'block_C_pulses_time.png')
 
# Frequency-domain plot
fig, axes = plt.subplots(2, 2, figsize=(13, 7), facecolor='#0d1117')
fig.suptitle('Block C — Pulse Shapes (Frequency Domain)', color='white',
             fontsize=13)
for ax, (name, p), col in zip(axes.flatten(), pulses.items(), NEON):
    freqs, mag = pulse_spectrum(p, SAMPLES_PER_BIT)
    f_norm = freqs / SAMPLES_PER_BIT
    mask = np.abs(f_norm) <= 2.5
    ax.plot(f_norm[mask], mag[mask] / np.max(mag[mask]), color=col, lw=1.3)
    ax.set_title(name.upper(), color='white')
    ax.set_xlabel('Normalised freq  (f · T)', color='#aaa', fontsize=8)
    ax.set_ylabel('|P(f)| / max', color='#aaa', fontsize=8)
    style_ax(ax)
plt.tight_layout()
save_show(fig, 'block_C_pulses_freq.png')
 
# Eye diagrams for all four pulses
shaped_dict = {}
for name, p in pulses.items():
    shaped_dict[name] = apply_pulse_shaping(sym_lc, p, SAMPLES_PER_BIT)
 
eye_dict = {f'{n.upper()} pulse': shaped_dict[n] for n in pulse_names}
fig_c3 = plot_all_eye_diagrams(eye_dict, SAMPLES_PER_BIT,
                               suptitle='Block C — Eye Diagrams (all pulses)')
fig_c3.patch.set_facecolor('#0d1117')
save_show(fig_c3, 'block_C_eye_diagrams.png')
 
print(f"  RC roll-off β = {ROLLOFF}")
print(f"  Pulse names: {pulse_names}")
 
 
# =============================================================================
# BLOCK D — AWGN Channel
# =============================================================================
hdr("BLOCK D — AWGN Channel")
 
# Full-pipeline: symbol impulse train → shape → add noise
pulse_rc = pulses['rc']
shaped_rc = shape_signal_full(sym_lc, pulse_rc, SAMPLES_PER_BIT)
noisy_rc, noise_vec = add_awgn_noise(shaped_rc, SNR_DB)
 
P_s = np.mean(shaped_rc ** 2)
P_n = np.mean(noise_vec ** 2)
actual_snr = 10 * np.log10(P_s / P_n)
print(f"  TX pulse shape     : RC  (β = {ROLLOFF})")
print(f"  Target SNR         : {SNR_DB} dB")
print(f"  Measured SNR       : {actual_snr:.2f} dB")
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor='#0d1117')
fig.suptitle(f'Block D — AWGN Channel  (SNR = {SNR_DB} dB)',
             color='white', fontsize=13)
# Use shaped_dict['rc'] for eye (trim-mode, cleaner display)
plot_eye_diagram(shaped_dict['rc'], SAMPLES_PER_BIT,
                 title='Eye — Before Channel  (RC shaped)',
                 ax=ax1, n_traces=N_EYE_TRACES, color=NEON[2])
# Noisy eye — add noise to the trimmed version for display
noisy_trim, _ = add_awgn_noise(shaped_dict['rc'], SNR_DB)
plot_eye_diagram(noisy_trim, SAMPLES_PER_BIT,
                 title=f'Eye — After AWGN  (SNR = {SNR_DB} dB)',
                 ax=ax2, n_traces=N_EYE_TRACES, color=NEON[1])
plt.tight_layout()
save_show(fig, 'block_D_awgn_eye.png')
 
 
# =============================================================================
# BLOCK E — Matched Filter
# =============================================================================
hdr("BLOCK E — Matched Filter")
 
mf_out, sym_offset = matched_filter(noisy_rc, pulse_rc)
print(f"  Matched filter     : RC pulse (time-reversed)")
print(f"  First symbol peak  : index {sym_offset}")
 
# For eye diagram display we use the trimmed noisy signal
noisy_trim2, _ = add_awgn_noise(shaped_dict['rc'], SNR_DB)
from scipy.signal import fftconvolve as _fftconv
mf_trim = _fftconv(noisy_trim2, pulse_rc[::-1], mode='same')
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor='#0d1117')
fig.suptitle('Block E — Matched Filter', color='white', fontsize=13)
plot_eye_diagram(noisy_trim2, SAMPLES_PER_BIT,
                 title='Eye — Receiver Input (noisy)',
                 ax=ax1, n_traces=N_EYE_TRACES, color=NEON[1])
plot_eye_diagram(mf_trim, SAMPLES_PER_BIT,
                 title='Eye — After Matched Filter',
                 ax=ax2, n_traces=N_EYE_TRACES, color=NEON[3])
plt.tight_layout()
save_show(fig, 'block_E_matched_filter_eye.png')
 
 
# =============================================================================
# BLOCK F — Bit Detection & BER (single SNR point)
# =============================================================================
hdr("BLOCK F — Bit Detection & BER")
 
bits_rx, samp_vals = sample_and_decide(
    mf_out, sym_lc, SAMPLES_PER_BIT, sym_offset, LINE_CODE)
 
ber, n_errs, n_total = compute_ber(bits, bits_rx)
 
print(f"  Total bits         : {n_total}")
print(f"  Bit errors         : {n_errs}")
print(f"  BER                : {ber:.6f}  ({ber:.2e})")
print(f"  SNR                : {SNR_DB} dB")
 
# Show first 64 bits comparison
N_SHOW = min(64, len(bits), len(bits_rx))
mismatches = [i for i in range(N_SHOW) if bits[i] != bits_rx[i]]
print(f"  TX bits (first {N_SHOW}): {list(map(int, bits[:N_SHOW]))}")
print(f"  RX bits (first {N_SHOW}): {list(map(int, bits_rx[:N_SHOW]))}")
print(f"  Error positions    : {mismatches}")
 
# Decision scatter plot
fig, ax = plt.subplots(figsize=(11, 4), facecolor='#0d1117')
n_scat = min(300, len(samp_vals))
col_arr = [NEON[1] if (i < len(bits) and bits[i] != bits_rx[i])
           else NEON[0] for i in range(n_scat)]
ax.scatter(np.arange(n_scat), samp_vals[:n_scat],
           c=col_arr, s=18, alpha=0.85)
ax.axhline(0, color='white', lw=1, ls='--', label='Decision threshold')
ax.set_title(f'Block F — MF Output Samples  '
             f'(cyan=correct, red=error)  BER = {ber:.2e}',
             color='white', fontsize=11)
ax.set_xlabel('Bit index', color='#aaa')
ax.set_ylabel('Amplitude', color='#aaa')
ax.legend(fontsize=8)
style_ax(ax)
plt.tight_layout()
save_show(fig, 'block_F_detection_ber.png')
 
 
# =============================================================================
# BLOCK G — BER vs SNR Waterfall Curve
# =============================================================================
hdr("BLOCK G — BER vs SNR Waterfall Curve")
 
print(f"  Bits per SNR point : {N_BITS_BER:,}")
print(f"  SNR range          : {SNR_RANGE_DB} dB")
print(f"  Line code          : {LINE_CODE.upper()}")
print(f"  Running sweep …")
 
rng = np.random.default_rng(1234)
bits_ber = rng.integers(0, 2, size=N_BITS_BER)
 
# NOTE: For the waterfall curve we use SPB=8 (reduced oversampling).
# High oversampling (SPB=64) gives near-perfect recovery even at low SNR,
# making the waterfall flat. SPB=8 matches the expected 0-12 dB SNR range.
SPB_BER = 8
pulse_ber = get_pulse('rc', SPB_BER, ROLLOFF, n_periods=4)
 
snr_ax, ber_vals = ber_vs_snr(
    bits_tx        = bits_ber,
    pulse          = pulse_ber,
    samples_per_bit= SPB_BER,
    snr_range_dB   = SNR_RANGE_DB,
    code           = LINE_CODE,
    amplitude      = AMPLITUDE,
)
 
print(f"\n  {'SNR (dB)':>8}  │  BER")
print(f"  {'─'*8}─┼─{'─'*14}")
for s, b in zip(snr_ax, ber_vals):
    print(f"  {s:8.1f}  │  {b:.4e}")
 
fig, ax = plt.subplots(figsize=(9, 6), facecolor='#0d1117')
ax.semilogy(snr_ax, ber_vals, color=NEON[0], lw=2.2,
            marker='o', ms=9,
            markerfacecolor=NEON[3], markeredgecolor='white', mew=0.8,
            label=f'{LINE_CODE.upper()} + RC (β={ROLLOFF})')
ax.set_xlabel('SNR (dB)', color='#aaa', fontsize=11)
ax.set_ylabel('Bit Error Rate (BER)', color='#aaa', fontsize=11)
ax.set_title('Block G — BER vs SNR  (Waterfall Curve)',
             color='white', fontsize=13)
ax.grid(True, which='both', color='#2a2a2a', ls='--', lw=0.6)
ax.legend(fontsize=9, facecolor='#1a1a2e',
          labelcolor='white', edgecolor='#555')
ax.yaxis.set_major_locator(LogLocator(base=10))
style_ax(ax)
plt.tight_layout()
save_show(fig, 'block_G_waterfall_ber.png')
 
 
# =============================================================================
# FINAL SUMMARY
# =============================================================================
hdr("SUMMARY — All Blocks Complete")
print(f"  Source signal      : {SIGNAL_FREQS} Hz,  fs_analog = {FS_ANALOG} Hz")
print(f"  Sampling rate      : {FS_SAMPLE} Hz")
print(f"  Quantization       : L = {L_LEVELS}  →  {adc['bits_per_sample']} b/sample"
      f"  SQNR = {adc['sqnr_dB']:.1f} dB")
print(f"  Line code          : {LINE_CODE.upper()}")
print(f"  Pulse shape        : RC  (β = {ROLLOFF})")
print(f"  SNR (demo)         : {SNR_DB} dB")
print(f"  BER (demo)         : {ber:.4e}")
print(f"\n  PNG files saved:")
pngs = [
    'block_A_sampling_quantization.png',
    'block_B_line_coding.png',
    'block_B_eye_line_code.png',
    'block_C_pulses_time.png',
    'block_C_pulses_freq.png',
    'block_C_eye_diagrams.png',
    'block_D_awgn_eye.png',
    'block_E_matched_filter_eye.png',
    'block_F_detection_ber.png',
    'block_G_waterfall_ber.png',
]
for f in pngs:
    print(f"    {f}")
print("\n  Done.")