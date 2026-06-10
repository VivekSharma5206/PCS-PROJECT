"""
=============================================================================
MODULE 5: Eye Diagram
=============================================================================
Generate eye diagram data and plots for any waveform.
Mimics MATLAB's eyediagram() function behaviour.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 5A. Eye diagram data extractor
# ---------------------------------------------------------------------------
def eye_diagram_data(waveform, samples_per_bit, n_traces=None, offset=0):
    """
    Slice a waveform into overlapping 2-symbol windows for the eye diagram.

    Parameters
    ----------
    waveform        : ndarray – (possibly shaped/filtered) signal
    samples_per_bit : int     – samples per symbol period
    n_traces        : int     – max number of traces to overlay (None = all)
    offset          : int     – sample offset within each window

    Returns
    -------
    traces : list of ndarray – each trace is 2*samples_per_bit samples long
    t_eye  : ndarray         – normalised time axis  (-1 … +1) symbol periods
    """
    window = 2 * samples_per_bit
    traces = []
    start = offset
    while start + window <= len(waveform):
        traces.append(waveform[start: start + window])
        start += samples_per_bit

    if n_traces is not None:
        traces = traces[:n_traces]

    t_eye = np.linspace(-1, 1, window, endpoint=False)
    return traces, t_eye


# ---------------------------------------------------------------------------
# 5B. Plot eye diagram
# ---------------------------------------------------------------------------
def plot_eye_diagram(waveform, samples_per_bit, title='Eye Diagram',
                     ax=None, n_traces=200, offset=0,
                     color='#00C8FF', alpha=0.15):
    """
    Draw an eye diagram on matplotlib axes.

    Parameters
    ----------
    waveform        : ndarray
    samples_per_bit : int
    title           : str
    ax              : matplotlib Axes (creates new figure if None)
    n_traces        : int   – number of traces to overlay
    offset          : int   – alignment offset
    color           : str   – trace colour
    alpha           : float – trace transparency

    Returns
    -------
    ax : matplotlib Axes
    """
    traces, t_eye = eye_diagram_data(waveform, samples_per_bit,
                                     n_traces=n_traces, offset=offset)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    for trace in traces:
        ax.plot(t_eye, trace, color=color, alpha=alpha, linewidth=0.8)

    ax.axvline(0, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.axhline(0, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.set_title(title, color='white', fontsize=11, pad=8)
    ax.set_xlabel('Time (symbol periods)', color='#aaaaaa', fontsize=9)
    ax.set_ylabel('Amplitude', color='#aaaaaa', fontsize=9)
    ax.tick_params(colors='#888888')
    ax.set_facecolor('#0d1117')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    return ax


# ---------------------------------------------------------------------------
# 5C. Plot all four eye diagrams in a 2×2 grid
# ---------------------------------------------------------------------------
def plot_all_eye_diagrams(waveforms_dict, samples_per_bit,
                          suptitle='Eye Diagrams'):
    """
    Plot multiple eye diagrams side by side.

    Parameters
    ----------
    waveforms_dict : dict  {label: waveform_array}
    samples_per_bit: int

    Returns
    -------
    fig : matplotlib Figure
    """
    n = len(waveforms_dict)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(13, 4 * nrows),
                             facecolor='#0d1117')
    axes_flat = axes.flatten() if n > 1 else [axes]

    colors = ['#00C8FF', '#FF6B6B', '#6BFF9E', '#FFD700',
              '#FF69B4', '#DA70D6']
    for i, (label, wfm) in enumerate(waveforms_dict.items()):
        plot_eye_diagram(wfm, samples_per_bit,
                         title=label,
                         ax=axes_flat[i],
                         color=colors[i % len(colors)],
                         alpha=0.15)

    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(suptitle, color='white', fontsize=14, y=1.01)
    plt.tight_layout()
    return fig
