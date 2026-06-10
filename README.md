## Overview

A Python-based simulation of an end-to-end baseband digital communication system. The project models the complete transmitter–receiver chain, including PCM encoding, line coding, pulse shaping, AWGN channel transmission, matched filtering, eye-diagram visualization, and BER performance analysis.

## Features

* Signal generation using configurable sinusoidal sources
* Uniform sampling and quantization
* Pulse Code Modulation (PCM)
* Polar NRZ, Unipolar NRZ, and Bipolar AMI line coding
* Rectangular and Raised Cosine pulse shaping
* AWGN channel simulation
* Matched filter receiver
* Eye diagram analysis
* Bit detection and BER computation
* BER vs SNR (Waterfall Curve) generation

## System Architecture

```text
Analog Signal
      │
      ▼
Sampling
      │
      ▼
Quantization
      │
      ▼
PCM Encoding
      │
      ▼
Line Coding
      │
      ▼
Pulse Shaping
      │
      ▼
AWGN Channel
      │
      ▼
Matched Filter
      │
      ▼
Bit Detection
      │
      ▼
BER Calculation
```

## Project Structure

```text
PCS-PROJECT/
│
├── sampling_quantization.py
├── line_codes.py
├── pulse_shaping.py
├── channel_noise.py
├── eye_diagrams.py
├── test_dcs.py
├── Output_Plots/
├── plots.pdf
└── README.md
```
## Running the Simulation

```bash
python test_code.py
```

All major simulation parameters, including sampling frequency, quantization levels, line coding scheme, pulse shaping parameters, and channel SNR, can be configured directly in `test_code.py`.

## Results

The `Output_Plots` directory contains generated results for:

* Sampling and Quantization
* PCM Waveforms
* Line Coding Schemes
* Eye Diagrams
* Matched Filter Output
* BER vs SNR Waterfall Curve

Representative outputs are also included in `plots.pdf`.

## Concepts Implemented

* Nyquist Sampling Theorem
* Uniform Quantization
* Pulse Code Modulation (PCM)
* Line Coding Techniques
* Pulse Shaping and Bandwidth Control
* Additive White Gaussian Noise (AWGN)
* Matched Filter Detection
* Eye Diagram Analysis
* Bit Error Rate (BER) Estimation
## References

* B. P. Lathi and Zhi Ding, *Modern Digital and Analog Communication Systems*
* Upamanyu Madhow, *Introduction to Communication Systems*

## Author

**Vivek**

Course Project – *Principles of Communication Systems (PCS)*


