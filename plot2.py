import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Simulating EEG data (4 channels, 10 seconds, sampled at 250Hz)
fs = 250  # Sampling frequency (Hz)
t = np.linspace(0, 10, fs * 10)  # Time vector (10 seconds)
eeg_signals = [
    np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.5, len(t)),  # Alpha (10 Hz)
    np.sin(2 * np.pi * 20 * t) + np.random.normal(0, 0.5, len(t)),  # Beta (20 Hz)
    np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.5, len(t)),   # Theta (5 Hz)
    np.sin(2 * np.pi * 40 * t) + np.random.normal(0, 0.5, len(t))   # Gamma (40 Hz)
]

# Plot Power Spectral Density (PSD)
plt.figure(figsize=(12, 6))
for i, eeg in enumerate(eeg_signals):
    freqs, psd = welch(eeg, fs, nperseg=1024)  # Compute PSD using Welch's method
    plt.semilogy(freqs, psd, label=f"Channel {i+1}")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (µV²/Hz)")
plt.title("EEG Frequency Spectrum (Power Spectral Density - PSD)")
plt.legend()
plt.grid(True)
plt.show()
