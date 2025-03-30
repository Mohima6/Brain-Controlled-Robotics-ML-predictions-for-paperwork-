import numpy as np
import matplotlib.pyplot as plt

# Simulating EEG data (4 channels, 10 seconds, sampled at 250Hz)
fs = 250  # Sampling frequency (Hz)
t = np.linspace(0, 10, fs * 10)  # Time vector (10 seconds)
eeg_signals = [
    np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.2, len(t)),  # Simulated Alpha Wave (10 Hz)
    np.sin(2 * np.pi * 20 * t) + np.random.normal(0, 0.2, len(t)),  # Beta Wave (20 Hz)
    np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.2, len(t)),   # Theta Wave (5 Hz)
    np.sin(2 * np.pi * 40 * t) + np.random.normal(0, 0.2, len(t))   # Gamma Wave (40 Hz)
]

# Plot EEG time-series data
plt.figure(figsize=(12, 6))
for i, eeg in enumerate(eeg_signals):
    plt.plot(t, eeg + (i * 2), label=f"Channel {i+1}")  # Offset each channel for visibility

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (µV)")
plt.title("Simulated EEG Signal Time-Series")
plt.legend()
plt.grid(True)
plt.show()
