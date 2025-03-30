import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

# Simulate EEG signal (sine wave + noise for simplicity)
def simulate_eeg_signal(length=1000, noise_level=0.1):
    time = np.linspace(0, 10, length)
    eeg_signal = np.sin(2 * np.pi * 1 * time) + noise_level * np.random.randn(length)
    return eeg_signal

# Generate synthetic EEG data
eeg_data = simulate_eeg_signal()

# Split data into training and testing sets
train_size = int(len(eeg_data) * 0.8)
train_data, test_data = eeg_data[:train_size], eeg_data[train_size:]

# Fit AutoRegressive model
lag = 10  # Number of previous points to consider
model = AutoReg(train_data, lags=lag).fit()

# Make predictions
predictions = model.predict(start=len(train_data), end=len(eeg_data)-1, dynamic=False)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(range(len(eeg_data)), eeg_data, label="Actual EEG Signal")
plt.plot(range(train_size, len(eeg_data)), predictions, label="Predicted EEG Signal", linestyle='dashed')
plt.axvline(x=train_size, color='r', linestyle='--', label="Train/Test Split")
plt.xlabel("Time Steps")
plt.ylabel("EEG Signal Amplitude")
plt.legend()
plt.title("EEG Signal Prediction using AutoRegressive Model")
plt.show()
