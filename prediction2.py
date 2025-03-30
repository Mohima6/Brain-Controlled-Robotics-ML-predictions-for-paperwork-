import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, welch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Simulate EEG signal with movement intentions
def simulate_eeg_signal(length=1000, noise_level=0.1):
    time = np.linspace(0, 10, length)
    eeg_signal = np.sin(2 * np.pi * 10 * time) + noise_level * np.random.randn(length)
    labels = np.random.choice([0, 1], size=length // 100)  # Adjust label count to match feature count
    return eeg_signal, labels

# Bandpass filter for EEG frequency bands
def bandpass_filter(data, lowcut, highcut, fs=100, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# Feature extraction (Power Spectral Density - PSD)
def extract_features(eeg_signal, fs=100):
    freqs, psd = welch(eeg_signal, fs, nperseg=fs)
    alpha_power = np.mean(psd[(freqs >= 8) & (freqs <= 12)])
    beta_power = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
    return np.array([alpha_power, beta_power])

# Generate synthetic EEG data
eeg_data, labels = simulate_eeg_signal()
features = np.array([extract_features(eeg_data[i:i+100]) for i in range(0, len(eeg_data)-100, 100)])

# Ensure labels match the number of feature samples
labels = labels[:len(features)]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Visualization
plt.figure(figsize=(12, 6))

# Plot EEG signal
plt.subplot(2, 2, 1)
plt.plot(eeg_data[:500])
plt.title("Raw EEG Signal")
plt.xlabel("Time Steps")
plt.ylabel("Amplitude")

# Plot extracted features
plt.subplot(2, 2, 2)
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm', edgecolors='k', label='Actual Labels')
plt.title("Feature Distribution (Alpha vs Beta Power)")
plt.xlabel("Alpha Power")
plt.ylabel("Beta Power")
plt.legend()

# Plot model predictions
plt.subplot(2, 2, 3)
plt.scatter(features[:len(y_pred), 0], features[:len(y_pred), 1], c=y_pred, cmap='coolwarm', edgecolors='k', marker='s', label='Predicted Labels')
plt.title("Predicted Classes")
plt.xlabel("Alpha Power")
plt.ylabel("Beta Power")
plt.legend()

# Plot accuracy with step plot for clarity
test_indices = np.arange(len(y_pred))  # Ensure matching length for predictions
plt.subplot(2, 2, 4)
plt.step(test_indices, y_test[:len(y_pred)], where='mid', label="Actual", color='b')
plt.step(test_indices, y_pred, where='mid', label="Predicted", color='r', linestyle='dashed')
plt.title("Model Predictions vs Actual")
plt.xlabel("Sample Index")
plt.ylabel("Class")
plt.legend()

plt.tight_layout()
plt.show()

