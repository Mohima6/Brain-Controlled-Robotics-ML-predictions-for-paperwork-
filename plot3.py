import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Simulating EEG data (5 channels, 1000 samples)
fs = 250  # Sampling frequency (Hz)
t = np.linspace(0, 10, fs * 10)  # Time vector (10 seconds)
eeg_signals = np.array([
    np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.5, len(t)),  # Alpha (10 Hz)
    np.sin(2 * np.pi * 20 * t) + np.random.normal(0, 0.5, len(t)),  # Beta (20 Hz)
    np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.5, len(t)),   # Theta (5 Hz)
    np.sin(2 * np.pi * 40 * t) + np.random.normal(0, 0.5, len(t)),  # Gamma (40 Hz)
    np.random.normal(0, 1, len(t))  # Noise
])

# Standardize the data
scaler = StandardScaler()
eeg_signals_scaled = scaler.fit_transform(eeg_signals.T).T

# Apply PCA (Reducing to 2 principal components for visualization)
pca = PCA(n_components=2)
eeg_signals_pca = pca.fit_transform(eeg_signals_scaled.T).T

# Plot the PCA results
plt.figure(figsize=(10, 6))
plt.scatter(eeg_signals_pca[0], eeg_signals_pca[1], c='blue', label='EEG Components')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of EEG Signals - Dimensionality Reduction")
plt.grid(True)
plt.legend()
plt.show()

# Display the explained variance ratio
print(f"Explained Variance Ratio of each component: {pca.explained_variance_ratio_}")
