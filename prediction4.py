import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Simulating EEG-like data for brain-robot interaction (use real EEG data in practice)
np.random.seed(42)
data_size = 1000
# Example EEG features: [alpha_band_power, beta_band_power, theta_band_power]
X = np.random.rand(data_size, 3)  # Random EEG-like data
y = np.random.choice([0, 1, 2], size=data_size)  # Actions: 0 = forward, 1 = left, 2 = stop

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a simple Sequential model with LSTM (for time-series-like data)
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer
model.add(Dense(32, activation='relu'))  # Hidden layer
model.add(Dense(3, activation='softmax'))  # Output layer (3 classes)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy[1] * 100:.2f}%")

# Predict the classes for test data
y_pred = model.predict(X_test)

# Visualize predictions (in this case, we'll show predicted vs actual movements)
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred.argmax(axis=1), cmap='coolwarm', label="Predicted")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', cmap='coolwarm', label="Actual", alpha=0.6)
plt.title('Predicted vs Actual Robot Movements from EEG Data')
plt.xlabel('EEG Alpha Band Power')
plt.ylabel('EEG Beta Band Power')
plt.legend(loc='upper right')
plt.colorbar(label='Movement Type')
plt.show()
