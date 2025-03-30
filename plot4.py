import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Step 1: Simulate EEG-like data
# Simulating 64 EEG channels and 1000 samples (each sample corresponds to a time point)
X = np.random.rand(1000, 64)  # 1000 samples, 64 features (EEG channels)

# Simulating corresponding labels (binary classification)
# For simplicity, we assume there are two classes: 0 = no movement, 1 = movement
y = np.random.randint(2, size=1000)  # Random binary labels (0 or 1)

# Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define a simple neural network model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # 64 input features (EEG channels)
    layers.Dense(32, activation='relu'),  # Hidden layer
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Step 3: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Step 5: Evaluate the model
print("Evaluating the model on test data...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Step 6: Visualize the training process (accuracy and loss over epochs)
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Model Training Progress')
plt.xlabel('Epochs')
plt.ylabel('Accuracy / Loss')
plt.legend(loc='upper left')
plt.show()
