import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Simulated Data (Sensor readings: Temperature, Pressure, Radiation)
np.random.seed(42)
time_steps = np.linspace(0, 100, 1000)  # Simulating over 100 time units
temperature = np.sin(time_steps / 10) + 0.1 * np.random.randn(1000)  # Simulated temperature
pressure = np.cos(time_steps / 15) + 0.1 * np.random.randn(1000)  # Simulated pressure
radiation = np.sin(time_steps / 5) + 0.1 * np.random.randn(1000)  # Simulated radiation

# Simulated Robot Performance (Function of environment)
robot_performance = (2 * temperature - 1.5 * pressure + 0.5 * radiation +
                     0.1 * np.random.randn(1000) - 0.02 * time_steps)  # Decreasing over time due to fatigue

# Environmental Classification: Safe, Difficult, Risk Zone
env_types = []
for i in range(len(temperature)):
    if temperature[i] < 0.5 and pressure[i] > -0.5 and radiation[i] < 0.5:
        env_types.append('Safe')
    elif (0.5 <= temperature[i] <= 1.5) and (0.5 <= pressure[i] <= 1.5) and (0.5 <= radiation[i] <= 1.5):
        env_types.append('Difficult')
    else:
        env_types.append('Risk Zone')

env_types = np.array(env_types)  # Convert to NumPy array

# Create dataset
X = np.vstack([temperature, pressure, radiation, time_steps]).T
y = robot_performance

# Split Data
X_train, X_test, y_train, y_test, env_train, env_test = train_test_split(X, y, env_types, test_size=0.2, random_state=42)

# Model Training (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Recalibration Prediction (Maintenance Trigger)
recalibration_threshold = -8  # Hypothetical threshold for poor performance requiring maintenance
maintenance_needed = y_pred < recalibration_threshold

# Define colors for environment zones
colors = {'Safe': 'green', 'Difficult': 'orange', 'Risk Zone': 'red'}

# --------- PLOTTING ---------

plt.figure(figsize=(15, 12))

# 🔹 Plot 1: Robot Performance Prediction Over Time
plt.subplot(3, 1, 1)
plt.plot(X_test[:, 3], y_test, 'bo', markersize=3, label='Actual Performance')  # Blue dots for actual
plt.plot(X_test[:, 3], y_pred, 'r-', linewidth=1.5, label='Predicted Performance')  # Red line for predicted
plt.xlabel('Time Steps')
plt.ylabel('Robot Performance')
plt.title('Actual vs. Predicted Robot Performance Over Time')
plt.legend()
plt.grid()

# 🔹 Plot 2: Environmental Zones Over Time
plt.subplot(3, 1, 2)
for i, env in enumerate(env_test):
    plt.scatter(X_test[i, 3], y_test[i], color=colors[env], s=10, label=f'{env}' if i == 0 else "")
plt.xlabel('Time Steps')
plt.ylabel('Robot Performance')
plt.title('Environmental Zones Affecting Robot Performance')
plt.legend()
plt.grid()

# 🔹 Plot 3: System Recalibration Prediction
plt.subplot(3, 1, 3)
plt.plot(X_test[:, 3], y_pred, 'g-', linewidth=1.5, label='Predicted Performance')
plt.scatter(X_test[:, 3][maintenance_needed], y_pred[maintenance_needed], color='red', s=20, label='Recalibration Needed')
plt.axhline(y=recalibration_threshold, color='black', linestyle='--', label='Maintenance Threshold')
plt.xlabel('Time Steps')
plt.ylabel('Performance')
plt.title('System Reliability & Recalibration Prediction')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
