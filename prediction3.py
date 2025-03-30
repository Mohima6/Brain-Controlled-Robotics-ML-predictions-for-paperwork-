import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Robot class to simulate movement
class Robot:
    def __init__(self, initial_position):
        self.position = np.array(initial_position)
        self.velocity = np.array([0.0, 0.0])  # Initial velocity

    def update_state(self, control_input):
        """
        Update robot's position based on control inputs (acceleration).
        """
        self.velocity += control_input  # Update velocity (control_input is acceleration)
        self.position += self.velocity  # Update position using velocity

# Simulate robot movement with brain signals (for now, using random control inputs)
def simulate_robot_motion(num_steps=50, horizon=10):
    # Create robot at initial position (0, 0)
    robot = Robot([0.0, 0.0])

    actual_positions = [robot.position.copy()]
    predicted_positions = [robot.position.copy()]

    for step in range(num_steps):
        # Generate random control inputs (acceleration in x and y directions)
        control_input = np.random.randn(2)  # Random control input for simplicity

        # Update robot's state
        robot.update_state(control_input)

        # Store the robot's actual position
        actual_positions.append(robot.position.copy())

        # Linear regression to predict the next position
        if step >= horizon:  # Perform regression after collecting enough data
            X = np.array([np.arange(step - horizon, step)]).T  # Time steps as input
            y = np.array([actual_positions[i] for i in range(step - horizon, step)])  # Positions

            # Perform linear regression for X and Y
            model_x = LinearRegression().fit(X, y[:, 0])  # Fit regression for X
            model_y = LinearRegression().fit(X, y[:, 1])  # Fit regression for Y

            predicted_x = model_x.predict(np.array([[step]]))  # Predict X position
            predicted_y = model_y.predict(np.array([[step]]))  # Predict Y position
            predicted_positions.append([predicted_x[0], predicted_y[0]])

    # Convert positions to numpy array for easy plotting
    actual_positions = np.array(actual_positions)
    predicted_positions = np.array(predicted_positions)

    return actual_positions, predicted_positions

# Simulate the robot's motion
actual_positions, predicted_positions = simulate_robot_motion()

# Clustering (e.g., KMeans) to segment the path
kmeans = KMeans(n_clusters=3)
kmeans.fit(actual_positions)  # Fit KMeans to actual positions

# Predict the cluster labels for each data point
labels = kmeans.predict(actual_positions)

# Plot the actual and predicted paths
plt.figure(figsize=(10, 6))

# Plot Actual Path with Blue Dots (continuous line)
plt.plot(actual_positions[:, 0], actual_positions[:, 1], color='blue', label='Actual Path', marker='o', markersize=5, linestyle='-', linewidth=2)

# Plot Predicted Path with Red Dots (continuous line)
plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], color='red', label='Predicted Path', marker='o', markersize=5, linestyle='-', linewidth=2)

# Plot Clusters with different colors
for i in range(3):
    cluster_points = actual_positions[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}', s=50, alpha=0.6)

# Titles and Labels
plt.title('Predicted vs Actual Robot Movement Path with Clusters')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)

# Show Plot
plt.show()
