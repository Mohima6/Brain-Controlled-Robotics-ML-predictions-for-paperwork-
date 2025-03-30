import numpy as np
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Simulated sensor data
time = np.arange(0, 10, 0.1)  # 10 seconds (sampled every 0.1s)
temperature = 25 + 5 * np.sin(0.5 * time)  # Temperature variations (°C)
radiation = np.abs(10 * np.cos(0.3 * time))  # Radiation levels (mSv)

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(time, temperature, label="Temperature (°C)", color='r', linestyle='-', linewidth=2)
plt.plot(time, radiation, label="Radiation (mSv)", color='b', linestyle='--', linewidth=2)

# Formatting the plot
plt.xlabel("Time (s)")
plt.ylabel("Sensor Readings")
plt.legend()
plt.title("Environmental Adaptation: Temperature & Radiation Over Time")
plt.grid(True, linestyle="--", alpha=0.7)  # Add a light grid for better readability

# **Save the plot as a file**
plt.savefig("environmental_adaptation_plot.png", dpi=300)  # Saves in the current working directory

# Show the plot
plt.show()

