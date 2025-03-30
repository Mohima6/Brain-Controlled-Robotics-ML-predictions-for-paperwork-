import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

# Example Dataset (replace with your actual sensor data)
# Features: temperature, pressure, radiation
X = np.random.rand(1000, 3)  # 1000 samples, 3 features (sensor data: temp, pressure, radiation)

# Targets:
# 1. Regression: Predict robot performance (e.g., speed, efficiency)
y_regression = np.random.rand(1000)  # Continuous output (e.g., robot performance score)

# 2. Classification: Hazardous vs Safe (1 for hazardous, 0 for safe)
y_classification = np.random.randint(2, size=1000)  # 0 = safe, 1 = hazardous

# Split data into training and testing sets
X_train, X_test, y_train_reg, y_test_reg, y_train_class, y_test_class = train_test_split(
    X, y_regression, y_classification, test_size=0.2, random_state=42)

# Regression Model: Predict robot performance (e.g., speed or efficiency)
reg_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
reg_model.fit(X_train, y_train_reg)

# Predict and evaluate regression model
y_pred_reg = reg_model.predict(X_test)
reg_mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Regression Mean Squared Error: {reg_mse}")

# Classification Model: Predict hazardous vs safe environment
class_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
class_model.fit(X_train, y_train_class)

# Predict and evaluate classification model
y_pred_class = class_model.predict(X_test)
class_accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"Classification Accuracy: {class_accuracy}")

# Plotting the predictions vs actual values for both models

# 1. Regression Plot
plt.figure(figsize=(12, 6))

# Plot the regression predictions vs actuals
plt.subplot(1, 2, 1)
plt.scatter(range(len(y_test_reg)), y_test_reg, color='blue', label='Actual', alpha=0.6)
plt.scatter(range(len(y_pred_reg)), y_pred_reg, color='red', label='Predicted', alpha=0.6)
plt.title('Predicted vs Actual (Regression: Robot Performance)')
plt.xlabel('Sample Index')
plt.ylabel('Performance')
plt.legend()

# 2. Classification Plot
plt.subplot(1, 2, 2)
plt.scatter(range(len(y_test_class)), y_test_class, color='blue', label='Actual', alpha=0.6)
plt.scatter(range(len(y_pred_class)), y_pred_class, color='red', label='Predicted', alpha=0.6)
plt.title('Predicted vs Actual (Classification: Hazardous Environment)')
plt.xlabel('Sample Index')
plt.ylabel('Class (0 = Safe, 1 = Hazardous)')
plt.legend()

plt.tight_layout()
plt.show()
