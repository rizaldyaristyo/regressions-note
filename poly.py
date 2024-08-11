import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = [10, 323, 23, 98, 71, 45, 39, 212, 67, 463]
X = np.array(range(len(data))).reshape(-1, 1)
y = np.array(data).reshape(-1, 1)

# Define the degree of the polynomial
degree = 3

# Generate polynomial features
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# Fit polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Generate polynomial features for future positions
next_positions = np.array(range(len(data) + 1, len(data) + 6)).reshape(-1, 1)
next_positions_poly = poly_features.transform(next_positions)

# Predict using the polynomial regression model
next_numbers = model.predict(next_positions_poly)

for i in range(len(next_numbers)):
    print("Predicted number for position", len(data) + i + 1, ":", next_numbers[i][0])

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, model.predict(X_poly), color='red', linewidth=2, label='Polynomial Regression')
plt.scatter(next_positions, next_numbers, color='green', label='Predictions')
plt.xlabel('Position')
plt.ylabel('Number')
plt.title('Polynomial Regression Prediction')
plt.legend()
plt.grid(True)
plt.show()
