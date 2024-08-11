import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = [10, 323, 23, 98, 71, 45, 39, 212, 67, 463]
X = np.array(range(len(data))).reshape(-1, 1)
y = np.array(data).reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)

next_positions = np.array(range(len(data) + 1, len(data) + 6)).reshape(-1, 1)
next_numbers = model.predict(next_positions)

for i in range(len(next_numbers)):
    print("Predicted number for position", len(data) + i + 1, ":", next_numbers[i][0])

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Linear Regression')
plt.scatter(next_positions, next_numbers, color='green', label='Predictions')
plt.xlabel('Position')
plt.ylabel('Number')
plt.title('Linear Regression Prediction')
plt.legend()
plt.grid(True)
plt.show()
