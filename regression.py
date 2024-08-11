import numpy as np
from sklearn.linear_model import LinearRegression
data = [10, 323, 23, 98, 71, 45, 39, 212, 67, 463]
X = np.array(range(len(data))).reshape(-1, 1) # Independent variable
y = np.array(data).reshape(-1, 1) # Dependent variable
model = LinearRegression()
model.fit(X, y)
next_position = len(data) # Position of the next number
print("Next position:", next_position)
next_position = np.array([next_position]).reshape(-1, 1) # Reshape for prediction
next_number = model.predict(next_position)
print("Predicted next number:", next_number[0][0])