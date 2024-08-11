import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Sales history data
data = [10, 323, 23, 98, 71, 45, 39, 212, 67, 463]

# Fit ARIMA model
model = ARIMA(data, order=(1,1,1))  # Example order, you can adjust this
model_fit = model.fit()

# Forecast next 5 sales
forecast = model_fit.forecast(steps=5)

# Print forecasted sales
for i, sales in enumerate(forecast):
    print(f"Predicted sales for tomorrow + {i+1} day: {sales:.2f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data, color='blue', label='Actual Sales')
plt.plot(range(len(data), len(data) + 5), forecast, color='red', linestyle='--', marker='o', label='Forecasted Sales')
plt.xlabel('Day')
plt.ylabel('Sales')
plt.title('ARIMA Forecasting')
plt.legend()
plt.grid(True)
plt.show()
