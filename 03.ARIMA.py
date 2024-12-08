!pip install yfinance

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

stock = "AAPL"
start_date = "2020-01-01"
end_date = pd.to_datetime("today").strftime("%Y-%m-%d")

stock_data = yf.download(stock, start=start_date, end=end_date)

data = stock_data["Close"]

from statsmodels.tsa.stattools import adfuller

result = adfuller(data)
print("ADF Statistic:", result[0])
print("p-value:", result[1])

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(data, order=(0, 1, 0))
fitted_model = model.fit()
print(fitted_model.summary())

predictions = fitted_model.forecast(steps=10)
print(predictions)

plt.figure(figsize=(14, 7))
plt.plot(data, label="Actual AAPL Stock Price", color="blue")
plt.plot(predictions, label="Predicted Stock Price", color="orange", linestyle="--")

plt.title("AAPL Stock Price and ARIMA Predictions")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.show()