!pip install yfinance

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

stock = "AAPL"
start_date = "2020-01-01"
end_date = pd.to_datetime("today").strftime("%Y-%m-%d")

stock_data = yf.download(stock, start=start_date, end=end_date)

data = stock_data["Close"]

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1] <= 0.05:
        print("The data is stationary.")
    else:
        print("The data is not stationary.")

check_stationarity(data)

best_aic = np.inf
best_order = None
best_model = None

for p in range(0, 3):  
    for d in range(0, 2):  
        for q in range(0, 3):  
            try:
                model = ARIMA(data, order=(p, d, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
                    best_model = results
            except:
                continue

print(f"Best ARIMA{best_order} AIC: {best_aic}")

fitted_model = best_model

forecast_steps = 10
forecast = fitted_model.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq="B")
forecast_series = pd.Series(forecast, index=forecast_index)

plt.figure(figsize=(14, 7))
plt.plot(data, label="Actual AAPL Stock Price", color="blue")
plt.plot(forecast_series, label="Forecasted Stock Price", color="orange", linestyle="--")

plt.title("AAPL Stock Price and ARIMA Forecast")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.show()