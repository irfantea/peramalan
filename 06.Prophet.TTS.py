!pip install yfinance
!pip install prophet

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.model_selection import train_test_split

stock = "AAPL"
start_date = "2020-01-01"
end_date = pd.to_datetime("today").strftime("%Y-%m-%d")

stock_data = yf.download(stock, start=start_date, end=end_date)

data = stock_data[["Close"]].reset_index()
data.columns = ["ds", "y"] 

train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

model = Prophet()
model.fit(train_data)

future = model.make_future_dataframe(periods=len(test_data)) 
forecast = model.predict(future)

forecast_test = forecast[forecast["ds"].isin(test_data["ds"])]

plt.figure(figsize=(14, 7))
plt.plot(train_data["ds"], train_data["y"], label="Train Data", color="blue")
plt.plot(test_data["ds"], test_data["y"], label="Test Data", color="orange")
plt.plot(forecast["ds"], forecast["yhat"], label="Forecasted Data", color="green", linestyle="--")
plt.scatter(forecast_test["ds"], forecast_test["yhat"], color="red", label="Forecasted Test Data", marker="x")
plt.title("AAPL Stock Price Forecast using Prophet")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.show()

fig = model.plot_components(forecast)
plt.show()