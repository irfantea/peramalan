!pip install yfinance
!pip install prophet

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

stock = "AAPL"
start_date = "2020-01-01"
end_date = pd.to_datetime("today").strftime("%Y-%m-%d")

stock_data = yf.download(stock, start=start_date, end=end_date)

data = stock_data[["Close"]].reset_index()
data.columns = ["ds", "y"] 

model = Prophet()
model.fit(data)

future = model.make_future_dataframe(periods=30) 
forecast = model.predict(future)

plt.figure(figsize=(14, 7))
model.plot(forecast)
plt.title("AAPL Stock Price Forecast using Prophet")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid()
plt.show()