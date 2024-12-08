!pip install yfinance

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

stock = "AAPL"
start_date = "2020-01-01"
end_date = pd.to_datetime("today").strftime("%Y-%m-%d")

stock_data = yf.download(stock, start=start_date, end=end_date)

sma_short = stock_data["Close"].rolling(window=20).mean() 
sma_medium = stock_data["Close"].rolling(window=50).mean()  
sma_long = stock_data["Close"].rolling(window=100).mean()  

ema_short = stock_data["Close"].ewm(span=20, adjust=False).mean()  
ema_medium = stock_data["Close"].ewm(span=50, adjust=False).mean()  
ema_long = stock_data["Close"].ewm(span=100, adjust=False).mean()  

plt.figure(figsize=(14, 7))
plt.plot(stock_data["Close"], label="AAPL Stock Price", color="blue")
plt.plot(ema_short, label="20-Day EMA", color="orange")
plt.plot(ema_medium, label="50-Day EMA", color="green")
plt.plot(ema_long, label="100-Day EMA", color="red")

plt.title("AAPL Stock Price and Exponential Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(stock_data["Close"], label="AAPL Stock Price", color="blue")
plt.plot(sma_short, label="20-Day SMA", color="orange", linestyle="--")
plt.plot(sma_medium, label="50-Day SMA", color="green", linestyle="--")
plt.plot(sma_long, label="100-Day SMA", color="red", linestyle="--")
plt.plot(ema_short, label="20-Day EMA", color="purple")
plt.plot(ema_medium, label="50-Day EMA", color="brown")
plt.plot(ema_long, label="100-Day EMA", color="pink")

plt.title("AAPL Stock Price, Simple and Exponential Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.show()