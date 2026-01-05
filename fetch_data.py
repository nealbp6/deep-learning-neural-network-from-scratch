# fetch_data.py
import yfinance as yf
import numpy as np

def get_data(symbol="^SPX", interval="1h", period="max"):
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    data = data.dropna()

    data["return"] = data["Close"].pct_change()
    data = data["return"].dropna()

    return data.values