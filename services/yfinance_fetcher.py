import yfinance as yf
import pandas as pd

def fetch_price_data(symbol: str, period: str = "6mo", interval: str = "1h"):
    """
    Fetch historical price data for a given symbol using yfinance.
    """
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df
