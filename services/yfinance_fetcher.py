import yfinance as yf
import pandas as pd

def fetch_price_data(symbol: str, period: str = "6mo", interval: str = "1h"):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)

    # ✅ Ensure flat column names — Backtrader requires this
    if isinstance(df.columns[0], tuple):
        df.columns = [col[0] for col in df.columns]  # Flatten multi-index

    # ✅ Reorder and trim only required columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    return df
