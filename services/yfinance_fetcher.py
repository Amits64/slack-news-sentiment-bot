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

def get_top_volume_symbols(limit=20):
    major_cryptos = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
        "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD",
        "LINK-USD", "LTC-USD", "BCH-USD", "TRX-USD",
        "XLM-USD", "APT-USD", "ARB-USD", "ATOM-USD"
    ]

    volume_data = []
    for symbol in major_cryptos:
        try:
            df = yf.download(symbol, period="1d", interval="1h", progress=False)
            if not df.empty and "Volume" in df.columns:
                volume = df["Volume"].sum()
                volume_data.append((symbol, volume))
                print(f"✅ {symbol} volume: {volume}")
            else:
                print(f"⚠️ {symbol} returned empty or no 'Volume'")
        except Exception as e:
            print(f"❌ Failed to fetch {symbol}: {e}")

    volume_data = [item for item in volume_data if isinstance(item[1], (int, float))]
    volume_data.sort(key=lambda x: x[1], reverse=True)

    top = [s[0] for s in volume_data[:limit]]
    print(f"\n🎯 Top {len(top)} Symbols by Volume: {top}")
    return top
