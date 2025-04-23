import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from datetime import datetime
import yaml
from services.yfinance_fetcher import fetch_price_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CryptoHeatmap")

# Load Slack credentials
with open("config/credentials.yaml", "r") as file:
    config = yaml.safe_load(file)

SLACK_TOKEN = config["SLACK"]["BOT_TOKEN"]
SLACK_CHANNEL_ID = config["SLACK"]["CHANNEL_ID"]
slack_client = WebClient(token=SLACK_TOKEN)


def get_top_crypto_symbols():
    return [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
        "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD",
        "SHIB-USD", "TRX-USD", "LTC-USD", "UNI-USD", "LINK-USD",
        "ATOM-USD", "XLM-USD", "ETC-USD", "XMR-USD", "APT-USD",
        "IMX-USD", "NEAR-USD", "ICP-USD", "FIL-USD", "HBAR-USD"
    ]


def fetch_crypto_returns(symbols):
    data = {}
    for symbol in symbols:
        try:
            df = fetch_price_data(symbol)
            if len(df) < 2:
                continue
            latest_close = df["Close"].iloc[-1]
            previous_close = df["Close"].iloc[-2]
            percent_change = ((latest_close - previous_close) / previous_close) * 100
            data[symbol] = {
                "latest": latest_close,
                "change": round(percent_change, 2)
            }
        except Exception as e:
            logger.warning(f"Error fetching data for {symbol}: {e}")
    return data


def plot_crypto_heatmap(coin_data, save_path):
    fig, ax = plt.subplots(figsize=(18, 10), dpi=200)
    ax.set_facecolor("black")
    ax.axis("off")

    coins = list(coin_data.items())
    cols = 6
    rows = (len(coins) + cols - 1) // cols

    for i, (symbol, data) in enumerate(coins):
        row = i // cols
        col = i % cols

        name = symbol.replace("-USD", "")
        price = f"${data['latest']:.2f}"
        change = data['change']
        color = "#d9534f" if change < 0 else "#5cb85c"
        label = f"{name}\n{price}\n{change:+.2f}%"

        rect = Rectangle((col, rows - row), 1, 1, color=color, ec="black")
        ax.add_patch(rect)
        ax.text(col + 0.5, rows - row + 0.5, label, va='center', ha='center',
                fontsize=10, color="white", fontweight="bold")

    plt.xlim(0, cols)
    plt.ylim(0, rows + 1)
    plt.title(f"Crypto Heatmap — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
              fontsize=18, color="white", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def generate_gainers_losers_summary(data, top_n=3):
    sorted_data = sorted(data.items(), key=lambda x: x[1]['change'], reverse=True)

    gainers = sorted_data[:top_n]
    losers = sorted_data[-top_n:]

    gainers_text = "\n".join([f"• {sym.replace('-USD','')}: +{info['change']}%" for sym, info in gainers])
    losers_text = "\n".join([f"• {sym.replace('-USD','')}: {info['change']}%" for sym, info in reversed(losers)])

    summary = (
        "*:chart_with_upwards_trend: Top Gainers:*\n"
        f"{gainers_text}\n\n"
        "*:chart_with_downwards_trend: Top Losers:*\n"
        f"{losers_text}"
    )
    return summary


def send_to_slack(image_path, summary_text):
    try:
        with open(image_path, "rb") as file_content:
            slack_client.files_upload_v2(
                file=file_content,
                title="Crypto Heatmap",
                channel=SLACK_CHANNEL_ID,
                initial_comment=summary_text,
            )
        logger.info("✅ Heatmap uploaded to Slack.")
    except SlackApiError as e:
        logger.error(f"Slack upload error: {e.response.get('error')}")


def run_crypto_heatmap_post():
    symbols = get_top_crypto_symbols()
    returns_data = fetch_crypto_returns(symbols)
    if not returns_data:
        logger.warning("⚠️ No return data available.")
        return

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    image_file = f"crypto_heatmap_{timestamp}.png"

    plot_crypto_heatmap(returns_data, image_file)
    summary_text = generate_gainers_losers_summary(returns_data)
    send_to_slack(image_file, summary_text)
    os.remove(image_file)


if __name__ == "__main__":
    run_crypto_heatmap_post()
