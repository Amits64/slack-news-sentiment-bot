import yaml
import logging
import threading
import time
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from scheduler.news_scheduler import start_scheduler
from app.slack_events import app
from services.yfinance_fetcher import fetch_price_data
from services.backtest_engine import run_backtest
from models.crypto_bert_model import load_crypto_bert_pipeline
from services.news_fetcher import fetch_news_articles
from services.crypto_heatmap import run_crypto_heatmap_post
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
try:
    with open("config/credentials.yaml", "r") as file:
        config = yaml.safe_load(file)
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    exit(1)

# Initialize Slack client
slack_client = WebClient(token=config["SLACK"]["BOT_TOKEN"])
channel = config["SLACK"].get("CHANNEL", "#general")

# 🔁 Backtest Trigger with Slack Image Upload
def run_sentiment_backtest(symbol: str, topic: str):
    try:
        logger.info(f"Running backtest for {symbol} using topic: {topic}")

        price_df = fetch_price_data(symbol)
        articles = fetch_news_articles(topic)
        if not articles:
            logger.warning("No news articles found for topic.")
            return None

        titles = [a.get("title", "") for a in articles]
        published_times = [a.get("publishedAt", "") for a in articles]
        crypto_pipeline = load_crypto_bert_pipeline()
        sentiments = crypto_pipeline(titles)

        sentiment_df = pd.DataFrame(sentiments)
        sentiment_df["timestamp"] = pd.to_datetime(published_times)
        sentiment_df.set_index("timestamp", inplace=True)
        aligned_df = sentiment_df.resample("1H").first().dropna()

        performance_stats = run_backtest(
            price_df,
            aligned_df,
            slack_client=slack_client,
            channel=channel
        )
        return performance_stats

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return None

# 🔁 Periodically post heatmap using Slack files_upload_v2

def run_scheduled_heatmap():
    while True:
        try:
            logger.info("📊 Posting crypto heatmap to Slack...")
            run_crypto_heatmap_post()  # Make sure crypto_heatmap supports this param
        except Exception as e:
            logger.error(f"Error while posting heatmap: {e}")
        time.sleep(3600)  # 1 hour

# Run app
if __name__ == "__main__":
    threading.Thread(target=start_scheduler, daemon=True).start()
    threading.Thread(target=run_scheduled_heatmap, daemon=True).start()

    try:
        handler = SocketModeHandler(app, config["SLACK"]["APP_TOKEN"])
        logger.info("🚀 Starting Slack bot and scheduler...")
        handler.start()
    except Exception as e:
        logger.error(f"Failed to start Slack app: {e}")
