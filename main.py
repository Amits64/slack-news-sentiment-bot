import os
import yaml
import logging
import threading
from slack_bolt.adapter.socket_mode import SocketModeHandler
from scheduler.news_scheduler import start_scheduler
from app.slack_events import app
from services.yfinance_fetcher import fetch_price_data
from services.backtest_engine import run_backtest
from models.crypto_bert_model import load_crypto_bert_pipeline
from services.news_fetcher import fetch_news_articles
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

# Set environment variables for Slack (optional)
os.environ["SLACK_BOT_TOKEN"] = config["SLACK"]["BOT_TOKEN"]
os.environ["SLACK_APP_TOKEN"] = config["SLACK"]["APP_TOKEN"]

# 🔁 Optional Backtest Trigger (for testing purpose)
def run_sentiment_backtest(symbol: str, topic: str):
    try:
        logger.info(f"Running backtest for {symbol} using topic: {topic}")

        # 1. Get historical OHLCV data
        price_df = fetch_price_data(symbol)

        # 2. Fetch sentiment-scored news
        articles = fetch_news_articles(topic)
        if not articles:
            logger.warning("No news articles found for topic.")
            return

        titles = [a.get("title", "") for a in articles]
        published_times = [a.get("publishedAt", "") for a in articles]

        # 3. Apply CryptoBERT
        crypto_pipeline = load_crypto_bert_pipeline()
        sentiments = crypto_pipeline(titles)

        # 4. Create timestamped sentiment DataFrame
        sentiment_df = pd.DataFrame(sentiments)
        sentiment_df["timestamp"] = pd.to_datetime(published_times)
        sentiment_df.set_index("timestamp", inplace=True)
        aligned_df = sentiment_df.resample("1H").first().dropna()

        # 5. Run backtest
        performance_stats = run_backtest(price_df, aligned_df)
        logger.info("📊 Backtest completed!")
        logger.info(performance_stats)

    except Exception as e:
        logger.error(f"Backtest error: {e}")

# Run app
if __name__ == "__main__":
    # Run scheduler in background
    threading.Thread(target=start_scheduler, daemon=True).start()

    # Launch Slack app via Socket Mode
    try:
        handler = SocketModeHandler(app, config["SLACK"]["APP_TOKEN"])
        handler.start()
    except Exception as e:
        logger.error(f"Failed to start Slack app: {e}")
