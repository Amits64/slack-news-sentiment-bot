import schedule
import time
import yaml
import logging
from services.news_fetcher import fetch_news_articles
from services.sentiment_analyzer import analyze_crypto_sentiment
from services.message_formatter import format_sentiment_results
from models.crypto_bert_model import load_crypto_bert_pipeline
from slack_bolt import App

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load Slack config
with open("config/credentials.yaml", "r") as file:
    config = yaml.safe_load(file)

SLACK_BOT_TOKEN = config["SLACK"]["BOT_TOKEN"]
SLACK_CHANNEL = config["SLACK"].get("CHANNEL", "#general")

app = App(token=SLACK_BOT_TOKEN)
crypto_pipeline = load_crypto_bert_pipeline()

def scheduled_news_update():
    topics = [
        "bitcoin", "ethereum", "solana", "cardano",
        "crypto", "etf", "btc-usd", "eth-usd", "xrp-usd"
    ]
    for topic in topics:
        logger.info(f"🔁 Scheduled fetch for: {topic}")
        articles = fetch_news_articles(topic)

        if not articles:
            logger.warning(f"⚠️ No articles found for {topic}")
            continue

        titles = [a.get("title", "") for a in articles]
        if not titles:
            logger.warning(f"⚠️ No titles extracted from articles for {topic}")
            continue

        sentiments = analyze_crypto_sentiment(titles, crypto_pipeline)

        if not sentiments:
            logger.warning(f"⚠️ Sentiment analysis failed for topic: {topic}")
            continue

        message = f"*📰 Latest News Sentiment on {topic.title()}*\n"
        message += format_sentiment_results(articles, sentiments)

        try:
            app.client.chat_postMessage(
                channel=SLACK_CHANNEL,
                text=message
            )
            logger.info(f"✅ Sentiment update sent for {topic}")
        except Exception as e:
            logger.error(f"❌ Failed to post update to Slack for {topic}: {e}")

def start_scheduler():
    schedule.every().hour.do(scheduled_news_update)
    while True:
        schedule.run_pending()
        time.sleep(1)
