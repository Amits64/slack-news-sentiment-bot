import schedule
import time
import threading
import yaml
from services.news_fetcher import fetch_news_articles
from services.sentiment_analyzer import analyze_crypto_sentiment
from services.message_formatter import format_sentiment_results
from models.crypto_bert_model import load_crypto_bert_pipeline
from slack_bolt import App

with open("config/credentials.yaml", "r") as file:
    config = yaml.safe_load(file)

app = App(token=config["SLACK"]["BOT_TOKEN"])
crypto_pipeline = load_crypto_bert_pipeline()

def scheduled_news_update():
    channel = config["SLACK"].get("CHANNEL", "#general")
    for topic in ["bitcoin", "ethereum", "solana", "cardano"]:
        articles = fetch_news_articles(topic)
        if articles:
            titles = [a.get("title", "") for a in articles]
            sentiments = analyze_crypto_sentiment(titles, crypto_pipeline)
            message = f"*Latest {topic.title()} News:*\n" + format_sentiment_results(articles, sentiments)
            app.client.chat_postMessage(channel=channel, text=message)

def start_scheduler():
    schedule.every().hour.do(scheduled_news_update)
    while True:
        schedule.run_pending()
        time.sleep(1)
