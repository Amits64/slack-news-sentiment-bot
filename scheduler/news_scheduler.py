import schedule
import time
import yaml
from services.news_fetcher import fetch_news_articles
from services.sentiment_analyzer import analyze_crypto_sentiment
from models.crypto_bert_model import load_crypto_bert_pipeline
from slack_bolt import App

with open("config/credentials.yaml", "r") as file:
    config = yaml.safe_load(file)

app = App(token=config["SLACK"]["BOT_TOKEN"])
crypto_pipeline = load_crypto_bert_pipeline()

def scheduled_news_update():
    channel = config["SLACK"].get("CHANNEL", "#general")
    for topic in ["bitcoin", "ethereum", "solana", "cardano", "cryptocoin", "digital asset", "finance"]:
        articles = fetch_news_articles(topic)
        if articles:
            # Add source tag if not already present
            for a in articles:
                if "twitter.com" in a.get("url", "").lower():
                    a["source"] = "Twitter"
                else:
                    a["source"] = "NewsAPI"

            titles = [a.get("title", "") for a in articles]
            sentiments = analyze_crypto_sentiment(titles, crypto_pipeline)

            # Append source tag into formatted message
            message = f"*🗞️ Latest {topic.title()} News:*\n"
            for article, sentiment in zip(articles, sentiments):
                label = sentiment.get("label", "Unknown")
                score = sentiment.get("score", 0.0)
                url = article.get("url", "")
                title = article.get("title", "No Title")
                published_at = article.get("publishedAt", "Unknown time")
                source = article.get("source", "Unknown")

                message += (
                    f"• [{source}] *{title}*\n"
                    f"  Sentiment: *{label}* (Score: {score:.2f})\n"
                    f"  Published: {published_at}\n"
                    f"  <{url}|Read more>\n\n"
                )

            app.client.chat_postMessage(channel=channel, text=message)

def start_scheduler():
    schedule.every().hour.do(scheduled_news_update)
    while True:
        schedule.run_pending()
        time.sleep(1)
