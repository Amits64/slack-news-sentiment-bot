import re
import yaml
import logging
import pandas as pd
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.models.blocks import SectionBlock, ActionsBlock, ButtonElement
from services.news_fetcher import fetch_news_articles
from services.sentiment_analyzer import analyze_crypto_sentiment
from services.message_formatter import format_sentiment_results
from models.crypto_bert_model import load_crypto_bert_pipeline
from services.yfinance_fetcher import fetch_price_data
from services.backtest_engine import run_backtest

# Load Slack config
with open("config/credentials.yaml", "r") as file:
    config = yaml.safe_load(file)

SLACK_BOT_TOKEN = config["SLACK"]["BOT_TOKEN"]
app = App(token=SLACK_BOT_TOKEN)
crypto_pipeline = load_crypto_bert_pipeline()
logger = logging.getLogger(__name__)

user_states = {}

# List of topics for user to select
TOPICS = ["BTC-USD", "ETH-USD", "SOL-USD",
          "XRP-USD", "nifty50", "sensex", "ETF",
          "trump", "cryptocoin", "Stock Market"]

@app.event("app_mention")
def handle_app_mention_events(body, say):
    user = body["event"]["user"]
    text = body["event"]["text"]

    if user in user_states and user_states[user]["awaiting_topic"]:
        topic = text.split('>', 1)[-1].strip()
        if topic:
            say(f"📡 Fetching news articles for *{topic}*...")
            articles = fetch_news_articles(topic)

            if articles:
                titles = [a.get("title", "") for a in articles]
                published_times = [a.get("publishedAt", "") for a in articles]

                # Sentiment analysis
                sentiments = analyze_crypto_sentiment(titles, crypto_pipeline)

                # Post news + sentiment results
                formatted_news = format_sentiment_results(articles, sentiments)
                say(formatted_news)

                # Run Backtest
                try:
                    say("🧠 Running backtest using sentiment-aligned strategy...")
                    price_df = fetch_price_data(topic)  # BTC-USD etc.

                    sentiment_df = pd.DataFrame(sentiments)
                    sentiment_df["timestamp"] = pd.to_datetime(published_times)
                    sentiment_df.set_index("timestamp", inplace=True)
                    aligned_df = sentiment_df.resample("1h").first().dropna()

                    stats = run_backtest(price_df, aligned_df)

                    result_msg = (
                        f"*Backtest Results for {topic}:*\n"
                        f"• 💰 Final Portfolio Value: ${stats['Final Portfolio Value']:.2f}\n"
                        f"• 📈 Sharpe Ratio: {stats['Sharpe Ratio'].get('sharperatio', 'N/A'):.2f}\n"
                        f"• 🧾 Trades: Total={stats['Trade Analysis'].get('total', 0)}, "
                        f"Win={stats['Trade Analysis'].get('won', 0)}, "
                        f"Loss={stats['Trade Analysis'].get('lost', 0)}"
                    )

                    say(result_msg)

                except Exception as e:
                    logger.error(f"Backtest failed: {e}")
                    say("❌ Backtest failed due to an error.")
            else:
                say(f"No recent news articles found for *{topic}*.")
            user_states[user]["awaiting_topic"] = False
        else:
            say("Please provide a valid topic.")
    else:
        blocks = [
            SectionBlock(text="👋 Hello! Please select a topic you'd like analysis for:"),
            ActionsBlock(elements=[
                ButtonElement(text=topic, action_id=f"select_topic_{topic.lower()}")
                for topic in TOPICS
            ])
        ]
        say(blocks=blocks)
        user_states[user] = {"awaiting_topic": True}

@app.action(re.compile(r"select_topic_(.*)"))
def handle_topic_selection(ack, body, say):
    ack()
    user = body["user"]["id"]
    topic = body["actions"][0]["text"]["text"]
    say(f"📡 Fetching news articles for *{topic}*...")
    articles = fetch_news_articles(topic)

    if articles:
        titles = [a.get("title", "") for a in articles]
        published_times = [a.get("publishedAt", "") for a in articles]

        # Sentiment analysis
        sentiments = analyze_crypto_sentiment(titles, crypto_pipeline)

        # Post news + sentiment results
        formatted_news = format_sentiment_results(articles, sentiments)
        say(formatted_news)

        # Run Backtest
        try:
            say("🧠 Running backtest using sentiment-aligned strategy...")
            price_df = fetch_price_data(topic)  # BTC-USD etc.

            sentiment_df = pd.DataFrame(sentiments)
            sentiment_df["timestamp"] = pd.to_datetime(published_times)
            sentiment_df.set_index("timestamp", inplace=True)
            aligned_df = sentiment_df.resample("1h").first().dropna()

            stats = run_backtest(price_df, aligned_df)

            result_msg = (
                f"*Backtest Results for {topic}:*\n"
                f"• 💰 Final Portfolio Value: ${stats['Final Portfolio Value']:.2f}\n"
                f"• 📈 Sharpe Ratio: {stats['Sharpe Ratio'].get('sharperatio', 'N/A'):.2f}\n"
                f"• 🧾 Trades: Total={stats['Trade Analysis'].get('total', 0)}, "
                f"Win={stats['Trade Analysis'].get('won', 0)}, "
                f"Loss={stats['Trade Analysis'].get('lost', 0)}"
            )

            say(result_msg)

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            say("❌ Backtest failed due to an error.")
    else:
        say(f"No recent news articles found for *{topic}*.")
    user_states[user]["awaiting_topic"] = False

if __name__ == "__main__":
    handler = SocketModeHandler(app, config["SLACK"]["APP_TOKEN"])
    handler.start()
