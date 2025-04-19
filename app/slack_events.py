import re
import yaml
import logging
import pandas as pd
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.models.blocks import SectionBlock, ActionsBlock, ButtonElement
from services.news_fetcher import fetch_news_articles
from services.sentiment_analyzer import analyze_crypto_sentiment
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

def handle_topic_analysis(topic, say):
    say(f"📡 Fetching news articles for *{topic}*...")
    articles = fetch_news_articles(topic)

    if articles:
        articles.sort(key=lambda x: x.get("publishedAt", ""), reverse=True)
        titles = [a.get("title", "") for a in articles]
        published_times = [a.get("publishedAt", "") for a in articles]

        # Sentiment analysis
        raw_sentiments = analyze_crypto_sentiment(titles, crypto_pipeline)

        def safe_extract(entry):
            if isinstance(entry, list) and entry:
                entry = entry[0]
            if isinstance(entry, tuple) and entry:
                entry = entry[0]
            if isinstance(entry, dict):
                label = entry.get("label", "neutral")
                score = entry.get("score", 0.0)
                if isinstance(label, (list, tuple)):
                    label = label[0]
                return {
                    "label": label.lower() if isinstance(label, str) else "neutral",
                    "score": score
                }
            return {"label": "neutral", "score": 0.0}

        sentiments = [safe_extract(s) for s in raw_sentiments]

        # ✅ Optional: Log first few cleaned items
        import pprint
        logger.debug("✅ Cleaned Sentiments Sample:")
        logger.debug(pprint.pformat(sentiments[:3]))

        # Post news + sentiment results with emojis
        message_lines = [f"*🗞️ Latest {topic.title()} News:*\n"]
        for article, sentiment in zip(articles, sentiments):
            label = sentiment.get("label", "unknown")
            score = sentiment.get("score", 0.0)

            # ✅ Safe guard against tuples
            if isinstance(label, (list, tuple)):
                label = label[0]
            label = label.lower() if isinstance(label, str) else "neutral"

            emoji = "🟢" if label == "bullish" else "🔴" if label == "bearish" else "⚪"
            url = article.get("url", "")
            title = article.get("title", "No Title")
            published_at = article.get("publishedAt", "Unknown time")

            message_lines.append(
                f"{emoji} *{title}*\nSentiment: *{label}* (Score: {score:.2f})\nPublished: {published_at}\n<{url}|Read more>\n"
            )

        formatted_news = "\n".join(message_lines)
        MAX_CHARS = 3900
        if len(formatted_news) > MAX_CHARS:
            say(formatted_news[:MAX_CHARS] + "\n…truncated due to Slack limit.")
        else:
            say(formatted_news)

        # Run Backtest
        try:
            say("🧠 Running backtest using sentiment-aligned strategy...")
            price_df = fetch_price_data(topic)

            if price_df.empty:
                say(f"⚠️ No price data found for *{topic}*. Backtest skipped.")
                return

            sentiment_df = pd.DataFrame(sentiments)
            sentiment_df["timestamp"] = pd.to_datetime(published_times)
            sentiment_df.set_index("timestamp", inplace=True)
            aligned_df = sentiment_df.resample("1h").first().dropna()

            stats = run_backtest(price_df, aligned_df)

            # Safely extract stats with fallback
            # Extract and flatten all values safely
            portfolio_value = stats.get("Final Portfolio Value", 0.0)
            sharpe_ratio = stats.get("Sharpe Ratio", {}).get("sharperatio", "N/A")

            trades = stats.get("Trade Analysis", {})
            total_trades = trades.get("total", 0)
            open_trades = trades.get("open", 0)
            closed_trades = trades.get("closed", 0)

            # Flatten win stats
            won = trades.get("won", {})
            win_total = won.get("total", 0)
            win_pnl = won.get("pnl", {}).get("total", 0.0)

            # Flatten loss stats
            lost = trades.get("lost", {})
            loss_total = lost.get("total", 0)
            loss_pnl = lost.get("pnl", {}).get("total", 0.0)

            # Format Sharpe
            sharpe_str = f"{sharpe_ratio:.2f}" if isinstance(sharpe_ratio, (int, float)) else "N/A"

            # ✅ Construct final message
            result_msg = (
                f"*📊 Backtest Summary for `{topic}`*\n\n"
                f"*💰 Portfolio Performance:*\n"
                f"• Final Value: `${portfolio_value:,.2f}`\n"
                f"• Sharpe Ratio: `{sharpe_str}`\n\n"
                f"*🧾 Trade Breakdown:*\n"
                f"• Total Trades: `{total_trades}` (✅ Closed: `{closed_trades}` / 🔁 Open: `{open_trades}`)\n"
                f"• ✅ Wins: `{win_total}` | Total PnL: `${win_pnl:.4f}`\n"
                f"• ❌ Losses: `{loss_total}` | Total Loss: `${loss_pnl:.4f}`\n"
            )

            say(result_msg)

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            say("❌ Backtest failed due to an error.")
    else:
        say(f"⚠️ No tweets or news articles found for *{topic}*. Rate limit might have occurred.")

@app.event("app_mention")
def handle_app_mention_events(body, say):
    user = body["event"]["user"]
    text = body["event"]["text"]

    if user in user_states and user_states[user]["awaiting_topic"]:
        topic = text.split('>', 1)[-1].strip()
        if topic:
            handle_topic_analysis(topic, say)
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
    handle_topic_analysis(topic, say)
    user_states[user]["awaiting_topic"] = False

if __name__ == "__main__":
    handler = SocketModeHandler(app, config["SLACK"]["APP_TOKEN"])
    handler.start()
