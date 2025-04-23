import re
import yaml
import logging
import pandas as pd
<<<<<<< HEAD
import tempfile
import matplotlib.pyplot as plt
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
=======
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
>>>>>>> d8e31c57aae9838d2c072ba3edd4080cd03e4182
from slack_sdk.models.blocks import SectionBlock, ActionsBlock, ButtonElement
from services.news_fetcher import fetch_news_articles
from services.sentiment_analyzer import analyze_crypto_sentiment
from models.crypto_bert_model import load_crypto_bert_pipeline
from services.yfinance_fetcher import fetch_price_data
from services.backtest_engine import run_backtest
<<<<<<< HEAD
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
=======
>>>>>>> d8e31c57aae9838d2c072ba3edd4080cd03e4182

# Load Slack config
with open("config/credentials.yaml", "r") as file:
    config = yaml.safe_load(file)

SLACK_BOT_TOKEN = config["SLACK"]["BOT_TOKEN"]
<<<<<<< HEAD
SLACK_CHANNEL = config["SLACK"].get("CHANNEL", "#general")
app = App(token=SLACK_BOT_TOKEN)
slack_client = WebClient(token=SLACK_BOT_TOKEN)
crypto_pipeline = load_crypto_bert_pipeline()

user_states = {}
TOPICS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "nifty50", "sensex",
          "ETF", "trump", "cryptocoin", "Stock Market"]


def generate_trade_plot(price_df, trades, topic):
    """Generate and save trade execution plot."""
    try:
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot price
        price_df['Close'].plot(ax=ax, label='Price', color='navy', alpha=0.7)

        # Plot trades
        for _, row in trades.iterrows():
            entry_color = 'green' if row['side'] == 'bullish' else 'red'
            exit_color = 'darkgreen' if row['side'] == 'bullish' else 'darkred'

            ax.scatter(row['entry_dt'], row['entry_price'],
                       marker='^', s=100, color=entry_color, label='Entry')
            ax.scatter(row['exit_dt'], row['exit_price'],
                       marker='v', s=100, color=exit_color, label='Exit')

        ax.set_title(f'Trade Execution for {topic}', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)

        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.grid(True)

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plt.savefig(temp_file.name, bbox_inches='tight', dpi=100)
        plt.close()

        return temp_file.name

    except Exception as e:
        logger.error(f"Error generating plot: {e}")
        return None


def upload_plot_to_slack(file_path, topic):
    """Upload plot to Slack and return permalink."""
    try:
        response = slack_client.files_upload_v2(
            channels=SLACK_CHANNEL,
            file=file_path,
            title=f"Backtest Results for {topic}",
            initial_comment="Here's the trade execution chart:"
        )
        return response['file']['permalink']
    except Exception as e:
        logger.error(f"Error uploading plot to Slack: {e}")
        return None


def handle_topic_analysis(topic, say):
    """Handle complete topic analysis workflow."""
    try:
        # Fetch news articles
        say(f"📡 Fetching news articles for *{topic}*...")
        articles = fetch_news_articles(topic)

        if not articles:
            say(f"⚠️ No news articles found for *{topic}*.")
            return

        # Process articles
=======
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
>>>>>>> d8e31c57aae9838d2c072ba3edd4080cd03e4182
        articles.sort(key=lambda x: x.get("publishedAt", ""), reverse=True)
        titles = [a.get("title", "") for a in articles]
        published_times = [a.get("publishedAt", "") for a in articles]

<<<<<<< HEAD
        # Analyze sentiment
        raw_sentiments = analyze_crypto_sentiment(titles, crypto_pipeline)

        def safe_extract(entry):
            if isinstance(entry, (list, tuple)) and entry:
                entry = entry[0]
            if isinstance(entry, dict):
                return {
                    "label": entry.get("label", "neutral").lower(),
                    "score": float(entry.get("score", 0.0))
=======
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
>>>>>>> d8e31c57aae9838d2c072ba3edd4080cd03e4182
                }
            return {"label": "neutral", "score": 0.0}

        sentiments = [safe_extract(s) for s in raw_sentiments]

<<<<<<< HEAD
        # Post news summary
        message_lines = [f"*🗞️ Latest {topic.title()} News:*\n"]
        for article, sentiment in zip(articles, sentiments):
            label = sentiment["label"]
            score = sentiment["score"]
            emoji = "🟢" if label == "bullish" else "🔴" if label == "bearish" else "⚪"
            message_lines.append(
                f"{emoji} *{article.get('title', 'No Title')}*\n"
                f"Sentiment: *{label}* (Score: {score:.2f})\n"
                f"Published: {article.get('publishedAt', 'Unknown time')}\n"
                f"<{article.get('url', '')}|Read more>\n"
            )

        say("\n".join(message_lines)[:3900])  # Truncate if too long

        # Run backtest
        say("🧠 Running backtest using sentiment-aligned strategy...")
        price_df = fetch_price_data(topic)

        if price_df.empty:
            say(f"⚠️ No price data found for *{topic}*. Backtest skipped.")
            return

        # Prepare sentiment data
        sentiment_df = pd.DataFrame(sentiments)
        sentiment_df["timestamp"] = pd.to_datetime(published_times)
        sentiment_df.set_index("timestamp", inplace=True)
        aligned_df = sentiment_df.resample("1h").first().dropna()

        # Execute backtest
        stats = run_backtest(
            price_df=price_df,
            sentiment_df=aligned_df,
            slack_client=slack_client,
            channel=SLACK_CHANNEL
        )

        # Generate and upload plot
        plot_path = generate_trade_plot(price_df, stats.get('all_trades', pd.DataFrame()), topic)
        plot_url = None

        if plot_path:
            plot_url = upload_plot_to_slack(plot_path, topic)
            try:
                os.unlink(plot_path)
            except:
                pass

        # Prepare results message
        result_msg = [
            f"*📊 Backtest Summary for `{topic}`*",
            "",
            "*💰 Portfolio Performance:*",
            f"• Final Value: ${stats.get('final_value', 0):,.2f}",
            f"• Sharpe Ratio: {stats.get('sharpe_ratio', 'N/A')}",
            "",
            "*🧾 Trade Breakdown:*",
            f"• Total Trades: {len(stats.get('all_trades', []))}",
            f"• Average PnL: ${stats.get('avg_pnl', 0):.2f}",
            f"• Win Rate: {stats.get('win_rate', 0):.1%}"
        ]

        if plot_url:
            result_msg.extend(["", f"*📈 Trade Execution Chart:* {plot_url}"])

        say("\n".join(result_msg))

    except Exception as e:
        logger.error(f"Error in topic analysis: {e}", exc_info=True)
        say(f"❌ Error processing {topic}: {str(e)}")


@app.event("app_mention")
def handle_app_mention_events(body, say):
    """Handle app mention events."""
=======
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
>>>>>>> d8e31c57aae9838d2c072ba3edd4080cd03e4182
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

<<<<<<< HEAD

@app.action(re.compile(r"select_topic_(.*)"))
def handle_topic_selection(ack, body, say):
    """Handle topic selection from buttons."""
=======
@app.action(re.compile(r"select_topic_(.*)"))
def handle_topic_selection(ack, body, say):
>>>>>>> d8e31c57aae9838d2c072ba3edd4080cd03e4182
    ack()
    user = body["user"]["id"]
    topic = body["actions"][0]["text"]["text"]
    handle_topic_analysis(topic, say)
    user_states[user]["awaiting_topic"] = False

<<<<<<< HEAD

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting Slack bot...")
        handler = SocketModeHandler(app, config["SLACK"]["APP_TOKEN"])
        handler.start()
    except Exception as e:
        logger.error(f"Failed to start Slack app: {e}")
=======
if __name__ == "__main__":
    handler = SocketModeHandler(app, config["SLACK"]["APP_TOKEN"])
    handler.start()
>>>>>>> d8e31c57aae9838d2c072ba3edd4080cd03e4182
