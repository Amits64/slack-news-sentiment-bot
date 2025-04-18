import os
import yaml
import logging
import requests
import schedule
import time
import threading
from datetime import datetime, timedelta
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from transformers import pipeline
from dateutil import parser
from cachetools import TTLCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from YAML file
try:
    with open("config/credentials.yaml", "r") as file:
        config = yaml.safe_load(file)
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    exit(1)

# Set environment variables for Slack Bolt
os.environ["SLACK_BOT_TOKEN"] = config["SLACK"]["BOT_TOKEN"]
os.environ["SLACK_APP_TOKEN"] = config["SLACK"]["APP_TOKEN"]

# Initialize Slack app
app = App(token=os.environ["SLACK_BOT_TOKEN"])

# Initialize sentiment analysis pipeline
try:
    sentiment_pipeline = pipeline(
        "text-classification",
        model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    )
except Exception as e:
    logger.error(f"Failed to initialize sentiment analysis pipeline: {e}")
    exit(1)

# Initialize TTLCache for news articles
news_cache = TTLCache(maxsize=100, ttl=300)  # Cache up to 100 topics for 5 minutes
cache_lock = threading.Lock()

def fetch_news_articles(topic):
    """Fetch or retrieve cached news articles for a topic."""
    with cache_lock:
        if topic in news_cache:
            logger.info(f"Using cached data for topic: {topic}")
            return news_cache[topic]

    # Fetch from NewsAPI if not in cache
    api_key = config["NEWSAPI"]["API_KEY"]
    base_url = "https://newsapi.org/v2/everything"
    from_date = (datetime.utcnow() - timedelta(days=7)).date().isoformat()
    to_date = datetime.utcnow().date().isoformat()

    params = {
        "q": topic,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5,
        "apiKey": api_key
    }

    try:
        resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        with cache_lock:
            news_cache[topic] = articles
        return articles
    except requests.RequestException as e:
        logger.error(f"Error fetching news articles: {e}")
        return []

def format_sentiment_analysis(articles):
    """Format articles with sentiment and publication time for Slack."""
    messages = []
    for article in articles:
        title = article.get("title", "No Title")
        url = article.get("url", "")
        published_at = article.get("publishedAt", "")

        # Parse and format publication date
        if published_at:
            try:
                dt = parser.isoparse(published_at)
                pub_str = dt.strftime("%B %d, %Y at %I:%M %p")
            except Exception:
                pub_str = published_at
        else:
            pub_str = "Unknown date"

        try:
            sentiment = sentiment_pipeline(title)[0]
            label, score = sentiment["label"], sentiment["score"]
            messages.append(
                f"*{title}*\nPublished on: {pub_str}\n"
                f"Sentiment: *{label}* (Score: {score:.2f})\n<{url}|Read more>\n"
            )
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            messages.append(
                f"*{title}*\nPublished on: {pub_str}\n"
                f"Sentiment analysis failed.\n<{url}|Read more>\n"
            )
    return "\n".join(messages)

# Handle on-demand analysis via @app_mention
user_states = {}

@app.event("app_mention")
def handle_app_mention_events(body, say):
    user = body["event"]["user"]
    text = body["event"]["text"]

    if user in user_states and user_states[user]["awaiting_topic"]:
        topic = text.split('>', 1)[-1].strip()
        if topic:
            say(f"Fetching news articles for *{topic}*...")
            arts = fetch_news_articles(topic)
            if arts:
                say(format_sentiment_analysis(arts))
            else:
                say(f"No recent news articles found for *{topic}*.")
            user_states[user]["awaiting_topic"] = False
        else:
            say("Please provide a valid topic for analysis.")
    else:
        say("Hello! Please enter the cryptocoin/stock or any news topic for analysis...")
        user_states[user] = {"awaiting_topic": True}

def scheduled_news_update():
    """Scheduled job to fetch and post updates for tracked topics."""
    channel = config["SLACK"].get("CHANNEL", "#general")
    for topic in ["bitcoin", "ethereum", "solana", "cardano"]:
        articles = fetch_news_articles(topic)
        if articles:
            msg = f"*Latest {topic.title()} News:*\n" + format_sentiment_analysis(articles)
            app.client.chat_postMessage(channel=channel, text=msg)

def start_scheduler():
    """Continuously run scheduled pending jobs."""
    # Schedule the update every hour
    schedule.every().hour.do(scheduled_news_update)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    # Start scheduler in background thread
    threading.Thread(target=start_scheduler, daemon=True).start()

    # Start Slack app in Socket Mode (blocking)
    try:
        handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
        handler.start()
    except Exception as e:
        logger.error(f"Failed to start Slack app: {e}")
