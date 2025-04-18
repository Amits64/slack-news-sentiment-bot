import yaml
from slack_bolt import App
from services.news_fetcher import fetch_news_articles
from services.sentiment_analyzer import analyze_crypto_sentiment
from services.message_formatter import format_sentiment_results
from models.crypto_bert_model import load_crypto_bert_pipeline

# Load Slack token from YAML
with open("config/credentials.yaml", "r") as file:
    config = yaml.safe_load(file)

SLACK_BOT_TOKEN = config["SLACK"]["BOT_TOKEN"]

# Initialize Slack App with token
app = App(token=SLACK_BOT_TOKEN)
crypto_pipeline = load_crypto_bert_pipeline()

user_states = {}

@app.event("app_mention")
def handle_app_mention_events(body, say):
    user = body["event"]["user"]
    text = body["event"]["text"]

    if user in user_states and user_states[user]["awaiting_topic"]:
        topic = text.split('>', 1)[-1].strip()
        if topic:
            say(f"Fetching news articles for *{topic}*...")
            articles = fetch_news_articles(topic)
            if articles:
                titles = [a.get("title", "") for a in articles]
                sentiments = analyze_crypto_sentiment(titles, crypto_pipeline)
                say(format_sentiment_results(articles, sentiments))
            else:
                say(f"No recent news articles found for *{topic}*.")
            user_states[user]["awaiting_topic"] = False
        else:
            say("Please provide a valid topic.")
    else:
        say("Hello! Enter the cryptocoin/stock/topic you'd like analysis for...")
        user_states[user] = {"awaiting_topic": True}
