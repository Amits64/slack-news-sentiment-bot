import requests
import tweepy
import yaml
import logging
import threading
from datetime import datetime, timedelta
from cachetools import TTLCache

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Cache to avoid excessive API hits
news_cache = TTLCache(maxsize=100, ttl=300)
cache_lock = threading.Lock()

# Load API keys from config
with open("config/credentials.yaml", "r") as file:
    config = yaml.safe_load(file)

# Credentials
newsapi_key = config["NEWSAPI"]["API_KEY"]
bearer_token = config["TWITTER"]["X_BEARER_TOKEN"]

# Twitter client
twitter_client = tweepy.Client(bearer_token=bearer_token)

def fetch_from_newsapi(topic: str, max_results: int = 5):
    base_url = "https://newsapi.org/v2/everything"
    from_date = (datetime.utcnow() - timedelta(days=7)).date().isoformat()
    to_date = datetime.utcnow().date().isoformat()

    params = {
        "q": topic,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": max_results,
        "apiKey": newsapi_key
    }

    try:
        logger.info(f"📡 Fetching NewsAPI results for topic: {topic}")
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])

        return [{
            "title": article["title"],
            "publishedAt": article["publishedAt"],
            "url": article["url"]
        } for article in articles]

    except requests.RequestException as e:
        logger.warning(f"NewsAPI fetch failed: {e}")
        return []

def fetch_from_twitter(topic: str, max_results: int = 10):
    query = f"{topic} -is:retweet lang:en"
    start_time = (datetime.utcnow() - timedelta(hours=12)).isoformat("T") + "Z"

    try:
        logger.info(f"📡 Fetching Twitter results for topic: {topic}")
        tweets = twitter_client.search_recent_tweets(
            query=query,
            max_results=max_results,
            start_time=start_time,
            tweet_fields=["created_at", "text", "lang"]
        )

        if not tweets.data:
            return []

        return [{
            "title": tweet.text,
            "publishedAt": tweet.created_at.isoformat(),
            "url": f"https://twitter.com/i/web/status/{tweet.id}"
        } for tweet in tweets.data if tweet.lang == "en"]

    except Exception as e:
        logger.warning(f"Twitter fetch failed: {e}")
        return []

def fetch_news_articles(topic: str, max_results: int = 10):
    with cache_lock:
        if topic in news_cache:
            logger.info(f"✅ Using cached news for topic: {topic}")
            return news_cache[topic]

    # Fetch from both sources
    newsapi_articles = fetch_from_newsapi(topic, max_results)
    twitter_articles = fetch_from_twitter(topic, max_results)

    combined = newsapi_articles + twitter_articles
    logger.info(f"📰 Fetched total {len(combined)} articles for topic: {topic}")

    with cache_lock:
        news_cache[topic] = combined

    return combined
