import requests
from datetime import datetime, timedelta
from cachetools import TTLCache
import threading
import logging
import yaml

logger = logging.getLogger(__name__)
news_cache = TTLCache(maxsize=100, ttl=300)
cache_lock = threading.Lock()

with open("config/credentials.yaml", "r") as file:
    config = yaml.safe_load(file)

def fetch_news_articles(topic):
    with cache_lock:
        if topic in news_cache:
            logger.info(f"Using cached data for topic: {topic}")
            return news_cache[topic]

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
        articles = resp.json().get("articles", [])
        with cache_lock:
            news_cache[topic] = articles
        return articles
    except requests.RequestException as e:
        logger.error(f"Error fetching news articles: {e}")
        return []
