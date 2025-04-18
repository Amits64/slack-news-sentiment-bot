import logging

logger = logging.getLogger(__name__)

def analyze_crypto_sentiment(titles, pipeline):
    if not titles:
        logger.warning("No titles provided for sentiment analysis.")
        return []

    try:
        logger.info(f"Analyzing sentiment for {len(titles)} titles...")
        raw_results = pipeline(titles)

        # ✅ Flatten top_k=1 output: convert [[{...}], [{...}]] to [{...}, {...}]
        flattened = [r[0] if isinstance(r, list) else r for r in raw_results]

        logger.info(f"Sample sentiment output: {flattened[:1]}")
        return flattened

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return []
