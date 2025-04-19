import logging

logger = logging.getLogger(__name__)

def analyze_crypto_sentiment(titles, pipeline):
    if not titles:
        logger.warning("No titles provided for sentiment analysis.")
        return []

    try:
        logger.info(f"Analyzing sentiment for {len(titles)} titles...")
        raw_results = pipeline(titles)

        def safe_extract(entry):
            if isinstance(entry, list) and entry:
                entry = entry[0]
            if isinstance(entry, tuple) and entry:
                entry = entry[0]
            if isinstance(entry, dict):
                label = entry.get("label", "neutral")
                score = entry.get("score", 0.0)
                return {
                    "label": label.lower() if isinstance(label, str) else "neutral",
                    "score": score
                }
            return {"label": "neutral", "score": 0.0}

        cleaned = [safe_extract(r) for r in raw_results]

        logger.info(f"Sample sentiment output: {cleaned[:1]}")
        return cleaned

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return []
