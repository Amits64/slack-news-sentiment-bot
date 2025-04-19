from dateutil import parser

def format_sentiment_results(articles, sentiments):
    messages = []
    for article, sentiment in zip(articles, sentiments):
        title = article.get("title", "No Title")
        url = article.get("url", "")
        published_at = article.get("publishedAt", "")

        try:
            pub_str = parser.isoparse(published_at).strftime("%B %d, %Y at %I:%M %p") if published_at else "Unknown date"
        except Exception:
            pub_str = published_at or "Unknown date"

        # ✅ FIX: unwrap top_k=1 result which is a single-element list
        if isinstance(sentiment, list):
            sentiment = sentiment[0]

        label = sentiment.get("label", "Unknown")
        score = sentiment.get("score", 0.0)
        messages.append(
            f"*{title}*\nPublished on: {pub_str}\nSentiment: *{label}* (Score: {score:.2f})\n<{url}|Read more>\n"
        )
    return "\n".join(messages)
