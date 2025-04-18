[![CI](https://github.com/Amits64/slack-news-sentiment-bot/actions/workflows/docker-image.yml/badge.svg)](https://github.com/Amits64/slack-news-sentiment-bot/actions/workflows/docker-image.yml)
# Slack News Sentiment Bot

![Python](https://img.shields.io/badge/python-3.9-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A Slack bot that fetches the latest news articles on specified topics, performs sentiment analysis using the CryptoBERT model (fine-tuned on financial/crypto social media text), runs a sentiment-aligned trading backtest using Backtrader, and posts the summarized results directly into your Slack channels.

## Features

- Fetches recent news articles using NewsAPI.
- Performs sentiment analysis using a domain-specific CryptoBERT model.
- Schedules regular sentiment updates.
- Displays formatted summaries in Slack with sentiment scores and source links.
- Performs backtesting using historical OHLCV data from Yahoo Finance (`yfinance`).
- Posts a trading performance summary including portfolio value, Sharpe ratio, and trade stats.
- Interactive Slack UI with button-based topic selection.

## Installation

### Prerequisites

- Python 3.9+
- Slack App with bot and socket mode permissions
- NewsAPI API key

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/slack-news-sentiment-bot.git
   cd slack-news-sentiment-bot
   ```

2. **Set up a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set your configuration:**

   Create a `config/credentials.yaml` file:

   ```yaml
   SLACK:
     BOT_TOKEN: xoxb-...
     APP_TOKEN: xapp-...
     CHANNEL_ID: C123456789

   NEWSAPI:
     API_KEY: your-newsapi-key
   ```

## Usage

Run the bot using:

```bash
python main.py
```

Once running, you can:

- Mention the bot in any channel: `@NewsSentimentBot BTC-USD`
- Or use interactive buttons like `Bitcoin`, `Ethereum`, etc., to trigger analysis

The bot will:

1. Fetch news for the topic
2. Perform sentiment analysis
3. Backtest the strategy using past price data
4. Post both sentiment results and backtest performance in Slack

## Technologies Used

- [Python 3.9+](https://www.python.org/)
- [Slack Bolt](https://slack.dev/bolt-python/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [CryptoBERT Model](https://huggingface.co/ElKulako/cryptobert)
- [Backtrader](https://www.backtrader.com/)
- [NewsAPI](https://newsapi.org/)
- [YFinance](https://pypi.org/project/yfinance/)
- [Schedule](https://pypi.org/project/schedule/)

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and commit
4. Push and open a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the pre-trained NLP models
- [Slack](https://api.slack.com/) for the Bolt framework
- [NewsAPI](https://newsapi.org/) for real-time news data
- [Yahoo Finance](https://finance.yahoo.com/) for historical pricing

