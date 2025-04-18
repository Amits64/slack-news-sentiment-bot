[![CI](https://github.com/Amits64/slack-news-sentiment-bot/actions/workflows/docker-image.yml/badge.svg)](https://github.com/Amits64/slack-news-sentiment-bot/actions/workflows/docker-image.yml)
# Slack News Sentiment Bot

![Python](https://img.shields.io/badge/python-3.9-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A Slack bot that fetches the latest news articles on specified topics, performs sentiment analysis using a fine-tuned DistilRoBERTa model, and posts the summarized results directly into your Slack channels.

## Features

- Fetches recent news articles from NewsAPI based on user-specified topics.
- Performs sentiment analysis on article titles using a financial news sentiment model.
- Posts formatted summaries with sentiment scores to Slack channels.
- Caches recent queries to minimize redundant API calls.
- Supports scheduled updates for predefined topics.

## Installation

### Prerequisites

- Python 3.9
- Slack workspace with permissions to add bots
- NewsAPI account for API key

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/slack-news-sentiment-bot.git
   cd slack-news-sentiment-bot
   ```


2. **Set up a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```


3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```


4. **Configure environment variables:**

   Create a `.env` file in the root directory and add the following:

   ```env
   SLACK_BOT_TOKEN=your-slack-bot-token
   SLACK_APP_TOKEN=your-slack-app-token
   NEWSAPI_API_KEY=your-newsapi-key
   SLACK_CHANNEL=your-slack-channel-id
   ```


   Alternatively, set these variables directly in your environment.

## Usage

Run the bot using the following command:


```bash
python bot.py
```


Once running, you can interact with the bot in Slack by mentioning it and providing a topic:


```
@NewsSentimentBot Bitcoin
```


The bot will respond with the latest news articles on Bitcoin along with sentiment analysis.

## Technologies Used

- [Python 3.9](https://www.python.org/)
- [Slack Bolt for Python](https://slack.dev/bolt-python/)
- [Transformers](https://huggingface.co/transformers/)
- [NewsAPI](https://newsapi.org/)
- [Cachetools](https://pypi.org/project/cachetools/)
- [Schedule](https://pypi.org/project/schedule/)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library and pre-trained models.
- [Slack](https://api.slack.com/) for the Slack API and Bolt framework.
- [NewsAPI](https://newsapi.org/) for providing news data.

---

