import os
import yaml
import logging
import threading
from slack_bolt.adapter.socket_mode import SocketModeHandler
from scheduler.news_scheduler import start_scheduler
from app.slack_events import app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
try:
    with open("config/credentials.yaml", "r") as file:
        config = yaml.safe_load(file)
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    exit(1)

# Optional: set as env vars (if needed later)
os.environ["SLACK_BOT_TOKEN"] = config["SLACK"]["BOT_TOKEN"]
os.environ["SLACK_APP_TOKEN"] = config["SLACK"]["APP_TOKEN"]

if __name__ == "__main__":
    # Start scheduled background job
    threading.Thread(target=start_scheduler, daemon=True).start()

    # Start Slack bot using Socket Mode
    try:
        handler = SocketModeHandler(app, config["SLACK"]["APP_TOKEN"])
        handler.start()
    except Exception as e:
        logger.error(f"Failed to start Slack app: {e}")
