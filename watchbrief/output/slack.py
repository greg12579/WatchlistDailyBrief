"""Slack delivery via webhook."""

import json
from urllib.request import Request, urlopen
from urllib.error import URLError

from watchbrief.config import SlackConfig


def send_slack(config: SlackConfig, payload: dict) -> bool:
    """Send a message to Slack via webhook.

    Args:
        config: Slack configuration with webhook URL
        payload: Slack message payload (blocks format)

    Returns:
        True if sent successfully, False otherwise
    """
    if not config.webhook_url:
        print("Warning: Slack webhook URL not configured")
        return False

    try:
        data = json.dumps(payload).encode("utf-8")

        request = Request(
            config.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=10) as response:
            if response.status == 200:
                print("Slack message sent successfully")
                return True
            else:
                print(f"Slack webhook returned status {response.status}")
                return False

    except URLError as e:
        print(f"URL error sending to Slack: {e}")
        return False
    except Exception as e:
        print(f"Error sending to Slack: {e}")
        return False


def send_slack_text(config: SlackConfig, text: str) -> bool:
    """Send a simple text message to Slack.

    Args:
        config: Slack configuration
        text: Plain text message

    Returns:
        True if sent successfully, False otherwise
    """
    return send_slack(config, {"text": text})
