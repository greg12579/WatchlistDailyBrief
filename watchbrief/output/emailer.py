"""Email delivery via SMTP."""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from watchbrief.config import EmailConfig


def send_email(
    config: EmailConfig,
    subject: str,
    html_body: str,
    text_body: str,
) -> bool:
    """Send an email via SMTP with STARTTLS.

    Args:
        config: Email configuration
        subject: Email subject
        html_body: HTML version of the email
        text_body: Plain text version of the email

    Returns:
        True if sent successfully, False otherwise
    """
    # Get password from environment
    password = os.environ.get("SMTP_PASSWORD")
    if not password:
        print("Warning: SMTP_PASSWORD not set, cannot send email")
        return False

    if not config.from_addr or not config.to_addrs:
        print("Warning: Email from/to addresses not configured")
        return False

    try:
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = config.from_addr
        msg["To"] = ", ".join(config.to_addrs)

        # Attach both versions (text first, then HTML)
        part1 = MIMEText(text_body, "plain")
        part2 = MIMEText(html_body, "html")
        msg.attach(part1)
        msg.attach(part2)

        # Connect and send
        with smtplib.SMTP(config.smtp_host, config.smtp_port) as server:
            server.starttls()
            server.login(config.from_addr, password)
            server.sendmail(config.from_addr, config.to_addrs, msg.as_string())

        print(f"Email sent to {', '.join(config.to_addrs)}")
        return True

    except smtplib.SMTPException as e:
        print(f"SMTP error sending email: {e}")
        return False
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
