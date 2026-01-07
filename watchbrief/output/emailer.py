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
    # Get credentials - config values take precedence, then env vars
    smtp_user = config.smtp_user or config.from_addr or os.environ.get("SMTP_USER", "")
    smtp_password = config.smtp_password or os.environ.get("SMTP_PASSWORD", "")

    if not smtp_password:
        print("Warning: SMTP_PASSWORD not set in config or environment, cannot send email")
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

        # Add CC if specified
        if config.cc_addrs:
            msg["Cc"] = ", ".join(config.cc_addrs)

        # Attach both versions (text first, then HTML)
        part1 = MIMEText(text_body, "plain")
        part2 = MIMEText(html_body, "html")
        msg.attach(part1)
        msg.attach(part2)

        # Build recipient list (To + CC)
        all_recipients = config.to_addrs + (config.cc_addrs or [])

        # Connect and send
        with smtplib.SMTP(config.smtp_host, config.smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(config.from_addr, all_recipients, msg.as_string())

        cc_str = f" (cc: {', '.join(config.cc_addrs)})" if config.cc_addrs else ""
        print(f"Email sent to {', '.join(config.to_addrs)}{cc_str}")
        return True

    except smtplib.SMTPException as e:
        print(f"SMTP error sending email: {e}")
        return False
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
