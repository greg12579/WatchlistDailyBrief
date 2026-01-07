"""Render briefs for email and Slack delivery."""

from dataclasses import dataclass
from typing import Optional

from watchbrief.features.triggers import TriggerResult
from watchbrief.llm.explain import Explanation


@dataclass
class BriefItem:
    """A rendered item for the brief."""

    ticker: str
    rank: int
    score: float
    result: TriggerResult
    explanation: Explanation
    brief_id: Optional[int] = None


def render_email(
    subject: str,
    items: list[BriefItem],
    base_url: str,
    brief_id: int,
) -> tuple[str, str, str]:
    """Render email brief with HTML and text versions.

    Args:
        subject: Email subject line
        items: List of BriefItem objects to render
        base_url: Base URL for feedback links
        brief_id: Brief ID for feedback links

    Returns:
        Tuple of (subject, html_body, text_body)
    """
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.5; max-width: 600px; margin: 0 auto; padding: 20px; }",
        ".item { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }",
        ".actionable { border-left: 4px solid #dc3545; }",
        ".monitor { border-left: 4px solid #ffc107; }",
        ".ticker { font-size: 1.3em; font-weight: bold; }",
        ".label { padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin-left: 8px; }",
        ".label-actionable { background: #dc3545; color: white; }",
        ".label-monitor { background: #ffc107; color: black; }",
        ".metrics { background: #f8f9fa; padding: 12px; border-radius: 4px; margin: 12px 0; }",
        ".drivers { margin: 12px 0; }",
        ".driver { margin: 4px 0; }",
        ".confidence { font-style: italic; color: #666; }",
        ".why-matters { background: #e8f4fd; padding: 12px; border-radius: 4px; margin: 12px 0; }",
        ".feedback { margin-top: 12px; }",
        ".feedback a { margin-right: 12px; text-decoration: none; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{subject}</h1>",
    ]

    text_parts = [f"{subject}\n{'=' * len(subject)}\n"]

    for item in items:
        r = item.result
        exp = item.explanation

        # Determine label styling
        label_class = "actionable" if r.label == "ACTIONABLE" else "monitor"
        label_emoji = "🔴" if r.label == "ACTIONABLE" else "🟡"

        # HTML item
        html_parts.append(f'<div class="item {label_class}">')
        html_parts.append(f'<div><span class="ticker">{r.ticker}</span>')
        html_parts.append(f'<span class="label label-{label_class}">{label_emoji} {r.label}</span></div>')

        # Metrics
        html_parts.append('<div class="metrics">')
        html_parts.append("<strong>What Changed:</strong><br>")
        html_parts.append(f"• Price: {r.pct_change_1d:+.2f}% (1d), {r.pct_change_5d:+.2f}% (5d)<br>")
        html_parts.append(f"• Last Close: ${r.last_close:.2f}<br>")
        html_parts.append(f"• Volume: {r.volume_multiple:.1f}x average<br>")
        html_parts.append(f"• Price Z-score: {r.price_z:.2f}<br>")
        html_parts.append(f"• vs SPY Z-score: {r.rel_vs_spy_z:.2f}")
        if r.rel_vs_sector_z is not None:
            html_parts.append(f"<br>• vs {r.sector_etf} Z-score: {r.rel_vs_sector_z:.2f}")
        html_parts.append("</div>")

        # Drivers
        html_parts.append('<div class="drivers">')
        html_parts.append("<strong>Why (Ranked):</strong><br>")
        for driver in exp.drivers:
            html_parts.append(f'<div class="driver">{driver.rank}. {driver.text} ({driver.weight_pct}%)</div>')
        html_parts.append("</div>")

        # Confidence
        html_parts.append(f'<div class="confidence">Confidence: {exp.confidence}</div>')

        # Why it matters
        html_parts.append(f'<div class="why-matters"><strong>Why It Matters:</strong> {exp.why_it_matters}</div>')

        # Feedback links
        html_parts.append('<div class="feedback">')
        html_parts.append(f'<a href="{base_url}/f/{brief_id}/{r.ticker}/up">👍 Helpful</a>')
        html_parts.append(f'<a href="{base_url}/f/{brief_id}/{r.ticker}/down">👎 Not helpful</a>')
        html_parts.append(f'<a href="{base_url}/d/{brief_id}/{r.ticker}/yes">✅ Influenced decision</a>')
        html_parts.append(f'<a href="{base_url}/d/{brief_id}/{r.ticker}/no">❌ No impact</a>')
        html_parts.append("</div>")

        html_parts.append("</div>")

        # Text version
        text_parts.append(f"\n{'-' * 40}")
        text_parts.append(f"{label_emoji} {r.ticker} - {r.label}")
        text_parts.append(f"\nWhat Changed:")
        text_parts.append(f"  • Price: {r.pct_change_1d:+.2f}% (1d), {r.pct_change_5d:+.2f}% (5d)")
        text_parts.append(f"  • Last Close: ${r.last_close:.2f}")
        text_parts.append(f"  • Volume: {r.volume_multiple:.1f}x average")
        text_parts.append(f"  • Price Z-score: {r.price_z:.2f}")
        text_parts.append(f"  • vs SPY Z-score: {r.rel_vs_spy_z:.2f}")
        if r.rel_vs_sector_z is not None:
            text_parts.append(f"  • vs {r.sector_etf} Z-score: {r.rel_vs_sector_z:.2f}")

        text_parts.append(f"\nWhy (Ranked):")
        for driver in exp.drivers:
            text_parts.append(f"  {driver.rank}. {driver.text} ({driver.weight_pct}%)")

        text_parts.append(f"\nConfidence: {exp.confidence}")
        text_parts.append(f"\nWhy It Matters: {exp.why_it_matters}")

        text_parts.append(f"\nFeedback:")
        text_parts.append(f"  👍 Helpful: {base_url}/f/{brief_id}/{r.ticker}/up")
        text_parts.append(f"  👎 Not helpful: {base_url}/f/{brief_id}/{r.ticker}/down")

    html_parts.extend(["</body>", "</html>"])

    return subject, "\n".join(html_parts), "\n".join(text_parts)


def render_slack(
    subject: str,
    items: list[BriefItem],
    base_url: str,
    brief_id: int,
) -> dict:
    """Render Slack message payload.

    Args:
        subject: Message title
        items: List of BriefItem objects
        base_url: Base URL for feedback links
        brief_id: Brief ID for feedback links

    Returns:
        Slack webhook payload dictionary
    """
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": subject},
        },
        {"type": "divider"},
    ]

    for item in items:
        r = item.result
        exp = item.explanation
        label_emoji = "🔴" if r.label == "ACTIONABLE" else "🟡"

        # Main text block
        text_lines = [
            f"*{label_emoji} {r.ticker}* - {r.label}",
            "",
            "*What Changed:*",
            f"• Price: {r.pct_change_1d:+.2f}% (1d) | Volume: {r.volume_multiple:.1f}x",
            f"• Price Z: {r.price_z:.2f} | vs SPY Z: {r.rel_vs_spy_z:.2f}",
        ]

        if r.rel_vs_sector_z is not None:
            text_lines.append(f"• vs {r.sector_etf} Z: {r.rel_vs_sector_z:.2f}")

        text_lines.append("")
        text_lines.append("*Why:*")
        for driver in exp.drivers[:2]:  # Limit to 2 drivers for Slack
            text_lines.append(f"{driver.rank}. {driver.text} ({driver.weight_pct}%)")

        text_lines.append("")
        text_lines.append(f"_Confidence: {exp.confidence}_")
        text_lines.append(f">{exp.why_it_matters}")

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(text_lines)},
        })

        # Feedback buttons
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "👍"},
                    "url": f"{base_url}/f/{brief_id}/{r.ticker}/up",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "👎"},
                    "url": f"{base_url}/f/{brief_id}/{r.ticker}/down",
                },
            ],
        })

        blocks.append({"type": "divider"})

    return {"blocks": blocks}
