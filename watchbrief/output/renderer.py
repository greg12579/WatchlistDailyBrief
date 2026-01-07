"""Render briefs for email and Slack delivery."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from watchbrief.data.events import Event
from watchbrief.features.triggers import TriggerResult
from watchbrief.llm.explain import Explanation

if TYPE_CHECKING:
    from watchbrief.data.news_processor import NewsEvidence


@dataclass
class BriefItem:
    """A rendered item for the brief."""

    ticker: str
    rank: int
    score: float
    result: TriggerResult
    explanation: Explanation
    events: list[Event] = field(default_factory=list)  # News, SEC filings, earnings
    brief_id: Optional[int] = None
    news_evidence: Optional["NewsEvidence"] = None  # Processed news for transparency


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
        ".driver { margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 4px; }",
        ".driver-category { font-weight: bold; color: #495057; font-size: 0.85em; }",
        ".driver-evidence { font-size: 0.85em; color: #6c757d; margin-top: 4px; }",
        ".confidence { font-style: italic; color: #666; }",
        ".why-matters { background: #e8f4fd; padding: 12px; border-radius: 4px; margin: 12px 0; }",
        ".missing-checks { background: #fff3cd; padding: 8px; border-radius: 4px; margin: 12px 0; font-size: 0.85em; color: #856404; }",
        ".events { background: #f0f7ff; padding: 12px; border-radius: 4px; margin: 12px 0; }",
        ".event { margin: 6px 0; font-size: 0.9em; }",
        ".event-type { font-weight: bold; color: #0066cc; }",
        ".event-type.sec { color: #6c757d; }",
        ".event-type.news { color: #28a745; }",
        ".event-link { color: #0066cc; text-decoration: none; }",
        ".event-link:hover { text-decoration: underline; }",
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

        # Determine label styling - use ASCII markers for Windows compatibility
        label_class = "actionable" if r.label == "ACTIONABLE" else "monitor"
        label_marker = "[!]" if r.label == "ACTIONABLE" else "[*]"

        # HTML item
        html_parts.append(f'<div class="item {label_class}">')
        html_parts.append(f'<div><span class="ticker">{r.ticker}</span>')
        html_parts.append(f'<span class="label label-{label_class}">{label_marker} {r.label}</span></div>')

        # Metrics
        html_parts.append('<div class="metrics">')
        html_parts.append("<strong>What Changed:</strong><br>")
        html_parts.append(f"  Price: {r.pct_change_1d:+.2f}% (1d), {r.pct_change_5d:+.2f}% (5d)<br>")
        html_parts.append(f"  Last Close: ${r.last_close:.2f}<br>")
        html_parts.append(f"  Volume: {r.volume_multiple:.1f}x average<br>")
        html_parts.append(f"  Price Z-score: {r.price_z:.2f}<br>")
        html_parts.append(f"  vs SPY Z-score: {r.rel_vs_spy_z:.2f}")
        if r.rel_vs_sector_z is not None:
            html_parts.append(f"<br>  vs {r.sector_etf} Z-score: {r.rel_vs_sector_z:.2f}")
        html_parts.append("</div>")

        # Recent News & Filings section with transparency
        html_parts.append('<div class="events">')
        html_parts.append("<strong>Recent News & Filings:</strong>")

        # Use processed news_evidence if available for better transparency
        news_displayed = False
        if item.news_evidence is not None:
            if not item.news_evidence.checked:
                # News not checked (feed unavailable)
                html_parts.append('<div class="event" style="color:#856404;font-style:italic;">')
                html_parts.append("News not checked (feed unavailable)")
                html_parts.append('</div>')
                news_displayed = True
            elif item.news_evidence.no_company_specific_catalyst_found:
                # Checked but no catalyst found
                lookback = item.news_evidence.lookback_hours
                html_parts.append('<div class="event" style="color:#6c757d;font-style:italic;">')
                html_parts.append(f"No relevant company-specific headlines found (checked last {lookback}h)")
                html_parts.append('</div>')
                news_displayed = True
            elif item.news_evidence.top_clusters:
                # Show processed clusters with event type classification
                for cluster in item.news_evidence.top_clusters[:3]:
                    event_type = cluster.event_type.value
                    weak = " (weak)" if cluster.weak_evidence else ""
                    time_str = cluster.published_at.strftime("%m/%d %H:%M")
                    title = cluster.headline[:75] + "..." if len(cluster.headline) > 75 else cluster.headline

                    html_parts.append(f'<div class="event">')
                    html_parts.append(f'<span class="event-type news">[{event_type}]</span> ')
                    html_parts.append(f'{time_str}: ')
                    if cluster.url:
                        html_parts.append(f'<a href="{cluster.url}" class="event-link">{title}</a>')
                    else:
                        html_parts.append(title)
                    html_parts.append(f' <span style="color:#888">({cluster.source}{weak})</span>')
                    html_parts.append('</div>')
                news_displayed = True

        # Fallback to raw events if no news_evidence or add SEC filings
        if item.events:
            sec_events = [e for e in item.events if e.type == "sec_filing"]
            earnings_events = [e for e in item.events if e.type == "earnings"]
            other_events = [e for e in item.events if e.type not in ("sec_filing", "earnings", "news")]

            # Show SEC filings (always from raw events)
            for event in sec_events[:3]:
                html_parts.append(f'<div class="event">')
                html_parts.append(f'<span class="event-type sec">[SEC FILING]</span> ')
                html_parts.append(f'{event.date.strftime("%m/%d")}: ')
                if event.url:
                    title = event.title[:75] + "..." if len(event.title) > 75 else event.title
                    html_parts.append(f'<a href="{event.url}" class="event-link">{title}</a>')
                else:
                    html_parts.append(event.title[:75])
                html_parts.append(f' <span style="color:#888">({event.source})</span>')
                html_parts.append('</div>')

            # Show earnings if present
            for event in earnings_events[:1]:
                html_parts.append(f'<div class="event">')
                html_parts.append(f'<span class="event-type" style="color:#dc3545;">[EARNINGS]</span> ')
                html_parts.append(f'{event.date.strftime("%m/%d")}: {event.title}')
                html_parts.append('</div>')

            # If no news_evidence, show raw news events
            if not news_displayed:
                news_events = [e for e in item.events if e.type == "news"]
                for event in news_events[:3]:
                    html_parts.append(f'<div class="event">')
                    html_parts.append(f'<span class="event-type news">[NEWS]</span> ')
                    html_parts.append(f'{event.date.strftime("%m/%d")}: ')
                    if event.url:
                        title = event.title[:75] + "..." if len(event.title) > 75 else event.title
                        html_parts.append(f'<a href="{event.url}" class="event-link">{title}</a>')
                    else:
                        html_parts.append(event.title[:75])
                    html_parts.append(f' <span style="color:#888">({event.source})</span>')
                    html_parts.append('</div>')

        html_parts.append("</div>")

        # Drivers with categories and evidence
        html_parts.append('<div class="drivers">')
        html_parts.append("<strong>Why (Ranked):</strong>")
        for driver in exp.drivers:
            html_parts.append('<div class="driver">')
            # Category badge if available
            if driver.category:
                html_parts.append(f'<span class="driver-category">[{driver.category}]</span> ')
            html_parts.append(f'{driver.rank}. {driver.text} ({driver.weight_pct}%)')
            # Evidence if available
            if driver.evidence:
                evidence_str = ", ".join(driver.evidence[:3])
                html_parts.append(f'<div class="driver-evidence">Evidence: {evidence_str}</div>')
            html_parts.append('</div>')
        html_parts.append("</div>")

        # Confidence
        html_parts.append(f'<div class="confidence">Confidence: {exp.confidence}</div>')

        # Missing checks warning
        if exp.missing_checks:
            checks_str = ", ".join(exp.missing_checks)
            html_parts.append(f'<div class="missing-checks">Note: Not checked: {checks_str}</div>')

        # Why it matters
        html_parts.append(f'<div class="why-matters"><strong>Why It Matters:</strong> {exp.why_it_matters}</div>')

        # Feedback links
        html_parts.append('<div class="feedback">')
        html_parts.append(f'<a href="{base_url}/f/{brief_id}/{r.ticker}/up">[+] Helpful</a>')
        html_parts.append(f'<a href="{base_url}/f/{brief_id}/{r.ticker}/down">[-] Not helpful</a>')
        html_parts.append(f'<a href="{base_url}/d/{brief_id}/{r.ticker}/yes">[Y] Influenced decision</a>')
        html_parts.append(f'<a href="{base_url}/d/{brief_id}/{r.ticker}/no">[N] No impact</a>')
        html_parts.append("</div>")

        html_parts.append("</div>")

        # Text version
        text_parts.append(f"\n{'-' * 40}")
        text_parts.append(f"{label_marker} {r.ticker} - {r.label}")
        text_parts.append(f"\nWhat Changed:")
        text_parts.append(f"  Price: {r.pct_change_1d:+.2f}% (1d), {r.pct_change_5d:+.2f}% (5d)")
        text_parts.append(f"  Last Close: ${r.last_close:.2f}")
        text_parts.append(f"  Volume: {r.volume_multiple:.1f}x average")
        text_parts.append(f"  Price Z-score: {r.price_z:.2f}")
        text_parts.append(f"  vs SPY Z-score: {r.rel_vs_spy_z:.2f}")
        if r.rel_vs_sector_z is not None:
            text_parts.append(f"  vs {r.sector_etf} Z-score: {r.rel_vs_sector_z:.2f}")

        # Recent News & Filings (text version) with transparency
        text_parts.append(f"\nRecent News & Filings:")

        text_news_displayed = False
        if item.news_evidence is not None:
            if not item.news_evidence.checked:
                text_parts.append("  News not checked (feed unavailable)")
                text_news_displayed = True
            elif item.news_evidence.no_company_specific_catalyst_found:
                lookback = item.news_evidence.lookback_hours
                text_parts.append(f"  No relevant company-specific headlines found (checked last {lookback}h)")
                text_news_displayed = True
            elif item.news_evidence.top_clusters:
                for cluster in item.news_evidence.top_clusters[:3]:
                    event_type = cluster.event_type.value
                    weak = " (weak)" if cluster.weak_evidence else ""
                    time_str = cluster.published_at.strftime("%m/%d %H:%M")
                    text_parts.append(f"  [{event_type}] {time_str}: {cluster.headline[:65]}")
                    if cluster.url:
                        text_parts.append(f"    {cluster.url}")
                text_news_displayed = True

        if item.events:
            sec_events = [e for e in item.events if e.type == "sec_filing"]
            earnings_events = [e for e in item.events if e.type == "earnings"]

            for event in sec_events[:3]:
                text_parts.append(f"  [SEC FILING] {event.date.strftime('%m/%d')}: {event.title[:65]}")
                if event.url:
                    text_parts.append(f"    {event.url}")

            for event in earnings_events[:1]:
                text_parts.append(f"  [EARNINGS] {event.date.strftime('%m/%d')}: {event.title}")

            if not text_news_displayed:
                news_events = [e for e in item.events if e.type == "news"]
                for event in news_events[:3]:
                    text_parts.append(f"  [NEWS] {event.date.strftime('%m/%d')}: {event.title[:65]}")
                    if event.url:
                        text_parts.append(f"    {event.url}")

        text_parts.append(f"\nWhy (Ranked):")
        for driver in exp.drivers:
            category_str = f"[{driver.category}] " if driver.category else ""
            text_parts.append(f"  {driver.rank}. {category_str}{driver.text} ({driver.weight_pct}%)")
            if driver.evidence:
                evidence_str = ", ".join(driver.evidence[:3])
                text_parts.append(f"     Evidence: {evidence_str}")

        text_parts.append(f"\nConfidence: {exp.confidence}")

        if exp.missing_checks:
            text_parts.append(f"\nNote: Not checked: {', '.join(exp.missing_checks)}")

        text_parts.append(f"\nWhy It Matters: {exp.why_it_matters}")

        text_parts.append(f"\nFeedback:")
        text_parts.append(f"  [+] Helpful: {base_url}/f/{brief_id}/{r.ticker}/up")
        text_parts.append(f"  [-] Not helpful: {base_url}/f/{brief_id}/{r.ticker}/down")

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
        label_marker = "[!]" if r.label == "ACTIONABLE" else "[*]"

        # Main text block
        text_lines = [
            f"*{label_marker} {r.ticker}* - {r.label}",
            "",
            "*What Changed:*",
            f"  Price: {r.pct_change_1d:+.2f}% (1d) | Volume: {r.volume_multiple:.1f}x",
            f"  Price Z: {r.price_z:.2f} | vs SPY Z: {r.rel_vs_spy_z:.2f}",
        ]

        if r.rel_vs_sector_z is not None:
            text_lines.append(f"  vs {r.sector_etf} Z: {r.rel_vs_sector_z:.2f}")

        # Recent events (Slack version - abbreviated)
        if item.events:
            text_lines.append("")
            text_lines.append("*Recent News/Filings:*")
            for event in item.events[:3]:  # Limit to 3 for Slack
                type_label = event.type.upper().replace("_", " ")
                title = event.title[:50] + "..." if len(event.title) > 50 else event.title
                if event.url:
                    text_lines.append(f"  [{type_label}] <{event.url}|{title}>")
                else:
                    text_lines.append(f"  [{type_label}] {title}")

        text_lines.append("")
        text_lines.append("*Why:*")
        for driver in exp.drivers[:2]:  # Limit to 2 drivers for Slack
            category_str = f"[{driver.category}] " if driver.category else ""
            text_lines.append(f"{driver.rank}. {category_str}{driver.text} ({driver.weight_pct}%)")
            if driver.evidence:
                evidence_str = ", ".join(driver.evidence[:2])
                text_lines.append(f"   _{evidence_str}_")

        text_lines.append("")
        text_lines.append(f"_Confidence: {exp.confidence}_")

        if exp.missing_checks:
            text_lines.append(f"_Not checked: {', '.join(exp.missing_checks)}_")

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
                    "text": {"type": "plain_text", "text": "[+]"},
                    "url": f"{base_url}/f/{brief_id}/{r.ticker}/up",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "[-]"},
                    "url": f"{base_url}/f/{brief_id}/{r.ticker}/down",
                },
            ],
        })

        blocks.append({"type": "divider"})

    return {"blocks": blocks}
