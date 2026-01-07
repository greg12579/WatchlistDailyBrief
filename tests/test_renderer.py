"""Tests for brief rendering."""

import pytest
from watchbrief.output.renderer import BriefItem, render_email, render_slack
from watchbrief.features.triggers import TriggerResult
from watchbrief.llm.explain import Explanation, Driver


def create_sample_items() -> list[BriefItem]:
    """Create sample BriefItems for testing."""
    result = TriggerResult(
        ticker="AAPL",
        last_close=185.50,
        pct_change_1d=3.25,
        pct_change_5d=5.10,
        volume_last=80_000_000,
        volume_avg=50_000_000,
        volume_multiple=1.6,
        price_z=2.1,
        rel_vs_spy_z=1.8,
        rel_vs_sector_z=1.5,
        triggered=True,
        triggered_reasons=["price_z=2.1", "vol=1.6x"],
        label="ACTIONABLE",
        sector_etf="XLK",
        spy_pct_change_1d=0.5,
        sector_pct_change_1d=0.8,
    )

    explanation = Explanation(
        drivers=[
            Driver(rank=1, text="Strong iPhone sales reported", weight_pct=50),
            Driver(rank=2, text="Tech sector momentum", weight_pct=30),
            Driver(rank=3, text="Broader market rally", weight_pct=20),
        ],
        confidence="High",
        why_it_matters="Apple's outperformance signals continued tech leadership.",
    )

    return [BriefItem(
        ticker="AAPL",
        rank=1,
        score=8.5,
        result=result,
        explanation=explanation,
    )]


class TestRenderEmail:
    def test_returns_three_parts(self):
        """Test that render_email returns subject, html, and text."""
        items = create_sample_items()
        subject, html, text = render_email(
            "Test Brief",
            items,
            "http://localhost:8000",
            brief_id=1,
        )

        assert isinstance(subject, str)
        assert isinstance(html, str)
        assert isinstance(text, str)

    def test_html_contains_ticker(self):
        """Test that HTML contains ticker symbol."""
        items = create_sample_items()
        _, html, _ = render_email("Test", items, "http://localhost", 1)

        assert "AAPL" in html

    def test_html_contains_label_emoji(self):
        """Test that HTML contains appropriate emoji."""
        items = create_sample_items()
        _, html, _ = render_email("Test", items, "http://localhost", 1)

        # ACTIONABLE should have red emoji
        assert "🔴" in html or "ACTIONABLE" in html

    def test_html_contains_feedback_links(self):
        """Test that feedback links are present."""
        items = create_sample_items()
        _, html, _ = render_email("Test", items, "http://localhost:8000", 123)

        assert "/f/123/AAPL/up" in html
        assert "/f/123/AAPL/down" in html

    def test_text_version_is_readable(self):
        """Test that text version is human-readable."""
        items = create_sample_items()
        _, _, text = render_email("Test Brief", items, "http://localhost", 1)

        assert "AAPL" in text
        assert "3.25%" in text or "+3.25" in text
        assert "What Changed" in text
        assert "Why" in text

    def test_contains_all_drivers(self):
        """Test that all drivers are included."""
        items = create_sample_items()
        _, html, text = render_email("Test", items, "http://localhost", 1)

        assert "iPhone" in html
        assert "Tech sector" in html
        assert "iPhone" in text

    def test_contains_why_it_matters(self):
        """Test that 'why it matters' is included."""
        items = create_sample_items()
        _, html, text = render_email("Test", items, "http://localhost", 1)

        assert "leadership" in html.lower()
        assert "leadership" in text.lower()


class TestRenderSlack:
    def test_returns_dict_with_blocks(self):
        """Test that Slack payload has blocks."""
        items = create_sample_items()
        payload = render_slack("Test Brief", items, "http://localhost", 1)

        assert isinstance(payload, dict)
        assert "blocks" in payload

    def test_has_header_block(self):
        """Test that payload has header."""
        items = create_sample_items()
        payload = render_slack("Test Brief", items, "http://localhost", 1)

        headers = [b for b in payload["blocks"] if b.get("type") == "header"]
        assert len(headers) >= 1

    def test_contains_ticker_in_text(self):
        """Test that ticker appears in Slack message."""
        items = create_sample_items()
        payload = render_slack("Test", items, "http://localhost", 1)

        # Check that AAPL appears somewhere in the payload
        payload_str = str(payload)
        assert "AAPL" in payload_str

    def test_has_feedback_buttons(self):
        """Test that feedback buttons are present."""
        items = create_sample_items()
        payload = render_slack("Test", items, "http://localhost:8000", 123)

        actions = [b for b in payload["blocks"] if b.get("type") == "actions"]
        assert len(actions) >= 1

        # Check that action has button with feedback URL
        payload_str = str(payload)
        assert "/f/123/AAPL/" in payload_str
