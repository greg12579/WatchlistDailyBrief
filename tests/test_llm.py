"""Tests for LLM explanation validation."""

import pytest
from watchbrief.llm.explain import (
    parse_explanation_json,
    create_fallback_explanation,
    Explanation,
    Driver,
)
from watchbrief.features.triggers import TriggerResult


class TestParseExplanationJson:
    def test_parses_valid_json(self):
        """Test parsing valid LLM response."""
        response = """
        {
            "drivers": [
                {"rank": 1, "text": "Strong earnings beat", "weight_pct": 60},
                {"rank": 2, "text": "Sector rotation", "weight_pct": 40}
            ],
            "confidence": "High",
            "why_it_matters": "This signals momentum."
        }
        """

        result = parse_explanation_json(response)

        assert result is not None
        assert len(result.drivers) == 2
        assert result.drivers[0].text == "Strong earnings beat"
        assert result.drivers[0].weight_pct == 60
        assert result.confidence == "High"
        assert "momentum" in result.why_it_matters

    def test_handles_json_with_extra_text(self):
        """Test parsing JSON with surrounding text."""
        response = """
        Here's my analysis:

        {
            "drivers": [{"rank": 1, "text": "Test", "weight_pct": 100}],
            "confidence": "Medium",
            "why_it_matters": "Test matter."
        }

        Let me know if you need more.
        """

        result = parse_explanation_json(response)

        assert result is not None
        assert result.confidence == "Medium"

    def test_returns_none_for_invalid_json(self):
        """Test that invalid JSON returns None."""
        response = "This is not JSON at all"
        result = parse_explanation_json(response)
        assert result is None

    def test_returns_none_for_missing_fields(self):
        """Test that missing required fields returns None."""
        response = '{"drivers": [], "confidence": "High"}'  # Missing why_it_matters
        result = parse_explanation_json(response)
        assert result is None

    def test_normalizes_invalid_confidence(self):
        """Test that invalid confidence is normalized to Low."""
        response = """
        {
            "drivers": [{"rank": 1, "text": "Test", "weight_pct": 100}],
            "confidence": "VeryHigh",
            "why_it_matters": "Test."
        }
        """

        result = parse_explanation_json(response)

        assert result is not None
        assert result.confidence == "Low"  # Normalized


class TestCreateFallbackExplanation:
    @pytest.fixture
    def sample_result(self):
        return TriggerResult(
            ticker="TEST",
            last_close=100.0,
            pct_change_1d=5.0,
            pct_change_5d=8.0,
            volume_last=2_000_000,
            volume_avg=1_000_000,
            volume_multiple=2.0,
            price_z=2.5,
            rel_vs_spy_z=1.8,
            rel_vs_sector_z=1.5,
            triggered=True,
            triggered_reasons=["price_z=2.5", "vol=2.0x"],
            label="ACTIONABLE",
            sector_etf="XLK",
        )

    def test_creates_valid_fallback(self, sample_result):
        """Test that fallback creates valid explanation."""
        explanation = create_fallback_explanation(sample_result, [])

        assert explanation is not None
        assert explanation.is_fallback is True
        assert len(explanation.drivers) >= 1
        assert explanation.confidence in ("Low", "Medium", "High")
        assert len(explanation.why_it_matters) > 0

    def test_weights_sum_to_100(self, sample_result):
        """Test that driver weights sum to approximately 100."""
        explanation = create_fallback_explanation(sample_result, [])

        total_weight = sum(d.weight_pct for d in explanation.drivers)
        assert 95 <= total_weight <= 105  # Allow small rounding error

    def test_mentions_high_volume(self, sample_result):
        """Test that high volume is mentioned when it's the primary driver."""
        # Set low price_z so volume becomes primary driver
        sample_result.price_z = 1.0
        sample_result.volume_multiple = 2.5
        explanation = create_fallback_explanation(sample_result, [])

        driver_texts = " ".join(d.text for d in explanation.drivers)
        assert "volume" in driver_texts.lower() or "2.5x" in driver_texts

    def test_includes_event_when_present(self, sample_result):
        """Test that events are included in explanation."""
        from datetime import date
        from watchbrief.data.events import Event

        events = [
            Event(
                type="earnings",
                date=date.today(),
                title="Q4 Earnings Beat",
                source="test",
            )
        ]

        explanation = create_fallback_explanation(sample_result, events)

        driver_texts = " ".join(d.text.lower() for d in explanation.drivers)
        assert "earnings" in driver_texts or "event" in driver_texts
