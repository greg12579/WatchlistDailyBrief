"""Tests for ranking functionality."""

import pytest
from watchbrief.features.ranking import compute_score, rank_and_select
from watchbrief.features.triggers import TriggerResult


def create_trigger_result(
    ticker: str,
    price_z: float = 0.0,
    volume_multiple: float = 1.0,
    rel_vs_spy_z: float = 0.0,
    rel_vs_sector_z: float = None,
    triggered: bool = True,
) -> TriggerResult:
    """Create a TriggerResult for testing."""
    return TriggerResult(
        ticker=ticker,
        last_close=100.0,
        pct_change_1d=price_z * 0.5,  # Approximate
        pct_change_5d=0.0,
        volume_last=1_000_000 * volume_multiple,
        volume_avg=1_000_000,
        volume_multiple=volume_multiple,
        price_z=price_z,
        rel_vs_spy_z=rel_vs_spy_z,
        rel_vs_sector_z=rel_vs_sector_z,
        triggered=triggered,
        triggered_reasons=["test"],
        label="MONITOR",
    )


class TestComputeScore:
    def test_score_formula(self):
        """Test the scoring formula matches spec."""
        result = create_trigger_result(
            ticker="TEST",
            price_z=2.0,      # contributes 4.0 (2.0 * 2)
            volume_multiple=2.0,  # contributes 1.0 ((2.0-1) * 1)
            rel_vs_spy_z=1.0,     # contributes 1.5 (1.0 * 1.5)
            rel_vs_sector_z=1.0,  # contributes 1.5 (1.0 * 1.5)
        )

        score = compute_score(result)
        expected = 4.0 + 1.0 + 1.5 + 1.5
        assert abs(score - expected) < 0.01

    def test_score_without_sector(self):
        """Test scoring without sector data."""
        result = create_trigger_result(
            ticker="TEST",
            price_z=2.0,
            volume_multiple=1.5,
            rel_vs_spy_z=1.0,
            rel_vs_sector_z=None,
        )

        score = compute_score(result)
        expected = 4.0 + 0.5 + 1.5  # No sector contribution
        assert abs(score - expected) < 0.01

    def test_negative_price_z_uses_absolute(self):
        """Test that negative z-scores use absolute value."""
        result = create_trigger_result(
            ticker="TEST",
            price_z=-2.0,
            volume_multiple=1.0,
            rel_vs_spy_z=-1.0,
        )

        score = compute_score(result)
        expected = 4.0 + 0.0 + 1.5
        assert abs(score - expected) < 0.01

    def test_volume_below_1_contributes_zero(self):
        """Test that volume < 1x contributes nothing."""
        result = create_trigger_result(
            ticker="TEST",
            price_z=1.0,
            volume_multiple=0.5,
            rel_vs_spy_z=0.0,
        )

        score = compute_score(result)
        expected = 2.0 + 0.0 + 0.0
        assert abs(score - expected) < 0.01


class TestRankAndSelect:
    def test_ranks_by_score_descending(self):
        """Test that results are ranked by score descending."""
        results = [
            create_trigger_result("LOW", price_z=1.0),
            create_trigger_result("HIGH", price_z=3.0),
            create_trigger_result("MED", price_z=2.0),
        ]

        ranked = rank_and_select(results, max_items=5)

        assert len(ranked) == 3
        assert ranked[0][0].ticker == "HIGH"
        assert ranked[1][0].ticker == "MED"
        assert ranked[2][0].ticker == "LOW"

    def test_limits_to_max_items(self):
        """Test that selection is limited to max_items."""
        results = [
            create_trigger_result(f"TICK{i}", price_z=float(i))
            for i in range(10)
        ]

        ranked = rank_and_select(results, max_items=3)

        assert len(ranked) == 3

    def test_filters_untriggered(self):
        """Test that untriggered items are filtered out."""
        results = [
            create_trigger_result("TRIG", price_z=2.0, triggered=True),
            create_trigger_result("NOTRIG", price_z=3.0, triggered=False),
        ]

        ranked = rank_and_select(results, max_items=5)

        assert len(ranked) == 1
        assert ranked[0][0].ticker == "TRIG"

    def test_includes_rank_number(self):
        """Test that rank numbers are correct."""
        results = [
            create_trigger_result("A", price_z=3.0),
            create_trigger_result("B", price_z=2.0),
            create_trigger_result("C", price_z=1.0),
        ]

        ranked = rank_and_select(results, max_items=5)

        assert ranked[0][2] == 1  # First item has rank 1
        assert ranked[1][2] == 2
        assert ranked[2][2] == 3

    def test_deterministic_ranking(self):
        """Test that ranking is deterministic."""
        results = [
            create_trigger_result("A", price_z=2.0),
            create_trigger_result("B", price_z=2.0),  # Same score
        ]

        ranked1 = rank_and_select(results, max_items=5)
        ranked2 = rank_and_select(results, max_items=5)

        # Order should be consistent
        assert ranked1[0][0].ticker == ranked2[0][0].ticker
        assert ranked1[1][0].ticker == ranked2[1][0].ticker
