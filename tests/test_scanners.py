"""Tests for Phase 2.5 scanner functions."""

import pytest
from watchbrief.features.scanners import (
    TapeRow,
    compute_daily_tape,
    compute_near_highs,
    compute_broken_lows,
    compute_candidate_tags,
)


def _make_tape_row(
    ticker: str,
    ret_1d: float = 0,
    ret_5d: float = 0,
    ret_252d: float = 0,
    rel_vs_iwm_1d: float = 0,
    rel_vs_iwm_5d: float = 0,
    rel_vs_iwm_252d: float = 0,
    pct_from_52w_high: float = -10,
    pct_from_52w_low: float = 50,
    z_1d: float = None,
    z_252d: float = None,
) -> TapeRow:
    """Helper to create TapeRow for testing."""
    return TapeRow(
        ticker=ticker,
        name=f"Test {ticker}",
        ret_1d=ret_1d,
        ret_5d=ret_5d,
        ret_252d=ret_252d,
        rel_vs_iwm_1d=rel_vs_iwm_1d,
        rel_vs_iwm_5d=rel_vs_iwm_5d,
        rel_vs_iwm_252d=rel_vs_iwm_252d,
        state_label="Range Bound Mid",
        pct_from_52w_high=pct_from_52w_high,
        pct_from_52w_low=pct_from_52w_low,
        z_1d=z_1d,
        z_252d=z_252d,
    )


class TestDailyTape:
    """Tests for compute_daily_tape function."""

    def test_basic_ranking_by_abs_return(self):
        """Test that tape correctly selects top N by absolute 1D return."""
        rows = {
            "AAA": _make_tape_row("AAA", ret_1d=-8.0),  # |8| = 8
            "BBB": _make_tape_row("BBB", ret_1d=-3.0),  # |3| = 3
            "CCC": _make_tape_row("CCC", ret_1d=0.5),   # |0.5| = 0.5
            "DDD": _make_tape_row("DDD", ret_1d=5.0),   # |5| = 5
            "EEE": _make_tape_row("EEE", ret_1d=10.0),  # |10| = 10
        }

        result = compute_daily_tape(rows, top_n=3)

        # Should be sorted by absolute value (largest first)
        assert len(result) == 3
        assert result[0].ticker == "EEE"  # |10| = 10
        assert result[1].ticker == "AAA"  # |8| = 8
        assert result[2].ticker == "DDD"  # |5| = 5

    def test_small_watchlist(self):
        """Test with fewer tickers than requested top_n."""
        rows = {
            "AAA": _make_tape_row("AAA", ret_1d=-2.0),
            "BBB": _make_tape_row("BBB", ret_1d=3.0),
        }

        result = compute_daily_tape(rows, top_n=5)

        # Should return all available, not fail
        assert len(result) == 2

    def test_empty_watchlist(self):
        """Test with empty watchlist."""
        rows = {}
        result = compute_daily_tape(rows, top_n=5)

        assert len(result) == 0

    def test_sorting_by_absolute_value(self):
        """Verify results are sorted by absolute 1D return (largest first)."""
        rows = {
            "A": _make_tape_row("A", ret_1d=-1.0),
            "B": _make_tape_row("B", ret_1d=-5.0),
            "C": _make_tape_row("C", ret_1d=3.0),
        }

        result = compute_daily_tape(rows, top_n=3)

        # Should be sorted: B (|5|), C (|3|), A (|1|)
        assert result[0].ticker == "B"
        assert result[1].ticker == "C"
        assert result[2].ticker == "A"

    def test_mixes_positive_and_negative(self):
        """Verify tape includes both positive and negative movers."""
        rows = {
            "UP": _make_tape_row("UP", ret_1d=7.0),
            "DOWN": _make_tape_row("DOWN", ret_1d=-6.0),
            "FLAT": _make_tape_row("FLAT", ret_1d=0.1),
        }

        result = compute_daily_tape(rows, top_n=3)

        # Both big movers should be at top
        tickers = [r.ticker for r in result[:2]]
        assert "UP" in tickers
        assert "DOWN" in tickers


class TestNearHighs:
    """Tests for compute_near_highs function."""

    def test_filters_by_threshold(self):
        """Test that only tickers within threshold are included."""
        rows = {
            "NEAR": _make_tape_row("NEAR", pct_from_52w_high=-2.0),  # Within 3%
            "FAR": _make_tape_row("FAR", pct_from_52w_high=-25.0),  # Not within 3%
        }

        result = compute_near_highs(rows, threshold_pct=-3.0, top_n=5)

        assert len(result) == 1
        assert result[0].ticker == "NEAR"

    def test_sorts_by_proximity_to_high(self):
        """Test that results are sorted closest to high first."""
        rows = {
            "A": _make_tape_row("A", pct_from_52w_high=-2.5),
            "B": _make_tape_row("B", pct_from_52w_high=-0.5),  # Closest
            "C": _make_tape_row("C", pct_from_52w_high=-1.8),
        }

        result = compute_near_highs(rows, threshold_pct=-3.0, top_n=5)

        assert len(result) == 3
        assert result[0].ticker == "B"  # -0.5% is closest to 0
        assert result[0].pct_from_52w_high >= result[1].pct_from_52w_high

    def test_respects_top_n(self):
        """Test that only top_n results are returned."""
        rows = {
            "A": _make_tape_row("A", pct_from_52w_high=-1.0),
            "B": _make_tape_row("B", pct_from_52w_high=-0.5),
            "C": _make_tape_row("C", pct_from_52w_high=-2.0),
        }

        result = compute_near_highs(rows, threshold_pct=-3.0, top_n=2)

        assert len(result) == 2

    def test_at_high(self):
        """Test ticker exactly at 52w high."""
        rows = {
            "AT_HIGH": _make_tape_row("AT_HIGH", pct_from_52w_high=0.0),
        }

        result = compute_near_highs(rows, threshold_pct=-3.0, top_n=5)

        assert len(result) == 1
        assert result[0].ticker == "AT_HIGH"


class TestBrokenLows:
    """Tests for compute_broken_lows function."""

    def test_filters_near_low_and_weak(self):
        """Test that filters require both near-low AND weak performance."""
        rows = {
            # Near low AND weak -> should be included
            "BROKEN": _make_tape_row(
                "BROKEN",
                pct_from_52w_low=3.0,
                ret_252d=-35.0,
                rel_vs_iwm_252d=-25.0,
            ),
            # Near low but strong performance -> should be excluded
            "NEAR_BUT_STRONG": _make_tape_row(
                "NEAR_BUT_STRONG",
                pct_from_52w_low=2.0,
                ret_252d=15.0,
                rel_vs_iwm_252d=10.0,
            ),
            # Weak but not near low -> should be excluded
            "WEAK_BUT_FAR": _make_tape_row(
                "WEAK_BUT_FAR",
                pct_from_52w_low=50.0,
                ret_252d=-40.0,
                rel_vs_iwm_252d=-30.0,
            ),
        }

        result = compute_broken_lows(
            rows,
            near_low_threshold_pct=5.0,
            broken_return_threshold=-20.0,
            broken_rel_iwm_threshold=-10.0,
            top_n=5,
        )

        assert len(result) == 1
        assert result[0].ticker == "BROKEN"

    def test_includes_if_ret_252d_weak(self):
        """Test inclusion with weak 252d return even if rel vs IWM is OK."""
        rows = {
            "WEAK_ABS": _make_tape_row(
                "WEAK_ABS",
                pct_from_52w_low=4.0,
                ret_252d=-25.0,  # Weak absolute return
                rel_vs_iwm_252d=-5.0,  # OK relative (not below -10%)
            ),
        }

        result = compute_broken_lows(
            rows,
            near_low_threshold_pct=5.0,
            broken_return_threshold=-20.0,
            broken_rel_iwm_threshold=-10.0,
            top_n=5,
        )

        assert len(result) == 1

    def test_includes_if_rel_iwm_weak(self):
        """Test inclusion with weak rel vs IWM even if absolute return is OK."""
        rows = {
            "WEAK_REL": _make_tape_row(
                "WEAK_REL",
                pct_from_52w_low=4.0,
                ret_252d=-15.0,  # OK absolute (not below -20%)
                rel_vs_iwm_252d=-12.0,  # Weak relative
            ),
        }

        result = compute_broken_lows(
            rows,
            near_low_threshold_pct=5.0,
            broken_return_threshold=-20.0,
            broken_rel_iwm_threshold=-10.0,
            top_n=5,
        )

        assert len(result) == 1

    def test_sorts_by_proximity_to_low(self):
        """Test that results are sorted closest to low first."""
        rows = {
            "A": _make_tape_row("A", pct_from_52w_low=4.0, ret_252d=-30.0),
            "B": _make_tape_row("B", pct_from_52w_low=1.0, ret_252d=-30.0),  # Closest
            "C": _make_tape_row("C", pct_from_52w_low=3.0, ret_252d=-30.0),
        }

        result = compute_broken_lows(
            rows,
            near_low_threshold_pct=5.0,
            broken_return_threshold=-20.0,
            broken_rel_iwm_threshold=-10.0,
            top_n=5,
        )

        assert len(result) == 3
        assert result[0].ticker == "B"  # 1.0% is closest to 0


class TestCandidateTags:
    """Tests for compute_candidate_tags function."""

    def test_shock_down_via_z_score(self):
        """Test ShockDown tag via z-score."""
        row = _make_tape_row("TEST", z_1d=-2.5)
        tags = compute_candidate_tags(row)
        assert "ShockDown" in tags

    def test_shock_down_via_rel_iwm(self):
        """Test ShockDown tag via relative IWM move."""
        row = _make_tape_row("TEST", rel_vs_iwm_1d=-4.0, z_1d=None)
        tags = compute_candidate_tags(row)
        assert "ShockDown" in tags

    def test_shock_up_via_z_score(self):
        """Test ShockUp tag via z-score."""
        row = _make_tape_row("TEST", z_1d=2.5)
        tags = compute_candidate_tags(row)
        assert "ShockUp" in tags

    def test_shock_up_via_rel_iwm(self):
        """Test ShockUp tag via relative IWM move."""
        row = _make_tape_row("TEST", rel_vs_iwm_1d=4.0, z_1d=None)
        tags = compute_candidate_tags(row)
        assert "ShockUp" in tags

    def test_near_52w_high(self):
        """Test Near52wHigh tag."""
        row = _make_tape_row("TEST", pct_from_52w_high=-2.0)
        tags = compute_candidate_tags(row)
        assert "Near52wHigh" in tags

    def test_near_52w_low(self):
        """Test Near52wLow tag."""
        row = _make_tape_row("TEST", pct_from_52w_low=3.0)
        tags = compute_candidate_tags(row)
        assert "Near52wLow" in tags

    def test_broken_long_term_via_z(self):
        """Test BrokenLongTerm tag via z-score."""
        row = _make_tape_row("TEST", z_252d=-2.0)
        tags = compute_candidate_tags(row)
        assert "BrokenLongTerm" in tags

    def test_broken_long_term_via_return(self):
        """Test BrokenLongTerm tag via return threshold."""
        row = _make_tape_row("TEST", ret_252d=-35.0, z_252d=None)
        tags = compute_candidate_tags(row)
        assert "BrokenLongTerm" in tags

    def test_strong_long_term_via_z(self):
        """Test StrongLongTerm tag via z-score."""
        row = _make_tape_row("TEST", z_252d=2.0)
        tags = compute_candidate_tags(row)
        assert "StrongLongTerm" in tags

    def test_strong_long_term_via_return(self):
        """Test StrongLongTerm tag via return threshold."""
        row = _make_tape_row("TEST", ret_252d=35.0, z_252d=None)
        tags = compute_candidate_tags(row)
        assert "StrongLongTerm" in tags

    def test_multiple_tags(self):
        """Test that multiple tags can be assigned."""
        row = _make_tape_row(
            "TEST",
            z_1d=-2.5,  # ShockDown
            pct_from_52w_low=2.0,  # Near52wLow
            z_252d=-2.0,  # BrokenLongTerm
        )
        tags = compute_candidate_tags(row)

        assert "ShockDown" in tags
        assert "Near52wLow" in tags
        assert "BrokenLongTerm" in tags

    def test_no_tags(self):
        """Test that a neutral stock gets no tags."""
        row = _make_tape_row(
            "TEST",
            z_1d=0.5,  # Not extreme
            rel_vs_iwm_1d=0.5,  # Not extreme
            pct_from_52w_high=-20.0,  # Not near high
            pct_from_52w_low=30.0,  # Not near low
            z_252d=0.5,  # Not extreme
            ret_252d=10.0,  # Not extreme
        )
        tags = compute_candidate_tags(row)

        assert len(tags) == 0
