"""Tests for trigger computation."""

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from watchbrief.config import ThresholdsConfig
from watchbrief.features.triggers import (
    compute_triggers,
    compute_zscore,
    compute_actionability_label,
)


def create_test_df(
    n_days: int = 30,
    base_price: float = 100.0,
    daily_return: float = 0.0,
    last_return: float = 0.0,
    base_volume: float = 1_000_000,
    last_volume_multiple: float = 1.0,
) -> pd.DataFrame:
    """Create a test DataFrame with predictable values."""
    dates = [date.today() - timedelta(days=n_days - i - 1) for i in range(n_days)]

    # Generate prices with specified daily return
    closes = [base_price]
    for i in range(n_days - 2):
        closes.append(closes[-1] * (1 + daily_return))
    # Last day with specified return
    closes.append(closes[-1] * (1 + last_return))

    volumes = [base_volume] * (n_days - 1) + [base_volume * last_volume_multiple]

    df = pd.DataFrame({
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": volumes,
    }, index=dates)

    df["return"] = df["close"].pct_change()
    return df


class TestComputeZscore:
    def test_zscore_normal_distribution(self):
        """Test z-score with normal-ish data."""
        series = pd.Series([0.01, 0.02, -0.01, 0.0, 0.01, -0.02, 0.01])
        value = 0.05  # Should be positive z-score

        z = compute_zscore(value, series)
        assert z > 0

    def test_zscore_returns_zero_for_short_series(self):
        """Test that short series returns 0."""
        series = pd.Series([0.01])
        z = compute_zscore(0.05, series)
        assert z == 0.0

    def test_zscore_returns_zero_for_zero_std(self):
        """Test that constant series returns 0."""
        series = pd.Series([0.01, 0.01, 0.01, 0.01])
        z = compute_zscore(0.05, series)
        assert z == 0.0


class TestComputeTriggers:
    @pytest.fixture
    def default_thresholds(self):
        return ThresholdsConfig(
            lookback_days=30,
            vol_lookback_days=20,
            price_move_z=1.5,
            volume_multiple=1.5,
            rel_move_vs_sector_z=1.2,
            rel_move_vs_index_z=1.5,
        )

    def test_no_trigger_on_normal_move(self, default_thresholds):
        """Test that normal moves don't trigger."""
        df = create_test_df(n_days=30, last_return=0.005)  # 0.5% move
        spy_df = create_test_df(n_days=30, last_return=0.005)

        result = compute_triggers(
            ticker="TEST",
            df=df,
            spy_df=spy_df,
            sector_df=None,
            thresholds=default_thresholds,
        )

        assert result is not None
        assert result.triggered is False
        assert result.label == "IGNORE"

    def test_trigger_on_large_price_move(self, default_thresholds):
        """Test that large price moves trigger."""
        # Create data with varied historical returns, then a large spike
        dates = [date.today() - timedelta(days=30 - i - 1) for i in range(30)]

        # Historical returns: small random-ish returns
        np.random.seed(42)
        hist_returns = np.random.normal(0.001, 0.01, 28)  # Mean 0.1%, std 1%

        closes = [100.0]
        for r in hist_returns:
            closes.append(closes[-1] * (1 + r))
        # Last day: 5% spike
        closes.append(closes[-1] * 1.05)

        df = pd.DataFrame({
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1_000_000] * 30,
        }, index=dates)
        df["return"] = df["close"].pct_change()

        spy_df = create_test_df(n_days=30, last_return=0.001)

        result = compute_triggers(
            ticker="TEST",
            df=df,
            spy_df=spy_df,
            sector_df=None,
            thresholds=default_thresholds,
        )

        assert result is not None
        assert result.triggered is True
        assert any("price_z" in r for r in result.triggered_reasons)

    def test_trigger_on_high_volume(self, default_thresholds):
        """Test that high volume triggers."""
        df = create_test_df(n_days=30, last_volume_multiple=2.0)
        spy_df = create_test_df(n_days=30)

        result = compute_triggers(
            ticker="TEST",
            df=df,
            spy_df=spy_df,
            sector_df=None,
            thresholds=default_thresholds,
        )

        assert result is not None
        assert result.triggered is True
        assert any("vol" in r for r in result.triggered_reasons)

    def test_returns_none_for_insufficient_data(self, default_thresholds):
        """Test that insufficient data returns None."""
        df = create_test_df(n_days=5)  # Too short
        spy_df = create_test_df(n_days=30)

        result = compute_triggers(
            ticker="TEST",
            df=df,
            spy_df=spy_df,
            sector_df=None,
            thresholds=default_thresholds,
        )

        assert result is None


class TestActionabilityLabel:
    def test_actionable_on_strong_price_and_relative(self):
        """Test ACTIONABLE label with strong signals."""
        label = compute_actionability_label(
            price_z=2.5,
            volume_multiple=1.3,
            rel_vs_spy_z=1.8,
            rel_vs_sector_z=None,
            triggered=True,
        )
        assert label == "ACTIONABLE"

    def test_actionable_on_high_volume_with_big_move(self):
        """Test ACTIONABLE label with high volume AND big price move.

        Under tightened rules, volume alone is not enough for ACTIONABLE.
        Need either:
        - Path A: price_z >= 2.0 AND volume >= 2.0
        - Path B: has_company_catalyst AND notable move
        """
        label = compute_actionability_label(
            price_z=2.0,  # Big move (threshold)
            volume_multiple=2.5,  # High volume (confirmation)
            rel_vs_spy_z=0.5,
            rel_vs_sector_z=None,
            triggered=True,
        )
        assert label == "ACTIONABLE"

    def test_monitor_on_high_volume_without_big_move(self):
        """Test MONITOR label with high volume but small price move.

        Under tightened rules, volume alone is not enough for ACTIONABLE.
        """
        label = compute_actionability_label(
            price_z=1.0,  # Not big enough (< 2.0)
            volume_multiple=2.5,  # High volume, but no confirmation
            rel_vs_spy_z=0.5,
            rel_vs_sector_z=None,
            triggered=True,
        )
        assert label == "MONITOR"

    def test_monitor_on_moderate_signals(self):
        """Test MONITOR label with moderate signals."""
        label = compute_actionability_label(
            price_z=1.6,
            volume_multiple=1.6,
            rel_vs_spy_z=1.0,
            rel_vs_sector_z=None,
            triggered=True,
        )
        assert label == "MONITOR"

    def test_ignore_when_not_triggered(self):
        """Test IGNORE label when not triggered."""
        label = compute_actionability_label(
            price_z=0.5,
            volume_multiple=1.0,
            rel_vs_spy_z=0.3,
            rel_vs_sector_z=None,
            triggered=False,
        )
        assert label == "IGNORE"
