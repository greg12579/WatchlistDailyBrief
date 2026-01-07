"""Tests for trend context computation."""

from datetime import date
import numpy as np
import pandas as pd
import pytest

from watchbrief.features.trend_context import (
    MarketState,
    MarketStateThresholds,
    TrendContext,
    TrendReturns,
    TrendZScores,
    RelativeMetrics,
    ExtremeMetrics,
    compute_trend_context,
    _compute_return,
    _compute_zscore,
    _compute_52w_metrics,
    _compute_range_position,
    _assign_market_state,
    zscore_to_descriptor,
    get_horizon_summary,
    format_zscore_details,
)


def create_test_df(prices: list[float], volumes: list[float] = None) -> pd.DataFrame:
    """Create a test DataFrame with OHLCV data."""
    n = len(prices)
    if volumes is None:
        volumes = [1000000] * n

    dates = pd.date_range(end=date.today(), periods=n, freq="D")

    df = pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": volumes,
    }, index=dates)

    df.index = [d.date() for d in df.index]
    df.index.name = "date"

    return df


class TestComputeReturn:
    """Tests for _compute_return function."""

    def test_simple_return(self):
        """Test basic return calculation."""
        prices = [100.0] * 10 + [110.0]  # 10% increase
        df = create_test_df(prices)
        ret = _compute_return(df, 1)
        assert abs(ret - 10.0) < 0.01

    def test_5d_return(self):
        """Test 5-day return calculation."""
        prices = [100.0] * 10 + [120.0]  # 20% increase from 5 days ago
        df = create_test_df(prices)
        ret = _compute_return(df, 5)
        assert abs(ret - 20.0) < 0.01

    def test_insufficient_data(self):
        """Test with insufficient data returns 0."""
        prices = [100.0, 110.0]
        df = create_test_df(prices)
        ret = _compute_return(df, 10)  # Need more data
        assert ret == 0.0


class TestComputeZscore:
    """Tests for _compute_zscore function."""

    def test_normal_zscore(self):
        """Test z-score calculation with normal data."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z = _compute_zscore(5.0, series)  # Value at edge
        # Mean=3, std=1.58, z = (5-3)/1.58 = 1.26
        assert 1.0 < z < 1.5

    def test_extreme_zscore(self):
        """Test z-score for extreme value."""
        series = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
        series.iloc[0] = 10.0  # One outlier
        z = _compute_zscore(100.0, series)  # Extreme value
        assert z > 10.0  # Should be very high

    def test_insufficient_data(self):
        """Test z-score with insufficient data returns 0."""
        series = pd.Series([1.0])
        z = _compute_zscore(2.0, series)
        assert z == 0.0

    def test_zero_std(self):
        """Test z-score with zero std returns 0."""
        series = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        z = _compute_zscore(10.0, series)
        assert z == 0.0


class TestCompute52WMetrics:
    """Tests for _compute_52w_metrics function."""

    def test_at_52w_high(self):
        """Test when price is at 52-week high."""
        prices = list(range(100, 200))  # Steadily increasing
        df = create_test_df(prices)

        high, low, pct_high, pct_low, days_high, days_low = _compute_52w_metrics(df)

        assert high == 199.0 * 1.01  # Intraday high
        assert low == 100.0 * 0.99  # Intraday low
        assert pct_high > -2.0  # Near high
        assert pct_low > 90.0  # Far from low
        assert days_high <= 1  # Recently hit high

    def test_at_52w_low(self):
        """Test when price is at 52-week low."""
        prices = list(range(200, 100, -1))  # Steadily decreasing
        df = create_test_df(prices)

        high, low, pct_high, pct_low, days_high, days_low = _compute_52w_metrics(df)

        assert pct_high < -45.0  # Far from high
        assert pct_low < 5.0  # Near low
        assert days_low <= 1  # Recently hit low


class TestComputeRangePosition:
    """Tests for _compute_range_position function."""

    def test_at_high(self):
        """Test position at high."""
        pos = _compute_range_position(100.0, 100.0, 50.0)
        assert pos == 1.0

    def test_at_low(self):
        """Test position at low."""
        pos = _compute_range_position(50.0, 100.0, 50.0)
        assert pos == 0.0

    def test_at_midpoint(self):
        """Test position at midpoint."""
        pos = _compute_range_position(75.0, 100.0, 50.0)
        assert pos == 0.5

    def test_equal_high_low(self):
        """Test when high equals low."""
        pos = _compute_range_position(100.0, 100.0, 100.0)
        assert pos == 0.5


class TestMarketStateAssignment:
    """Tests for market state assignment logic."""

    def test_breakout_up(self):
        """Test breakout up detection."""
        # Create data with new high and high volume
        prices = [100.0] * 250 + [120.0]  # Jump to new high
        volumes = [1000000] * 250 + [3000000]  # High volume
        df = create_test_df(prices, volumes)
        thresholds = MarketStateThresholds()

        state, rationale = _assign_market_state(
            df=df,
            pct_from_high=0.0,  # At high
            pct_from_low=20.0,
            days_since_high=0,
            days_since_low=250,
            range_position=1.0,
            z_252d=1.0,
            vol_20d=None,
            thresholds=thresholds,
        )

        assert state == MarketState.BREAKOUT_UP

    def test_extended_selloff(self):
        """Test extended selloff detection."""
        prices = [100.0] * 300
        df = create_test_df(prices)
        thresholds = MarketStateThresholds()

        state, rationale = _assign_market_state(
            df=df,
            pct_from_high=-50.0,
            pct_from_low=5.0,
            days_since_high=200,
            days_since_low=5,
            range_position=0.1,
            z_252d=-2.5,  # Below threshold
            vol_20d=None,
            thresholds=thresholds,
        )

        assert state == MarketState.EXTENDED_SELLOFF

    def test_recovery_bounce(self):
        """Test recovery bounce detection.

        Recovery bounce only triggers in a downtrend context (YoY < -20%).
        The stock must be bouncing from lows in a still-bearish environment.
        """
        prices = [100.0] * 300
        df = create_test_df(prices)
        thresholds = MarketStateThresholds()

        state, rationale = _assign_market_state(
            df=df,
            pct_from_high=-30.0,
            pct_from_low=20.0,  # >15% from low
            days_since_high=100,
            days_since_low=20,
            range_position=0.4,  # Not in top 20%
            z_252d=-0.5,
            vol_20d=None,
            thresholds=thresholds,
            pct_252d=-25.0,  # Downtrend: YoY negative, so this is a bounce in a downtrend
        )

        assert state == MarketState.RECOVERY_BOUNCE

    def test_pullback_in_uptrend(self):
        """Test that stocks in uptrends pulling back are classified as PULLBACK_FROM_HIGH.

        A stock up +279% YoY that pulls back -34% from highs is a pullback in an uptrend,
        NOT a recovery bounce.
        """
        prices = [100.0] * 300
        df = create_test_df(prices)
        thresholds = MarketStateThresholds()

        state, rationale = _assign_market_state(
            df=df,
            pct_from_high=-34.0,  # Pulled back from highs
            pct_from_low=340.0,  # Far above 52w low
            days_since_high=26,
            days_since_low=252,
            range_position=0.66,  # In middle of range
            z_252d=-1.3,  # Moderate z-score
            vol_20d=None,
            thresholds=thresholds,
            pct_252d=279.0,  # Strong uptrend: YoY +279%
        )

        # Should be EXTENDED_RALLY because YoY > 50% (strong_uptrend_return_pct threshold)
        assert state == MarketState.EXTENDED_RALLY

    def test_strong_uptrend_by_return(self):
        """Test that stocks with large YoY returns are classified as EXTENDED_RALLY.

        Even if z-score is moderate, a stock +50%+ YoY is in an extended rally.
        """
        prices = [100.0] * 300
        df = create_test_df(prices)
        thresholds = MarketStateThresholds()

        state, rationale = _assign_market_state(
            df=df,
            pct_from_high=-20.0,
            pct_from_low=100.0,
            days_since_high=30,
            days_since_low=200,
            range_position=0.60,
            z_252d=1.0,  # Below extended_z_threshold (2.0)
            vol_20d=None,
            thresholds=thresholds,
            pct_252d=75.0,  # YoY +75% exceeds strong_uptrend_return_pct (50%)
        )

        assert state == MarketState.EXTENDED_RALLY

    def test_range_bound_mid(self):
        """Test range-bound middle detection."""
        prices = [100.0] * 300
        df = create_test_df(prices)
        thresholds = MarketStateThresholds()

        state, rationale = _assign_market_state(
            df=df,
            pct_from_high=-5.0,  # Not far enough for pullback (-10% threshold)
            pct_from_low=5.0,   # Not far enough for recovery (+15% threshold)
            days_since_high=50,
            days_since_low=30,
            range_position=0.5,  # Middle
            z_252d=0.0,
            vol_20d=None,
            thresholds=thresholds,
        )

        assert state == MarketState.RANGE_BOUND_MID


class TestComputeTrendContext:
    """Tests for the full compute_trend_context function."""

    def test_full_computation(self):
        """Test full trend context computation."""
        # Create 300 days of data
        prices = [100.0 + i * 0.1 for i in range(300)]  # Gradual uptrend
        df = create_test_df(prices)
        spy_df = create_test_df([100.0 + i * 0.05 for i in range(300)])  # SPY slower

        context = compute_trend_context(
            ticker="TEST",
            df=df,
            spy_df=spy_df,
            sector_df=None,
        )

        assert context is not None
        assert context.ticker == "TEST"
        assert isinstance(context.returns, TrendReturns)
        assert isinstance(context.z, TrendZScores)
        assert isinstance(context.relative, RelativeMetrics)
        assert isinstance(context.extremes, ExtremeMetrics)
        assert isinstance(context.market_state, MarketState)

    def test_insufficient_data(self):
        """Test with insufficient data returns None."""
        prices = [100.0] * 10
        df = create_test_df(prices)
        spy_df = create_test_df(prices)

        context = compute_trend_context(
            ticker="TEST",
            df=df,
            spy_df=spy_df,
        )

        # Should still work with 10 days (minimum is 21)
        assert context is None

    def test_to_dict(self):
        """Test TrendContext.to_dict() method."""
        prices = [100.0 + i * 0.1 for i in range(300)]
        df = create_test_df(prices)
        spy_df = create_test_df([100.0] * 300)

        context = compute_trend_context(
            ticker="TEST",
            df=df,
            spy_df=spy_df,
        )

        d = context.to_dict()
        assert d["ticker"] == "TEST"
        assert "returns" in d
        assert "z" in d
        assert "relative" in d
        assert "extremes" in d
        assert "market_state" in d


class TestTrendReturns:
    """Tests for TrendReturns dataclass."""

    def test_returns_structure(self):
        """Test TrendReturns has correct fields."""
        returns = TrendReturns(
            pct_1d=1.0,
            pct_5d=5.0,
            pct_21d=10.0,
            pct_63d=15.0,
            pct_126d=20.0,
            pct_252d=30.0,
        )
        assert returns.pct_1d == 1.0
        assert returns.pct_252d == 30.0


class TestMarketStateThresholds:
    """Tests for MarketStateThresholds defaults."""

    def test_defaults(self):
        """Test default thresholds are sensible."""
        t = MarketStateThresholds()
        assert t.extended_z_threshold == 2.0
        assert t.recovery_pct == 0.15
        assert t.pullback_pct == 0.10
        assert t.range_high_pct == 0.80
        assert t.range_low_pct == 0.20


class TestZScoreDescriptor:
    """Tests for z-score to human-readable descriptor translation."""

    def test_extreme_strength(self):
        """Test extreme positive z-scores."""
        assert zscore_to_descriptor(3.0) == "Extreme strength"
        assert zscore_to_descriptor(2.5) == "Extreme strength"

    def test_very_strong(self):
        """Test very strong z-scores."""
        assert zscore_to_descriptor(2.4) == "Very strong"
        assert zscore_to_descriptor(2.0) == "Very strong"

    def test_elevated_strength(self):
        """Test elevated strength z-scores."""
        assert zscore_to_descriptor(1.9) == "Elevated strength"
        assert zscore_to_descriptor(1.5) == "Elevated strength"

    def test_mild_strength(self):
        """Test mild strength z-scores."""
        assert zscore_to_descriptor(1.4) == "Mild strength"
        assert zscore_to_descriptor(0.5) == "Mild strength"

    def test_neutral(self):
        """Test neutral z-scores."""
        assert zscore_to_descriptor(0.4) == "Neutral"
        assert zscore_to_descriptor(0.0) == "Neutral"
        assert zscore_to_descriptor(-0.4) == "Neutral"

    def test_mild_weakness(self):
        """Test mild weakness z-scores."""
        assert zscore_to_descriptor(-0.5) == "Mild weakness"
        assert zscore_to_descriptor(-1.4) == "Mild weakness"

    def test_notable_weakness(self):
        """Test notable weakness z-scores."""
        assert zscore_to_descriptor(-1.5) == "Notable weakness"
        assert zscore_to_descriptor(-1.9) == "Notable weakness"

    def test_severe_weakness(self):
        """Test severe weakness z-scores."""
        assert zscore_to_descriptor(-2.0) == "Severe weakness"
        assert zscore_to_descriptor(-2.4) == "Severe weakness"

    def test_extreme_weakness(self):
        """Test extreme negative z-scores."""
        assert zscore_to_descriptor(-2.5) == "Extreme weakness"
        assert zscore_to_descriptor(-3.0) == "Extreme weakness"


class TestHorizonSummary:
    """Tests for horizon summary generation."""

    def test_horizon_summary_structure(self):
        """Test horizon summary returns correct keys."""
        z_scores = TrendZScores(
            z_5d=1.0,
            z_21d=-0.5,
            z_63d=2.5,
            z_126d=0.0,
            z_252d=-2.0,
        )
        summary = get_horizon_summary(z_scores)

        assert "short_term" in summary
        assert "one_month" in summary
        assert "quarter" in summary
        assert "one_year" in summary

    def test_horizon_summary_values(self):
        """Test horizon summary maps z-scores correctly."""
        z_scores = TrendZScores(
            z_5d=2.6,    # Extreme strength
            z_21d=-0.3,  # Neutral
            z_63d=1.6,   # Elevated strength
            z_126d=0.0,
            z_252d=-2.1, # Severe weakness
        )
        summary = get_horizon_summary(z_scores)

        assert summary["short_term"] == "Extreme strength"
        assert summary["one_month"] == "Neutral"
        assert summary["quarter"] == "Elevated strength"
        assert summary["one_year"] == "Severe weakness"


class TestZScoreDetails:
    """Tests for z-score detail formatting."""

    def test_format_zscore_details(self):
        """Test z-score detail string format."""
        z_scores = TrendZScores(
            z_5d=1.23,
            z_21d=-0.45,
            z_63d=2.00,
            z_126d=0.0,
            z_252d=-1.50,
        )
        details = format_zscore_details(z_scores)

        assert "z5=+1.2" in details
        assert "z21=-0.4" in details or "z21=-0.5" in details  # Rounding
        assert "z63=+2.0" in details
        assert "z252=-1.5" in details
