"""Trend context computation for Phase 2 market state analysis.

This module provides global context for price moves, separate from Phase 1 attribution.
Phase 1 explains "why" the stock moved (causes).
Phase 2 explains "where" this move sits within the stock's broader historical context.

Key constraints:
- Quantitative-only input (no news/events)
- No causality claims
- No trading advice
- Deterministic market state labels
"""

from dataclasses import dataclass, field, asdict
from datetime import date
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class MarketState(str, Enum):
    """Market state taxonomy for classifying stock positioning.

    States are mutually exclusive and assigned via explicit thresholds.
    """

    BREAKOUT_UP = "breakout_up"  # New 52w high with high volume
    BREAKOUT_DOWN = "breakout_down"  # New 52w low with high volume
    EXTENDED_RALLY = "extended_rally"  # >2 std dev above 252d mean
    EXTENDED_SELLOFF = "extended_selloff"  # >2 std dev below 252d mean
    RECOVERY_BOUNCE = "recovery_bounce"  # Up >15% from 52w low
    PULLBACK_FROM_HIGH = "pullback_from_high"  # Down >10% from 52w high
    RANGE_BOUND_HIGH = "range_bound_high"  # In top 20% of 52w range
    RANGE_BOUND_MID = "range_bound_mid"  # In middle 60% of 52w range
    RANGE_BOUND_LOW = "range_bound_low"  # In bottom 20% of 52w range
    TREND_CONTINUATION = "trend_continuation"  # Strong trend, not at extreme
    VOLATILITY_SPIKE = "volatility_spike"  # Current vol >2x 20d avg


@dataclass
class MarketStateThresholds:
    """Configurable thresholds for market state assignment."""

    new_high_window_days: int = 5  # "New high" if within last N days
    extended_z_threshold: float = 2.0  # Z-score for extended state
    recovery_pct: float = 0.15  # 15% for recovery bounce
    pullback_pct: float = 0.10  # 10% for pullback from high
    range_high_pct: float = 0.80  # Top 20% = above 80th percentile
    range_low_pct: float = 0.20  # Bottom 20% = below 20th percentile
    vol_spike_multiple: float = 2.0  # 2x avg volatility
    trend_z_threshold: float = 1.5  # Z-score for trend continuation
    # Return magnitude thresholds (for when z-score alone is insufficient)
    strong_uptrend_return_pct: float = 50.0  # YoY return that indicates uptrend
    strong_downtrend_return_pct: float = -30.0  # YoY return that indicates downtrend


@dataclass
class TrendReturns:
    """Multi-horizon returns."""

    pct_1d: float
    pct_5d: float
    pct_21d: float  # ~1 month
    pct_63d: float  # ~3 months
    pct_126d: float  # ~6 months
    pct_252d: float  # ~1 year


@dataclass
class TrendZScores:
    """Z-scores at each horizon (where is price vs historical distribution)."""

    z_5d: float
    z_21d: float
    z_63d: float
    z_126d: float
    z_252d: float


@dataclass
class RelativeMetrics:
    """Relative performance vs benchmarks."""

    vs_spy_z_63d: float
    vs_spy_z_252d: float
    vs_sector_z_63d: Optional[float] = None
    vs_sector_z_252d: Optional[float] = None


@dataclass
class ExtremeMetrics:
    """52-week positioning."""

    pct_from_52w_high: float  # e.g., -48.2 = 48% below high
    pct_from_52w_low: float  # e.g., +3.1 = 3% above low
    days_since_52w_high: int
    days_since_52w_low: int


@dataclass
class TrendContext:
    """Complete trend context for Phase 2 LLM input."""

    ticker: str
    returns: TrendReturns
    z: TrendZScores
    relative: RelativeMetrics
    extremes: ExtremeMetrics

    # Computed market state (deterministic)
    market_state: MarketState
    market_state_rationale: str

    # Optional: volatility regime
    vol_20d_annualized: Optional[float] = None
    vol_regime: Optional[str] = None  # "low", "normal", "elevated", "extreme"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "returns": asdict(self.returns),
            "z": asdict(self.z),
            "relative": asdict(self.relative),
            "extremes": asdict(self.extremes),
            "market_state": self.market_state.value,
            "market_state_rationale": self.market_state_rationale,
            "vol_20d_annualized": self.vol_20d_annualized,
            "vol_regime": self.vol_regime,
        }


def zscore_to_descriptor(z: float) -> str:
    """Translate z-score to human-readable descriptor.

    This is the authoritative mapping for PM-facing output.
    The z-score determines the state; the descriptor communicates it.

    Mapping:
        ≥ +2.5      → Extreme strength
        +2.0 to +2.5 → Very strong
        +1.5 to +2.0 → Elevated strength
        -1.5 to +1.5 → Neutral
        -2.0 to -1.5 → Notable weakness
        -2.5 to -2.0 → Severe weakness
        ≤ -2.5      → Extreme weakness
    """
    if z >= 2.5:
        return "Extreme strength"
    elif z >= 2.0:
        return "Very strong"
    elif z >= 1.5:
        return "Elevated strength"
    elif z >= 0.5:
        return "Mild strength"
    elif z > -0.5:
        return "Neutral"
    elif z > -1.5:
        return "Mild weakness"
    elif z > -2.0:
        return "Notable weakness"
    elif z > -2.5:
        return "Severe weakness"
    else:
        return "Extreme weakness"


def get_horizon_summary(z: TrendZScores) -> dict[str, str]:
    """Get human-readable horizon summaries from z-scores.

    Returns a dict mapping horizon labels to descriptors:
    - short_term: Based on z_5d
    - one_month: Based on z_21d
    - quarter: Based on z_63d
    - one_year: Based on z_252d

    These are the primary narrative surface for PMs.
    """
    return {
        "short_term": zscore_to_descriptor(z.z_5d),
        "one_month": zscore_to_descriptor(z.z_21d),
        "quarter": zscore_to_descriptor(z.z_63d),
        "one_year": zscore_to_descriptor(z.z_252d),
    }


def format_zscore_details(z: TrendZScores) -> str:
    """Format raw z-scores as a secondary detail line.

    This preserves transparency and auditability while keeping
    raw numbers as secondary information.
    """
    return f"z5={z.z_5d:+.1f}, z21={z.z_21d:+.1f}, z63={z.z_63d:+.1f}, z252={z.z_252d:+.1f}"


def _compute_return(df: pd.DataFrame, days: int) -> float:
    """Compute return over N days.

    Returns 0.0 if insufficient data.
    """
    if len(df) < days + 1:
        return 0.0
    return (df["close"].iloc[-1] / df["close"].iloc[-(days + 1)] - 1) * 100


def _compute_zscore(value: float, series: pd.Series) -> float:
    """Compute z-score of value against a series.

    Returns 0.0 if there's not enough data or std is 0.
    """
    if len(series) < 2:
        return 0.0

    mean = series.mean()
    std = series.std()

    if std == 0 or np.isnan(std):
        return 0.0

    return (value - mean) / std


def _compute_return_zscore(df: pd.DataFrame, window: int) -> float:
    """Compute z-score of current price position within window.

    Uses rolling returns to assess where current return sits historically.
    """
    if len(df) < window + 1:
        return 0.0

    # Compute rolling N-day returns
    returns = df["close"].pct_change(window) * 100
    current_return = returns.iloc[-1]

    if np.isnan(current_return):
        return 0.0

    # Use historical returns for z-score (excluding current)
    hist_returns = returns.iloc[:-1].dropna()
    return _compute_zscore(current_return, hist_returns)


def _compute_relative_zscore(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    window: int,
) -> float:
    """Compute z-score of stock's relative performance vs benchmark."""
    if len(stock_df) < window + 1 or len(benchmark_df) < window + 1:
        return 0.0

    # Align dates
    common_dates = stock_df.index.intersection(benchmark_df.index)
    if len(common_dates) < window + 1:
        return 0.0

    stock_aligned = stock_df.loc[common_dates]
    bench_aligned = benchmark_df.loc[common_dates]

    # Compute rolling relative returns
    stock_returns = stock_aligned["close"].pct_change(window)
    bench_returns = bench_aligned["close"].pct_change(window)
    relative_returns = (stock_returns - bench_returns) * 100

    current_rel = relative_returns.iloc[-1]
    if np.isnan(current_rel):
        return 0.0

    hist_rel = relative_returns.iloc[:-1].dropna()
    return _compute_zscore(current_rel, hist_rel)


def _compute_52w_metrics(df: pd.DataFrame) -> tuple[float, float, float, float, int, int]:
    """Compute 52-week high/low metrics.

    Returns:
        Tuple of (high_52w, low_52w, pct_from_high, pct_from_low,
                  days_since_high, days_since_low)
    """
    # Use ~252 trading days for 52 weeks
    lookback = min(252, len(df))
    recent_df = df.iloc[-lookback:]

    high_52w = recent_df["high"].max()
    low_52w = recent_df["low"].min()
    current_price = df["close"].iloc[-1]

    pct_from_high = ((current_price / high_52w) - 1) * 100
    pct_from_low = ((current_price / low_52w) - 1) * 100

    # Find days since high/low
    high_date_idx = recent_df["high"].idxmax()
    low_date_idx = recent_df["low"].idxmin()

    # Convert to days since
    dates = list(recent_df.index)
    days_since_high = len(dates) - 1 - dates.index(high_date_idx)
    days_since_low = len(dates) - 1 - dates.index(low_date_idx)

    return high_52w, low_52w, pct_from_high, pct_from_low, days_since_high, days_since_low


def _compute_volatility(df: pd.DataFrame) -> tuple[Optional[float], Optional[str]]:
    """Compute annualized volatility and regime.

    Returns:
        Tuple of (vol_20d_annualized, vol_regime)
    """
    if len(df) < 21:
        return None, None

    # Compute daily returns
    returns = df["close"].pct_change().dropna()
    if len(returns) < 20:
        return None, None

    # 20-day realized volatility (annualized)
    vol_20d = returns.iloc[-20:].std() * np.sqrt(252) * 100

    # Compute historical volatility for regime classification
    if len(returns) < 60:
        return vol_20d, "normal"

    hist_vol = returns.rolling(20).std().iloc[:-20].dropna() * np.sqrt(252) * 100

    if len(hist_vol) < 10:
        return vol_20d, "normal"

    vol_percentile = (hist_vol < vol_20d).mean()

    if vol_percentile < 0.25:
        regime = "low"
    elif vol_percentile < 0.75:
        regime = "normal"
    elif vol_percentile < 0.95:
        regime = "elevated"
    else:
        regime = "extreme"

    return vol_20d, regime


def _compute_range_position(current_price: float, high_52w: float, low_52w: float) -> float:
    """Compute position within 52-week range (0.0 = at low, 1.0 = at high)."""
    if high_52w == low_52w:
        return 0.5
    return (current_price - low_52w) / (high_52w - low_52w)


def _assign_market_state(
    df: pd.DataFrame,
    pct_from_high: float,
    pct_from_low: float,
    days_since_high: int,
    days_since_low: int,
    range_position: float,
    z_252d: float,
    vol_20d: Optional[float],
    thresholds: MarketStateThresholds,
    pct_252d: float = 0.0,  # YoY return for magnitude-based classification
) -> tuple[MarketState, str]:
    """Assign market state based on decision tree.

    The decision tree prioritizes:
    1. Breakouts (new 52w extremes with volume)
    2. Extended moves (by z-score OR by return magnitude)
    3. Trend continuation (strong underlying trend)
    4. Pullbacks within uptrends / bounces within downtrends
    5. Recovery bounces / pullbacks from highs (no underlying trend)
    6. Volatility spikes
    7. Range-bound states

    Returns:
        Tuple of (MarketState, rationale string)
    """
    volume_multiple = 1.0
    if len(df) >= 21:
        vol_last = df["volume"].iloc[-1]
        vol_avg = df["volume"].iloc[-21:-1].mean()
        if vol_avg > 0:
            volume_multiple = vol_last / vol_avg

    # 1. Check for 52-week extremes (breakouts)
    if days_since_high <= thresholds.new_high_window_days and volume_multiple >= thresholds.vol_spike_multiple:
        return MarketState.BREAKOUT_UP, f"New 52w high within {days_since_high}d, volume {volume_multiple:.1f}x average"

    if days_since_low <= thresholds.new_high_window_days and volume_multiple >= thresholds.vol_spike_multiple:
        return MarketState.BREAKOUT_DOWN, f"New 52w low within {days_since_low}d, volume {volume_multiple:.1f}x average"

    # 2. Check for extended moves (z-score based OR return magnitude based)
    # A stock up 50%+ YoY is in an extended rally even if z-score is moderate
    if z_252d >= thresholds.extended_z_threshold:
        return MarketState.EXTENDED_RALLY, f"Price z-score {z_252d:.1f} (>2.0 std dev above mean)"

    if pct_252d >= thresholds.strong_uptrend_return_pct:
        return MarketState.EXTENDED_RALLY, f"YoY return {pct_252d:+.1f}% indicates strong uptrend"

    if z_252d <= -thresholds.extended_z_threshold:
        return MarketState.EXTENDED_SELLOFF, f"Price z-score {z_252d:.1f} (<-2.0 std dev below mean)"

    if pct_252d <= thresholds.strong_downtrend_return_pct:
        return MarketState.EXTENDED_SELLOFF, f"YoY return {pct_252d:.1f}% indicates strong downtrend"

    # 3. Check for trend continuation (moderate but persistent trend)
    # This comes before recovery/pullback to avoid misclassifying trending stocks
    if abs(z_252d) >= thresholds.trend_z_threshold and thresholds.range_low_pct < range_position < thresholds.range_high_pct:
        direction = "up" if z_252d > 0 else "down"
        return MarketState.TREND_CONTINUATION, f"Strong {direction}trend (z={z_252d:.1f}), not at extremes"

    # 4. Check for pullback within uptrend or bounce within downtrend
    # If YoY return is positive but stock is pulling back from highs = PULLBACK_FROM_HIGH
    # If YoY return is negative but stock is bouncing from lows = RECOVERY_BOUNCE
    is_uptrend = pct_252d > 20  # Meaningfully positive YoY
    is_downtrend = pct_252d < -20  # Meaningfully negative YoY

    # Pullback from high in an uptrend (still bullish context)
    if pct_from_high < -thresholds.pullback_pct * 100 and is_uptrend:
        return MarketState.PULLBACK_FROM_HIGH, f"{pct_from_high:.1f}% from 52w high, but YoY still +{pct_252d:.1f}%"

    # Recovery bounce in a downtrend (still bearish context)
    if pct_from_low > thresholds.recovery_pct * 100 and is_downtrend and range_position < thresholds.range_high_pct:
        return MarketState.RECOVERY_BOUNCE, f"+{pct_from_low:.1f}% from 52w low, but YoY still {pct_252d:.1f}%"

    # 5. Generic pullback from high (no clear trend)
    if pct_from_high < -thresholds.pullback_pct * 100 and range_position > thresholds.range_low_pct:
        return MarketState.PULLBACK_FROM_HIGH, f"{pct_from_high:.1f}% from 52w high, range position {range_position:.0%}"

    # 6. Check for volatility spike
    if vol_20d is not None and len(df) >= 60:
        returns = df["close"].pct_change().dropna()
        hist_vol = returns.iloc[-60:-20].std() * np.sqrt(252) * 100
        if hist_vol > 0 and vol_20d / hist_vol >= thresholds.vol_spike_multiple:
            return MarketState.VOLATILITY_SPIKE, f"Volatility {vol_20d:.1f}% ({vol_20d/hist_vol:.1f}x historical)"

    # 7. Default to range positioning
    if range_position >= thresholds.range_high_pct:
        return MarketState.RANGE_BOUND_HIGH, f"In top 20% of 52w range ({range_position:.0%})"

    if range_position <= thresholds.range_low_pct:
        return MarketState.RANGE_BOUND_LOW, f"In bottom 20% of 52w range ({range_position:.0%})"

    return MarketState.RANGE_BOUND_MID, f"In middle of 52w range ({range_position:.0%})"


def compute_trend_context(
    ticker: str,
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
    sector_df: Optional[pd.DataFrame] = None,
    thresholds: Optional[MarketStateThresholds] = None,
) -> Optional[TrendContext]:
    """Compute complete trend context for Phase 2.

    Args:
        ticker: The ticker symbol
        df: Extended OHLCV DataFrame (260+ days recommended)
        spy_df: Extended SPY DataFrame for relative metrics
        sector_df: Optional sector ETF DataFrame for sector-relative metrics
        thresholds: Market state thresholds (uses defaults if None)

    Returns:
        TrendContext with all computed metrics, or None if insufficient data.
    """
    if df is None or len(df) < 21:
        return None

    if thresholds is None:
        thresholds = MarketStateThresholds()

    # Compute multi-horizon returns
    returns = TrendReturns(
        pct_1d=_compute_return(df, 1),
        pct_5d=_compute_return(df, 5),
        pct_21d=_compute_return(df, 21),
        pct_63d=_compute_return(df, 63),
        pct_126d=_compute_return(df, 126),
        pct_252d=_compute_return(df, 252),
    )

    # Compute z-scores at each horizon
    z_scores = TrendZScores(
        z_5d=_compute_return_zscore(df, 5),
        z_21d=_compute_return_zscore(df, 21),
        z_63d=_compute_return_zscore(df, 63),
        z_126d=_compute_return_zscore(df, 126),
        z_252d=_compute_return_zscore(df, 252),
    )

    # Compute relative metrics vs SPY
    vs_spy_z_63d = _compute_relative_zscore(df, spy_df, 63) if spy_df is not None else 0.0
    vs_spy_z_252d = _compute_relative_zscore(df, spy_df, 252) if spy_df is not None else 0.0

    # Compute relative metrics vs sector
    vs_sector_z_63d = None
    vs_sector_z_252d = None
    if sector_df is not None and len(sector_df) > 0:
        vs_sector_z_63d = _compute_relative_zscore(df, sector_df, 63)
        vs_sector_z_252d = _compute_relative_zscore(df, sector_df, 252)

    relative = RelativeMetrics(
        vs_spy_z_63d=vs_spy_z_63d,
        vs_spy_z_252d=vs_spy_z_252d,
        vs_sector_z_63d=vs_sector_z_63d,
        vs_sector_z_252d=vs_sector_z_252d,
    )

    # Compute 52-week metrics
    high_52w, low_52w, pct_from_high, pct_from_low, days_since_high, days_since_low = _compute_52w_metrics(df)

    extremes = ExtremeMetrics(
        pct_from_52w_high=pct_from_high,
        pct_from_52w_low=pct_from_low,
        days_since_52w_high=days_since_high,
        days_since_52w_low=days_since_low,
    )

    # Compute volatility
    vol_20d, vol_regime = _compute_volatility(df)

    # Compute range position for market state
    current_price = df["close"].iloc[-1]
    range_position = _compute_range_position(current_price, high_52w, low_52w)

    # Assign market state
    market_state, rationale = _assign_market_state(
        df=df,
        pct_from_high=pct_from_high,
        pct_from_low=pct_from_low,
        days_since_high=days_since_high,
        days_since_low=days_since_low,
        range_position=range_position,
        z_252d=z_scores.z_252d,
        vol_20d=vol_20d,
        thresholds=thresholds,
        pct_252d=returns.pct_252d,  # Pass YoY return for magnitude-based classification
    )

    return TrendContext(
        ticker=ticker,
        returns=returns,
        z=z_scores,
        relative=relative,
        extremes=extremes,
        market_state=market_state,
        market_state_rationale=rationale,
        vol_20d_annualized=vol_20d,
        vol_regime=vol_regime,
    )
