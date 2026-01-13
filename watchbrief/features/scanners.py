"""Daily Tape scanners and TapeRow model for Phase 2.5."""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from watchbrief.data.market_data import get_company_name
from watchbrief.features.trend_context import TrendContext


@dataclass
class TapeRow:
    """Lightweight model for top-of-email scanner tables."""

    ticker: str
    name: str  # Company name

    # Absolute returns (as percentages)
    ret_1d: float
    ret_5d: float
    ret_252d: float

    # Relative returns vs IWM (as percentages)
    rel_vs_iwm_1d: float
    rel_vs_iwm_5d: float
    rel_vs_iwm_252d: float

    # Trend context (from existing computation)
    state_label: str  # e.g., "Trend Continuation"
    pct_from_52w_high: float  # e.g., -16.3
    pct_from_52w_low: float  # e.g., +98.4

    # Deterministic tags for routing
    candidate_tags: list[str] = field(default_factory=list)

    # Optional z-scores (if already computed)
    z_1d: Optional[float] = None
    z_252d: Optional[float] = None
    rel_vs_iwm_z_1d: Optional[float] = None
    rel_vs_iwm_z_252d: Optional[float] = None


def compute_daily_tape(
    ticker_data: dict[str, TapeRow],
    top_n: int = 10,
) -> list[TapeRow]:
    """
    Compute the Daily Tape (Top N by absolute 1D return).

    Returns top N tickers sorted by absolute 1D return (largest moves first).
    """
    rows = list(ticker_data.values())

    # Sort by absolute 1D return (largest magnitude first)
    sorted_rows = sorted(rows, key=lambda r: abs(r.ret_1d), reverse=True)

    return sorted_rows[:top_n]


def compute_near_highs(
    ticker_data: dict[str, TapeRow],
    threshold_pct: float = -3.0,  # Within 3% of 52w high
    top_n: int = 5,
) -> list[TapeRow]:
    """
    Compute Near Highs scanner (Top N closest to 52w high).

    Filters to pct_from_52w_high >= threshold (e.g., >= -3.0).
    Returns tickers closest to 52w high.
    """
    candidates = [
        r for r in ticker_data.values() if r.pct_from_52w_high >= threshold_pct
    ]

    # Sort by proximity to high (closest first = largest value, closest to 0)
    sorted_candidates = sorted(
        candidates, key=lambda r: r.pct_from_52w_high, reverse=True
    )

    return sorted_candidates[:top_n]


def compute_broken_lows(
    ticker_data: dict[str, TapeRow],
    near_low_threshold_pct: float = 5.0,  # Within 5% of 52w low
    broken_return_threshold: float = -20.0,  # YoY return <= -20%
    broken_rel_iwm_threshold: float = -10.0,  # Rel vs IWM <= -10%
    top_n: int = 5,
) -> list[TapeRow]:
    """
    Compute "Trading Like Shit" scanner (Top N near 52w low with weak performance).

    Filters:
    - pct_from_52w_low <= threshold (e.g., <= 5.0)
    - AND (ret_252d <= -20% OR rel_vs_iwm_252d <= -10%)

    Returns tickers near 52w low with weak long-term performance.
    """
    candidates = [
        r
        for r in ticker_data.values()
        if r.pct_from_52w_low <= near_low_threshold_pct
        and (
            r.ret_252d <= broken_return_threshold
            or r.rel_vs_iwm_252d <= broken_rel_iwm_threshold
        )
    ]

    # Sort by proximity to low (closest first = smallest value)
    # Tiebreaker: most negative rel_vs_iwm_252d
    sorted_candidates = sorted(
        candidates, key=lambda r: (r.pct_from_52w_low, r.rel_vs_iwm_252d)
    )

    return sorted_candidates[:top_n]


def compute_candidate_tags(row: TapeRow) -> list[str]:
    """
    Assign deterministic tags based on simple conditions.

    These tags help route into deep-dive logic.
    """
    tags = []

    # Shock moves
    if row.z_1d is not None and row.z_1d <= -2.0:
        tags.append("ShockDown")
    elif row.rel_vs_iwm_1d <= -3.0:
        tags.append("ShockDown")

    if row.z_1d is not None and row.z_1d >= 2.0:
        tags.append("ShockUp")
    elif row.rel_vs_iwm_1d >= 3.0:
        tags.append("ShockUp")

    # Extremes positioning
    if row.pct_from_52w_high >= -3.0:
        tags.append("Near52wHigh")

    if row.pct_from_52w_low <= 5.0:
        tags.append("Near52wLow")

    # Long-term trend
    if row.z_252d is not None and row.z_252d <= -1.5:
        tags.append("BrokenLongTerm")
    elif row.ret_252d <= -30.0:
        tags.append("BrokenLongTerm")

    if row.z_252d is not None and row.z_252d >= 1.5:
        tags.append("StrongLongTerm")
    elif row.ret_252d >= 30.0:
        tags.append("StrongLongTerm")

    return tags


def _compute_return(df: pd.DataFrame, days: int) -> Optional[float]:
    """Compute N-day return from close prices."""
    if df is None or len(df) < days + 1:
        return None
    closes = df["close"].values
    return (closes[-1] / closes[-days - 1]) - 1.0


def build_tape_rows(
    watchlist: list[str],
    ticker_dfs: dict[str, pd.DataFrame],  # Pre-fetched OHLCV data
    iwm_df: Optional[pd.DataFrame],
    trend_contexts: dict[str, TrendContext],  # From existing Phase 2
) -> dict[str, TapeRow]:
    """
    Build TapeRow for each ticker in watchlist.

    Requires: ticker OHLCV, IWM OHLCV, and optional TrendContext.
    """
    rows = {}

    # Compute IWM returns at each horizon
    iwm_ret_1d = _compute_return(iwm_df, 1) if iwm_df is not None else None
    iwm_ret_5d = _compute_return(iwm_df, 5) if iwm_df is not None else None
    iwm_ret_252d = _compute_return(iwm_df, 252) if iwm_df is not None else None

    for ticker in watchlist:
        df = ticker_dfs.get(ticker)
        if df is None or len(df) < 5:
            continue

        # Compute ticker returns
        ret_1d = _compute_return(df, 1)
        ret_5d = _compute_return(df, 5)
        ret_252d = _compute_return(df, 252) if len(df) >= 253 else None

        if ret_1d is None:
            continue

        # Compute relative returns vs IWM
        rel_1d = (ret_1d - iwm_ret_1d) if iwm_ret_1d is not None else 0.0
        rel_5d = (ret_5d - iwm_ret_5d) if ret_5d and iwm_ret_5d else 0.0
        rel_252d = (ret_252d - iwm_ret_252d) if ret_252d and iwm_ret_252d else 0.0

        # Get trend context if available
        tc = trend_contexts.get(ticker)
        state_label = (
            tc.market_state.value.replace("_", " ").title() if tc else "Unknown"
        )
        pct_from_high = tc.extremes.pct_from_52w_high if tc else 0.0
        pct_from_low = tc.extremes.pct_from_52w_low if tc else 0.0
        z_1d = tc.z.z_5d if tc else None  # Closest proxy
        z_252d = tc.z.z_252d if tc else None

        row = TapeRow(
            ticker=ticker,
            name=get_company_name(ticker),
            ret_1d=ret_1d * 100,
            ret_5d=(ret_5d or 0) * 100,
            ret_252d=(ret_252d or 0) * 100,
            rel_vs_iwm_1d=rel_1d * 100,
            rel_vs_iwm_5d=rel_5d * 100,
            rel_vs_iwm_252d=rel_252d * 100,
            state_label=state_label,
            pct_from_52w_high=pct_from_high,
            pct_from_52w_low=pct_from_low,
            z_1d=z_1d,
            z_252d=z_252d,
        )

        # Compute tags
        row.candidate_tags = compute_candidate_tags(row)
        rows[ticker] = row

    return rows
