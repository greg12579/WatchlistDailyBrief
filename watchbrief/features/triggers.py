"""Trigger computation for detecting significant price/volume changes."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from watchbrief.config import ThresholdsConfig


@dataclass
class TriggerResult:
    """Result of trigger computation for a single ticker."""

    ticker: str

    # Raw values
    last_close: float
    pct_change_1d: float
    pct_change_5d: float
    volume_last: float
    volume_avg: float
    volume_multiple: float

    # Z-scores
    price_z: float
    rel_vs_spy_z: float
    rel_vs_sector_z: Optional[float]

    # Trigger status
    triggered: bool
    triggered_reasons: list[str] = field(default_factory=list)

    # Actionability label
    label: str = "IGNORE"  # "ACTIONABLE" | "MONITOR" | "IGNORE"

    # Context for explanation
    spy_pct_change_1d: float = 0.0
    sector_pct_change_1d: Optional[float] = None
    sector_etf: Optional[str] = None


def compute_zscore(value: float, series: pd.Series) -> float:
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


def compute_triggers(
    ticker: str,
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
    sector_df: Optional[pd.DataFrame],
    thresholds: ThresholdsConfig,
    sector_etf: Optional[str] = None,
) -> Optional[TriggerResult]:
    """Compute all triggers for a ticker.

    Args:
        ticker: The ticker symbol
        df: DataFrame with OHLCV data and 'return' column for the ticker
        spy_df: DataFrame with OHLCV data and 'return' column for SPY
        sector_df: Optional DataFrame for sector ETF, same format
        thresholds: Threshold configuration
        sector_etf: Optional sector ETF symbol for context

    Returns:
        TriggerResult with all computed values, or None if insufficient data.
    """
    if df is None or len(df) < thresholds.vol_lookback_days:
        return None

    if "return" not in df.columns:
        df = df.copy()
        df["return"] = df["close"].pct_change()

    # Get most recent values
    latest_date = df.index[-1]
    r0 = df["return"].iloc[-1]

    if np.isnan(r0):
        return None

    last_close = df["close"].iloc[-1]
    volume_last = df["volume"].iloc[-1]

    # Compute 5-day return
    if len(df) >= 6:
        pct_change_5d = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100
    else:
        pct_change_5d = 0.0

    pct_change_1d = r0 * 100

    # Historical returns for z-score computation (excluding most recent)
    lookback = thresholds.vol_lookback_days
    hist_returns = df["return"].iloc[-(lookback + 1) : -1].dropna()

    # 1) Price move z-score
    price_z = compute_zscore(r0, hist_returns)

    # 2) Volume multiple
    hist_volume = df["volume"].iloc[-lookback:].mean()
    volume_multiple = volume_last / hist_volume if hist_volume > 0 else 0.0
    volume_avg = hist_volume

    # 3) Relative move vs SPY
    spy_r0 = 0.0
    rel_vs_spy_z = 0.0

    if spy_df is not None and len(spy_df) > 0:
        # Align by date
        if latest_date in spy_df.index:
            spy_r0 = spy_df.loc[latest_date, "return"]
            if not np.isnan(spy_r0):
                rel_ret = r0 - spy_r0

                # Compute historical relative returns
                common_dates = df.index.intersection(spy_df.index)
                if len(common_dates) > lookback:
                    stock_rets = df.loc[common_dates, "return"]
                    spy_rets = spy_df.loc[common_dates, "return"]
                    rel_rets = (stock_rets - spy_rets).iloc[-(lookback + 1) : -1].dropna()
                    rel_vs_spy_z = compute_zscore(rel_ret, rel_rets)

    # 4) Relative move vs sector
    rel_vs_sector_z = None
    sector_pct_change_1d = None

    if sector_df is not None and len(sector_df) > 0:
        if latest_date in sector_df.index:
            sector_r0 = sector_df.loc[latest_date, "return"]
            if not np.isnan(sector_r0):
                sector_pct_change_1d = sector_r0 * 100
                rel_ret_sector = r0 - sector_r0

                common_dates = df.index.intersection(sector_df.index)
                if len(common_dates) > lookback:
                    stock_rets = df.loc[common_dates, "return"]
                    sector_rets = sector_df.loc[common_dates, "return"]
                    rel_rets = (stock_rets - sector_rets).iloc[-(lookback + 1) : -1].dropna()
                    rel_vs_sector_z = compute_zscore(rel_ret_sector, rel_rets)

    # Determine if triggered
    triggered_reasons = []

    if abs(price_z) >= thresholds.price_move_z:
        triggered_reasons.append(f"price_z={price_z:.2f}")

    if volume_multiple >= thresholds.volume_multiple:
        triggered_reasons.append(f"vol={volume_multiple:.1f}x")

    if abs(rel_vs_spy_z) >= thresholds.rel_move_vs_index_z:
        triggered_reasons.append(f"rel_vs_spy_z={rel_vs_spy_z:.2f}")

    if rel_vs_sector_z is not None and abs(rel_vs_sector_z) >= thresholds.rel_move_vs_sector_z:
        triggered_reasons.append(f"rel_vs_sector_z={rel_vs_sector_z:.2f}")

    triggered = len(triggered_reasons) > 0

    # Compute actionability label
    label = compute_actionability_label(
        price_z=price_z,
        volume_multiple=volume_multiple,
        rel_vs_spy_z=rel_vs_spy_z,
        rel_vs_sector_z=rel_vs_sector_z,
        triggered=triggered,
    )

    return TriggerResult(
        ticker=ticker,
        last_close=last_close,
        pct_change_1d=pct_change_1d,
        pct_change_5d=pct_change_5d,
        volume_last=volume_last,
        volume_avg=volume_avg,
        volume_multiple=volume_multiple,
        price_z=price_z,
        rel_vs_spy_z=rel_vs_spy_z,
        rel_vs_sector_z=rel_vs_sector_z,
        triggered=triggered,
        triggered_reasons=triggered_reasons,
        label=label,
        spy_pct_change_1d=spy_r0 * 100 if spy_r0 else 0.0,
        sector_pct_change_1d=sector_pct_change_1d,
        sector_etf=sector_etf,
    )


def compute_actionability_label(
    price_z: float,
    volume_multiple: float,
    rel_vs_spy_z: float,
    rel_vs_sector_z: Optional[float],
    triggered: bool,
) -> str:
    """Compute the actionability label based on trigger values.

    Rules (from BuildPlan):
    - ACTIONABLE if (abs(price_z) >= 2.0 AND (abs(rel_vs_spy_z) >= 1.5 OR abs(rel_vs_sector_z) >= 1.5))
                 OR volume_multiple >= 2.0
    - MONITOR if triggered but not actionable
    - IGNORE if not triggered

    This is deterministic - LLM must not override it.
    """
    if not triggered:
        return "IGNORE"

    # Check actionable conditions
    price_condition = abs(price_z) >= 2.0
    rel_spy_condition = abs(rel_vs_spy_z) >= 1.5
    rel_sector_condition = rel_vs_sector_z is not None and abs(rel_vs_sector_z) >= 1.5
    volume_condition = volume_multiple >= 2.0

    if (price_condition and (rel_spy_condition or rel_sector_condition)) or volume_condition:
        return "ACTIONABLE"

    return "MONITOR"
