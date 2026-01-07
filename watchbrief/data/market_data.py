"""Market data retrieval using yfinance."""

from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Optional

import pandas as pd
import yfinance as yf


class MarketDataError(Exception):
    """Raised when market data cannot be retrieved."""

    pass


def get_ohlcv(
    ticker: str,
    start_date: date,
    end_date: date,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data for a ticker from yfinance.

    Args:
        ticker: yfinance-compatible ticker symbol
        start_date: Start date for data range
        end_date: End date for data range (inclusive)

    Returns:
        DataFrame with columns: open, high, low, close, volume
        Index is date. Returns None if data cannot be retrieved.
    """
    try:
        yf_ticker = yf.Ticker(ticker)

        # yfinance end date is exclusive, so add 1 day
        df = yf_ticker.history(
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=True,
        )

        if df.empty:
            return None

        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Keep only OHLCV columns
        df = df[["open", "high", "low", "close", "volume"]].copy()

        # Convert index to date only (remove time component)
        df.index = pd.to_datetime(df.index).date
        df.index.name = "date"

        return df

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def compute_returns(df: pd.DataFrame) -> pd.Series:
    """Compute daily returns from close prices.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series of daily returns (close-to-close percentage change)
    """
    return df["close"].pct_change()


def get_market_data_for_analysis(
    ticker: str,
    reference_date: date,
    lookback_days: int = 30,
) -> Optional[pd.DataFrame]:
    """Get market data with computed returns for trigger analysis.

    Args:
        ticker: yfinance-compatible ticker symbol
        reference_date: The date to analyze (most recent date in range)
        lookback_days: Number of trading days to look back

    Returns:
        DataFrame with OHLCV data and 'return' column, or None if unavailable.
    """
    # Add buffer for weekends/holidays
    buffer_days = int(lookback_days * 1.5) + 10
    start_date = reference_date - timedelta(days=buffer_days)

    df = get_ohlcv(ticker, start_date, reference_date)

    if df is None or len(df) < 5:
        return None

    # Compute returns
    df["return"] = compute_returns(df)

    # Filter to only include data up to reference_date
    df = df[df.index <= reference_date]

    return df


class MarketDataCache:
    """Simple in-memory cache for market data to avoid repeated API calls."""

    def __init__(self):
        self._cache: dict[tuple[str, date, date], Optional[pd.DataFrame]] = {}

    def get(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.DataFrame]:
        """Get cached data or fetch if not cached."""
        key = (ticker, start_date, end_date)

        if key not in self._cache:
            self._cache[key] = get_ohlcv(ticker, start_date, end_date)

        return self._cache[key]

    def clear(self):
        """Clear the cache."""
        self._cache.clear()


# Global cache instance
_cache = MarketDataCache()


def get_ohlcv_cached(
    ticker: str,
    start_date: date,
    end_date: date,
) -> Optional[pd.DataFrame]:
    """Cached version of get_ohlcv."""
    return _cache.get(ticker, start_date, end_date)


def clear_cache():
    """Clear the global market data cache."""
    _cache.clear()
