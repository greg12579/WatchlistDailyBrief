"""Event data retrieval (earnings, analyst actions)."""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import yfinance as yf


@dataclass
class Event:
    """Represents a company event."""

    type: str  # "earnings" | "analyst" | "other"
    date: date
    title: str
    source: str


def get_events(
    ticker: str,
    start_date: date,
    end_date: date,
) -> list[Event]:
    """Get events for a ticker within a date range.

    Currently only implements earnings dates from yfinance.
    Returns empty list if data is unavailable.

    Args:
        ticker: yfinance-compatible ticker symbol
        start_date: Start of date range
        end_date: End of date range

    Returns:
        List of Event objects
    """
    events = []

    try:
        yf_ticker = yf.Ticker(ticker)

        # Try to get earnings dates
        try:
            calendar = yf_ticker.calendar
            if calendar is not None and not calendar.empty:
                # calendar can be a DataFrame or dict depending on yfinance version
                if hasattr(calendar, "to_dict"):
                    cal_dict = calendar.to_dict()
                else:
                    cal_dict = calendar

                # Look for earnings date
                earnings_date = None
                if "Earnings Date" in cal_dict:
                    earnings_date = cal_dict["Earnings Date"]
                elif "earnings" in str(cal_dict).lower():
                    # Try to find earnings in the structure
                    for key, value in cal_dict.items():
                        if "earn" in str(key).lower() and "date" in str(key).lower():
                            earnings_date = value
                            break

                if earnings_date:
                    # Handle various date formats
                    if hasattr(earnings_date, "date"):
                        ed = earnings_date.date()
                    elif isinstance(earnings_date, str):
                        from datetime import datetime

                        ed = datetime.strptime(earnings_date, "%Y-%m-%d").date()
                    elif isinstance(earnings_date, date):
                        ed = earnings_date
                    else:
                        ed = None

                    if ed and start_date <= ed <= end_date:
                        events.append(
                            Event(
                                type="earnings",
                                date=ed,
                                title=f"{ticker} earnings release",
                                source="yfinance",
                            )
                        )
        except Exception:
            pass  # Calendar data not available

        # Try to get recent earnings history
        try:
            earnings_hist = yf_ticker.earnings_dates
            if earnings_hist is not None and not earnings_hist.empty:
                for idx in earnings_hist.index:
                    if hasattr(idx, "date"):
                        ed = idx.date()
                    else:
                        ed = idx

                    if isinstance(ed, date) and start_date <= ed <= end_date:
                        # Check if we already have this date
                        if not any(e.date == ed and e.type == "earnings" for e in events):
                            events.append(
                                Event(
                                    type="earnings",
                                    date=ed,
                                    title=f"{ticker} earnings reported",
                                    source="yfinance",
                                )
                            )
        except Exception:
            pass  # Earnings history not available

    except Exception as e:
        print(f"Error fetching events for {ticker}: {e}")

    return events


def get_upcoming_earnings(ticker: str) -> Optional[date]:
    """Get the next upcoming earnings date for a ticker.

    Returns None if not available.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        calendar = yf_ticker.calendar

        if calendar is not None:
            if hasattr(calendar, "get"):
                earnings_date = calendar.get("Earnings Date")
            elif hasattr(calendar, "to_dict"):
                cal_dict = calendar.to_dict()
                earnings_date = cal_dict.get("Earnings Date")
            else:
                return None

            if earnings_date:
                if hasattr(earnings_date, "date"):
                    return earnings_date.date()
                elif isinstance(earnings_date, date):
                    return earnings_date

    except Exception:
        pass

    return None
