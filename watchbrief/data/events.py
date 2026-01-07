"""Event data retrieval (earnings, news, SEC filings)."""

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import requests
import yfinance as yf


@dataclass
class Event:
    """Represents a company event."""

    type: str  # "earnings" | "news" | "sec_filing" | "analyst" | "other"
    date: date
    title: str
    source: str
    url: Optional[str] = None  # Link to the news article or filing


@dataclass
class NewsItem:
    """A news article about a company."""

    title: str
    publisher: str
    link: str
    published_date: datetime
    type: str = "news"  # Could be "news" or "sec_filing"


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


def get_news(
    ticker: str,
    lookback_days: int = 7,
    company_name: Optional[str] = None,
) -> tuple[list[NewsItem], bool]:
    """Fetch recent news for a ticker using available news APIs.

    Tries providers in order:
    1. Polygon.io (if POLYGON_API_KEY set) - Excellent stock-specific news
    2. Finnhub (if FINNHUB_API_KEY set) - Good stock-specific news
    3. NewsAPI (if NEWSAPI_KEY set) - General news coverage

    Args:
        ticker: Stock ticker symbol
        lookback_days: How many days back to look for news
        company_name: Optional company name for better search

    Returns:
        Tuple of (list of NewsItem objects, success boolean)
        success=True means we successfully checked (even if no news found)
        success=False means no API key configured or error
    """
    # Try Polygon first (best for stock-specific news)
    polygon_key = os.getenv("POLYGON_API_KEY")
    if polygon_key:
        news, success = _get_polygon_news(ticker, lookback_days, polygon_key)
        if success:
            return news, True

    # Try Finnhub
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if finnhub_key:
        news, success = _get_finnhub_news(ticker, lookback_days, finnhub_key)
        if success:
            return news, True

    # Try NewsAPI
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if newsapi_key:
        search_term = company_name or ticker
        news, success = _get_newsapi_news(search_term, lookback_days, newsapi_key)
        if success:
            return news, True

    # No API keys configured
    return [], False


def _get_polygon_news(
    ticker: str,
    lookback_days: int,
    api_key: str,
) -> tuple[list[NewsItem], bool]:
    """Fetch news from Polygon.io API.

    Polygon provides high-quality stock-specific news.
    See: https://polygon.io/docs/stocks/get_v2_reference_news
    """
    news_items = []
    cutoff = datetime.now() - timedelta(days=lookback_days)

    # Format dates for Polygon API (YYYY-MM-DD)
    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    try:
        url = "https://api.polygon.io/v2/reference/news"
        params = {
            "ticker": ticker,
            "published_utc.gte": from_date,
            "order": "desc",
            "limit": 10,
            "apiKey": api_key,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "OK" and "results" in data:
            for article in data["results"]:
                # Polygon returns ISO timestamp in 'published_utc'
                pub_date_str = article.get("published_utc")
                if pub_date_str:
                    try:
                        # Parse ISO format: "2026-01-07T17:57:34Z"
                        pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                        pub_date = pub_date.replace(tzinfo=None)
                    except ValueError:
                        continue

                    if pub_date >= cutoff:
                        title = article.get("title", "")
                        publisher = article.get("publisher", {}).get("name", "Unknown")
                        link = article.get("article_url", "")

                        is_sec = _is_sec_filing(title, publisher)

                        news_items.append(NewsItem(
                            title=title,
                            publisher=publisher,
                            link=link,
                            published_date=pub_date,
                            type="sec_filing" if is_sec else "news",
                        ))

        return news_items, True

    except Exception as e:
        print(f"Polygon error for {ticker}: {e}")
        return [], False


def _get_finnhub_news(
    ticker: str,
    lookback_days: int,
    api_key: str,
) -> tuple[list[NewsItem], bool]:
    """Fetch news from Finnhub API.

    Finnhub provides company-specific news which is ideal for stock analysis.
    Free tier: 60 calls/minute.
    """
    news_items = []
    cutoff = datetime.now() - timedelta(days=lookback_days)
    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    try:
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": from_date,
            "to": to_date,
            "token": api_key,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            for article in data[:10]:  # Limit to 10 articles
                # Finnhub returns Unix timestamp in 'datetime' field
                timestamp = article.get("datetime")
                if timestamp:
                    pub_date = datetime.fromtimestamp(timestamp)
                    if pub_date >= cutoff:
                        title = article.get("headline", "")
                        publisher = article.get("source", "Unknown")
                        link = article.get("url", "")

                        is_sec = _is_sec_filing(title, publisher)

                        news_items.append(NewsItem(
                            title=title,
                            publisher=publisher,
                            link=link,
                            published_date=pub_date,
                            type="sec_filing" if is_sec else "news",
                        ))

        return news_items, True

    except Exception as e:
        print(f"Finnhub error for {ticker}: {e}")
        return [], False


def _get_newsapi_news(
    search_term: str,
    lookback_days: int,
    api_key: str,
) -> tuple[list[NewsItem], bool]:
    """Fetch news from NewsAPI.

    NewsAPI provides general news search. Free tier: 100 requests/day.
    """
    news_items = []
    cutoff = datetime.now() - timedelta(days=lookback_days)
    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f'"{search_term}" stock OR shares OR market',
            "from": from_date,
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 10,
            "apiKey": api_key,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "ok":
            for article in data.get("articles", []):
                pub_date_str = article.get("publishedAt")
                if pub_date_str:
                    try:
                        pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                        pub_date = pub_date.replace(tzinfo=None)
                    except ValueError:
                        continue

                    if pub_date >= cutoff:
                        title = article.get("title", "")
                        publisher = article.get("source", {}).get("name", "Unknown")
                        link = article.get("url", "")

                        is_sec = _is_sec_filing(title, publisher)

                        news_items.append(NewsItem(
                            title=title,
                            publisher=publisher,
                            link=link,
                            published_date=pub_date,
                            type="sec_filing" if is_sec else "news",
                        ))

        return news_items, True

    except Exception as e:
        print(f"NewsAPI error for {search_term}: {e}")
        return [], False


def _is_sec_filing(title: str, publisher: str) -> bool:
    """Detect if a news item is likely an SEC filing.

    Args:
        title: News headline
        publisher: Publisher name

    Returns:
        True if this appears to be an SEC filing
    """
    title_lower = title.lower()
    publisher_lower = publisher.lower()

    # SEC filing indicators
    sec_keywords = [
        "10-k", "10-q", "8-k", "6-k",
        "form 4", "form 3", "form 13",
        "sec filing", "sec form",
        "proxy statement", "def 14a",
        "s-1", "s-3", "s-4",
        "quarterly report", "annual report",
    ]

    sec_publishers = [
        "sec", "edgar", "securities and exchange",
    ]

    for keyword in sec_keywords:
        if keyword in title_lower:
            return True

    for pub in sec_publishers:
        if pub in publisher_lower:
            return True

    return False


def news_to_events(
    news_items: list[NewsItem],
    start_date: date,
    end_date: date,
) -> list[Event]:
    """Convert NewsItem objects to Event objects for attribution.

    Args:
        news_items: List of NewsItem objects
        start_date: Start of date range
        end_date: End of date range

    Returns:
        List of Event objects
    """
    events = []

    for item in news_items:
        item_date = item.published_date.date()

        if start_date <= item_date <= end_date:
            events.append(Event(
                type=item.type,
                date=item_date,
                title=item.title,
                source=item.publisher,
                url=item.link,
            ))

    return events


def get_sec_filings(
    ticker: str,
    lookback_days: int = 7,
) -> tuple[list[Event], bool]:
    """Fetch recent SEC filings for a ticker from SEC EDGAR.

    Uses the SEC EDGAR company filings RSS feed.

    Args:
        ticker: Stock ticker symbol
        lookback_days: How many days back to look

    Returns:
        Tuple of (list of Event objects, success boolean)
    """
    events = []
    cutoff = datetime.now() - timedelta(days=lookback_days)

    # First, we need to get the CIK for this ticker
    # SEC provides a ticker-to-CIK mapping
    try:
        # Get CIK from SEC ticker mapping
        cik = _get_cik_for_ticker(ticker)
        if not cik:
            return [], True  # No CIK found, but not an error

        # Fetch recent filings from EDGAR
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        headers = {
            "User-Agent": "WatchlistDailyBrief/1.0 (contact@example.com)",
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Parse recent filings
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        filing_dates = filings.get("filingDate", [])
        descriptions = filings.get("primaryDocument", [])
        accession_numbers = filings.get("accessionNumber", [])

        # Important form types to track
        important_forms = {
            "10-K": "Annual Report",
            "10-Q": "Quarterly Report",
            "8-K": "Current Report",
            "4": "Insider Trading",
            "13F": "Institutional Holdings",
            "S-1": "IPO Registration",
            "DEF 14A": "Proxy Statement",
            "6-K": "Foreign Issuer Report",
        }

        for i, form in enumerate(forms[:20]):  # Check last 20 filings
            if form in important_forms:
                try:
                    filing_date = datetime.strptime(filing_dates[i], "%Y-%m-%d")
                    if filing_date >= cutoff:
                        accession = accession_numbers[i].replace("-", "")
                        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{descriptions[i]}"

                        events.append(Event(
                            type="sec_filing",
                            date=filing_date.date(),
                            title=f"{form}: {important_forms[form]}",
                            source="SEC EDGAR",
                            url=filing_url,
                        ))
                except (ValueError, IndexError):
                    continue

        return events, True

    except Exception as e:
        print(f"SEC EDGAR error for {ticker}: {e}")
        return [], False


def _get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Get SEC CIK number for a ticker symbol.

    Returns CIK as zero-padded 10-digit string, or None if not found.
    """
    try:
        # SEC provides a ticker-to-CIK mapping file
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {
            "User-Agent": "WatchlistDailyBrief/1.0 (contact@example.com)",
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Search for ticker (case-insensitive)
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = entry.get("cik_str")
                # Zero-pad to 10 digits
                return str(cik).zfill(10)

        return None

    except Exception:
        return None


def get_events_with_news(
    ticker: str,
    start_date: date,
    end_date: date,
    lookback_days: int = 7,
) -> tuple[list[Event], bool, bool]:
    """Get all events including news and SEC filings for a ticker.

    Combines earnings events, news articles, and SEC filings.

    Args:
        ticker: yfinance-compatible ticker symbol
        start_date: Start of date range
        end_date: End of date range
        lookback_days: How many days back to look for news

    Returns:
        Tuple of (events list, news_checked bool, sec_checked bool)
    """
    # Get existing earnings events
    events = get_events(ticker, start_date, end_date)

    # Fetch news
    news_items, news_success = get_news(ticker, lookback_days)

    # Convert news to events and add
    if news_success and news_items:
        news_events = news_to_events(news_items, start_date, end_date)
        events.extend(news_events)

    # Fetch SEC filings directly from EDGAR
    sec_events, sec_success = get_sec_filings(ticker, lookback_days)
    if sec_success and sec_events:
        # Filter to date range and add (avoid duplicates)
        existing_titles = {e.title for e in events}
        for sec_event in sec_events:
            if start_date <= sec_event.date <= end_date:
                if sec_event.title not in existing_titles:
                    events.append(sec_event)

    # Sort by date descending (most recent first)
    events.sort(key=lambda e: e.date, reverse=True)

    return events, news_success, sec_success


def get_processed_news_evidence(
    ticker: str,
    trigger_time: datetime,
    lookback_hours: int = 72,
) -> "NewsEvidence":
    """Get processed news evidence using the news processing pipeline.

    This implements the news_processing_spec_for_claude.md specification:
    - Clusters and de-duplicates headlines
    - Classifies event types
    - Scores relevance
    - Returns structured evidence for attribution

    Args:
        ticker: Stock ticker symbol
        trigger_time: When the alert triggered
        lookback_hours: How far back to look for news

    Returns:
        NewsEvidence object with processed clusters
    """
    from watchbrief.data.news_processor import (
        NewsEvidence,
        process_news_for_attribution,
    )

    lookback_days = lookback_hours // 24 + 1

    # Fetch raw news
    news_items, news_success = get_news(ticker, lookback_days)

    if not news_success:
        return NewsEvidence(
            checked=False,
            lookback_hours=lookback_hours,
            top_clusters=[],
            no_company_specific_catalyst_found=True,
        )

    # Convert to format expected by processor
    headlines = []
    for item in news_items:
        headlines.append({
            "headline": item.title,
            "source": item.publisher,
            "published_at": item.published_date,
            "url": item.link,
        })

    # Process through the pipeline
    return process_news_for_attribution(
        headlines=headlines,
        trigger_time=trigger_time,
        lookback_hours=lookback_hours,
    )
