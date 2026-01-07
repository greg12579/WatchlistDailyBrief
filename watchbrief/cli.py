"""CLI entrypoint for Watchlist Daily Brief."""

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file from current directory (if exists)
load_dotenv()

from watchbrief.config import Config, load_config
from watchbrief.data.events import get_events_with_news, get_processed_news_evidence
from watchbrief.data.market_data import (
    compute_returns,
    get_ohlcv,
    clear_cache,
)
from watchbrief.features.attribution import build_attribution_context
from watchbrief.features.ranking import rank_and_select
from watchbrief.features.triggers import compute_triggers
from watchbrief.llm.client import create_llm_client
from watchbrief.llm.explain import explanation_to_dict, get_explanation_v2
from watchbrief.output.emailer import send_email
from watchbrief.output.renderer import BriefItem, render_email, render_slack
from watchbrief.output.slack import send_slack
from watchbrief.storage.db import init_db, session_scope
from watchbrief.storage.models import Brief, BriefItem as DBBriefItem


def run_daily(config: Config, target_date: date) -> None:
    """Run the daily brief generation.

    Args:
        config: Loaded configuration
        target_date: The date to analyze (usually today)
    """
    print(f"Running daily brief for {target_date}")
    print(f"Watchlist: {len(config.watchlist)} tickers")

    # Initialize database
    init_db(config.storage.sqlite_path)

    # Calculate date range
    lookback = max(config.thresholds.lookback_days, config.thresholds.vol_lookback_days)
    buffer_days = int(lookback * 1.5) + 10
    start_date = target_date - timedelta(days=buffer_days)

    # Fetch SPY data
    print("Fetching SPY data...")
    spy_df = get_ohlcv(config.thresholds.index_ticker, start_date, target_date)
    if spy_df is None:
        print("Error: Could not fetch SPY data")
        return
    spy_df["return"] = compute_returns(spy_df)

    # Cache sector ETF data
    print("Fetching sector ETF data...")
    sector_dfs = {}
    unique_sectors = set(config.sector_map.values())
    for sector_etf in unique_sectors:
        df = get_ohlcv(sector_etf, start_date, target_date)
        if df is not None:
            df["return"] = compute_returns(df)
            sector_dfs[sector_etf] = df

    # Cache peer data
    print("Fetching peer data...")
    peer_dfs = {}
    all_peers = set()
    for peers in config.peer_map.values():
        all_peers.update(peers)
    for peer_ticker in all_peers:
        if peer_ticker not in config.watchlist:  # Avoid duplicate fetches
            df = get_ohlcv(peer_ticker, start_date, target_date)
            if df is not None:
                df["return"] = compute_returns(df)
                peer_dfs[peer_ticker] = df

    # Process each ticker
    print(f"Processing {len(config.watchlist)} tickers...")
    results = []
    skipped = 0

    for i, ticker in enumerate(config.watchlist):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(config.watchlist)}...")

        # Get ticker data
        df = get_ohlcv(ticker, start_date, target_date)
        if df is None or len(df) < config.thresholds.vol_lookback_days:
            skipped += 1
            continue

        df["return"] = compute_returns(df)

        # Get sector data if available
        sector_etf = config.sector_map.get(ticker)
        sector_df = sector_dfs.get(sector_etf) if sector_etf else None

        # Compute triggers
        result = compute_triggers(
            ticker=ticker,
            df=df,
            spy_df=spy_df,
            sector_df=sector_df,
            thresholds=config.thresholds,
            sector_etf=sector_etf,
        )

        if result and result.triggered:
            results.append(result)

    print(f"Found {len(results)} triggered items (skipped {skipped} tickers)")

    # Rank and select top items
    ranked = rank_and_select(results, config.thresholds.max_items)

    if not ranked:
        print("No items triggered. No brief to send.")
        return

    print(f"Top {len(ranked)} items selected")

    # Create LLM client
    try:
        llm_client = create_llm_client(config.llm)
    except Exception as e:
        print(f"Warning: Could not create LLM client: {e}")
        print("Using fallback explanations only")
        llm_client = None

    # Generate explanations and build brief items
    brief_items = []
    for result, score, rank in ranked:
        print(f"  Generating explanation for {result.ticker}...")

        # Get events with news and SEC filings
        events, news_checked, sec_checked = get_events_with_news(
            result.ticker, start_date, target_date, lookback_days=7
        )
        if events:
            news_count = sum(1 for e in events if e.type == "news")
            sec_count = sum(1 for e in events if e.type == "sec_filing")
            earnings_count = sum(1 for e in events if e.type == "earnings")
            print(f"    Found {len(events)} events ({earnings_count} earnings, {news_count} news, {sec_count} SEC)")

        # Get processed news evidence (clustering, classification, relevance scoring)
        trigger_time = datetime.combine(target_date, datetime.min.time())
        news_evidence = get_processed_news_evidence(
            result.ticker, trigger_time, lookback_hours=72
        )
        if news_evidence.checked and news_evidence.top_clusters:
            cluster_count = len(news_evidence.top_clusters)
            catalyst_found = "yes" if not news_evidence.no_company_specific_catalyst_found else "no"
            print(f"    News: {cluster_count} clusters, company catalyst: {catalyst_found}")

        # Get sector data for this ticker
        sector_etf = config.sector_map.get(result.ticker)
        sector_df = sector_dfs.get(sector_etf) if sector_etf else None

        # Get peer data for this ticker
        peer_tickers = config.peer_map.get(result.ticker, [])
        peer_data = {}
        for peer in peer_tickers:
            # Check if peer data was fetched, or if peer is in watchlist
            if peer in peer_dfs:
                peer_data[peer] = peer_dfs[peer]
            elif peer in config.watchlist:
                # Peer is in watchlist, fetch it
                df = get_ohlcv(peer, start_date, target_date)
                if df is not None:
                    df["return"] = compute_returns(df)
                    peer_data[peer] = df

        # Build attribution context with news evidence
        attribution_context = build_attribution_context(
            result=result,
            events=events,
            spy_df=spy_df,
            sector_df=sector_df,
            sector_etf=sector_etf,
            peer_data=peer_data,
            news_checked=news_checked,
            sec_checked=sec_checked,
            news_evidence=news_evidence,
        )

        # Get explanation with attribution context
        if llm_client:
            explanation = get_explanation_v2(attribution_context, llm_client)
        else:
            from watchbrief.llm.explain import create_fallback_explanation
            explanation = create_fallback_explanation(result, events, context=attribution_context)

        brief_items.append(BriefItem(
            ticker=result.ticker,
            rank=rank,
            score=score,
            result=result,
            explanation=explanation,
            events=events,  # Include news, SEC filings, earnings
            news_evidence=news_evidence,  # Processed news for transparency
        ))

    # Store in database
    with session_scope() as session:
        brief = Brief(
            date_sent=datetime.utcnow(),
            subject=f"Watchlist Brief - {target_date}",
            delivery_mode=config.delivery.mode,
        )
        session.add(brief)
        session.flush()  # Get the brief ID

        for item in brief_items:
            db_item = DBBriefItem(
                brief_id=brief.id,
                ticker=item.ticker,
                label=item.result.label,
                rank=item.rank,
                score=item.score,
                facts_json=json.dumps({
                    "last_close": item.result.last_close,
                    "pct_change_1d": item.result.pct_change_1d,
                    "pct_change_5d": item.result.pct_change_5d,
                    "volume_multiple": item.result.volume_multiple,
                    "price_z": item.result.price_z,
                    "rel_vs_spy_z": item.result.rel_vs_spy_z,
                    "rel_vs_sector_z": item.result.rel_vs_sector_z,
                    "triggered_reasons": item.result.triggered_reasons,
                }),
                llm_json=json.dumps(explanation_to_dict(item.explanation)),
            )
            session.add(db_item)

        brief_id = brief.id
        print(f"Stored brief #{brief_id} with {len(brief_items)} items")

    # Render and send
    subject = f"Watchlist Brief - {target_date}"

    if config.delivery.mode == "email":
        subject, html_body, text_body = render_email(
            subject, brief_items, config.feedback.base_url, brief_id
        )
        success = send_email(config.delivery.email, subject, html_body, text_body)
    else:
        payload = render_slack(subject, brief_items, config.feedback.base_url, brief_id)
        success = send_slack(config.delivery.slack, payload)

    if success:
        print("Brief sent successfully!")
    else:
        print("Failed to send brief (check configuration)")

    # Print summary (use ASCII for Windows compatibility)
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for item in brief_items:
        r = item.result
        marker = "[!]" if r.label == "ACTIONABLE" else "[*]"
        print(f"{marker} {r.ticker}: {r.pct_change_1d:+.2f}% | {r.label} | Score: {item.score:.2f}")
    print("=" * 50)

    # Clear cache
    clear_cache()


def serve_feedback(host: str, port: int, config: Config) -> None:
    """Run the feedback server.

    Args:
        host: Host to bind to
        port: Port to listen on
        config: Configuration for database path
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
        import uvicorn
    except ImportError:
        print("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")
        return

    # Initialize database
    init_db(config.storage.sqlite_path)

    app = FastAPI(title="Watchlist Brief Feedback")

    @app.get("/f/{brief_id}/{ticker}/{vote}")
    def record_vote(brief_id: int, ticker: str, vote: str):
        """Record a vote (up/down) for a brief item."""
        if vote not in ("up", "down"):
            return HTMLResponse("<h1>Invalid vote</h1>", status_code=400)

        from watchbrief.storage.feedback import record_vote as db_record_vote

        with session_scope() as session:
            feedback = db_record_vote(session, brief_id, ticker, vote)
            if feedback:
                emoji = "👍" if vote == "up" else "👎"
                return HTMLResponse(
                    f"<h1>Thanks! {emoji}</h1><p>Your feedback for {ticker} has been recorded.</p>"
                )
            else:
                return HTMLResponse("<h1>Item not found</h1>", status_code=404)

    @app.get("/d/{brief_id}/{ticker}/{impact}")
    def record_impact(brief_id: int, ticker: str, impact: str):
        """Record decision impact (yes/no) for a brief item."""
        if impact not in ("yes", "no"):
            return HTMLResponse("<h1>Invalid impact value</h1>", status_code=400)

        from watchbrief.storage.feedback import record_impact as db_record_impact

        with session_scope() as session:
            feedback = db_record_impact(session, brief_id, ticker, impact)
            if feedback:
                emoji = "✅" if impact == "yes" else "❌"
                return HTMLResponse(
                    f"<h1>Thanks! {emoji}</h1><p>Decision impact for {ticker} recorded.</p>"
                )
            else:
                return HTMLResponse("<h1>Item not found</h1>", status_code=404)

    @app.get("/stats")
    def get_stats():
        """Get feedback statistics."""
        from watchbrief.storage.feedback import get_feedback_stats

        with session_scope() as session:
            return get_feedback_stats(session)

    print(f"Starting feedback server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Watchlist Daily Brief - Market change detection and explanation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run-daily command
    daily_parser = subparsers.add_parser("run-daily", help="Run the daily brief")
    daily_parser.add_argument(
        "--date",
        type=str,
        help="Target date (YYYY-MM-DD), defaults to today",
    )
    daily_parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )

    # serve-feedback command
    feedback_parser = subparsers.add_parser("serve-feedback", help="Run the feedback server")
    feedback_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    feedback_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on",
    )
    feedback_parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )

    # init-db command
    init_parser = subparsers.add_parser("init-db", help="Initialize the database")
    init_parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    if args.command == "run-daily":
        if args.date:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        else:
            target_date = date.today()
        run_daily(config, target_date)

    elif args.command == "serve-feedback":
        serve_feedback(args.host, args.port, config)

    elif args.command == "init-db":
        init_db(config.storage.sqlite_path)
        print(f"Database initialized at {config.storage.sqlite_path}")


if __name__ == "__main__":
    main()
