"""CLI entrypoint for Watchlist Daily Brief."""

import argparse
from dataclasses import replace
import json
from datetime import date, datetime, timedelta, UTC
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
    get_extended_ohlcv,
    clear_cache,
)
from watchbrief.features.attribution import build_attribution_context, recompute_actionability_label
from watchbrief.features.ranking import rank_and_select, rank_and_select_enhanced, EnhancedScore
from watchbrief.features.triggers import compute_triggers
from watchbrief.features.trend_context import compute_trend_context
from watchbrief.llm.client import create_llm_client
from watchbrief.llm.explain import explanation_to_dict, get_explanation_v2
from watchbrief.llm.trend_prompt import get_phase2_context
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

    # Phase 1: Get candidate pool (larger than final selection)
    # We fetch 2x max_items to allow evidence-aware re-ranking
    candidate_pool_size = config.thresholds.max_items * 2
    candidates = rank_and_select(results, candidate_pool_size)

    if not candidates:
        print("No items triggered. No brief to send.")
        return

    print(f"Selected {len(candidates)} candidates for evidence-aware ranking")

    # Create LLM client
    try:
        llm_client = create_llm_client(config.llm)
    except Exception as e:
        print(f"Warning: Could not create LLM client: {e}")
        print("Using fallback explanations only")
        llm_client = None

    # Phase 2: Build attribution context for each candidate
    # This enables evidence-aware scoring (catalysts, drift, etc.)
    print("Building attribution context for candidates...")
    candidate_contexts = []  # List of (result, context, explanation)

    for result, base_score, _ in candidates:
        # Get events with news and SEC filings
        events, news_checked, sec_checked = get_events_with_news(
            result.ticker, start_date, target_date, lookback_days=7
        )

        # Get processed news evidence (clustering, classification, relevance scoring)
        trigger_time = datetime.combine(target_date, datetime.min.time())
        news_evidence = get_processed_news_evidence(
            result.ticker, trigger_time, lookback_hours=72
        )

        # Get sector data for this ticker
        sector_etf = config.sector_map.get(result.ticker)
        sector_df = sector_dfs.get(sector_etf) if sector_etf else None

        # Get peer data for this ticker
        peer_tickers = config.peer_map.get(result.ticker, [])
        peer_data = {}
        for peer in peer_tickers:
            if peer in peer_dfs:
                peer_data[peer] = peer_dfs[peer]
            elif peer in config.watchlist:
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

        # Recompute actionability label now that we have catalyst evidence
        # (the initial label was computed without company catalyst info)
        updated_label = recompute_actionability_label(attribution_context)
        if updated_label != result.label:
            result = replace(result, label=updated_label)
            # Also update the trigger_result in context to stay in sync
            attribution_context = replace(attribution_context, trigger_result=result)

        candidate_contexts.append((result, attribution_context, None))  # explanation=None for now

    # Phase 3: Re-rank using enhanced scoring with evidence
    print("Re-ranking candidates with evidence-aware scoring...")
    ranked_items = rank_and_select_enhanced(
        candidate_contexts,
        max_items=config.thresholds.max_items
    )

    print(f"Final {len(ranked_items)} items after evidence-aware ranking:")
    for item in ranked_items:
        es = item.enhanced_score
        adj_str = ", ".join(es.adjustment_reasons) if es.adjustment_reasons else "none"
        print(f"  {item.rank}. {item.result.ticker}: base={es.base_score:.1f}, adj=[{adj_str}], final={es.final_score:.1f}")

    # Phase 4: Generate explanations for final ranked items
    brief_items = []
    for ranked_item in ranked_items:
        result = ranked_item.result
        attribution_context = ranked_item.context

        print(f"  Generating explanation for {result.ticker}...")

        # Retrieve events and news_evidence from context
        events = attribution_context.events if attribution_context else []
        news_evidence = attribution_context.news_evidence if attribution_context else None

        if events:
            news_count = sum(1 for e in events if e.type == "news")
            sec_count = sum(1 for e in events if e.type == "sec_filing")
            earnings_count = sum(1 for e in events if e.type == "earnings")
            print(f"    Found {len(events)} events ({earnings_count} earnings, {news_count} news, {sec_count} SEC)")

        if news_evidence and news_evidence.checked and news_evidence.top_clusters:
            cluster_count = len(news_evidence.top_clusters)
            catalyst_found = "yes" if not news_evidence.no_company_specific_catalyst_found else "no"
            print(f"    News: {cluster_count} clusters, company catalyst: {catalyst_found}")

        # Get explanation with attribution context
        if llm_client:
            explanation = get_explanation_v2(attribution_context, llm_client)
        else:
            from watchbrief.llm.explain import create_fallback_explanation
            explanation = create_fallback_explanation(result, events, context=attribution_context)

        # Phase 2: Compute trend context (separate from Phase 1 attribution)
        print(f"    Computing trend context...")
        extended_df = get_extended_ohlcv(result.ticker, target_date, lookback_days=260)
        extended_spy_df = get_extended_ohlcv(config.thresholds.index_ticker, target_date, lookback_days=260)

        trend_context = None
        phase2_response = None
        if extended_df is not None and extended_spy_df is not None:
            # Get extended sector data if available
            extended_sector_df = None
            if sector_etf:
                extended_sector_df = get_extended_ohlcv(sector_etf, target_date, lookback_days=260)

            trend_context = compute_trend_context(
                ticker=result.ticker,
                df=extended_df,
                spy_df=extended_spy_df,
                sector_df=extended_sector_df,
            )

            if trend_context:
                print(f"    Market state: [{trend_context.market_state.value.upper()}]")
                # Get Phase 2 LLM response
                phase2_response = get_phase2_context(trend_context, llm_client)
        else:
            print(f"    Trend context: insufficient historical data")

        brief_items.append(BriefItem(
            ticker=result.ticker,
            rank=ranked_item.rank,
            score=ranked_item.enhanced_score.final_score,
            result=result,
            explanation=explanation,
            events=events,  # Include news, SEC filings, earnings
            news_evidence=news_evidence,  # Processed news for transparency
            trend_context=trend_context,  # Phase 2: trend context
            phase2_response=phase2_response,  # Phase 2: LLM response
            enhanced_score=ranked_item.enhanced_score,  # Score breakdown for transparency
        ))

    # Store in database
    with session_scope() as session:
        brief = Brief(
            date_sent=datetime.now(UTC),
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

    # clear-cache command
    cache_parser = subparsers.add_parser("clear-cache", help="Clear the ticker data cache")
    cache_parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )

    # reset-cooldown command
    cooldown_parser = subparsers.add_parser("reset-cooldown", help="Reset cooldown by clearing brief history")
    cooldown_parser.add_argument(
        "--ticker",
        type=str,
        help="Specific ticker to reset (clears all if not specified)",
    )
    cooldown_parser.add_argument(
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

    elif args.command == "clear-cache":
        clear_cache()
        print("Ticker data cache cleared. Failed tickers will be retried on next run.")

    elif args.command == "reset-cooldown":
        from watchbrief.storage.models import Brief, BriefItem as DBBriefItem

        init_db(config.storage.sqlite_path)

        with session_scope() as session:
            if args.ticker:
                # Delete specific ticker's appearances
                count = session.query(DBBriefItem).filter(DBBriefItem.ticker == args.ticker.upper()).delete()
                print(f"Deleted {count} appearance(s) of {args.ticker.upper()} from brief history.")
            else:
                # Delete all briefs (cascades to items)
                count = session.query(Brief).delete()
                print(f"Deleted {count} brief(s) from history. All cooldowns reset.")


if __name__ == "__main__":
    main()
