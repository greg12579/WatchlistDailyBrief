"""CLI v2.5 entrypoint - adds scanner sections to daily brief.

This is a separate entry point that extends the main CLI with Phase 2.5 features:
- Daily Tape vs IWM (Top 10: 5 underperformers + 5 outperformers)
- Near Highs (Watch): Top 5 closest to 52w high
- Trading Like Shit (Watch): Top 5 closest to 52w low with weak YoY performance

Once validated, these features can be merged into the main cli.py.
"""

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
from watchbrief.features.scanners import (
    build_tape_rows,
    compute_daily_tape,
    compute_near_highs,
    compute_broken_lows,
)
from watchbrief.llm.client import create_llm_client
from watchbrief.llm.explain import explanation_to_dict, get_explanation_v2
from watchbrief.llm.trend_prompt import get_phase2_context
from watchbrief.output.emailer import send_email
from watchbrief.output.renderer import BriefItem, render_email, render_slack
from watchbrief.output.slack import send_slack
from watchbrief.storage.db import init_db, session_scope
from watchbrief.storage.models import Brief, BriefItem as DBBriefItem


def run_daily_v25(config: Config, target_date: date) -> None:
    """Run the daily brief generation with v2.5 scanner sections.

    Args:
        config: Loaded configuration
        target_date: The date to analyze (usually today)
    """
    print(f"Running daily brief v2.5 for {target_date}")
    print(f"Watchlist: {len(config.watchlist)} tickers")

    # Initialize database
    init_db(config.storage.sqlite_path)

    # Calculate date range - need 260+ days for 52-week analysis
    lookback = max(config.thresholds.lookback_days, config.thresholds.vol_lookback_days, 260)
    buffer_days = int(lookback * 1.5) + 10
    start_date = target_date - timedelta(days=buffer_days)

    # Fetch benchmark data (SPY + IWM)
    print("Fetching benchmark data (SPY + IWM)...")
    spy_df = get_ohlcv(config.thresholds.index_ticker, start_date, target_date)
    if spy_df is None:
        print("Error: Could not fetch SPY data")
        return
    spy_df["return"] = compute_returns(spy_df)

    # Fetch IWM for scanner benchmarking
    iwm_df = get_ohlcv(config.thresholds.benchmark_ticker, start_date, target_date)
    if iwm_df is not None:
        iwm_df["return"] = compute_returns(iwm_df)
        print(f"  IWM data: {len(iwm_df)} days")
    else:
        print("Warning: Could not fetch IWM data, scanners will be limited")

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

    # Process each ticker - also cache data for scanners
    print(f"Processing {len(config.watchlist)} tickers...")
    results = []
    skipped = 0
    ticker_dfs = {}  # Cache all ticker dataframes for scanners

    for i, ticker in enumerate(config.watchlist):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(config.watchlist)}...")

        # Get ticker data
        df = get_ohlcv(ticker, start_date, target_date)
        if df is None or len(df) < config.thresholds.vol_lookback_days:
            skipped += 1
            continue

        df["return"] = compute_returns(df)
        ticker_dfs[ticker] = df  # Cache for scanner use

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

    # =========================================================================
    # Phase 2.5: Build scanner data for ALL tickers
    # =========================================================================
    print("\nBuilding scanner data for all tickers...")

    # Compute trend context for all tickers (needed for scanner state labels)
    trend_context_cache = {}
    for ticker, df in ticker_dfs.items():
        # Get extended data for 52-week analysis
        extended_df = get_extended_ohlcv(ticker, target_date, lookback_days=260)
        extended_spy_df = get_extended_ohlcv(config.thresholds.index_ticker, target_date, lookback_days=260)

        if extended_df is not None and extended_spy_df is not None and len(extended_df) >= 60:
            # Get extended sector data if available
            sector_etf = config.sector_map.get(ticker)
            extended_sector_df = None
            if sector_etf:
                extended_sector_df = get_extended_ohlcv(sector_etf, target_date, lookback_days=260)

            tc = compute_trend_context(
                ticker=ticker,
                df=extended_df,
                spy_df=extended_spy_df,
                sector_df=extended_sector_df,
            )
            if tc:
                trend_context_cache[ticker] = tc

    print(f"  Computed trend context for {len(trend_context_cache)} tickers")

    # Build TapeRows for all tickers
    tape_rows = build_tape_rows(
        watchlist=config.watchlist,
        ticker_dfs=ticker_dfs,
        iwm_df=iwm_df,
        trend_contexts=trend_context_cache,
    )
    print(f"  Built TapeRows for {len(tape_rows)} tickers")

    # Run scanners
    scan_results = {}

    if "tape" in config.thresholds.scan_sections_enabled:
        tape_movers = compute_daily_tape(
            tape_rows, top_n=config.thresholds.tape_top_n
        )
        scan_results["tape"] = tape_movers
        print(f"  Daily Tape: {len(tape_movers)} top movers")

    if "near_highs" in config.thresholds.scan_sections_enabled:
        near_highs = compute_near_highs(
            tape_rows,
            threshold_pct=-config.thresholds.near_high_threshold_pct,
            top_n=5,
        )
        scan_results["near_highs"] = near_highs
        print(f"  Near Highs: {len(near_highs)} tickers")

    if "broken" in config.thresholds.scan_sections_enabled:
        broken = compute_broken_lows(
            tape_rows,
            near_low_threshold_pct=config.thresholds.near_low_threshold_pct,
            broken_return_threshold=config.thresholds.broken_252d_return_threshold,
            broken_rel_iwm_threshold=config.thresholds.broken_rel_iwm_252d_threshold,
            top_n=5,
        )
        scan_results["broken"] = broken
        print(f"  Trading Like Shit: {len(broken)} tickers")

    # =========================================================================
    # Continue with standard pipeline
    # =========================================================================

    # Phase 1: Get candidate pool (larger than final selection)
    candidate_pool_size = config.thresholds.max_items * 2
    candidates = rank_and_select(results, candidate_pool_size)

    if not candidates:
        print("No items triggered. Sending scanner-only brief...")
        # Even with no triggered items, we can still send scan results
        if scan_results:
            _send_scanner_only_brief(config, target_date, scan_results)
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
    print("Building attribution context for candidates...")
    candidate_contexts = []

    for result, base_score, _ in candidates:
        # Get events with news and SEC filings
        events, news_checked, sec_checked = get_events_with_news(
            result.ticker, start_date, target_date, lookback_days=7
        )

        # Get processed news evidence
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

        # Recompute actionability label
        updated_label = recompute_actionability_label(attribution_context)
        if updated_label != result.label:
            result = replace(result, label=updated_label)
            attribution_context = replace(attribution_context, trigger_result=result)

        candidate_contexts.append((result, attribution_context, None))

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

        # Get trend context from cache (computed earlier for scanners)
        trend_context = trend_context_cache.get(result.ticker)
        phase2_response = None

        if trend_context:
            print(f"    Market state: [{trend_context.market_state.value.upper()}]")
            phase2_response = get_phase2_context(trend_context, llm_client)
        else:
            print(f"    Trend context: not available")

        brief_items.append(BriefItem(
            ticker=result.ticker,
            rank=ranked_item.rank,
            score=ranked_item.enhanced_score.final_score,
            result=result,
            explanation=explanation,
            events=events,
            news_evidence=news_evidence,
            trend_context=trend_context,
            phase2_response=phase2_response,
            enhanced_score=ranked_item.enhanced_score,
        ))

    # Store in database
    with session_scope() as session:
        brief = Brief(
            date_sent=datetime.now(UTC),
            subject=f"Watchlist Brief v2.5 - {target_date}",
            delivery_mode=config.delivery.mode,
        )
        session.add(brief)
        session.flush()

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

    # Render and send with scan results
    subject = f"Watchlist Brief v2.5 - {target_date}"

    if config.delivery.mode == "email":
        subject, html_body, text_body = render_email(
            subject, brief_items, config.feedback.base_url, brief_id,
            scan_results=scan_results,  # Pass scanner results to renderer
        )
        success = send_email(config.delivery.email, subject, html_body, text_body)
    else:
        # Slack doesn't support scan results yet
        payload = render_slack(subject, brief_items, config.feedback.base_url, brief_id)
        success = send_slack(config.delivery.slack, payload)

    if success:
        print("Brief sent successfully!")
    else:
        print("Failed to send brief (check configuration)")

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    # Print scan summary first
    if "tape" in scan_results:
        print("\nDaily Tape - Top Movers:")
        for row in scan_results["tape"][:5]:
            print(f"  {row.ticker}: {row.ret_1d:+.1f}% (vs IWM: {row.rel_vs_iwm_1d:+.1f}%)")

    print("\nTriggered Items:")
    for item in brief_items:
        r = item.result
        marker = "[!]" if r.label == "ACTIONABLE" else "[*]"
        print(f"{marker} {r.ticker}: {r.pct_change_1d:+.2f}% | {r.label} | Score: {item.score:.2f}")
    print("=" * 50)

    # Clear cache
    clear_cache()


def _send_scanner_only_brief(config: Config, target_date: date, scan_results: dict) -> None:
    """Send a brief with only scanner sections (no triggered items)."""
    with session_scope() as session:
        brief = Brief(
            date_sent=datetime.now(UTC),
            subject=f"Watchlist Scan v2.5 - {target_date}",
            delivery_mode=config.delivery.mode,
        )
        session.add(brief)
        session.flush()
        brief_id = brief.id

    subject = f"Watchlist Scan v2.5 - {target_date} (No Triggers)"

    if config.delivery.mode == "email":
        subject, html_body, text_body = render_email(
            subject, [], config.feedback.base_url, brief_id,
            scan_results=scan_results,
        )
        success = send_email(config.delivery.email, subject, html_body, text_body)
    else:
        print("Slack scanner-only briefs not supported yet")
        success = False

    if success:
        print("Scanner-only brief sent successfully!")
    else:
        print("Failed to send scanner-only brief")


def main():
    """Main CLI v2.5 entrypoint."""
    parser = argparse.ArgumentParser(
        description="Watchlist Daily Brief v2.5 - with scanner sections"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run-daily command
    daily_parser = subparsers.add_parser("run-daily", help="Run the daily brief with scanners")
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

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    if args.command == "run-daily":
        if args.date:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        else:
            target_date = date.today()
        run_daily_v25(config, target_date)


if __name__ == "__main__":
    main()
