"""Attribution engine for PM-grade 'Why' explanations.

This module gathers evidence and computes attribution hints BEFORE
the LLM is called, ensuring explanations are grounded in facts.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Optional

import pandas as pd

from watchbrief.data.events import Event
from watchbrief.features.triggers import TriggerResult

if TYPE_CHECKING:
    from watchbrief.data.news_processor import NewsEvidence


class AttributionCategory(str, Enum):
    """Categories for price move attribution."""

    COMPANY = "Company"  # Company-specific (earnings, news, filings)
    SECTOR_PEER = "Sector/Peer"  # Sector or peer group movement
    MACRO = "Macro"  # Broad market movement
    FLOW = "Flow"  # Technical/flow-driven (no fundamental reason)
    UNATTRIBUTED = "Unattributed"  # Cannot determine


class CatalystCheckStatus(str, Enum):
    """Status of catalyst data sources."""

    CHECKED = "checked"
    NOT_AVAILABLE = "not_available"
    ERROR = "error"


@dataclass
class PeerContext:
    """Performance context for a peer stock."""

    ticker: str
    pct_change_1d: float
    pct_change_5d: float


@dataclass
class SectorContext:
    """Performance context for sector ETF."""

    etf: str
    pct_change_1d: float
    pct_change_5d: float


@dataclass
class MacroContext:
    """Broad market context."""

    spy_pct_change_1d: float
    spy_pct_change_5d: float


@dataclass
class CatalystChecks:
    """Status of various catalyst data sources."""

    earnings_calendar: CatalystCheckStatus = CatalystCheckStatus.NOT_AVAILABLE
    news_feed: CatalystCheckStatus = CatalystCheckStatus.NOT_AVAILABLE
    sec_filings: CatalystCheckStatus = CatalystCheckStatus.NOT_AVAILABLE

    def to_dict(self) -> dict:
        return {
            "earnings_calendar": self.earnings_calendar.value,
            "news_feed": self.news_feed.value,
            "sec_filings": self.sec_filings.value,
        }

    def missing_checks(self) -> list[str]:
        """Return list of data sources not available."""
        missing = []
        if self.news_feed != CatalystCheckStatus.CHECKED:
            missing.append("news_feed")
        if self.sec_filings != CatalystCheckStatus.CHECKED:
            missing.append("sec_filings")
        return missing


@dataclass
class AttributionHint:
    """A hint about likely attribution with supporting evidence."""

    category: AttributionCategory
    evidence: list[str] = field(default_factory=list)
    strength: float = 0.0  # 0-1, higher = stronger evidence


@dataclass
class AttributionContext:
    """Full context for attribution, passed to LLM."""

    # Stock being analyzed
    ticker: str
    trigger_result: TriggerResult

    # Context data
    sector: Optional[SectorContext] = None
    peers: list[PeerContext] = field(default_factory=list)
    macro: Optional[MacroContext] = None
    events: list[Event] = field(default_factory=list)

    # What we checked
    catalyst_checks: CatalystChecks = field(default_factory=CatalystChecks)

    # Pre-computed hints for LLM
    attribution_hints: list[AttributionHint] = field(default_factory=list)

    # Processed news evidence (from news_processor)
    news_evidence: Optional["NewsEvidence"] = None


def compute_sector_context(
    sector_df: Optional[pd.DataFrame],
    sector_etf: Optional[str],
) -> Optional[SectorContext]:
    """Compute sector ETF performance context."""
    if sector_df is None or sector_etf is None or len(sector_df) < 2:
        return None

    if "return" not in sector_df.columns:
        sector_df = sector_df.copy()
        sector_df["return"] = sector_df["close"].pct_change()

    pct_1d = sector_df["return"].iloc[-1] * 100 if len(sector_df) >= 1 else 0.0

    # 5-day return
    if len(sector_df) >= 6:
        pct_5d = (sector_df["close"].iloc[-1] / sector_df["close"].iloc[-6] - 1) * 100
    else:
        pct_5d = 0.0

    return SectorContext(
        etf=sector_etf,
        pct_change_1d=pct_1d,
        pct_change_5d=pct_5d,
    )


def compute_macro_context(spy_df: Optional[pd.DataFrame]) -> Optional[MacroContext]:
    """Compute broad market context from SPY."""
    if spy_df is None or len(spy_df) < 2:
        return None

    if "return" not in spy_df.columns:
        spy_df = spy_df.copy()
        spy_df["return"] = spy_df["close"].pct_change()

    pct_1d = spy_df["return"].iloc[-1] * 100 if len(spy_df) >= 1 else 0.0

    # 5-day return
    if len(spy_df) >= 6:
        pct_5d = (spy_df["close"].iloc[-1] / spy_df["close"].iloc[-6] - 1) * 100
    else:
        pct_5d = 0.0

    return MacroContext(
        spy_pct_change_1d=pct_1d,
        spy_pct_change_5d=pct_5d,
    )


def compute_peer_context(
    peer_data: dict[str, pd.DataFrame],
) -> list[PeerContext]:
    """Compute performance context for peer stocks.

    Args:
        peer_data: Dict mapping peer ticker to their OHLCV DataFrame

    Returns:
        List of PeerContext objects
    """
    peers = []

    for ticker, df in peer_data.items():
        if df is None or len(df) < 2:
            continue

        if "return" not in df.columns:
            df = df.copy()
            df["return"] = df["close"].pct_change()

        pct_1d = df["return"].iloc[-1] * 100

        # 5-day return
        if len(df) >= 6:
            pct_5d = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100
        else:
            pct_5d = 0.0

        peers.append(PeerContext(
            ticker=ticker,
            pct_change_1d=pct_1d,
            pct_change_5d=pct_5d,
        ))

    return peers


def compute_attribution_hints(
    result: TriggerResult,
    sector: Optional[SectorContext],
    peers: list[PeerContext],
    macro: Optional[MacroContext],
    events: list[Event],
    news_evidence: Optional["NewsEvidence"] = None,
) -> list[AttributionHint]:
    """Compute attribution hints based on available evidence.

    Priority order (from build plan):
    1. Company-specific (from processed news_evidence or raw events)
    2. Sector/Peer correlation
    3. Macro (broad market)
    4. Flow/technical
    5. Unattributed

    When news_evidence is provided, uses the processed clusters for
    more accurate attribution per news_processing_spec_for_claude.md.
    """
    hints = []
    stock_move = result.pct_change_1d
    stock_direction = "up" if stock_move > 0 else "down"

    # 1. Check for company-specific catalysts
    # Prefer processed news_evidence over raw events
    company_specific_found = False

    if news_evidence and news_evidence.checked:
        # Use processed news clusters
        if not news_evidence.no_company_specific_catalyst_found:
            evidence = []
            for cluster in news_evidence.top_clusters[:3]:
                if not cluster.weak_evidence:
                    # Format: "EVENT_TYPE: Headline (Source, time)"
                    time_str = cluster.published_at.strftime("%m/%d %H:%M")
                    evidence.append(
                        f"{cluster.event_type.value}: {cluster.headline[:60]}... "
                        f"({cluster.source}, {time_str})"
                    )
                    company_specific_found = True

            if evidence:
                # Use the best cluster's relevance as strength
                strength = news_evidence.top_clusters[0].relevance_score
                hints.append(AttributionHint(
                    category=AttributionCategory.COMPANY,
                    evidence=evidence,
                    strength=min(0.95, strength + 0.2),  # Boost for having processed evidence
                ))

    # Fallback to raw events if no processed news evidence
    if not company_specific_found and events:
        evidence = []
        for e in events[:3]:
            evidence.append(f"{e.type}: {e.title} ({e.date.isoformat()})")

        hints.append(AttributionHint(
            category=AttributionCategory.COMPANY,
            evidence=evidence,
            strength=0.9,  # High confidence when we have events
        ))

    # 2. Check for sector/peer correlation
    sector_peer_evidence = []
    sector_correlation = False
    peer_correlation = False

    if sector:
        # Check if stock moved in same direction as sector with similar magnitude
        sector_move = sector.pct_change_1d
        same_direction = (stock_move > 0) == (sector_move > 0)

        if same_direction and abs(sector_move) > 0.5:
            sector_correlation = True
            sector_peer_evidence.append(
                f"{sector.etf} {sector_move:+.1f}% (1d)"
            )

    if peers:
        # Check if peers moved similarly
        peer_moves_same_dir = sum(
            1 for p in peers
            if (p.pct_change_1d > 0) == (stock_move > 0)
        )
        if peer_moves_same_dir >= len(peers) // 2 + 1:
            peer_correlation = True
            for p in peers[:3]:
                sector_peer_evidence.append(f"{p.ticker} {p.pct_change_1d:+.1f}%")

    if sector_correlation or peer_correlation:
        strength = 0.7 if (sector_correlation and peer_correlation) else 0.5
        hints.append(AttributionHint(
            category=AttributionCategory.SECTOR_PEER,
            evidence=sector_peer_evidence,
            strength=strength,
        ))

    # 3. Check for macro correlation
    if macro:
        spy_move = macro.spy_pct_change_1d
        same_direction = (stock_move > 0) == (spy_move > 0)

        # Strong market move in same direction
        if same_direction and abs(spy_move) > 0.5:
            # But only attribute to macro if stock didn't outperform significantly
            relative_move = abs(stock_move) - abs(spy_move)

            if relative_move < abs(spy_move) * 0.5:  # Didn't outperform by much
                hints.append(AttributionHint(
                    category=AttributionCategory.MACRO,
                    evidence=[f"SPY {spy_move:+.1f}% (1d)"],
                    strength=0.4,
                ))

    # 4. Flow/technical attribution (when we have volume but no other explanation)
    if result.volume_multiple >= 1.5 and not hints:
        hints.append(AttributionHint(
            category=AttributionCategory.FLOW,
            evidence=[f"Volume {result.volume_multiple:.1f}x average"],
            strength=0.3,
        ))

    # 5. Unattributed if nothing else
    if not hints:
        evidence = ["No clear catalyst identified"]
        if not events:
            evidence.append("No company events detected")
        hints.append(AttributionHint(
            category=AttributionCategory.UNATTRIBUTED,
            evidence=evidence,
            strength=0.1,
        ))

    # Sort by strength descending
    hints.sort(key=lambda h: h.strength, reverse=True)

    return hints


def build_attribution_context(
    result: TriggerResult,
    events: list[Event],
    spy_df: Optional[pd.DataFrame],
    sector_df: Optional[pd.DataFrame],
    sector_etf: Optional[str],
    peer_data: Optional[dict[str, pd.DataFrame]] = None,
    news_checked: bool = False,
    sec_checked: bool = False,
    news_evidence: Optional["NewsEvidence"] = None,
) -> AttributionContext:
    """Build complete attribution context for LLM explanation.

    Args:
        result: Trigger computation result
        events: List of company events (including news if fetched)
        spy_df: SPY OHLCV data
        sector_df: Sector ETF OHLCV data
        sector_etf: Sector ETF ticker
        peer_data: Dict mapping peer tickers to their OHLCV DataFrames
        news_checked: Whether news was successfully fetched
        sec_checked: Whether SEC filings were successfully checked
        news_evidence: Processed NewsEvidence from news_processor (optional)

    Returns:
        AttributionContext with all evidence gathered
    """
    # Compute context pieces
    sector = compute_sector_context(sector_df, sector_etf)
    macro = compute_macro_context(spy_df)
    peers = compute_peer_context(peer_data or {})

    # Track what we checked
    # If news_evidence provided, use its checked status
    if news_evidence is not None:
        news_status = CatalystCheckStatus.CHECKED if news_evidence.checked else CatalystCheckStatus.NOT_AVAILABLE
    else:
        news_status = CatalystCheckStatus.CHECKED if news_checked else CatalystCheckStatus.NOT_AVAILABLE

    catalyst_checks = CatalystChecks(
        earnings_calendar=CatalystCheckStatus.CHECKED,  # We always check this
        news_feed=news_status,
        sec_filings=CatalystCheckStatus.CHECKED if sec_checked else CatalystCheckStatus.NOT_AVAILABLE,
    )

    # Compute hints (now with news_evidence)
    hints = compute_attribution_hints(result, sector, peers, macro, events, news_evidence)

    return AttributionContext(
        ticker=result.ticker,
        trigger_result=result,
        sector=sector,
        peers=peers,
        macro=macro,
        events=events,
        catalyst_checks=catalyst_checks,
        attribution_hints=hints,
        news_evidence=news_evidence,
    )
