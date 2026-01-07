"""Ranking and scoring for triggered items.

This module implements a two-layer scoring system:

1. BASE SCORE: Pure quantitative signal (magnitude of move)
   - Price z-score (2x weight)
   - Volume multiple (1x weight)
   - Relative vs SPY (1.5x weight)
   - Relative vs Sector (1.5x weight)

2. ADJUSTMENTS: Evidence-aware modifiers
   - Evidence boost: Reward company-specific catalysts
   - Drift penalty: Penalize low-volume moves without catalysts
   - Cooldown penalty: Penalize repeated appearances

Design principle: "Magnitude gets attention. Evidence earns a spot in the email."
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from typing import TYPE_CHECKING, Optional

from watchbrief.features.triggers import TriggerResult

if TYPE_CHECKING:
    from watchbrief.features.attribution import AttributionContext, AttributionCategory
    from watchbrief.llm.explain import Explanation


@dataclass
class ScoringConfig:
    """Configuration for scoring adjustments.

    All values are additive modifiers to the base score.
    """

    # Evidence boosts (additive)
    company_catalyst_boost: float = 1.5  # Earnings, 8-K, Form 4, major headline
    sector_peer_corroboration_boost: float = 0.5  # Sector/peer evidence exists
    high_confidence_boost: float = 0.5  # Attribution confidence == High

    # Drift penalty
    drift_penalty: float = -0.5  # Low volume + no catalyst + checks done
    drift_volume_threshold: float = 0.8  # Volume multiple below this triggers check

    # Cooldown penalty
    cooldown_penalty: float = -1.0  # Appeared recently without improvement
    cooldown_days: int = 3  # How far back to check
    cooldown_score_improvement_threshold: float = 1.5  # Margin to override cooldown


def compute_base_score(result: TriggerResult) -> float:
    """Compute BASE ranking score for a trigger result.

    This is the original quantitative-only formula.
    Formula (from BuildPlan):
    score = 0
        + abs(price_z) * 2
        + max(0, volume_multiple - 1) * 1
        + abs(rel_vs_spy_z) * 1.5
        + abs(rel_vs_sector_z) * 1.5 (if exists)

    Higher scores indicate more statistically significant moves.
    """
    score = 0.0

    # Price z-score contribution (weight: 2)
    score += abs(result.price_z) * 2.0

    # Volume multiple contribution (weight: 1, only excess above 1x)
    score += max(0, result.volume_multiple - 1) * 1.0

    # Relative vs SPY contribution (weight: 1.5)
    score += abs(result.rel_vs_spy_z) * 1.5

    # Relative vs sector contribution (weight: 1.5, if available)
    if result.rel_vs_sector_z is not None:
        score += abs(result.rel_vs_sector_z) * 1.5

    return score


# Legacy alias for backwards compatibility
compute_score = compute_base_score


def compute_evidence_boost(
    context: "AttributionContext",
    explanation: Optional["Explanation"],
    config: ScoringConfig,
) -> tuple[float, list[str]]:
    """Compute evidence-based score boost.

    Args:
        context: Attribution context with evidence
        explanation: LLM explanation with confidence
        config: Scoring configuration

    Returns:
        Tuple of (boost_value, reasons_list)
    """
    from watchbrief.features.attribution import AttributionCategory, CoverageTier

    boost = 0.0
    reasons = []

    # Check for company-specific catalyst
    has_company_catalyst = False
    if context.attribution_hints:
        for hint in context.attribution_hints:
            if hint.category == AttributionCategory.COMPANY and hint.strength >= 0.5:
                has_company_catalyst = True
                break

    # Also check processed news evidence
    if context.news_evidence and context.news_evidence.checked:
        if not context.news_evidence.no_company_specific_catalyst_found:
            has_company_catalyst = True

    # Also check for earnings, 8-K, Form 4 in events
    if context.events:
        for event in context.events:
            if event.type in ("earnings", "sec_filing"):
                has_company_catalyst = True
                break

    if has_company_catalyst:
        boost += config.company_catalyst_boost
        reasons.append(f"+{config.company_catalyst_boost:.1f} company catalyst")

    # Check for sector/peer corroboration
    has_sector_peer = False
    if context.attribution_hints:
        for hint in context.attribution_hints:
            if hint.category == AttributionCategory.SECTOR_PEER and hint.strength >= 0.4:
                has_sector_peer = True
                break

    if has_sector_peer:
        boost += config.sector_peer_corroboration_boost
        reasons.append(f"+{config.sector_peer_corroboration_boost:.1f} sector/peer corroboration")

    # Check for high confidence attribution
    # Only award confidence boost if coverage is Full (we actually checked news and SEC)
    coverage = context.catalyst_checks.coverage
    if explanation and explanation.confidence == "High" and coverage == CoverageTier.FULL:
        boost += config.high_confidence_boost
        reasons.append(f"+{config.high_confidence_boost:.1f} high confidence")

    return boost, reasons


def compute_drift_penalty(
    result: TriggerResult,
    context: "AttributionContext",
    config: ScoringConfig,
) -> tuple[float, list[str]]:
    """Compute penalty for low-volume drift moves without catalysts.

    A drift move is:
    - Volume below threshold (default 0.8x average)
    - No company-specific catalyst found
    - News/SEC checks were performed (we actually looked)

    Args:
        result: Trigger result
        context: Attribution context
        config: Scoring configuration

    Returns:
        Tuple of (penalty_value, reasons_list) - penalty is negative
    """
    from watchbrief.features.attribution import AttributionCategory, CatalystCheckStatus

    penalty = 0.0
    reasons = []

    # Check low volume condition
    if result.volume_multiple >= config.drift_volume_threshold:
        return penalty, reasons  # Volume is fine, no penalty

    # Check if catalyst checks were done
    checks = context.catalyst_checks
    checks_performed = (
        checks.news_feed == CatalystCheckStatus.CHECKED
        or checks.sec_filings == CatalystCheckStatus.CHECKED
    )

    if not checks_performed:
        return penalty, reasons  # Can't penalize if we didn't check

    # Check for company catalyst
    has_company_catalyst = False
    if context.attribution_hints:
        for hint in context.attribution_hints:
            if hint.category == AttributionCategory.COMPANY and hint.strength >= 0.5:
                has_company_catalyst = True
                break

    if context.news_evidence and context.news_evidence.checked:
        if not context.news_evidence.no_company_specific_catalyst_found:
            has_company_catalyst = True

    if context.events:
        for event in context.events:
            if event.type in ("earnings", "sec_filing", "news"):
                has_company_catalyst = True
                break

    # Apply penalty only if: low volume AND no catalyst AND checks done
    if not has_company_catalyst:
        penalty = config.drift_penalty
        reasons.append(f"{config.drift_penalty:.1f} low-volume drift (vol={result.volume_multiple:.1f}x, no catalyst)")

    return penalty, reasons


def get_recent_appearances(
    ticker: str,
    days: int = 3,
) -> list[tuple[datetime, float]]:
    """Get recent appearances of a ticker in briefs.

    Args:
        ticker: The ticker to check
        days: How many days back to look

    Returns:
        List of (date_sent, score) tuples for recent appearances
    """
    try:
        from watchbrief.storage.db import session_scope
        from watchbrief.storage.models import Brief, BriefItem

        cutoff = datetime.now(UTC) - timedelta(days=days)

        with session_scope() as session:
            appearances = (
                session.query(Brief.date_sent, BriefItem.score)
                .join(BriefItem)
                .filter(BriefItem.ticker == ticker)
                .filter(Brief.date_sent >= cutoff)
                .order_by(Brief.date_sent.desc())
                .all()
            )
            return [(date, score) for date, score in appearances]
    except Exception:
        # If DB not available, no penalty
        return []


def compute_cooldown_penalty(
    ticker: str,
    current_base_score: float,
    config: ScoringConfig,
) -> tuple[float, list[str]]:
    """Compute penalty for repeated appearances.

    A cooldown penalty applies if:
    - Ticker appeared in email within last N days
    - Current score doesn't exceed prior score by significant margin

    Args:
        ticker: The ticker symbol
        current_base_score: Current base score (before adjustments)
        config: Scoring configuration

    Returns:
        Tuple of (penalty_value, reasons_list) - penalty is negative
    """
    penalty = 0.0
    reasons = []

    recent = get_recent_appearances(ticker, config.cooldown_days)

    if not recent:
        return penalty, reasons  # No recent appearances

    # Get the highest prior score
    max_prior_score = max(score for _, score in recent)
    # Handle both timezone-aware and naive datetimes from DB
    recent_date = recent[0][0]
    now = datetime.now(UTC)
    if recent_date.tzinfo is None:
        # DB returned naive datetime, make comparison with naive now
        now = datetime.utcnow()
    days_ago = (now - recent_date).days

    # Check if current score is significantly better
    improvement = current_base_score - max_prior_score

    if improvement >= config.cooldown_score_improvement_threshold:
        # Score improved significantly, no penalty
        return penalty, reasons

    # Apply cooldown penalty
    penalty = config.cooldown_penalty
    reasons.append(f"{config.cooldown_penalty:.1f} cooldown (appeared {days_ago}d ago, prior={max_prior_score:.1f})")

    return penalty, reasons


@dataclass
class EnhancedScore:
    """Complete score breakdown for a triggered item."""

    ticker: str
    base_score: float  # Original quantitative score
    evidence_boost: float  # From catalysts/confidence
    drift_penalty: float  # Low-volume penalty
    cooldown_penalty: float  # Repeat appearance penalty
    final_score: float  # base + boost + penalties
    adjustment_reasons: list[str]  # Explanation of adjustments

    @property
    def total_adjustment(self) -> float:
        return self.evidence_boost + self.drift_penalty + self.cooldown_penalty


def compute_enhanced_score(
    result: TriggerResult,
    context: Optional["AttributionContext"] = None,
    explanation: Optional["Explanation"] = None,
    config: Optional[ScoringConfig] = None,
    skip_cooldown: bool = False,
) -> EnhancedScore:
    """Compute fully enhanced score with all adjustments.

    Args:
        result: Trigger result
        context: Attribution context (optional, enables evidence boost/drift penalty)
        explanation: LLM explanation (optional, enables confidence boost)
        config: Scoring configuration (uses defaults if None)
        skip_cooldown: If True, skip cooldown check (for testing)

    Returns:
        EnhancedScore with full breakdown
    """
    if config is None:
        config = ScoringConfig()

    base = compute_base_score(result)
    reasons = []

    # Evidence boost (requires context)
    evidence_boost = 0.0
    if context is not None:
        boost, boost_reasons = compute_evidence_boost(context, explanation, config)
        evidence_boost = boost
        reasons.extend(boost_reasons)

    # Drift penalty (requires context)
    drift_penalty = 0.0
    if context is not None:
        penalty, penalty_reasons = compute_drift_penalty(result, context, config)
        drift_penalty = penalty
        reasons.extend(penalty_reasons)

    # Cooldown penalty (requires DB access)
    cooldown_penalty = 0.0
    if not skip_cooldown:
        penalty, penalty_reasons = compute_cooldown_penalty(result.ticker, base, config)
        cooldown_penalty = penalty
        reasons.extend(penalty_reasons)

    final = base + evidence_boost + drift_penalty + cooldown_penalty

    return EnhancedScore(
        ticker=result.ticker,
        base_score=base,
        evidence_boost=evidence_boost,
        drift_penalty=drift_penalty,
        cooldown_penalty=cooldown_penalty,
        final_score=final,
        adjustment_reasons=reasons,
    )


def rank_and_select(
    results: list[TriggerResult],
    max_items: int = 5,
) -> list[tuple[TriggerResult, float, int]]:
    """Rank triggered results and select top items (legacy interface).

    This is the ORIGINAL interface for backwards compatibility.
    Uses base score only (no evidence adjustments).

    Args:
        results: List of TriggerResult objects (should be pre-filtered to triggered only)
        max_items: Maximum number of items to return

    Returns:
        List of tuples: (TriggerResult, score, rank)
        Sorted by score descending, limited to max_items.
    """
    # Filter to only triggered items
    triggered = [r for r in results if r.triggered]

    # Compute scores
    scored = [(r, compute_base_score(r)) for r in triggered]

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Take top items and add rank
    top_items = scored[:max_items]
    ranked = [(r, score, rank + 1) for rank, (r, score) in enumerate(top_items)]

    return ranked


@dataclass
class RankedItem:
    """A ranked item with full scoring breakdown."""

    result: TriggerResult
    context: Optional["AttributionContext"]
    explanation: Optional["Explanation"]
    enhanced_score: EnhancedScore
    rank: int


def rank_and_select_enhanced(
    items: list[tuple[TriggerResult, Optional["AttributionContext"], Optional["Explanation"]]],
    max_items: int = 5,
    config: Optional[ScoringConfig] = None,
) -> list[RankedItem]:
    """Rank items using enhanced scoring with evidence adjustments.

    This is the NEW interface that uses all available evidence.

    Args:
        items: List of (TriggerResult, AttributionContext, Explanation) tuples
        max_items: Maximum number of items to return
        config: Scoring configuration (uses defaults if None)

    Returns:
        List of RankedItem objects, sorted by final_score descending
    """
    if config is None:
        config = ScoringConfig()

    # Filter to only triggered items and compute enhanced scores
    scored = []
    for result, context, explanation in items:
        if not result.triggered:
            continue

        enhanced = compute_enhanced_score(result, context, explanation, config)
        scored.append((result, context, explanation, enhanced))

    # Sort by final score descending
    scored.sort(key=lambda x: x[3].final_score, reverse=True)

    # Take top items and add rank
    ranked = []
    for rank, (result, context, explanation, enhanced) in enumerate(scored[:max_items]):
        ranked.append(RankedItem(
            result=result,
            context=context,
            explanation=explanation,
            enhanced_score=enhanced,
            rank=rank + 1,
        ))

    return ranked
