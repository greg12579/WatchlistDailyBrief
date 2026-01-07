"""LLM explanation generation with validation and fallback."""

import json
from dataclasses import dataclass, field
from typing import Optional

from watchbrief.config import LLMConfig
from watchbrief.data.events import Event
from watchbrief.features.attribution import AttributionContext, AttributionCategory, CoverageTier
from watchbrief.features.triggers import TriggerResult
from watchbrief.llm.client import LLMClient, create_llm_client
from watchbrief.llm.prompt import build_prompt, build_prompt_v2, get_system_prompt


def cap_confidence_by_coverage(confidence: str, coverage: CoverageTier) -> str:
    """Cap confidence level based on coverage tier.

    Rules:
    - Full coverage: High/Medium/Low allowed
    - Partial coverage: Medium/Low allowed (High -> Medium)
    - None coverage: Low only (High/Medium -> Low)

    Args:
        confidence: The raw confidence from LLM ("High", "Medium", "Low")
        coverage: The coverage tier from CatalystChecks

    Returns:
        Capped confidence level
    """
    if coverage == CoverageTier.FULL:
        return confidence  # No cap
    elif coverage == CoverageTier.PARTIAL:
        if confidence == "High":
            return "Medium"
        return confidence
    else:  # CoverageTier.NONE
        return "Low"


@dataclass
class Driver:
    """A ranked driver explanation."""

    rank: int
    text: str
    weight_pct: int
    category: str = ""  # New: attribution category
    evidence: list[str] = field(default_factory=list)  # New: supporting evidence


@dataclass
class Explanation:
    """LLM-generated explanation for a stock move."""

    drivers: list[Driver]
    confidence: str  # "Low" | "Medium" | "High"
    why_it_matters: str
    is_fallback: bool = False
    missing_checks: list[str] = field(default_factory=list)  # New: data sources not checked


def parse_explanation_json(response: str) -> Optional[Explanation]:
    """Parse LLM response JSON into Explanation object.

    Handles both legacy and v2 formats.
    Returns None if parsing fails.
    """
    try:
        # Try to extract JSON from response (in case of extra text)
        start = response.find("{")
        end = response.rfind("}") + 1
        if start == -1 or end == 0:
            return None

        json_str = response[start:end]
        data = json.loads(json_str)

        # Validate required fields
        if "drivers" not in data or "confidence" not in data or "why_it_matters" not in data:
            return None

        drivers = [
            Driver(
                rank=d.get("rank", i + 1),
                text=d.get("text", ""),
                weight_pct=d.get("weight_pct", 0),
                category=d.get("category", ""),  # New field
                evidence=d.get("evidence", []),  # New field
            )
            for i, d in enumerate(data["drivers"])
        ]

        # Validate confidence value
        confidence = data["confidence"]
        if confidence not in ("Low", "Medium", "High"):
            confidence = "Low"

        # Get missing_checks (new field, optional)
        missing_checks = data.get("missing_checks", [])

        return Explanation(
            drivers=drivers,
            confidence=confidence,
            why_it_matters=data["why_it_matters"],
            missing_checks=missing_checks,
        )

    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def create_fallback_explanation(
    result: TriggerResult,
    events: list[Event],
    context: Optional[AttributionContext] = None,
) -> Explanation:
    """Create a template-based fallback explanation when LLM fails.

    Args:
        result: Trigger computation result
        events: List of events (may be empty)
        context: Optional attribution context for enhanced fallback

    Returns:
        Fallback Explanation object
    """
    drivers = []
    missing_checks = ["news_feed", "sec_filings"]  # Phase 1: always missing these

    # Use attribution hints if available
    if context and context.attribution_hints:
        for i, hint in enumerate(context.attribution_hints[:3]):
            drivers.append(
                Driver(
                    rank=i + 1,
                    text=f"{hint.category.value}: {'; '.join(hint.evidence[:2])}",
                    weight_pct=int(hint.strength * 100 / sum(h.strength for h in context.attribution_hints[:3])),
                    category=hint.category.value,
                    evidence=hint.evidence,
                )
            )
    else:
        # Legacy fallback logic
        # Primary driver based on strongest trigger
        if abs(result.price_z) >= 2.0:
            direction = "up" if result.pct_change_1d > 0 else "down"
            drivers.append(
                Driver(
                    rank=1,
                    text=f"Significant price move {direction} ({result.pct_change_1d:+.1f}%) with z-score {result.price_z:.1f}",
                    weight_pct=50,
                    category="Flow",
                    evidence=[f"price_z={result.price_z:.1f}", f"pct_change_1d={result.pct_change_1d:+.1f}%"],
                )
            )
        elif result.volume_multiple >= 2.0:
            drivers.append(
                Driver(
                    rank=1,
                    text=f"Unusual volume ({result.volume_multiple:.1f}x average)",
                    weight_pct=50,
                    category="Flow",
                    evidence=[f"volume={result.volume_multiple:.1f}x"],
                )
            )
        else:
            drivers.append(
                Driver(
                    rank=1,
                    text="Price/volume deviation from recent patterns",
                    weight_pct=50,
                    category="Unattributed",
                    evidence=result.triggered_reasons,
                )
            )

        # Secondary driver: relative performance
        if abs(result.rel_vs_spy_z) >= 1.5:
            direction = "outperformed" if result.rel_vs_spy_z > 0 else "underperformed"
            drivers.append(
                Driver(
                    rank=2,
                    text=f"Stock {direction} market (SPY) significantly",
                    weight_pct=30,
                    category="Macro",
                    evidence=[f"rel_vs_spy_z={result.rel_vs_spy_z:.1f}"],
                )
            )
        elif result.rel_vs_sector_z is not None and abs(result.rel_vs_sector_z) >= 1.2:
            direction = "outperformed" if result.rel_vs_sector_z > 0 else "underperformed"
            drivers.append(
                Driver(
                    rank=2,
                    text=f"Stock {direction} sector ({result.sector_etf}) significantly",
                    weight_pct=30,
                    category="Sector/Peer",
                    evidence=[f"rel_vs_sector_z={result.rel_vs_sector_z:.1f}"],
                )
            )
        else:
            remaining = 100 - sum(d.weight_pct for d in drivers)
            drivers.append(
                Driver(
                    rank=2,
                    text="Macro/sector factors or flow dynamics",
                    weight_pct=remaining,
                    category="Unattributed",
                    evidence=[],
                )
            )

        # Add event driver if present
        if events:
            event_texts = [f"{e.type}: {e.title}" for e in events[:2]]
            remaining = 100 - sum(d.weight_pct for d in drivers)
            if remaining > 0:
                drivers.append(
                    Driver(
                        rank=len(drivers) + 1,
                        text=f"Company event(s): {'; '.join(event_texts)}",
                        weight_pct=remaining,
                        category="Company",
                        evidence=[f"{e.type}: {e.title}" for e in events[:2]],
                    )
                )

    # Normalize weights to sum to 100
    total_weight = sum(d.weight_pct for d in drivers)
    if total_weight != 100 and total_weight > 0:
        factor = 100 / total_weight
        for d in drivers:
            d.weight_pct = round(d.weight_pct * factor)

    # Confidence based on available signals
    if events and (abs(result.price_z) >= 2.0 or result.volume_multiple >= 2.0):
        confidence = "Medium"
    else:
        confidence = "Low"

    # Gate confidence by coverage if context available
    if context:
        coverage = context.catalyst_checks.coverage
        confidence = cap_confidence_by_coverage(confidence, coverage)

    # Generic why it matters
    direction = "gains" if result.pct_change_1d > 0 else "losses"
    why = f"{result.ticker} showing {result.label.lower()} signals with {abs(result.pct_change_1d):.1f}% {direction}; warrants attention."

    return Explanation(
        drivers=drivers,
        confidence=confidence,
        why_it_matters=why,
        is_fallback=True,
        missing_checks=missing_checks,
    )


def get_explanation(
    result: TriggerResult,
    events: list[Event],
    client: LLMClient,
) -> Explanation:
    """Get LLM explanation for a trigger result (legacy).

    Implements retry logic and fallback:
    1. Try LLM call
    2. If invalid JSON, retry with "return valid JSON only" message
    3. If still invalid, use template fallback

    Args:
        result: Trigger computation result
        events: List of events for context
        client: LLM client to use

    Returns:
        Explanation object (either from LLM or fallback)
    """
    prompt = build_prompt(result, events)
    system = get_system_prompt()

    # First attempt
    try:
        response = client.complete(prompt, system=system)
        explanation = parse_explanation_json(response)
        if explanation:
            return explanation
    except Exception as e:
        print(f"LLM error for {result.ticker}: {e}")

    # Retry with stricter instruction
    retry_prompt = f"""{prompt}

IMPORTANT: Your previous response was not valid JSON. Please respond with ONLY the JSON object, no other text."""

    try:
        response = client.complete(retry_prompt, system=system)
        explanation = parse_explanation_json(response)
        if explanation:
            return explanation
    except Exception as e:
        print(f"LLM retry error for {result.ticker}: {e}")

    # Fallback to template
    print(f"Using fallback explanation for {result.ticker}")
    return create_fallback_explanation(result, events)


def get_explanation_v2(
    context: AttributionContext,
    client: LLMClient,
) -> Explanation:
    """Get LLM explanation with full attribution context (v2).

    Implements retry logic and fallback:
    1. Try LLM call with enhanced prompt
    2. If invalid JSON, retry
    3. If still invalid, use template fallback with attribution hints
    4. Apply confidence cap based on coverage tier

    Args:
        context: Full attribution context
        client: LLM client to use

    Returns:
        Explanation object (either from LLM or fallback)
    """
    prompt = build_prompt_v2(context)
    system = get_system_prompt()

    # Get coverage tier for confidence gating
    coverage = context.catalyst_checks.coverage

    # First attempt
    try:
        response = client.complete(prompt, system=system)
        explanation = parse_explanation_json(response)
        if explanation:
            # Gate confidence by coverage
            capped_confidence = cap_confidence_by_coverage(explanation.confidence, coverage)
            if capped_confidence != explanation.confidence:
                explanation = Explanation(
                    drivers=explanation.drivers,
                    confidence=capped_confidence,
                    why_it_matters=explanation.why_it_matters,
                    is_fallback=explanation.is_fallback,
                    missing_checks=explanation.missing_checks,
                )
            return explanation
    except Exception as e:
        print(f"LLM error for {context.ticker}: {e}")

    # Retry with stricter instruction
    retry_prompt = f"""{prompt}

IMPORTANT: Your previous response was not valid JSON. Please respond with ONLY the JSON object, no other text."""

    try:
        response = client.complete(retry_prompt, system=system)
        explanation = parse_explanation_json(response)
        if explanation:
            # Gate confidence by coverage
            capped_confidence = cap_confidence_by_coverage(explanation.confidence, coverage)
            if capped_confidence != explanation.confidence:
                explanation = Explanation(
                    drivers=explanation.drivers,
                    confidence=capped_confidence,
                    why_it_matters=explanation.why_it_matters,
                    is_fallback=explanation.is_fallback,
                    missing_checks=explanation.missing_checks,
                )
            return explanation
    except Exception as e:
        print(f"LLM retry error for {context.ticker}: {e}")

    # Fallback to template with attribution hints
    print(f"Using fallback explanation for {context.ticker}")
    return create_fallback_explanation(
        context.trigger_result,
        context.events,
        context=context,
    )


def explanation_to_dict(exp: Explanation) -> dict:
    """Convert Explanation to dictionary for storage."""
    return {
        "drivers": [
            {
                "rank": d.rank,
                "text": d.text,
                "weight_pct": d.weight_pct,
                "category": d.category,
                "evidence": d.evidence,
            }
            for d in exp.drivers
        ],
        "confidence": exp.confidence,
        "why_it_matters": exp.why_it_matters,
        "is_fallback": exp.is_fallback,
        "missing_checks": exp.missing_checks,
    }
