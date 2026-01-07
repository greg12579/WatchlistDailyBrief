"""LLM explanation generation with validation and fallback."""

import json
from dataclasses import dataclass
from typing import Optional

from watchbrief.config import LLMConfig
from watchbrief.data.events import Event
from watchbrief.features.triggers import TriggerResult
from watchbrief.llm.client import LLMClient, create_llm_client
from watchbrief.llm.prompt import build_prompt, get_system_prompt


@dataclass
class Driver:
    """A ranked driver explanation."""

    rank: int
    text: str
    weight_pct: int


@dataclass
class Explanation:
    """LLM-generated explanation for a stock move."""

    drivers: list[Driver]
    confidence: str  # "Low" | "Medium" | "High"
    why_it_matters: str
    is_fallback: bool = False


def parse_explanation_json(response: str) -> Optional[Explanation]:
    """Parse LLM response JSON into Explanation object.

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
            )
            for i, d in enumerate(data["drivers"])
        ]

        # Validate confidence value
        confidence = data["confidence"]
        if confidence not in ("Low", "Medium", "High"):
            confidence = "Low"

        return Explanation(
            drivers=drivers,
            confidence=confidence,
            why_it_matters=data["why_it_matters"],
        )

    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def create_fallback_explanation(result: TriggerResult, events: list[Event]) -> Explanation:
    """Create a template-based fallback explanation when LLM fails.

    Args:
        result: Trigger computation result
        events: List of events (may be empty)

    Returns:
        Fallback Explanation object
    """
    drivers = []

    # Primary driver based on strongest trigger
    if abs(result.price_z) >= 2.0:
        direction = "up" if result.pct_change_1d > 0 else "down"
        drivers.append(
            Driver(
                rank=1,
                text=f"Significant price move {direction} ({result.pct_change_1d:+.1f}%) with z-score {result.price_z:.1f}",
                weight_pct=50,
            )
        )
    elif result.volume_multiple >= 2.0:
        drivers.append(
            Driver(
                rank=1,
                text=f"Unusual volume ({result.volume_multiple:.1f}x average)",
                weight_pct=50,
            )
        )
    else:
        drivers.append(
            Driver(
                rank=1,
                text="Price/volume deviation from recent patterns",
                weight_pct=50,
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
            )
        )
    elif result.rel_vs_sector_z is not None and abs(result.rel_vs_sector_z) >= 1.2:
        direction = "outperformed" if result.rel_vs_sector_z > 0 else "underperformed"
        drivers.append(
            Driver(
                rank=2,
                text=f"Stock {direction} sector ({result.sector_etf}) significantly",
                weight_pct=30,
            )
        )
    else:
        remaining = 100 - sum(d.weight_pct for d in drivers)
        drivers.append(
            Driver(
                rank=2,
                text="Macro/sector factors or flow dynamics",
                weight_pct=remaining,
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
                )
            )

    # Normalize weights to sum to 100
    total_weight = sum(d.weight_pct for d in drivers)
    if total_weight != 100:
        factor = 100 / total_weight
        for d in drivers:
            d.weight_pct = round(d.weight_pct * factor)

    # Confidence based on available signals
    if events and (abs(result.price_z) >= 2.0 or result.volume_multiple >= 2.0):
        confidence = "Medium"
    else:
        confidence = "Low"

    # Generic why it matters
    direction = "gains" if result.pct_change_1d > 0 else "losses"
    why = f"{result.ticker} showing {result.label.lower()} signals with {abs(result.pct_change_1d):.1f}% {direction}; warrants attention."

    return Explanation(
        drivers=drivers,
        confidence=confidence,
        why_it_matters=why,
        is_fallback=True,
    )


def get_explanation(
    result: TriggerResult,
    events: list[Event],
    client: LLMClient,
) -> Explanation:
    """Get LLM explanation for a trigger result.

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


def explanation_to_dict(exp: Explanation) -> dict:
    """Convert Explanation to dictionary for storage."""
    return {
        "drivers": [
            {"rank": d.rank, "text": d.text, "weight_pct": d.weight_pct}
            for d in exp.drivers
        ],
        "confidence": exp.confidence,
        "why_it_matters": exp.why_it_matters,
        "is_fallback": exp.is_fallback,
    }
