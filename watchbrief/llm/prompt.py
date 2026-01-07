"""Prompt construction for LLM explanations."""

import json
from typing import Optional

from watchbrief.data.events import Event
from watchbrief.features.triggers import TriggerResult


SYSTEM_PROMPT = """You are a financial analyst assistant providing concise explanations for stock price movements.

STRICT RULES:
1. Do NOT invent facts. Only use the data provided.
2. Do NOT provide trading advice or recommendations.
3. If no specific events are provided, attribute the move to macro/sector/flow factors.
4. Be concise and factual.
5. Output ONLY valid JSON in the exact format specified."""


def build_facts_json(
    result: TriggerResult,
    events: list[Event],
    spy_move: float,
    sector_move: Optional[float],
) -> dict:
    """Build a structured facts dictionary for the LLM.

    Args:
        result: Trigger computation result
        events: List of events for the ticker
        spy_move: SPY percentage change
        sector_move: Sector ETF percentage change (if available)

    Returns:
        Dictionary of facts
    """
    facts = {
        "ticker": result.ticker,
        "last_close": round(result.last_close, 2),
        "pct_change_1d": round(result.pct_change_1d, 2),
        "pct_change_5d": round(result.pct_change_5d, 2),
        "price_zscore": round(result.price_z, 2),
        "volume_multiple": round(result.volume_multiple, 2),
        "rel_vs_spy_zscore": round(result.rel_vs_spy_z, 2),
        "spy_pct_change_1d": round(spy_move, 2),
        "triggered_reasons": result.triggered_reasons,
        "label": result.label,
    }

    if result.sector_etf:
        facts["sector_etf"] = result.sector_etf

    if result.rel_vs_sector_z is not None:
        facts["rel_vs_sector_zscore"] = round(result.rel_vs_sector_z, 2)

    if sector_move is not None:
        facts["sector_pct_change_1d"] = round(sector_move, 2)

    if events:
        facts["events"] = [
            {
                "type": e.type,
                "date": e.date.isoformat(),
                "title": e.title,
            }
            for e in events
        ]
    else:
        facts["events"] = []

    return facts


def build_prompt(
    result: TriggerResult,
    events: list[Event],
) -> str:
    """Build the full prompt for LLM explanation.

    Args:
        result: Trigger computation result
        events: List of events for the ticker

    Returns:
        Complete prompt string
    """
    facts = build_facts_json(
        result=result,
        events=events,
        spy_move=result.spy_pct_change_1d,
        sector_move=result.sector_pct_change_1d,
    )

    facts_str = json.dumps(facts, indent=2)

    prompt = f"""Analyze this stock movement and provide an explanation.

FACTS:
{facts_str}

INSTRUCTIONS:
1. Based ONLY on the facts above, identify the most likely drivers of this move.
2. If no events are listed, say "No specific company event detected; likely macro/sector/flow."
3. Provide 1-3 ranked drivers with approximate percentage weights that sum to 100%.
4. Assess confidence: Low (no clear catalyst), Medium (some signals), High (clear event + strong signals).
5. Write one sentence explaining "Why it matters" from a portfolio manager perspective (no trade advice).

OUTPUT FORMAT (JSON only, no other text):
{{
  "drivers": [
    {{"rank": 1, "text": "Description of primary driver", "weight_pct": 60}},
    {{"rank": 2, "text": "Description of secondary driver", "weight_pct": 30}},
    {{"rank": 3, "text": "Description of tertiary driver", "weight_pct": 10}}
  ],
  "confidence": "Low|Medium|High",
  "why_it_matters": "One sentence for a PM..."
}}"""

    return prompt


def get_system_prompt() -> str:
    """Return the system prompt for LLM explanations."""
    return SYSTEM_PROMPT
