"""Prompt construction for LLM explanations with attribution context."""

import json
from typing import Optional

from watchbrief.data.events import Event
from watchbrief.features.attribution import AttributionContext
from watchbrief.features.triggers import TriggerResult


SYSTEM_PROMPT = """You are a financial analyst assistant providing credible, evidence-backed explanations for stock price movements.

STRICT RULES:
1. Do NOT invent facts. Only use the data provided in the FACTS section.
2. Do NOT provide trading advice or recommendations.
3. Each driver MUST include explicit evidence from the facts provided.
4. If a data source was not checked (see catalyst_checks), acknowledge this limitation.
5. Prefer "Unattributed" over false certainty - it's better to say "unknown" than guess.
6. Output ONLY valid JSON in the exact format specified.

ATTRIBUTION CATEGORIES (use these exact values):
- "Company": Company-specific events (earnings, news, filings)
- "Sector/Peer": Sector or peer group correlation
- "Macro": Broad market movement (SPY correlation)
- "Flow": Technical/flow-driven (volume without fundamental reason)
- "Unattributed": Cannot determine with available data"""


def build_facts_json_v2(context: AttributionContext) -> dict:
    """Build enhanced facts dictionary with attribution context.

    Args:
        context: Full attribution context

    Returns:
        Dictionary of facts for LLM
    """
    result = context.trigger_result

    facts = {
        "ticker": result.ticker,
        "stock": {
            "last_close": round(result.last_close, 2),
            "pct_change_1d": round(result.pct_change_1d, 2),
            "pct_change_5d": round(result.pct_change_5d, 2),
            "price_zscore": round(result.price_z, 2),
            "volume_multiple": round(result.volume_multiple, 2),
            "rel_vs_spy_zscore": round(result.rel_vs_spy_z, 2),
        },
        "triggered_reasons": result.triggered_reasons,
        "label": result.label,
    }

    # Add relative vs sector if available
    if result.rel_vs_sector_z is not None:
        facts["stock"]["rel_vs_sector_zscore"] = round(result.rel_vs_sector_z, 2)

    # Sector context
    if context.sector:
        facts["sector"] = {
            "etf": context.sector.etf,
            "pct_change_1d": round(context.sector.pct_change_1d, 2),
            "pct_change_5d": round(context.sector.pct_change_5d, 2),
        }

    # Peer context
    if context.peers:
        facts["peers"] = [
            {
                "ticker": p.ticker,
                "pct_change_1d": round(p.pct_change_1d, 2),
                "pct_change_5d": round(p.pct_change_5d, 2),
            }
            for p in context.peers
        ]

    # Macro context
    if context.macro:
        facts["macro"] = {
            "spy_pct_change_1d": round(context.macro.spy_pct_change_1d, 2),
            "spy_pct_change_5d": round(context.macro.spy_pct_change_5d, 2),
        }

    # Events
    if context.events:
        facts["events"] = [
            {
                "type": e.type,
                "date": e.date.isoformat(),
                "title": e.title,
            }
            for e in context.events
        ]
    else:
        facts["events"] = []

    # Catalyst check status
    facts["catalyst_checks"] = context.catalyst_checks.to_dict()

    # Pre-computed attribution hints
    if context.attribution_hints:
        facts["attribution_hints"] = [
            {
                "category": hint.category.value,
                "evidence": hint.evidence,
                "strength": round(hint.strength, 2),
            }
            for hint in context.attribution_hints
        ]

    return facts


def build_prompt_v2(context: AttributionContext) -> str:
    """Build the enhanced prompt for LLM explanation with attribution.

    Args:
        context: Full attribution context

    Returns:
        Complete prompt string
    """
    facts = build_facts_json_v2(context)
    facts_str = json.dumps(facts, indent=2)

    # Build missing checks warning
    missing = context.catalyst_checks.missing_checks()
    missing_warning = ""
    if missing:
        missing_warning = f"\nNOTE: The following data sources were NOT checked: {', '.join(missing)}. Acknowledge this limitation in your explanation."

    prompt = f"""Analyze this stock movement and provide a credible, evidence-backed attribution.

FACTS:
{facts_str}
{missing_warning}

INSTRUCTIONS:
1. Review the attribution_hints provided - these are pre-computed based on the data.
2. For each driver, you MUST cite specific evidence from the facts (e.g., "XLK +2.1%", "peer DHI +2.4%").
3. Use the correct attribution category for each driver.
4. If attribution is unclear, use "Unattributed" with honesty about what we don't know.
5. List any data sources that were not available in missing_checks.
6. Assess confidence based on evidence strength:
   - High: Clear event + corroborating data
   - Medium: Strong correlation but no direct catalyst
   - Low: Weak or conflicting signals
7. Write "why_it_matters" for a portfolio manager - what should they consider?

OUTPUT FORMAT (JSON only, no other text):
{{
  "drivers": [
    {{
      "rank": 1,
      "category": "Company|Sector/Peer|Macro|Flow|Unattributed",
      "text": "Description with specific numbers",
      "evidence": ["fact 1", "fact 2"],
      "weight_pct": 60
    }}
  ],
  "confidence": "Low|Medium|High",
  "missing_checks": ["news_feed", "sec_filings"],
  "why_it_matters": "One sentence for a PM..."
}}"""

    return prompt


def get_system_prompt() -> str:
    """Return the system prompt for LLM explanations."""
    return SYSTEM_PROMPT


# Legacy functions for backwards compatibility
def build_facts_json(
    result: TriggerResult,
    events: list[Event],
    spy_move: float,
    sector_move: Optional[float],
) -> dict:
    """Build a structured facts dictionary for the LLM (legacy).

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
    """Build the full prompt for LLM explanation (legacy).

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
