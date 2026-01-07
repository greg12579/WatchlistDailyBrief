"""Phase 2 LLM prompt builder for trend context analysis.

This module builds prompts for the second LLM phase that provides global context
for price moves. Unlike Phase 1 (attribution), Phase 2 uses only quantitative
inputs and never references news, filings, or events.

Key constraints:
- Quantitative-only input
- No causality claims (causes are Phase 1)
- No trading advice
- Context only, explaining state not recommending action
"""

import json
from dataclasses import dataclass
from typing import Optional

from watchbrief.features.trend_context import (
    TrendContext,
    get_horizon_summary,
    zscore_to_descriptor,
)


PHASE2_SYSTEM_PROMPT = """You are a portfolio context analyst providing neutral, factual market state context.

Your role: Given quantitative data about a stock's multi-horizon returns, z-scores, relative performance, and 52-week positioning, provide brief context about where this move sits historically.

CRITICAL CONSTRAINTS:
1. Use ONLY the quantitative data provided - no external knowledge
2. NEVER reference news, earnings, filings, or events (that's Phase 1's job)
3. NEVER make trading recommendations or predictions
4. NEVER use words like "should", "buy", "sell", "consider"
5. Provide CONTEXT only - describe the state, not the action
6. Be concise - 1-2 sentences per field

You must respond with valid JSON only."""


def build_phase2_prompt(trend_context: TrendContext) -> str:
    """Build the Phase 2 LLM prompt from trend context.

    Args:
        trend_context: Computed trend context with all quantitative metrics

    Returns:
        Formatted prompt string for LLM
    """
    context_json = json.dumps(trend_context.to_dict(), indent=2)

    prompt = f"""Analyze the following quantitative trend context and provide brief, neutral commentary.

TREND CONTEXT:
{context_json}

Based on this data, provide context in the following JSON format:

{{
  "trend_summary": "1-2 sentence neutral description of where the stock sits in its historical context",
  "positioning_insight": "1 sentence about the 52-week positioning and what the multi-horizon returns show",
  "speed_assessment": "1 sentence about how unusual the pace of this move is based on z-scores"
}}

Remember:
- Use ONLY the provided data
- NO references to news, events, or causes
- NO trading advice
- Be factual and neutral

Respond with valid JSON only."""

    return prompt


@dataclass
class Phase2Response:
    """Parsed response from Phase 2 LLM call."""

    trend_summary: str
    positioning_insight: str
    speed_assessment: str

    @classmethod
    def from_dict(cls, data: dict) -> "Phase2Response":
        """Create Phase2Response from parsed JSON dict."""
        return cls(
            trend_summary=data.get("trend_summary", ""),
            positioning_insight=data.get("positioning_insight", ""),
            speed_assessment=data.get("speed_assessment", ""),
        )

    def is_valid(self) -> bool:
        """Check if response has required fields."""
        return bool(self.trend_summary and self.positioning_insight)


def parse_phase2_response(response_text: str) -> Optional[Phase2Response]:
    """Parse the LLM response into a Phase2Response.

    Args:
        response_text: Raw LLM response text

    Returns:
        Phase2Response if valid JSON, None otherwise
    """
    try:
        # Try to find JSON in the response
        text = response_text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        data = json.loads(text)
        return Phase2Response.from_dict(data)

    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def create_fallback_phase2_response(trend_context: TrendContext) -> Phase2Response:
    """Create a fallback response using only the computed metrics.

    Used when LLM is unavailable or returns invalid response.
    Uses human-readable descriptors instead of raw z-scores.
    """
    r = trend_context.returns
    e = trend_context.extremes
    z = trend_context.z
    state = trend_context.market_state.value.replace("_", " ")

    # Get horizon descriptors for human-readable output
    horizon = get_horizon_summary(z)

    # Build trend summary using descriptors
    year_desc = horizon["one_year"].lower()
    if r.pct_252d < -20:
        trend_desc = f"significantly down {r.pct_252d:.1f}% over the past year"
    elif r.pct_252d > 20:
        trend_desc = f"significantly up {r.pct_252d:+.1f}% over the past year"
    elif r.pct_252d < -5:
        trend_desc = f"modestly down {r.pct_252d:.1f}% year-over-year"
    elif r.pct_252d > 5:
        trend_desc = f"modestly up {r.pct_252d:+.1f}% year-over-year"
    else:
        trend_desc = f"roughly flat ({r.pct_252d:+.1f}%) year-over-year"

    trend_summary = f"Stock is in {state} state, {trend_desc}."

    # Build positioning insight using descriptors
    if e.pct_from_52w_high > -5:
        pos_desc = "near its 52-week high"
    elif e.pct_from_52w_low < 10:
        pos_desc = "near its 52-week low"
    else:
        pos_desc = f"{e.pct_from_52w_high:.0f}% from 52w high, {e.pct_from_52w_low:+.0f}% from 52w low"

    # Use descriptors for horizon comparison
    short_desc = horizon["short_term"].lower()
    quarter_desc = horizon["quarter"].lower()
    positioning_insight = f"Currently {pos_desc}. Short-term shows {short_desc}, while the quarter shows {quarter_desc}."

    # Build speed assessment using descriptors (no raw z-scores in narrative)
    max_z = max(abs(z.z_5d), abs(z.z_21d), abs(z.z_63d))
    speed_descriptor = zscore_to_descriptor(max_z if max_z == abs(z.z_5d) else -max_z if z.z_5d < 0 else max_z)

    if max_z > 2.5:
        speed_desc = "unusually rapid"
    elif max_z > 1.5:
        speed_desc = "moderately fast"
    else:
        speed_desc = "typical"

    speed_assessment = f"The move represents a {speed_desc} pace."

    return Phase2Response(
        trend_summary=trend_summary,
        positioning_insight=positioning_insight,
        speed_assessment=speed_assessment,
    )


def get_phase2_context(
    trend_context: TrendContext,
    llm_client,
) -> Phase2Response:
    """Get Phase 2 context from LLM or fallback.

    Args:
        trend_context: Computed trend context
        llm_client: LLM client for making API calls

    Returns:
        Phase2Response with trend context commentary
    """
    if llm_client is None:
        return create_fallback_phase2_response(trend_context)

    prompt = build_phase2_prompt(trend_context)

    try:
        response_text = llm_client.complete(
            prompt=prompt,
            system=PHASE2_SYSTEM_PROMPT,
        )

        parsed = parse_phase2_response(response_text)
        if parsed and parsed.is_valid():
            return parsed

        # Retry once
        response_text = llm_client.complete(
            prompt=prompt + "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no explanations.",
            system=PHASE2_SYSTEM_PROMPT,
        )

        parsed = parse_phase2_response(response_text)
        if parsed and parsed.is_valid():
            return parsed

    except Exception as e:
        print(f"  Phase 2 LLM error: {e}")

    # Fallback to template-based response
    return create_fallback_phase2_response(trend_context)
