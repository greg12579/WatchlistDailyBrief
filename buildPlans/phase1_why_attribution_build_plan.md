# Phase 1 Build Plan: PM-Grade “Why” Attribution Engine

## Objective
Upgrade the system’s “Why (Ranked)” explanation from a restatement of technical triggers
to a **credible, evidence-backed attribution of stock price movement** that a portfolio
manager can trust.

This phase explicitly **does NOT rely on the LLM knowing news**.
Attribution must be grounded in **facts supplied by the system**.

---

## Problem Statement (Current State)
- The LLM is asked to infer “why” using only price, volume, and relative performance.
- Sector, peer, macro, and catalyst checks are **not actually performed**, but the output
  implies they were.
- This produces explanations that are technically correct but **epistemically weak**:
  - “Likely macro/sector/flow” without evidence
  - “Outperformed SPY” restated as a driver
- PM trust erodes because the system **sounds confident without checking**.

---

## Design Principles
1. **Attribution before language**
2. **No evidence → no attribution**
3. **Prefer “Unknown” over false certainty**
4. **Evidence always visible**

---

## Target Outcome (Definition of Done)
For every triggered stock:
- The “Why” section identifies **attribution categories**, not trigger mechanics.
- Each driver includes **explicit evidence** from supplied facts.
- Missing checks (news, filings) are disclosed.
- The explanation answers:  
  *“Is this move fundamental, thematic, macro, or flow-driven — and how do we know?”*

---

## Architecture Changes

```
trigger_detection
    ↓
attribution_engine
    ↓
attribution_hints
    ↓
LLM_explainer
    ↓
email / slack
```

---

## FACTS Payload v2

### Sector Context
```json
{
  "sector": {
    "etf": "XHB",
    "pct_change_1d": 3.2,
    "pct_change_5d": 5.8
  }
}
```

### Peer Context
```json
{
  "peers": [
    {"ticker": "DHI", "pct_change_1d": 2.4},
    {"ticker": "LEN", "pct_change_1d": 2.1}
  ]
}
```

### Catalyst Check Status
```json
{
  "catalyst_checks": {
    "earnings_calendar": "checked",
    "news_feed": "not_available",
    "sec_filings": "not_available"
  }
}
```

---

## Attribution Engine Logic
Priority order:
1. Company-specific
2. Sector / peer
3. Macro
4. Flow / technical
5. Unattributed

---

## Output Schema
```json
{
  "drivers": [
    {
      "rank": 1,
      "category": "Sector/Peer | Company | Macro | Flow | Unattributed",
      "text": "Description",
      "evidence": ["fact 1", "fact 2"],
      "weight_pct": 60
    }
  ],
  "confidence": "Low | Medium | High",
  "missing_checks": ["news_feed"],
  "why_it_matters": "PM-relevant sentence"
}
```

---

## Non-Goals
- No prediction
- No trade advice
- No transcript NLP
