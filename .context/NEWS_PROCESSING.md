# News Processing Pipeline - Detailed Reference

## Purpose

The news processing pipeline ensures PM-grade attribution by:
1. Detecting credible company-specific catalysts
2. Filtering low-signal content (recycled "why it moved" articles)
3. De-duplicating similar stories
4. Classifying event types
5. Scoring relevance
6. Providing minimal, citable evidence

**Key Principle**: The LLM should NOT "know the news." The pipeline must fetch, structure, and gate what the LLM can say.

---

## Pipeline Steps

### Step 1: Fetch Headlines

**API Priority**:
1. Polygon.io (POLYGON_API_KEY) - Best for stock-specific news
2. Finnhub (FINNHUB_API_KEY) - Good alternative
3. NewsAPI (NEWSAPI_KEY) - General news fallback

**Fields Stored**:
```python
{
    "headline": str,
    "source": str,      # e.g., "Reuters", "PR Newswire"
    "published_at": datetime,
    "url": str
}
```

**Lookback**: Default 72 hours from trigger time

---

### Step 2: Filter Low-Signal Headlines

**Patterns Filtered** (downranked, not removed):
```python
LOW_SIGNAL_PATTERNS = [
    r"\b(top\s*movers?|biggest\s*(gainers?|losers?))\b",
    r"\bwhy\s+.+\s+(stock|shares?)\s+(is|are)\s+(up|down)\b",
    r"\b(stock|shares?)\s+(jumps?|falls?)\s+(on\s+)?volume\b",
    r"\b(technical\s*analysis|breaks?\s*(resistance|support))\b",
    r"\b(what\s*happened|here'?s?\s*why)\b",
    r"\bmarket\s*(wrap|update|close)\b",
]
```

These are kept as **secondary** fallback if nothing else exists.

---

### Step 3: Cluster & De-duplicate

**Similarity Computation**:
```python
similarity = 0.4 * jaccard_tokens + 0.6 * sequence_ratio
```

**Clustering**:
- Threshold: 0.65 similarity
- Headlines sorted by timestamp (earliest first)
- Each headline assigned to first matching cluster

**Representative Selection** (per cluster):
1. Highest source authority
2. Earliest timestamp (tie-breaker)
3. Richest content (longer headline)

---

### Step 4: Event Classification

**Taxonomy**:

| Category | Event Types |
|----------|-------------|
| Company-Specific | EARNINGS, GUIDANCE, MNA, FINANCING, REGULATORY_LEGAL, PRODUCT, MANAGEMENT, CONTRACT_CUSTOMER, INSIDER_OWNERSHIP |
| Non-Company | ANALYST_ACTION, SECTOR, MACRO, OTHER, LOW_SIGNAL |

**Keyword Patterns** (examples):
```python
EVENT_PATTERNS = {
    EventType.EARNINGS: [
        r"\bq[1-4]\s*(results?|earnings?)\b",
        r"\b(beats?|misses?)\s*(estimates?|expectations?)\b",
    ],
    EventType.MNA: [
        r"\b(acquires?|acquisition|acquired)\b",
        r"\b(merger|merge|merging)\b",
        r"\b(in\s+talks?|exploring\s+(sale|options))\b",
    ],
    EventType.FINANCING: [
        r"\b(offering|offer)\s*(priced?|of|at)\b",
        r"\b(convertible|convert)\s*(notes?|bonds?)\b",
        r"\braises?\s*\$[\d.]+[mb]\b",
    ],
    # ... more patterns
}
```

**Confidence**:
- Multiple company-specific matches: 0.85
- Single company-specific match: 0.75
- Non-company match: 0.65
- Low-signal: 0.30

---

### Step 5: Source Authority Scoring

```python
SOURCE_AUTHORITY = {
    # Tier 1: Wire services
    "reuters": 1.0,
    "bloomberg": 1.0,
    "wall street journal": 0.95,

    # Tier 2: Official sources
    "pr newswire": 0.90,
    "business wire": 0.90,
    "sec edgar": 0.95,

    # Tier 3: Major business news
    "cnbc": 0.80,
    "marketwatch": 0.75,
    "barrons": 0.80,

    # Tier 4: Lower quality
    "seeking alpha": 0.65,
    "benzinga": 0.65,
    "motley fool": 0.50,

    # Default
    "unknown": 0.40,
}
```

---

### Step 6: Relevance Scoring

**Components**:

| Factor | Score Range | Examples |
|--------|-------------|----------|
| Recency | -0.10 to +0.35 | <12h: +0.35, <24h: +0.25, <48h: +0.15, older: -0.10 |
| Specificity | +0.05 to +0.30 | Company-specific: +0.30, Analyst: +0.20, Sector/Macro: +0.05 |
| Source Authority | +0.10 to +0.20 | Tier 1: +0.20, Tier 2-3: +0.10 |
| Cluster Size | +0.10 to +0.15 | 5+ sources: +0.15, 3+ sources: +0.10 |
| Low-Signal Penalty | -0.25 | If classified as LOW_SIGNAL |

**Threshold**: 0.55 (clusters below this marked as `weak_evidence=True`)

---

### Step 7: Evidence Selection

**Selection Rules**:
1. Sort clusters by `relevance_score` descending
2. Select top 3 clusters with `relevance_score >= 0.55`
3. If none qualify: keep most recent as fallback with `weak_evidence=True`

**Output**:
```python
NewsEvidence(
    checked=True,
    lookback_hours=72,
    top_clusters=[
        NewsCluster(
            event_type=EventType.FINANCING,
            event_confidence=0.82,
            relevance_score=0.74,
            headline="Company X prices $500M convertible notes",
            source="PR Newswire",
            published_at=datetime(...),
            url="...",
            weak_evidence=False,
            cluster_size=3,
            keywords_matched=["convertible", "notes", "priced"],
        )
    ],
    no_company_specific_catalyst_found=False,
)
```

---

## LLM Constraints

**Allowed**:
- Paraphrase the headline
- State "a news catalyst was detected"
- Cite the cluster (headline, source, time)

**NOT Allowed**:
- Infer details not present (terms, numbers, outcomes)
- Claim causality with certainty
- Use knowledge outside the evidence payload

**Required Transparency**:
- If `checked=False`: "News not checked (feed unavailable)"
- If `no_company_specific_catalyst_found=True`: "No relevant company-specific headlines found in last 72h"

---

## Rendering Output

### HTML
```html
<div class="events">
  <strong>Recent News & Filings:</strong>
  <div class="event">
    <span class="event-type news">[FINANCING]</span>
    01/07 14:30: <a href="..." class="event-link">Company prices $500M...</a>
    <span style="color:#888">(PR Newswire)</span>
  </div>
</div>
```

### Text
```
Recent News & Filings:
  [FINANCING] 01/07 14:30: Company prices $500M convertible notes
    https://...
```

### No Catalyst Found
```html
<div class="event" style="color:#6c757d;font-style:italic;">
  No relevant company-specific headlines found (checked last 72h)
</div>
```

---

## Common Failure Modes

| Issue | Solution |
|-------|----------|
| Duplicate story spam | Clustering with 0.65 similarity threshold |
| Recycled "why it moved" | Low-signal pattern filtering |
| Sector news as company news | Event type classification + specificity scoring |
| Old news causing misattribution | Recency scoring (-0.10 for >48h) |
| LLM inventing details | Strict evidence payload, hard constraints |
