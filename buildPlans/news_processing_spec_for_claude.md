# News Processing Spec for Phase 1 “Why” Attribution (Give to Claude)

## Goal
Use **news headlines/articles** as an **evidence source** to improve “Why” quality for triggered watchlist moves.
The system must:
1) Detect whether a **credible company-specific catalyst** exists in the move window  
2) Classify the catalyst type  
3) Provide **minimal, citeable evidence** to the LLM and the PM (no invented facts)

**Important:** The LLM should *not* “know the news.” The pipeline must fetch, structure, and gate what the LLM can say.

---

## Inputs
- `ticker`: string
- `trigger_time`: timestamp (when alert fired)
- `price_context`: returns/z-scores/volume, etc. (already in system)
- `window`: lookback horizon for catalyst search  
  - Default: `lookback_start = max(trigger_time - 72h, prev_close_time)`
  - Default: `lookback_end = trigger_time`

---

## Step 1 — Fetch news (headlines-first)
### 1.1 Query
Fetch news items where **ticker is mentioned** OR entity match hits company name.

### 1.2 Store raw fields (no transformation yet)
For each item:
- `news_id` (provider id)
- `source` (e.g., Reuters, PR Newswire, WSJ)
- `published_at` (UTC)
- `headline`
- `url` (or internal link)
- `tickers_mentioned` (from provider if available)
- `summary/snippet` (optional; keep short)
- `full_text` (optional; avoid if not needed)

**Best practice:** you can do Phase 1 with just headlines + timestamps + source.

---

## Step 2 — Normalize & clean
### 2.1 Standardize time & text
- Normalize `published_at` to UTC
- Strip whitespace, normalize quotes, remove trailing “- source”
- Lowercase for similarity checks (keep original for display)

### 2.2 Filter obvious low-signal / meta headlines
Drop or downrank headlines that are **not catalysts**, e.g.:
- “Top movers today…”
- “Stock jumps/falls on heavy volume…”
- “Why XYZ stock is up today” (repackaging)
- recycled “what happened” pieces
- “Technical analysis: XYZ breaks resistance…”

Keep these only if nothing else exists, and label as **secondary**.

---

## Step 3 — Cluster & de-duplicate (critical)
PMs hate repeated versions of the same story.

### 3.1 Headline clustering
Compute similarity between headlines (any method is fine):
- Min viable: token overlap / Jaccard + fuzzy match
- Better: embeddings cosine similarity

Cluster items where similarity > threshold (e.g., 0.80).

### 3.2 Choose a “cluster representative”
For each cluster, select the best representative:
1) Most authoritative source (wire > blog)  
2) Earliest timestamp (original break)  
3) Richest snippet if ties

Store:
- `cluster_id`
- `cluster_size`
- `rep_item` (headline, source, published_at, url)
- `cluster_items` (ids only)

---

## Step 4 — Event classification (structured tags)
### 4.1 Event type taxonomy (start small)
Classify each cluster representative into one primary `event_type`:

**Company-specific**
- `EARNINGS` (results, guidance, pre-announcement)
- `GUIDANCE` (raise/lower/withdraw)
- `MNA` (acquire, acquired, talks, rumor)
- `FINANCING` (offering, debt, convert, ATM)
- `REGULATORY_LEGAL` (DOJ/SEC, lawsuits, approvals)
- `PRODUCT` (launch, recall, outage)
- `MANAGEMENT` (CEO/CFO change)
- `CONTRACT_CUSTOMER` (major win/loss)
- `INSIDER_OWNERSHIP` (stake, activism, buyback)

**Non-company**
- `SECTOR` (industry-wide story)
- `MACRO` (rates, inflation, geopolitics)
- `ANALYST_ACTION` (upgrade/downgrade/target)
- `OTHER/LOW_SIGNAL`

### 4.2 How to classify (recommended)
Use a **deterministic keyword + pattern baseline**, optionally enhanced by an LLM classifier:
- Financing: “offering”, “ATM”, “convertible”, “notes”, “priced”
- M&A: “acquires”, “to buy”, “in talks”, “deal”
- Earnings/guidance: “Q4 results”, “beats/misses”, “raises guidance”
- Analyst: “upgrade”, “downgrade”, “price target”

Output per cluster:
- `event_type`
- `event_confidence` (0–1)
- `keywords_matched` (for audit)

**Hard rule:** If confidence is low, classify as `OTHER/LOW_SIGNAL`.

---

## Step 5 — Relevance scoring (for THIS move)
Compute `relevance_score` (0–1) using:
- **Recency**: closer to trigger_time = higher
- **Specificity**: company-specific > sector/macro
- **Authority**: reputable source > low quality
- **Novelty**: first time story appears vs repeats
- **Breadth**: cluster_size as a weak proxy

Example heuristic:
- company-specific in last 24h: +0.4
- reputable wire: +0.2
- cluster size >= 3: +0.1
- older than 48h: -0.2

Store:
- `relevance_score`
- `relevance_rationale` (short list)

---

## Step 6 — Select evidence to pass downstream
Pick up to **3 clusters** as candidate catalysts:
- Sort by `relevance_score` desc
- Keep clusters with `relevance_score >= 0.55`
- If none qualify, keep 1 “most recent” cluster as fallback but set `weak_evidence=true`

Create `news_evidence` payload:
```json
{
  "checked": true,
  "lookback_hours": 72,
  "top_clusters": [
    {
      "event_type": "FINANCING",
      "event_confidence": 0.82,
      "relevance_score": 0.74,
      "headline": "Company X prices $500M convertible notes offering",
      "source": "PR Newswire",
      "published_at": "2026-01-07T13:20:00Z",
      "url": "…",
      "weak_evidence": false
    }
  ],
  "no_company_specific_catalyst_found": false
}
```

If nothing credible:
```json
{
  "checked": true,
  "lookback_hours": 72,
  "top_clusters": [],
  "no_company_specific_catalyst_found": true
}
```

---

## Step 7 — Feed into attribution engine (not directly to “Why”)
### 7.1 Deterministic gating
Update attribution logic:
- If `no_company_specific_catalyst_found == false` and top cluster `relevance_score >= threshold`:
  - Add driver candidate `Company-specific catalyst` with evidence (headline + timestamp + source)
- Else:
  - Do **not** claim company catalyst from news

### 7.2 Mixed attribution is expected
News can be one driver among others (sector/macro/flow).

---

## Step 8 — What the LLM is allowed to say
### 8.1 Hard constraints
LLM must ONLY use `news_evidence.top_clusters` fields as facts.
It may:
- paraphrase the headline
- state that “a news catalyst was detected” and cite the cluster
It may NOT:
- infer details not present (terms, numbers, outcomes)
- claim causality with certainty (use “likely associated with”)

### 8.2 Required transparency
If `checked=false` or feed missing:
- LLM must state “News not checked” / “News feed unavailable”
If `no_company_specific_catalyst_found=true`:
- LLM must state “No relevant company-specific headlines found in last X hours”

---

## Step 9 — Rendering guidance for PM output
In email/slack, show:
- Driver category + weight
- One evidence line:
  - `Headline (Source, Time)`
- If none found:
  - “No relevant company-specific headlines found (checked last 72h)”
- If not checked:
  - “News not checked (feed unavailable)”

Keep it **1–2 lines** with a link out.

---

## Step 10 — Logging & feedback loop (moat)
Persist per alert:
- raw news ids used
- clusters + representative chosen
- classification + scores
- what was shown to PM
- PM feedback

This enables threshold tuning + auditing.

---

## Common failure modes to avoid
- Duplicate story spam (fix via clustering)
- Attributing moves to recycled “why it moved” articles (filter/downrank)
- Confusing sector news for company news (taxonomy + gating)
- Old news causing misattribution (recency penalty)
- LLM inventing details (strict evidence payload)

---

## Minimal viable implementation checklist
- [ ] Headline ingestion + timestamps
- [ ] De-dupe clustering
- [ ] Event type classifier (keyword baseline)
- [ ] Relevance scoring
- [ ] Evidence payload (`news_evidence`)
- [ ] Attribution engine gating
- [ ] Output includes checked/missing transparency
