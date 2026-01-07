# Watchlist Daily Brief - Data Flow Reference

## High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DAILY RUN PIPELINE                          │
└─────────────────────────────────────────────────────────────────────┘

1. CONFIG LOADING
   config.yaml ──┬──> Config dataclass
   watchlist.xlsx──┤   ├── watchlist: list[str] (293 tickers)
   sector_map.csv ─┤   ├── sector_map: dict[ticker, ETF]
   peer_map.csv ───┘   └── peer_map: dict[ticker, list[peers]]

2. MARKET DATA FETCH (yfinance)
   SPY ─────────────> spy_df (OHLCV + returns)
   Sector ETFs ─────> sector_dfs: dict[ETF, DataFrame]
   Peer tickers ────> peer_dfs: dict[ticker, DataFrame]
   Watchlist ───────> per-ticker DataFrames (on demand)

3. TRIGGER COMPUTATION (for each ticker)
   ticker_df + spy_df + sector_df + thresholds
                  ↓
   TriggerResult:
   ├── ticker, triggered, label (ACTIONABLE/MONITOR)
   ├── price_z, volume_multiple
   ├── pct_change_1d, pct_change_5d
   ├── rel_vs_spy_z, rel_vs_sector_z
   └── triggered_reasons: list[str]

4. RANKING & SELECTION
   All TriggerResults where triggered=True
                  ↓
   compute_score(result) → float
                  ↓
   Sort by score descending
                  ↓
   Top N (default 5) selected

5. FOR EACH SELECTED TICKER:

   5a. EVENTS FETCH
       ├── get_events() → earnings from yfinance
       ├── get_news() → Polygon.io / Finnhub / NewsAPI
       └── get_sec_filings() → SEC EDGAR direct

       Returns: (events: list[Event], news_checked: bool, sec_checked: bool)

   5b. NEWS PROCESSING
       Raw headlines
            ↓
       filter low-signal → cluster → classify → score
            ↓
       NewsEvidence:
       ├── checked: bool
       ├── lookback_hours: int
       ├── top_clusters: list[NewsCluster]
       └── no_company_specific_catalyst_found: bool

   5c. ATTRIBUTION CONTEXT BUILD
       Inputs: trigger_result, events, spy_df, sector_df, peer_data, news_evidence
            ↓
       AttributionContext:
       ├── sector: SectorContext (ETF, 1d/5d returns)
       ├── peers: list[PeerContext]
       ├── macro: MacroContext (SPY returns)
       ├── events: list[Event]
       ├── catalyst_checks: CatalystChecks
       ├── attribution_hints: list[AttributionHint]
       └── news_evidence: NewsEvidence

   5d. LLM EXPLANATION
       AttributionContext → build_prompt() → LLM call → parse response
            ↓
       Explanation:
       ├── drivers: list[Driver] (ranked, with category/weight/evidence)
       ├── confidence: str
       ├── why_it_matters: str
       └── missing_checks: list[str]

   5e. BUILD BRIEF ITEM
       BriefItem:
       ├── ticker, rank, score
       ├── result: TriggerResult
       ├── explanation: Explanation
       ├── events: list[Event]
       └── news_evidence: NewsEvidence

6. STORAGE
   Brief (id, date_sent, subject, delivery_mode)
      └── BriefItems (ticker, label, rank, score, facts_json, llm_json)

7. RENDER & DELIVER
   BriefItems → render_email() or render_slack()
             → send_email() or send_slack()
```

---

## Data Structures

### TriggerResult
```python
@dataclass
class TriggerResult:
    ticker: str
    triggered: bool
    label: str  # "ACTIONABLE" | "MONITOR" | "IGNORE"

    # Price metrics
    last_close: float
    pct_change_1d: float
    pct_change_5d: float
    price_z: float

    # Volume metrics
    volume_multiple: float

    # Relative metrics
    rel_vs_spy_z: float
    rel_vs_sector_z: Optional[float]
    sector_etf: Optional[str]

    # Trigger info
    triggered_reasons: list[str]
```

### Event
```python
@dataclass
class Event:
    type: str  # "earnings" | "news" | "sec_filing"
    date: date
    title: str
    source: str
    url: Optional[str]
```

### NewsCluster
```python
@dataclass
class NewsCluster:
    cluster_id: int
    event_type: EventType  # EARNINGS, MNA, FINANCING, etc.
    event_confidence: float
    relevance_score: float
    headline: str
    source: str
    published_at: datetime
    url: Optional[str]
    weak_evidence: bool
    cluster_size: int
    keywords_matched: list[str]
    relevance_rationale: list[str]
```

### NewsEvidence
```python
@dataclass
class NewsEvidence:
    checked: bool
    lookback_hours: int
    top_clusters: list[NewsCluster]
    no_company_specific_catalyst_found: bool
```

### AttributionContext
```python
@dataclass
class AttributionContext:
    ticker: str
    trigger_result: TriggerResult
    sector: Optional[SectorContext]
    peers: list[PeerContext]
    macro: Optional[MacroContext]
    events: list[Event]
    catalyst_checks: CatalystChecks
    attribution_hints: list[AttributionHint]
    news_evidence: Optional[NewsEvidence]
```

### Explanation
```python
@dataclass
class Explanation:
    drivers: list[Driver]  # rank, text, weight_pct, category, evidence
    confidence: str
    why_it_matters: str
    missing_checks: list[str]
```

### TrendContext (Phase 2)
```python
@dataclass
class TrendContext:
    ticker: str
    returns: TrendReturns      # pct_1d, pct_5d, pct_21d, pct_63d, pct_126d, pct_252d
    z: TrendZScores            # z_5d, z_21d, z_63d, z_126d, z_252d
    relative: RelativeMetrics  # vs_spy_z_63d, vs_spy_z_252d, vs_sector_z_*
    extremes: ExtremeMetrics   # pct_from_52w_high/low, days_since_*
    market_state: MarketState  # Enum: BREAKOUT_UP, RANGE_BOUND_MID, etc.
    market_state_rationale: str
    vol_20d_annualized: Optional[float]
    vol_regime: Optional[str]  # "low", "normal", "elevated", "extreme"
```

### BriefItem
```python
@dataclass
class BriefItem:
    ticker: str
    rank: int
    score: float
    result: TriggerResult
    explanation: Explanation
    events: list[Event]
    news_evidence: Optional[NewsEvidence]
    trend_context: Optional[TrendContext]    # Phase 2
    phase2_response: Optional[Phase2Response] # Phase 2 LLM output
```

---

## External API Interactions

### yfinance
- `yf.Ticker(symbol).history()` → OHLCV DataFrame
- `yf.Ticker(symbol).calendar` → Earnings dates
- Caching: In-memory during run

### Polygon.io News
```
GET https://api.polygon.io/v2/reference/news
?ticker=AAPL
&published_utc.gte=2026-01-04
&order=desc
&limit=10
&apiKey=...
```

### SEC EDGAR
```
GET https://data.sec.gov/submissions/CIK{cik}.json
Headers: User-Agent: WatchlistDailyBrief/1.0

# CIK lookup
GET https://www.sec.gov/files/company_tickers.json
```

### OpenAI / Anthropic
- Model: gpt-4o-mini (default) or claude-3-haiku
- Temperature: 0.2
- Structured JSON output expected

---

## Database Tables

```sql
CREATE TABLE briefs (
    id INTEGER PRIMARY KEY,
    date_sent TIMESTAMP,
    subject TEXT,
    delivery_mode TEXT
);

CREATE TABLE brief_items (
    id INTEGER PRIMARY KEY,
    brief_id INTEGER REFERENCES briefs(id),
    ticker TEXT,
    label TEXT,
    rank INTEGER,
    score REAL,
    facts_json TEXT,  -- TriggerResult as JSON
    llm_json TEXT     -- Explanation as JSON
);

CREATE TABLE feedback (
    id INTEGER PRIMARY KEY,
    brief_item_id INTEGER REFERENCES brief_items(id),
    vote TEXT,        -- "up" | "down"
    impact TEXT,      -- "yes" | "no"
    created_at TIMESTAMP
);
```

---

## Output Formats

### Email HTML Structure
```html
<div class="item actionable">
  <div class="ticker">AAPL <span class="label">[!] ACTIONABLE</span></div>
  <div class="metrics">What Changed: Price +5.2%...</div>
  <div class="events">
    Recent News & Filings:
    [EARNINGS] 01/07: AAPL reports Q1...
    [SEC FILING] 01/06: 8-K: Current Report
  </div>
  <div class="drivers">
    Why (Ranked):
    1. [Company] Earnings beat (45%)
    2. [Sector/Peer] Tech sector rally (30%)
  </div>
  <div class="confidence">Confidence: high</div>
  <div class="why-matters">Why It Matters: ...</div>
  <div class="feedback">[+] Helpful [-] Not helpful</div>
</div>
```

### Slack Block Kit
```json
{
  "blocks": [
    {"type": "header", "text": {"type": "plain_text", "text": "Watchlist Brief"}},
    {"type": "section", "text": {"type": "mrkdwn", "text": "*[!] AAPL*..."}},
    {"type": "actions", "elements": [
      {"type": "button", "text": {"type": "plain_text", "text": "[+]"}, "url": "..."}
    ]}
  ]
}
```
