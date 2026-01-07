# Watchlist Daily Brief - Architecture Context

## Overview

Watchlist Daily Brief is a Python application that monitors a watchlist of ~375 stock tickers, detects significant price/volume changes, explains them using LLM-powered attribution, and delivers daily briefs via email or Slack.

The system is designed for portfolio managers (PMs) who need concise, fact-grounded explanations for why stocks moved, not just that they moved.

## Core Philosophy

1. **PM-Grade Attribution**: Explanations must be grounded in facts, not LLM hallucinations
2. **Transparency**: Always disclose what data sources were/weren't checked
3. **Hierarchical Attribution**: Company-specific > Sector/Peer > Macro > Flow > Unattributed
4. **No Invented Facts**: LLM can only use evidence from the structured payload

---

## Directory Structure

```
watchbrief/
├── config.py           # Configuration loading (YAML, Excel watchlist, CSV mappings)
├── cli.py              # Main CLI entrypoint (run-daily, serve-feedback, init-db)
├── data/
│   ├── market_data.py  # yfinance OHLCV fetching with caching
│   ├── events.py       # Earnings, news, SEC filings fetching
│   └── news_processor.py # News clustering, classification, relevance scoring
├── features/
│   ├── triggers.py     # Z-score computation, trigger detection
│   ├── ranking.py      # Scoring and selection of top movers
│   └── attribution.py  # Attribution engine (builds context for LLM)
├── llm/
│   ├── client.py       # Pluggable LLM client (OpenAI/Anthropic)
│   ├── prompt.py       # Prompt construction with structured facts
│   └── explain.py      # Explanation generation and parsing
├── output/
│   ├── renderer.py     # Email/Slack formatting with transparency
│   ├── emailer.py      # SMTP email sending
│   └── slack.py        # Slack webhook integration
└── storage/
    ├── db.py           # SQLite session management
    ├── models.py       # SQLAlchemy models (Brief, BriefItem, Feedback)
    └── feedback.py     # Feedback recording functions
```

---

## Data Flow

```
1. Load Config (YAML + Excel watchlist + CSV mappings)
              ↓
2. Fetch Market Data (yfinance: OHLCV for all tickers + SPY + sector ETFs)
              ↓
3. Compute Triggers (z-scores, volume multiples, relative performance)
              ↓
4. Rank & Select Top N (deterministic scoring formula)
              ↓
5. For each triggered ticker:
   ├── Fetch Events (earnings, SEC filings)
   ├── Fetch & Process News (Polygon.io → cluster → classify → score)
   ├── Build Attribution Context (sector, peers, macro, events, news_evidence)
   └── Generate Explanation (LLM with structured facts)
              ↓
6. Store in SQLite (Brief + BriefItems)
              ↓
7. Render & Deliver (Email HTML/text or Slack webhook)
```

---

## Key Components

### 1. Trigger Detection (`features/triggers.py`)

Computes statistical measures to detect significant moves:

- **Price Z-score**: How unusual is today's move vs historical volatility?
- **Volume Multiple**: How does today's volume compare to 20-day average?
- **Relative vs SPY**: Is this stock-specific or market-wide?
- **Relative vs Sector**: Is this sector-wide or company-specific?

**Thresholds** (configurable):
- `price_z_threshold`: 2.0 (default)
- `volume_threshold`: 2.0x (default)
- `relative_z_threshold`: 1.5 (default)

**Labels**:
- `ACTIONABLE`: Strong signal, likely needs attention
- `MONITOR`: Moderate signal, worth watching
- `IGNORE`: Below thresholds

### 2. News Processing (`data/news_processor.py`)

Implements the news processing spec for PM-grade attribution:

**Pipeline**:
1. **Fetch** headlines from Polygon.io (or Finnhub/NewsAPI fallback)
2. **Filter** low-signal content ("Why XYZ is up", "Top movers", etc.)
3. **Cluster** similar headlines (Jaccard + sequence matching)
4. **Classify** into event types:
   - Company: EARNINGS, GUIDANCE, MNA, FINANCING, REGULATORY_LEGAL, PRODUCT, MANAGEMENT, CONTRACT_CUSTOMER, INSIDER_OWNERSHIP
   - Non-company: ANALYST_ACTION, SECTOR, MACRO, OTHER, LOW_SIGNAL
5. **Score** relevance (recency, specificity, source authority, cluster size)
6. **Select** top 3 clusters above threshold

**Source Authority** (0-1 scale):
- Tier 1 (1.0): Reuters, Bloomberg, WSJ, AP
- Tier 2 (0.9): PR Newswire, Business Wire, SEC EDGAR
- Tier 3 (0.75-0.80): CNBC, MarketWatch, Barron's
- Lower tiers: Seeking Alpha, Benzinga, Motley Fool, etc.

**Output**: `NewsEvidence` dataclass with:
- `checked`: bool - was news API available?
- `lookback_hours`: int - how far back we looked
- `top_clusters`: list of classified, scored clusters
- `no_company_specific_catalyst_found`: bool - transparency flag

### 3. Attribution Engine (`features/attribution.py`)

Builds structured context for LLM explanation:

**Attribution Categories** (priority order):
1. **Company**: Earnings, news, SEC filings
2. **Sector/Peer**: Sector ETF or peer group moving similarly
3. **Macro**: Broad market (SPY) correlation
4. **Flow**: Technical/volume-driven, no fundamental catalyst
5. **Unattributed**: Cannot determine cause

**Context includes**:
- `trigger_result`: All computed metrics
- `sector`: SectorContext (ETF, 1d/5d returns)
- `peers`: List of PeerContext
- `macro`: MacroContext (SPY returns)
- `events`: Raw events list
- `news_evidence`: Processed NewsEvidence
- `catalyst_checks`: What data sources were checked
- `attribution_hints`: Pre-computed likely attributions

### 4. LLM Explanation (`llm/explain.py`)

**Hard Constraints** (from spec):
- LLM can ONLY use facts from the structured context
- Must NOT infer details not present
- Must use "likely associated with" not certainty
- Must disclose if news not checked or no catalyst found

**Output**: `Explanation` dataclass with:
- `drivers`: Ranked list with category, weight, evidence
- `confidence`: high/medium/low
- `why_it_matters`: PM-focused insight
- `missing_checks`: Data sources not available

### 5. Phase 2: Trend Context (`features/trend_context.py`)

**Purpose**: Provide global context for the move, separate from Phase 1 attribution.
- Phase 1 explains "why" the stock moved (causes)
- Phase 2 explains "where" this move sits historically (context)

**Key Constraints**:
- Quantitative-only input (no news/events)
- No causality claims
- No trading advice
- Deterministic market state labels

**Market State Taxonomy**:
- `BREAKOUT_UP/DOWN`: New 52w high/low with high volume
- `EXTENDED_RALLY/SELLOFF`: >2 std dev from mean
- `RECOVERY_BOUNCE`: Up >15% from 52w low
- `PULLBACK_FROM_HIGH`: Down >10% from 52w high
- `RANGE_BOUND_HIGH/MID/LOW`: Position in 52w range
- `TREND_CONTINUATION`: Strong trend, not at extreme
- `VOLATILITY_SPIKE`: Current vol >2x average

**Output**: `TrendContext` with multi-horizon returns, z-scores, relative metrics, 52-week positioning.

### 6. Rendering (`output/renderer.py`)

**Transparency Requirements**:
- If news not checked: "News not checked (feed unavailable)"
- If checked but no catalyst: "No relevant company-specific headlines found (checked last 72h)"
- Show event type classification: `[FINANCING]`, `[MNA]`, `[EARNINGS]`
- Show source and timestamp for each headline
- Mark weak evidence: "(weak)"

**Phase 2 Trend Context Section**:
- Green accent (#e8f5e9) to differentiate from Phase 1
- Shows market state label, 52-week positioning
- Multi-horizon returns and z-scores
- Relative performance vs SPY and sector

**Formats**:
- Email: HTML + plain text
- Slack: Block Kit with mrkdwn

---

## Configuration

### `config.yaml`
```yaml
watchlist_file: "watchlist.xlsx"  # Excel with tickers

thresholds:
  price_z_threshold: 2.0
  volume_threshold: 2.0
  lookback_days: 20
  max_items: 5

delivery:
  mode: "email"  # or "slack"
  email:
    recipients: ["pm@example.com"]
    smtp_host: "smtp.gmail.com"

llm:
  provider: "openai"  # or "anthropic"
  model: "gpt-4o-mini"
```

### Environment Variables
```bash
OPENAI_API_KEY=...         # Required for LLM
ANTHROPIC_API_KEY=...      # Alternative LLM
POLYGON_API_KEY=...        # News API (recommended)
FINNHUB_API_KEY=...        # Alternative news
NEWSAPI_KEY=...            # Alternative news
SMTP_USER=...              # Email delivery
SMTP_PASSWORD=...          # Email delivery
```

### CSV Mappings
- `sector_map.csv`: ticker → sector ETF (e.g., AAPL → XLK)
- `peer_map.csv`: ticker → peer1, peer2, peer3

---

## Database Schema

```sql
-- Brief: One per daily run
briefs:
  id, date_sent, subject, delivery_mode

-- BriefItem: One per triggered ticker
brief_items:
  id, brief_id, ticker, label, rank, score, facts_json, llm_json

-- Feedback: PM feedback on alerts
feedback:
  id, brief_item_id, vote (up/down), impact (yes/no), created_at
```

---

## CLI Commands

```bash
# Run daily brief for today
python -m watchbrief.cli run-daily

# Run for specific date
python -m watchbrief.cli run-daily --date 2026-01-07

# Start feedback server
python -m watchbrief.cli serve-feedback --port 8000

# Initialize database
python -m watchbrief.cli init-db
```

---

## Key Design Decisions

1. **Headlines-First News**: Only fetch headlines + timestamps + source, not full articles
2. **Deterministic Scoring**: No LLM in trigger detection or ranking
3. **Structured LLM Input**: All facts computed before LLM call
4. **Fallback Explanations**: Template-based if LLM unavailable
5. **Feedback Loop**: Persist all evidence for threshold tuning
6. **Windows Compatible**: ASCII markers instead of emoji in console output

---

## Extension Points

1. **News Sources**: Add new APIs in `data/events.py` (`_get_*_news` functions)
2. **Event Types**: Extend `EventType` enum in `news_processor.py`
3. **Attribution Rules**: Modify `compute_attribution_hints` in `attribution.py`
4. **Delivery Channels**: Add new renderers in `output/`
5. **LLM Providers**: Extend `create_llm_client` in `llm/client.py`
