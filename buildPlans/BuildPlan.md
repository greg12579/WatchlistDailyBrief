# Project: Watchlist Daily Brief v0.1 (PM change-detection + explanation)

## Goal
Build a minimal system that:
1) Loads a fixed watchlist.
2) Computes "change triggers" for each ticker (price/volume/relative).
3) Pulls a small set of structured events (earnings + analyst actions if available).
4) Produces a DAILY brief with max 5 items: What changed, Why likely, Confidence, Actionability label.
5) Sends the brief by Email OR Slack webhook (configurable).
6) Captures feedback (👍/👎 + optional "decision impact") and stores it in SQLite.

IMPORTANT: The numbers decide what triggers. The LLM ONLY explains using provided facts.
Do NOT build a web UI. Do NOT build intraday. Do NOT build portfolios.

## Tech constraints
- Language: Python 3.11+
- Framework: FastAPI only if needed; otherwise plain scripts are fine.
- Storage: SQLite (via SQLAlchemy or sqlite3)
- Config: YAML file (watchlist + thresholds + delivery config)
- Scheduling: cron-compatible CLI entrypoint
- LLM: OpenAI or Anthropic (make provider pluggable via env vars); default to Anthropic if present.
- Data: Use ONE market data provider (Polygon OR Tiingo OR Alpaca OR Yahoo via yfinance). Prefer yfinance for v0.1 simplicity.
- Sector mapping: Use SPDR sector ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY) as proxies. For v0.1, map tickers to sector ETFs via a static mapping file `sector_map.csv` that I can fill in manually.

## Repo structure
watchbrief/
  README.md
  requirements.txt / pyproject.toml
  config.yaml.example
  sector_map.csv.example
  watchbrief/
    __init__.py
    config.py
    data/
      market_data.py
      events.py
    features/
      triggers.py
      ranking.py
    llm/
      client.py
      prompt.py
      explain.py
    output/
      renderer.py
      emailer.py
      slack.py
    storage/
      db.py
      models.py
      feedback.py
    cli.py
  scripts/
    run_daily.py

## Config format (YAML)
Create `config.yaml.example`:
watchlist:
  - AAPL
  - MSFT
thresholds:
  lookback_days: 30
  vol_lookback_days: 20
  price_move_z: 1.5
  volume_multiple: 1.5
  rel_move_vs_sector_z: 1.2
  rel_move_vs_index_z: 1.5
  max_items: 5
  index_ticker: SPY
delivery:
  mode: email  # email or slack
  email:
    smtp_host: smtp.gmail.com
    smtp_port: 587
    from_addr: ...
    to_addrs: ["..."]
  slack:
    webhook_url: "..."
llm:
  provider: anthropic  # anthropic or openai
  model: "claude-3-5-sonnet-latest" # or GPT model if openai
  temperature: 0.2
storage:
  sqlite_path: "./watchbrief.sqlite"

## Data retrieval requirements
### Market data
Implement `market_data.get_ohlcv(ticker, start_date, end_date)` returning a pandas DataFrame with:
- date (index)
- open, high, low, close, volume

Use yfinance for v0.1. Handle missing days (weekends/holidays).
Compute daily returns from close-to-close.

### Sector ETF mapping
Read `sector_map.csv` with columns: ticker, sector_etf
If missing mapping for a ticker, skip relative-to-sector computations but still compute absolute triggers.

### Events
Implement `events.get_events(ticker, start_date, end_date)` returning a list of structured events:
- type: "earnings" | "analyst" | "other"
- date
- title
- source (string)
For v0.1, implement ONLY earnings dates via yfinance calendar if possible; otherwise return empty.
Do NOT scrape the web.

## Trigger computation (core logic)
Implement in `features/triggers.py` a function:
compute_triggers(ticker, df, spy_df, sector_df_or_none, thresholds) -> dict

Compute:
1) price_move_z:
   - daily return r0 for most recent day
   - zscore = (r0 - mean(r[-vol_lookback_days:])) / std(r[-vol_lookback_days:])
   - trigger if abs(zscore) >= price_move_z
2) volume_multiple:
   - vol0 / mean(volume[-lookback_days:])
   - trigger if >= volume_multiple
3) rel_move_vs_index_z:
   - rel_ret = r_stock - r_spy for most recent day
   - compute zscore over vol_lookback_days
   - trigger if abs(z) >= rel_move_vs_index_z
4) rel_move_vs_sector_z (if sector df present):
   - rel_ret = r_stock - r_sector
   - compute zscore over vol_lookback_days
   - trigger if abs(z) >= rel_move_vs_sector_z
Also compute:
- last_close, pct_change_1d, pct_change_5d
- volume_multiple value
- triggered: bool
- triggered_reasons: list of strings with values embedded e.g. "price_z=2.1", "vol=2.3x", "rel_vs_spy_z=1.6"

Store all numeric values in the output dict.

## Ranking
In `features/ranking.py` implement a deterministic ranking score:
score = 0
+ abs(price_z) * 2
+ max(0, volume_multiple-1) * 1
+ abs(rel_vs_spy_z) * 1.5
+ abs(rel_vs_sector_z) * 1.5 (if exists)
Sort descending. Take top `max_items`.

## Actionability label (rule-based)
In triggers output, compute:
- label = "ACTIONABLE" if (abs(price_z) >= 2.0 and (abs(rel_vs_spy_z) >= 1.5 or abs(rel_vs_sector_z)>=1.5)) OR volume_multiple>=2.0
- label = "MONITOR" if triggered but not actionable
- label = "IGNORE" if not triggered (don’t include in brief)

This is deterministic. The LLM must not override it.

## LLM explanation (strict, fact-grounded)
### Prompt design
Create `llm/prompt.py` with a function build_prompt(item, events, context) that produces:
- a JSON blob of facts (all numeric values + sector ETF + SPY move + event list)
- instructions:
  - Do not invent facts.
  - If no events, say "No specific company event detected; likely macro/sector/flow."
  - Provide:
    1) Ranked likely drivers (3 bullets max) with approximate % weights that sum to 100 (can be coarse).
    2) Confidence: Low/Medium/High based on presence of clear events and strength of relative moves.
    3) One sentence "Why it matters" framed for a PM (no trade advice).

Return output MUST be JSON with keys:
{
  "drivers": [{"rank":1,"text":"...","weight_pct":60}, ...],
  "confidence": "Low|Medium|High",
  "why_it_matters": "..."
}

### Enforcer
In `llm/explain.py`, validate JSON parse; if invalid, retry once with a "return valid JSON only" message.
If still invalid, fall back to a templated explanation without LLM.

## Rendering the daily brief
In `output/renderer.py` create:
render_email(subject, items) -> (subject, html_body, text_body)
Each item section must include:
- Ticker + label emoji: ACTIONABLE 🔴 / MONITOR 🟡
- "What changed" bullets from numeric facts
- "Why (ranked)" from LLM drivers
- Confidence
- Why it matters
- Feedback links (see below)

Cap at 5 items.

## Feedback capture mechanism (no UI)
Implement feedback via unique links that hit a tiny FastAPI server OR a simple local endpoint.
Preferred v0.1: FastAPI server with 2 endpoints:
GET /f/{brief_id}/{ticker}/{vote} where vote in {up,down}
GET /d/{brief_id}/{ticker}/{impact} where impact in {yes,no}
These endpoints log to SQLite and return a simple "Thanks" HTML page.

If you cannot run a server, implement feedback via email reply keywords; but default to FastAPI.

## Storage (SQLite)
Create tables:
- briefs(id, date_sent, subject, delivery_mode)
- brief_items(id, brief_id, ticker, label, rank, score, facts_json, llm_json)
- feedback(id, brief_item_id, vote, impact, created_at)

Use SQLAlchemy models for simplicity.

## Delivery
### Email
Implement SMTP sender in `output/emailer.py`. Support STARTTLS.
### Slack
Implement webhook post in `output/slack.py` (simple JSON payload, text only).

## CLI entrypoint
Create `cli.py` with commands:
- run-daily --date YYYY-MM-DD (default today in US/Eastern)
- serve-feedback --host 0.0.0.0 --port 8000
- init-db

`scripts/run_daily.py` should call the package CLI for cron.

## Daily run flow
1) Load config + watchlist
2) Pull SPY data for lookback range
3) For each ticker:
   - pull data
   - pull sector ETF data if mapping exists
   - compute triggers
   - if triggered: compute ranking score
4) Select top max_items
5) For each selected item:
   - fetch events
   - call LLM explain (strict JSON)
   - store brief + items in SQLite
6) Render brief
7) Send via configured mode
8) Print summary to stdout

## Tests / sanity checks
Add minimal pytest tests:
- trigger computation zscore doesn’t crash with short history
- ranking stable
- LLM JSON validator works
- renderer outputs expected sections

## Non-goals (do not implement)
- No user accounts
- No multi-watchlist
- No intraday
- No backtesting
- No scraping
- No portfolio imports
- No “AI agent” that trades
- No charts beyond text numbers

## Deliverables
- Working `run-daily` that generates and sends a brief
- Working `serve-feedback` that logs thumbs up/down
- SQLite DB file created and populated
- README with setup steps + env vars + cron example
