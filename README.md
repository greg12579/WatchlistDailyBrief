# Watchlist Daily Brief

A minimal system that monitors a stock watchlist, detects significant price/volume changes, and delivers daily briefs with LLM-powered explanations.

## Features

- **Change Detection**: Monitors price z-scores, volume multiples, and relative performance vs SPY/sector
- **LLM Explanations**: Uses Claude or GPT to explain why stocks moved (fact-grounded, no speculation)
- **Actionability Labels**: Deterministic ACTIONABLE/MONITOR labels based on signal strength
- **Flexible Delivery**: Email (SMTP) or Slack webhook
- **Feedback Tracking**: Capture thumbs up/down and decision impact via embedded links

## Quick Start

### 1. Install

```bash
cd watchbrief
pip install -e .
```

### 2. Configure

Copy and edit the config file:

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` with your settings:
- Point `watchlist_file` to your Excel watchlist
- Set delivery mode (email or slack)
- Configure SMTP or Slack webhook

### 3. Set Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
# LLM API (at least one required)
ANTHROPIC_API_KEY=your_key_here
# or
OPENAI_API_KEY=your_key_here

# For email delivery
SMTP_PASSWORD=your_app_password
```

Or export them directly:

```bash
export ANTHROPIC_API_KEY=your_key_here
export SMTP_PASSWORD=your_app_password
```

### 4. Initialize Database

```bash
watchbrief init-db --config config.yaml
```

### 5. Run Daily Brief

```bash
watchbrief run-daily --config config.yaml
# or for a specific date
watchbrief run-daily --date 2024-01-15 --config config.yaml
```

### 6. Start Feedback Server (Optional)

```bash
watchbrief serve-feedback --host 0.0.0.0 --port 8000
```

## Configuration

### config.yaml

```yaml
# Watchlist source (Excel file with Bloomberg-style tickers)
watchlist_file: "watchlist.xlsx"

# Trigger thresholds
thresholds:
  lookback_days: 30
  vol_lookback_days: 20
  price_move_z: 1.5        # Z-score threshold for price moves
  volume_multiple: 1.5      # Volume vs average threshold
  rel_move_vs_sector_z: 1.2 # Relative vs sector threshold
  rel_move_vs_index_z: 1.5  # Relative vs SPY threshold
  max_items: 5              # Max items in brief
  index_ticker: SPY

# Delivery
delivery:
  mode: email  # or "slack"
  email:
    smtp_host: smtp.gmail.com
    smtp_port: 587
    from_addr: your-email@gmail.com
    to_addrs:
      - recipient@example.com
  slack:
    webhook_url: "https://hooks.slack.com/..."

# LLM
llm:
  provider: anthropic  # or "openai"
  model: "claude-sonnet-4-20250514"
  temperature: 0.2

# Storage
storage:
  sqlite_path: "./watchbrief.sqlite"

# Feedback server
feedback:
  base_url: "http://localhost:8000"
```

### Sector Mapping

Create `sector_map.csv` for sector-relative analysis:

```csv
ticker,sector_etf
AAPL,XLK
MSFT,XLK
AMZN,XLY
...
```

Available SPDR sector ETFs: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY

## Ticker Format

The watchlist Excel file should have Bloomberg-style tickers:
- `AAPL US` → `AAPL` (US stocks)
- `ADYEN NA` → `ADYEN.AS` (Amsterdam)
- `FLTR LN` → `FLTR.L` (London)

Supported exchanges: US, NA, LN, AU, SS, CN, CT, FP, LI

## Cron Setup

Run daily at 6:30 PM Eastern (after market close):

```bash
30 18 * * 1-5 cd /path/to/watchbrief && python scripts/run_daily.py >> /var/log/watchbrief.log 2>&1
```

## Architecture

```
watchbrief/
├── config.py          # Config loading, ticker conversion
├── data/
│   ├── market_data.py # yfinance OHLCV retrieval
│   └── events.py      # Earnings dates
├── features/
│   ├── triggers.py    # Z-score, volume, relative move computation
│   └── ranking.py     # Scoring and selection
├── llm/
│   ├── client.py      # Anthropic/OpenAI client
│   ├── prompt.py      # Prompt construction
│   └── explain.py     # Explanation generation with fallback
├── output/
│   ├── renderer.py    # HTML/text/Slack formatting
│   ├── emailer.py     # SMTP delivery
│   └── slack.py       # Webhook delivery
├── storage/
│   ├── models.py      # SQLAlchemy models
│   ├── db.py          # Database setup
│   └── feedback.py    # Feedback recording
└── cli.py             # CLI commands
```

## How Triggers Work

A ticker is triggered when ANY of these conditions are met:
1. **Price Z-score** ≥ threshold (default 1.5)
2. **Volume multiple** ≥ threshold (default 1.5x)
3. **Relative vs SPY Z-score** ≥ threshold (default 1.5)
4. **Relative vs Sector Z-score** ≥ threshold (default 1.2)

### Actionability Labels

- **ACTIONABLE** 🔴: Strong signals (price_z ≥ 2.0 AND rel_z ≥ 1.5) OR volume ≥ 2.0x
- **MONITOR** 🟡: Triggered but not actionable
- **IGNORE**: Not triggered (excluded from brief)

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

## Non-Goals

This is an MVP. It does NOT include:
- User accounts or authentication
- Multiple watchlists
- Intraday monitoring
- Backtesting
- Web scraping
- Portfolio management
- Trading recommendations
- Charts or visualizations

## License

MIT
