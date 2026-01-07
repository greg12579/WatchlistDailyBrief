"""Configuration loading and ticker parsing."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


# Bloomberg suffix to yfinance suffix mapping
EXCHANGE_MAP = {
    "US": "",      # US tickers have no suffix in yfinance
    "NA": ".AS",   # Amsterdam (Euronext)
    "LN": ".L",    # London
    "AU": ".AX",   # Australia
    "SS": ".ST",   # Stockholm
    "CN": ".TO",   # Toronto
    "CT": ".TO",   # Toronto (alternate)
    "FP": ".PA",   # Paris
    "LI": ".L",    # London (Liechtenstein listings often trade in London)
}


@dataclass
class ThresholdsConfig:
    lookback_days: int = 30
    vol_lookback_days: int = 20
    price_move_z: float = 1.5
    volume_multiple: float = 1.5
    rel_move_vs_sector_z: float = 1.2
    rel_move_vs_index_z: float = 1.5
    max_items: int = 5
    index_ticker: str = "SPY"


@dataclass
class EmailConfig:
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    from_addr: str = ""
    to_addrs: list[str] = field(default_factory=list)
    cc_addrs: list[str] = field(default_factory=list)


@dataclass
class SlackConfig:
    webhook_url: str = ""


@dataclass
class DeliveryConfig:
    mode: str = "email"
    email: EmailConfig = field(default_factory=EmailConfig)
    slack: SlackConfig = field(default_factory=SlackConfig)


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.2
    api_key: str = ""  # Can be set here or via env var


@dataclass
class StorageConfig:
    sqlite_path: str = "./watchbrief.sqlite"


@dataclass
class FeedbackConfig:
    base_url: str = "http://localhost:8000"


@dataclass
class Config:
    watchlist_file: str = ""
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    delivery: DeliveryConfig = field(default_factory=DeliveryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)

    # Computed fields
    watchlist: list[str] = field(default_factory=list)
    sector_map: dict[str, str] = field(default_factory=dict)
    peer_map: dict[str, list[str]] = field(default_factory=dict)


def convert_bloomberg_ticker(bloomberg_ticker: str) -> Optional[str]:
    """Convert Bloomberg-style ticker to yfinance format.

    Examples:
        "CVNA US" -> "CVNA"
        "ADYEN NA" -> "ADYEN.AS"
        "FLTR LN" -> "FLTR.L"
        "BRK/B US" -> "BRK-B"

    Returns None if the exchange is not recognized.
    """
    parts = bloomberg_ticker.strip().split()
    if len(parts) < 2:
        # No exchange suffix, assume US
        ticker = bloomberg_ticker.strip()
        exchange = "US"
    else:
        ticker = " ".join(parts[:-1])  # Handle tickers with spaces
        exchange = parts[-1].upper()

    # Handle special characters in ticker
    ticker = ticker.replace("/", "-")  # BRK/B -> BRK-B

    if exchange not in EXCHANGE_MAP:
        return None

    suffix = EXCHANGE_MAP[exchange]
    return f"{ticker}{suffix}"


def load_watchlist_from_excel(filepath: str, base_dir: Path) -> list[str]:
    """Load and parse watchlist from Excel file.

    Converts Bloomberg-style tickers to yfinance format,
    deduplicates, and filters out unrecognized exchanges.
    """
    excel_path = base_dir / filepath
    if not excel_path.exists():
        raise FileNotFoundError(f"Watchlist file not found: {excel_path}")

    df = pd.read_excel(excel_path)

    # Assume first column contains tickers
    ticker_column = df.columns[0]
    raw_tickers = df[ticker_column].dropna().astype(str).tolist()

    converted = []
    skipped = []

    for raw in raw_tickers:
        yf_ticker = convert_bloomberg_ticker(raw)
        if yf_ticker:
            converted.append(yf_ticker)
        else:
            skipped.append(raw)

    if skipped:
        print(f"Skipped {len(skipped)} tickers with unrecognized exchanges: {skipped[:5]}...")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in converted:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    print(f"Loaded {len(unique)} unique tickers from {len(raw_tickers)} raw entries")
    return unique


def load_sector_map(filepath: str, base_dir: Path) -> dict[str, str]:
    """Load sector ETF mapping from CSV file.

    Returns dict mapping ticker -> sector ETF symbol.
    """
    csv_path = base_dir / filepath
    if not csv_path.exists():
        print(f"Sector map not found: {csv_path}, using empty mapping")
        return {}

    df = pd.read_csv(csv_path)
    return dict(zip(df["ticker"], df["sector_etf"]))


def load_peer_map(filepath: str, base_dir: Path) -> dict[str, list[str]]:
    """Load peer ticker mapping from CSV file.

    Returns dict mapping ticker -> list of peer tickers.
    """
    csv_path = base_dir / filepath
    if not csv_path.exists():
        print(f"Peer map not found: {csv_path}, using empty mapping")
        return {}

    df = pd.read_csv(csv_path)
    peer_map = {}

    for _, row in df.iterrows():
        ticker = row["ticker"]
        peers = []
        for col in ["peer1", "peer2", "peer3"]:
            if col in row and pd.notna(row[col]):
                peers.append(row[col])
        if peers:
            peer_map[ticker] = peers

    return peer_map


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file.

    Also loads watchlist from Excel and sector map from CSV.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_dir = config_file.parent

    with open(config_file) as f:
        raw = yaml.safe_load(f)

    # Parse nested configs
    thresholds = ThresholdsConfig(**raw.get("thresholds", {}))

    delivery_raw = raw.get("delivery", {})
    email_config = EmailConfig(**delivery_raw.get("email", {}))
    slack_config = SlackConfig(**delivery_raw.get("slack", {}))
    delivery = DeliveryConfig(
        mode=delivery_raw.get("mode", "email"),
        email=email_config,
        slack=slack_config,
    )

    llm_raw = raw.get("llm", {})
    # Handle OPENAI_API_KEY or ANTHROPIC_API_KEY field names in config
    api_key = llm_raw.pop("OPENAI_API_KEY", "") or llm_raw.pop("ANTHROPIC_API_KEY", "") or llm_raw.get("api_key", "")
    llm = LLMConfig(
        provider=llm_raw.get("provider", "anthropic"),
        model=llm_raw.get("model", "claude-sonnet-4-20250514"),
        temperature=llm_raw.get("temperature", 0.2),
        api_key=api_key,
    )
    storage = StorageConfig(**raw.get("storage", {}))
    feedback = FeedbackConfig(**raw.get("feedback", {}))

    config = Config(
        watchlist_file=raw.get("watchlist_file", ""),
        thresholds=thresholds,
        delivery=delivery,
        llm=llm,
        storage=storage,
        feedback=feedback,
    )

    # Load watchlist from Excel
    if config.watchlist_file:
        config.watchlist = load_watchlist_from_excel(config.watchlist_file, base_dir)

    # Load sector map if exists
    sector_map_path = base_dir / "sector_map.csv"
    if sector_map_path.exists():
        config.sector_map = load_sector_map("sector_map.csv", base_dir)

    # Load peer map if exists
    peer_map_path = base_dir / "peer_map.csv"
    if peer_map_path.exists():
        config.peer_map = load_peer_map("peer_map.csv", base_dir)

    return config
