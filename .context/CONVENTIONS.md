# Watchlist Daily Brief - Coding Conventions

## Python Style

- Python 3.10+ features (type hints, dataclasses, match statements)
- Type hints on all public functions
- Docstrings for modules and public functions
- `from __future__ import annotations` not required (3.10+)

## Project Structure

- One module per concern (events, triggers, attribution, etc.)
- Dataclasses for structured data transfer
- Enums for fixed categories
- No global state; pass config/context explicitly

## Naming

- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

## Error Handling

- Use `Optional[T]` for nullable returns
- Return tuples like `(result, success_bool)` for operations that can fail
- Print errors to console, don't raise unless fatal
- Graceful degradation (fallback explanations, empty lists)

## Type Patterns

```python
# Dataclass for structured data
@dataclass
class TriggerResult:
    ticker: str
    triggered: bool
    price_z: float
    # ...

# Enum for fixed categories
class EventType(str, Enum):
    EARNINGS = "EARNINGS"
    MNA = "MNA"
    # ...

# Optional with default
def process(data: Optional[dict] = None) -> bool:
    # ...

# Return with success flag
def fetch_data(url: str) -> tuple[list[dict], bool]:
    # Returns (data, success)
```

## Import Order

1. Standard library
2. Third-party packages
3. Local imports
4. TYPE_CHECKING imports at end

```python
import os
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import pandas as pd
import requests

from watchbrief.config import Config
from watchbrief.data.events import Event

if TYPE_CHECKING:
    from watchbrief.data.news_processor import NewsEvidence
```

## Configuration

- YAML for static config
- Environment variables for secrets
- CSV for data mappings
- Excel for watchlist (user-provided)

## Testing Patterns

- `pytest` for all tests
- Test files mirror source structure: `tests/test_triggers.py`
- Fixtures for common test data
- Mock external APIs (yfinance, news APIs)

## Console Output

- Use `print()` for user-facing output
- ASCII characters only (Windows compatibility)
- Progress indicators: `Processed 50/293...`
- Markers: `[!]` for ACTIONABLE, `[*]` for MONITOR

## Database

- SQLAlchemy ORM with declarative base
- Session context manager: `with session_scope() as session:`
- JSON columns for flexible structured data
- UTC timestamps for all dates

## API Clients

- 10 second timeout on all HTTP requests
- User-Agent headers for SEC EDGAR
- Rate limiting awareness (Finnhub 60/min, NewsAPI 100/day)
- API key via environment variables

## Transparency

- Always track what data sources were checked
- Disclose missing checks in output
- Never claim certainty; use "likely associated with"
- Show evidence for all attributions
