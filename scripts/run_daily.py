#!/usr/bin/env python3
"""Cron-compatible wrapper script for daily brief generation.

Usage:
    python scripts/run_daily.py
    python scripts/run_daily.py --config /path/to/config.yaml
    python scripts/run_daily.py --date 2024-01-15

Cron example (run at 6:30 PM US/Eastern on weekdays):
    30 18 * * 1-5 cd /path/to/watchbrief && python scripts/run_daily.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from watchbrief.cli import main

if __name__ == "__main__":
    # Default to run-daily if no command specified
    if len(sys.argv) == 1:
        sys.argv.append("run-daily")
    elif sys.argv[1].startswith("--"):
        # If first arg is a flag, insert run-daily command
        sys.argv.insert(1, "run-daily")

    main()
