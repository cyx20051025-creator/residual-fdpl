#!/usr/bin/env python3
"""Data-free CPU smoke test for the public core API."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvfdpl.smoke import main  # noqa: E402

if __name__ == "__main__":
    main()
