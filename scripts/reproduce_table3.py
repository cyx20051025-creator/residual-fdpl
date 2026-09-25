#!/usr/bin/env python3
"""Source-checkout wrapper for ``cvfdpl.table3``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from cvfdpl.table3 import main as table3_main

    table3_main()


if __name__ == "__main__":
    main()
