#!/usr/bin/env python3
"""Backward-compatible wrapper for ``evaluate.py``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from cvfdpl.evaluation import main as evaluation_main

    evaluation_main()


if __name__ == "__main__":
    main()
