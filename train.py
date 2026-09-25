#!/usr/bin/env python3
"""Run the canonical Residual FDPL training pipeline from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    """Dispatch to the packaged CLI after adding the source tree."""

    from cvfdpl.training.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
