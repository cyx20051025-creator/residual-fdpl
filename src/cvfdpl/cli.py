"""Command-line entry points."""

from __future__ import annotations


def smoke() -> None:
    """Run the repository smoke script."""

    from cvfdpl.smoke import main

    main()

