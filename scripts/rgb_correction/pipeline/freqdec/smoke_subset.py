"""Create a small deterministic HDF5 subset for remote smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows <= 0:
        raise ValueError("rows must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.source, "r") as source, h5py.File(
        args.output, "w"
    ) as destination:
        rows = min(args.rows, int(source["noisy"].shape[0]))
        for key in ("noisy", "gt", "source_index"):
            if key not in source:
                continue
            destination.create_dataset(
                key,
                data=source[key][:rows],
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
        destination.attrs["source"] = str(args.source)
        destination.attrs["rows"] = rows
    print(args.output)


if __name__ == "__main__":
    main()
