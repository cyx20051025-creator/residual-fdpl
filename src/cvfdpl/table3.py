"""Reproduce the paired Table 3 comparison from released checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cvfdpl.evaluation import run_evaluation


@dataclass(frozen=True)
class Table3Row:
    """One row in the main 3-RCAB paired comparison."""

    row: int
    protocol: str
    seed: int
    no_fdpl_asset: str
    fdpl_asset: str
    reported_gain: float


TABLE3_ROWS: tuple[Table3Row, ...] = (
    Table3Row(
        1,
        "sliding",
        42,
        "rcab3_seed42_nofdpl_final.pth",
        "rcab3_seed42_fdpl_final.pth",
        0.6475,
    ),
    Table3Row(
        2,
        "sliding",
        43,
        "rcab3_seed43_nofdpl_final.pth",
        "rcab3_seed43_fdpl_final.pth",
        0.5985,
    ),
    Table3Row(
        3,
        "sliding",
        44,
        "rcab3_seed44_nofdpl_final.pth",
        "rcab3_seed44_fdpl_final.pth",
        0.4101,
    ),
    Table3Row(
        4,
        "direct",
        42,
        "rcab3_seed42_nofdpl_final.pth",
        "rcab3_seed42_fdpl_final.pth",
        0.5995,
    ),
    Table3Row(
        5,
        "direct",
        43,
        "rcab3_seed43_nofdpl_final.pth",
        "rcab3_seed43_fdpl_final.pth",
        0.6017,
    ),
)


def paired_statistics(
    no_fdpl_values: list[float],
    fdpl_values: list[float],
) -> dict[str, float | int | None]:
    """Compute paired deltas and normal-approximation PSNR statistics."""

    if not no_fdpl_values or len(no_fdpl_values) != len(fdpl_values):
        raise ValueError("paired metric lists must be non-empty and have equal length")

    deltas = [
        float(fdpl_value) - float(no_fdpl_value)
        for no_fdpl_value, fdpl_value in zip(no_fdpl_values, fdpl_values, strict=True)
    ]
    mean_delta = statistics.fmean(deltas)
    if len(deltas) == 1:
        std_delta = 0.0
        t_statistic = None if mean_delta != 0 else 0.0
        p_two_sided = 0.0 if mean_delta != 0 else 1.0
    else:
        std_delta = statistics.stdev(deltas)
        if std_delta == 0:
            t_statistic = None if mean_delta != 0 else 0.0
            p_two_sided = 0.0 if mean_delta != 0 else 1.0
        else:
            t_statistic = mean_delta / (std_delta / math.sqrt(len(deltas)))
            p_two_sided = math.erfc(abs(t_statistic) / math.sqrt(2.0))

    return {
        "images": len(deltas),
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "t_statistic": t_statistic,
        "p_two_sided": p_two_sided,
        "fdpl_better": sum(
            fdpl_value > no_fdpl_value
            for no_fdpl_value, fdpl_value in zip(
                no_fdpl_values,
                fdpl_values,
                strict=True,
            )
        ),
    }


def _validate_pair_alignment(
    no_fdpl_result: dict[str, Any],
    fdpl_result: dict[str, Any],
) -> None:
    no_fdpl_images = no_fdpl_result["per_image"]
    fdpl_images = fdpl_result["per_image"]
    if len(no_fdpl_images) != len(fdpl_images):
        raise ValueError("checkpoint evaluations returned different image counts")
    for index, (no_fdpl_image, fdpl_image) in enumerate(
        zip(no_fdpl_images, fdpl_images, strict=True)
    ):
        if (
            no_fdpl_image["noisy"] != fdpl_image["noisy"]
            or no_fdpl_image["gt"] != fdpl_image["gt"]
        ):
            raise ValueError(
                "checkpoint evaluations disagree at pair "
                f"{index}: {no_fdpl_image['noisy']!r} vs {fdpl_image['noisy']!r}"
            )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _evaluate_cached(
    *,
    checkpoint: Path,
    protocol: str,
    noisy_dir: Path,
    gt_dir: Path,
    output_dir: Path,
    cache: dict[tuple[str, str], dict[str, Any]],
    num_rcab: int,
    device: str | None,
    limit: int,
) -> dict[str, Any]:
    key = (checkpoint.name, protocol)
    if key in cache:
        return cache[key]

    result = run_evaluation(
        checkpoint,
        noisy_dir,
        gt_dir,
        protocol=protocol,
        tile_size=128,
        stride=96,
        num_rcab=num_rcab,
        device=device,
        limit=limit,
    )
    cache[key] = result
    _write_json(output_dir / "raw" / f"{checkpoint.stem}_{protocol}.json", result)
    return result


def reproduce_table3(
    *,
    checkpoint_dir: str | Path,
    noisy_dir: str | Path,
    gt_dir: str | Path,
    output_dir: str | Path,
    device: str | None = None,
    num_rcab: int = 3,
    limit: int = 0,
) -> dict[str, Any]:
    """Evaluate every released checkpoint and return Table 3 paired gains."""

    checkpoint_root = Path(checkpoint_dir)
    noisy_root = Path(noisy_dir)
    gt_root = Path(gt_dir)
    output_root = Path(output_dir)
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for row in TABLE3_ROWS:
        no_fdpl_checkpoint = checkpoint_root / row.no_fdpl_asset
        fdpl_checkpoint = checkpoint_root / row.fdpl_asset
        if not no_fdpl_checkpoint.is_file():
            raise FileNotFoundError(no_fdpl_checkpoint)
        if not fdpl_checkpoint.is_file():
            raise FileNotFoundError(fdpl_checkpoint)

        no_fdpl_result = _evaluate_cached(
            checkpoint=no_fdpl_checkpoint,
            protocol=row.protocol,
            noisy_dir=noisy_root,
            gt_dir=gt_root,
            output_dir=output_root,
            cache=cache,
            num_rcab=num_rcab,
            device=device,
            limit=limit,
        )
        fdpl_result = _evaluate_cached(
            checkpoint=fdpl_checkpoint,
            protocol=row.protocol,
            noisy_dir=noisy_root,
            gt_dir=gt_root,
            output_dir=output_root,
            cache=cache,
            num_rcab=num_rcab,
            device=device,
            limit=limit,
        )
        _validate_pair_alignment(no_fdpl_result, fdpl_result)

        no_fdpl_images = no_fdpl_result["per_image"]
        fdpl_images = fdpl_result["per_image"]
        psnr_stats = paired_statistics(
            [float(image["psnr"]) for image in no_fdpl_images],
            [float(image["psnr"]) for image in fdpl_images],
        )
        ssim_stats = paired_statistics(
            [float(image["ssim"]) for image in no_fdpl_images],
            [float(image["ssim"]) for image in fdpl_images],
        )
        no_fdpl_mean_psnr = float(no_fdpl_result["summary"]["psnr"]["mean"])
        fdpl_mean_psnr = float(fdpl_result["summary"]["psnr"]["mean"])
        no_fdpl_mean_ssim = float(no_fdpl_result["summary"]["ssim"]["mean"])
        fdpl_mean_ssim = float(fdpl_result["summary"]["ssim"]["mean"])

        rows.append(
            {
                **asdict(row),
                "images": psnr_stats["images"],
                "no_fdpl_mean_psnr": no_fdpl_mean_psnr,
                "fdpl_mean_psnr": fdpl_mean_psnr,
                "psnr_gain": psnr_stats["mean_delta"],
                "psnr_gain_std": psnr_stats["std_delta"],
                "psnr_t": psnr_stats["t_statistic"],
                "psnr_p_two_sided": psnr_stats["p_two_sided"],
                "images_fdpl_better_psnr": psnr_stats["fdpl_better"],
                "no_fdpl_mean_ssim": no_fdpl_mean_ssim,
                "fdpl_mean_ssim": fdpl_mean_ssim,
                "ssim_delta": ssim_stats["mean_delta"],
                "fdpl_images_fdpl_better_ssim": ssim_stats["fdpl_better"],
                "delta_from_reported_gain": psnr_stats["mean_delta"] - row.reported_gain,
            }
        )

    result = {
        "protocol_note": (
            "Paired same-seed evaluation. Direct and sliding values are not "
            "compared across seeds."
        ),
        "limit": limit if limit > 0 else None,
        "rows": rows,
    }
    _write_json(output_root / "table3_reproduced.json", result)
    _write_csv(output_root / "table3_reproduced.csv", rows)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output-dir", default="runs/reproduce_table3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-rcab", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = reproduce_table3(
        checkpoint_dir=args.checkpoint_dir,
        noisy_dir=args.noisy_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        device=args.device,
        num_rcab=args.num_rcab,
        limit=args.limit,
    )
    for row in result["rows"]:
        print(
            f"row={row['row']} protocol={row['protocol']} seed={row['seed']} "
            f"psnr_gain={row['psnr_gain']:.6f} "
            f"reported={row['reported_gain']:.4f} "
            f"delta={row['delta_from_reported_gain']:+.6f}"
        )
    print(f"table3_json={Path(args.output_dir) / 'table3_reproduced.json'}")


if __name__ == "__main__":
    main()
