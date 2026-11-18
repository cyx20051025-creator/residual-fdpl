#!/usr/bin/env python3
"""Draw the SIDD+ and SIDD-CC calibration-support diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SIDDPLUS_BUDGETS = (16, 64, 128, 256, 512)
SIDDCC_BUDGETS = (4, 8, 12, 16, 20, 24)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    samples: int = 5000,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(samples)
        ],
        dtype=np.float64,
    )
    return tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))


def selector_fixed(
    payload: dict,
) -> tuple[float, float, float]:
    selected = np.asarray(
        [
            record["evaluation"]["mean_psnr_delta"]
            for record in payload["records"]
        ],
        dtype=np.float64,
    )
    fixed = np.asarray(
        [
            record["evaluation_by_candidate"]["fixed_target"][
                "mean_psnr_delta"
            ]
            for record in payload["records"]
        ],
        dtype=np.float64,
    )
    difference = selected - fixed
    low, high = bootstrap_mean_interval(
        difference,
        seed=20261117 + len(payload["records"]),
    )
    return float(difference.mean()), low, high


def selected_gain(payload: dict) -> float:
    return float(
        np.mean(
            [
                record["evaluation"]["mean_psnr_delta"]
                for record in payload["records"]
            ]
        )
    )


def require(path: Path, *, allow_partial: bool, missing: list[Path]) -> bool:
    if path.is_file() and path.stat().st_size > 0:
        return True
    missing.append(path)
    return False


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    siddplus = root / "siddplus_calibration_budget_20261117"
    siddcc = root / "selector_no_source_siddcc_budget_20261117"
    cross = root / "crossbackbone_budget_20261117"
    missing: list[Path] = []

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.1,
            "lines.linewidth": 1.35,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "calibration-support-20261117",
        }
    )
    figure, (axis_a, axis_b) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.1),
        constrained_layout=True,
    )

    seed_colors = {
        62: "#2b6cb0",
        63: "#2f855a",
        64: "#b7791f",
    }
    for seed in (62, 63, 64):
        xs, centers, lows, highs = [], [], [], []
        for budget in SIDDPLUS_BUDGETS:
            path = (
                siddplus
                / f"sourcefree_core_seed{seed}"
                / (
                    "risk_selector_no_source_"
                    f"siddplus_seed{seed}_budget{budget}.json"
                )
            )
            if not require(path, allow_partial=args.allow_partial, missing=missing):
                continue
            center, low, high = selector_fixed(load(path))
            xs.append(budget)
            centers.append(center)
            lows.append(low)
            highs.append(high)
        if not xs:
            continue
        xs_array = np.asarray(xs, dtype=np.float64)
        centers_array = np.asarray(centers, dtype=np.float64)
        axis_a.fill_between(
            xs_array,
            lows,
            highs,
            color=seed_colors[seed],
            alpha=0.12,
            linewidth=0,
        )
        axis_a.plot(
            xs_array,
            centers_array,
            marker="o",
            markersize=4.0,
            color=seed_colors[seed],
            label=f"NAFNet seed {seed}",
        )

    axis_a.axhline(0.0, color="0.35", linewidth=0.75)
    axis_a.set_xscale("log", base=2)
    axis_a.set_xticks(SIDDPLUS_BUDGETS)
    axis_a.set_xticklabels([str(value) for value in SIDDPLUS_BUDGETS])
    axis_a.set_xlabel("SIDD+ calibration images")
    axis_a.set_ylabel("Selector − fixed target (dB)")
    axis_a.set_title("(a) Calibration support on SIDD+", loc="left", fontweight="bold")
    axis_a.grid(axis="y", color="0.9", linewidth=0.65)
    axis_a.set_axisbelow(True)
    axis_a.legend(frameon=False, loc="lower right")

    nafnet_curves: dict[int, dict[int, float]] = {}
    for seed in (62, 63, 64):
        values: dict[int, float] = {}
        for budget in SIDDCC_BUDGETS:
            if budget == 16:
                path = (
                    root
                    / "selector_no_source_20261117"
                    / "results"
                    / (
                        "risk_selector_no_source_naf_finetuned_"
                        f"seed{seed}_source56.json"
                    )
                )
            else:
                path = (
                    siddcc
                    / f"seed{seed}"
                    / (
                        "risk_selector_no_source_"
                        f"siddcc_seed{seed}_budget{budget}.json"
                    )
                )
            if not require(path, allow_partial=args.allow_partial, missing=missing):
                continue
            values[budget] = selected_gain(load(path))
        nafnet_curves[seed] = values
        if values:
            xs = sorted(values)
            color = seed_colors[seed]
            axis_b.plot(
                xs,
                [values[value] for value in xs],
                color=color,
                marker="o",
                markersize=3.6,
                alpha=0.72,
                linewidth=1.0,
                label=f"NAFNet seed {seed}",
            )
            sparse_xs = [value for value in xs if value > 16]
            if sparse_xs:
                axis_b.plot(
                    sparse_xs,
                    [values[value] for value in sparse_xs],
                    linestyle="none",
                    marker="o",
                    markersize=3.6,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=0.9,
                    alpha=0.9,
                )

    common_budgets = sorted(
        set.intersection(
            *[
                set(values)
                for values in nafnet_curves.values()
                if values
            ]
        )
    )
    if common_budgets:
        common_means = [
            np.mean(
                [nafnet_curves[seed][budget] for seed in (62, 63, 64)]
            )
            for budget in common_budgets
        ]
        axis_b.plot(
            common_budgets,
            common_means,
            color="#1a365d",
            marker="D",
            markersize=4.2,
            linewidth=1.8,
            label="NAFNet mean",
            zorder=4,
        )
        sparse_common = [
            budget for budget in common_budgets if budget > 16
        ]
        if sparse_common:
            axis_b.plot(
                sparse_common,
                [
                    common_means[common_budgets.index(budget)]
                    for budget in sparse_common
                ],
                linestyle="none",
                marker="D",
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor="#1a365d",
                markeredgewidth=0.9,
                zorder=5,
            )

    cross_specs = [
        (
            "SwinIR-M",
            cross / "swinir_seed62",
            "risk_selector_no_source_swinir_seed62_budget",
            "#2f855a",
            "s",
        ),
        (
            "Restormer",
            cross / "restormer",
            "risk_selector_no_source_restormer_budget",
            "#805ad5",
            "^",
        ),
    ]
    for label, directory, stem, color, marker in cross_specs:
        xs, values = [], []
        for budget in SIDDCC_BUDGETS:
            path = directory / f"{stem}{budget}.json"
            if not require(path, allow_partial=args.allow_partial, missing=missing):
                continue
            xs.append(budget)
            values.append(selected_gain(load(path)))
        if xs:
            axis_b.plot(
                xs,
                values,
                color=color,
                marker=marker,
                markersize=4.0,
                linewidth=1.35,
                label=label,
            )
            sparse_xs = [value for value in xs if value > 16]
            if sparse_xs:
                axis_b.plot(
                    sparse_xs,
                    [values[xs.index(value)] for value in sparse_xs],
                    linestyle="none",
                    marker=marker,
                    markersize=4.0,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=0.9,
                )

    axis_b.axhline(0.0, color="0.35", linewidth=0.75)
    axis_b.axvline(16, color="0.5", linestyle=(0, (3, 2)), linewidth=0.8)
    axis_b.set_xticks(SIDDCC_BUDGETS)
    axis_b.set_xlabel("SIDD-CC calibration scenes")
    axis_b.set_ylabel("Selected PSNR change (dB)")
    axis_b.set_title("(b) Calibration support on SIDD-CC", loc="left", fontweight="bold")
    axis_b.grid(axis="y", color="0.9", linewidth=0.65)
    axis_b.set_axisbelow(True)
    axis_b.legend(frameon=False, ncol=2, loc="upper left")

    if missing and not args.allow_partial:
        raise FileNotFoundError(f"missing results: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_dir / "figure7_calibration_support"
    figure.savefig(output_stem.with_suffix(".png"), dpi=320)
    figure.savefig(output_stem.with_suffix(".pdf"))
    figure.savefig(output_stem.with_suffix(".svg"))
    print(output_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
