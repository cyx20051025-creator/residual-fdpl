"""Command-line utilities for RGB correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rgb_correction.ops import (
    REC601_WEIGHTS,
    mse_reduction,
    preclip_psnr_delta,
    project_zero_luma,
)
from rgb_correction.selection import CandidateScore, RiskPolicy, select_action


def demo_payload() -> dict[str, object]:
    mean_error = np.asarray((0.030, -0.010, -0.020), dtype=np.float64)
    projected_factor = 0.65
    offset = project_zero_luma(mean_error) * projected_factor
    reduction = float(mse_reduction(mean_error, offset))
    base_mse = 0.012
    return {
        "luma_weights": REC601_WEIGHTS.tolist(),
        "mean_error": mean_error.tolist(),
        "projected_offset": offset.tolist(),
        "offset_luma": float(np.dot(REC601_WEIGHTS, offset)),
        "preclip_mse_reduction": reduction,
        "preclip_psnr_delta_db": float(
            preclip_psnr_delta(base_mse, mean_error, offset)
        ),
    }


def _candidate_from_mapping(payload: dict[str, object]) -> CandidateScore:
    fields = CandidateScore.__dataclass_fields__
    return CandidateScore(
        **{
            name: float(payload[name])
            for name in fields
            if name in payload
        }
    )


def _run_demo(args: argparse.Namespace) -> None:
    payload = demo_payload()
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _run_select(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    scores = {
        name: _candidate_from_mapping(values)
        for name, values in payload.items()
    }
    policy = RiskPolicy(
        min_mean_gain=args.min_mean_gain,
        max_severe_loss_rate=args.max_severe_loss_rate,
        capped_margin=args.capped_margin,
        uncapped_margin=args.uncapped_margin,
    )
    result = select_action(scores, policy).to_dict()
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo",
        help="run the dependency-light RGB geometry example",
    )
    demo.add_argument("--output", default="")
    demo.set_defaults(handler=_run_demo)

    selection = subparsers.add_parser(
        "select",
        help="apply the risk policy to candidate scores in JSON",
    )
    selection.add_argument("--input", required=True)
    selection.add_argument("--min-mean-gain", type=float, default=0.05)
    selection.add_argument("--max-severe-loss-rate", type=float, default=0.20)
    selection.add_argument("--capped-margin", type=float, default=0.05)
    selection.add_argument("--uncapped-margin", type=float, default=0.10)
    selection.set_defaults(handler=_run_select)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
