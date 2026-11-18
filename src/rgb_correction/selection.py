"""Risk-controlled selection among frozen correction actions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

SIMPLE_ACTIONS = ("source", "scaled_source", "fixed_target")
NO_CORRECTION = "no_correction"
CAPPED_RIDGE = "capped_ridge"
UNCAPPED_RIDGE = "uncapped_ridge"


@dataclass(frozen=True)
class CandidateScore:
    """Held-out calibration evidence for one correction action."""

    mean_psnr_delta: float
    bootstrap_p05: float
    severe_loss_rate: float
    ssim_delta: float = 0.0
    lpips_delta: float = 0.0
    dists_delta: float = 0.0


@dataclass(frozen=True)
class RiskPolicy:
    """Acceptance and promotion thresholds used by the paper."""

    min_mean_gain: float = 0.05
    min_bootstrap_p05: float = 0.0
    max_severe_loss_rate: float = 0.20
    max_ssim_drop: float = 0.0005
    max_lpips_rise: float = 0.0005
    max_dists_rise: float = 0.0005
    capped_margin: float = 0.05
    uncapped_margin: float = 0.10
    max_extra_harm_rate: float = 0.05

    def eligible(self, score: CandidateScore) -> bool:
        return (
            score.mean_psnr_delta >= self.min_mean_gain
            and score.bootstrap_p05 > self.min_bootstrap_p05
            and score.severe_loss_rate <= self.max_severe_loss_rate
            and score.ssim_delta >= -self.max_ssim_drop
            and score.lpips_delta <= self.max_lpips_rise
            and score.dists_delta <= self.max_dists_rise
        )


@dataclass(frozen=True)
class SelectionResult:
    """The selected action and the evidence used to choose it."""

    action: str
    comparison_action: str | None
    eligible_actions: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _better_ridge(
    candidate: str,
    incumbent: str,
    scores: Mapping[str, CandidateScore],
    policy: RiskPolicy,
    margin: float,
) -> bool:
    if candidate not in scores or incumbent not in scores:
        return False
    proposed = scores[candidate]
    current = scores[incumbent]
    return (
        policy.eligible(proposed)
        and proposed.mean_psnr_delta >= current.mean_psnr_delta + margin
        and proposed.severe_loss_rate
        <= current.severe_loss_rate + policy.max_extra_harm_rate
    )


def select_action(
    scores: Mapping[str, CandidateScore],
    policy: RiskPolicy = RiskPolicy(),
) -> SelectionResult:
    """Apply the predeclared simple-to-Ridge promotion rule."""
    eligible = tuple(
        name for name, score in scores.items() if policy.eligible(score)
    )
    eligible_simple = [
        name for name in SIMPLE_ACTIONS if name in eligible
    ]
    capped_eligible = CAPPED_RIDGE in eligible
    uncapped_eligible = UNCAPPED_RIDGE in eligible

    if eligible_simple:
        action = max(
            eligible_simple,
            key=lambda name: scores[name].mean_psnr_delta,
        )
        comparison = None
        if _better_ridge(
            CAPPED_RIDGE,
            action,
            scores,
            policy,
            policy.capped_margin,
        ):
            comparison = action
            action = CAPPED_RIDGE
        if action == CAPPED_RIDGE and _better_ridge(
            UNCAPPED_RIDGE,
            CAPPED_RIDGE,
            scores,
            policy,
            policy.uncapped_margin,
        ):
            comparison = action
            action = UNCAPPED_RIDGE
        return SelectionResult(
            action=action,
            comparison_action=comparison,
            eligible_actions=eligible,
            reason="best eligible simple action, then permitted Ridge promotion",
        )

    if capped_eligible:
        action = CAPPED_RIDGE
        comparison = None
        if _better_ridge(
            UNCAPPED_RIDGE,
            CAPPED_RIDGE,
            scores,
            policy,
            policy.uncapped_margin,
        ):
            comparison = CAPPED_RIDGE
            action = UNCAPPED_RIDGE
        return SelectionResult(
            action=action,
            comparison_action=comparison,
            eligible_actions=eligible,
            reason="no simple action qualified; use an eligible Ridge action",
        )

    if uncapped_eligible:
        return SelectionResult(
            action=UNCAPPED_RIDGE,
            comparison_action=None,
            eligible_actions=eligible,
            reason="uncapped Ridge was the only eligible correction",
        )

    return SelectionResult(
        action=NO_CORRECTION,
        comparison_action=None,
        eligible_actions=eligible,
        reason="no correction met the predeclared acceptance criteria",
    )
