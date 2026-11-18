from rgb_correction.selection import CandidateScore, select_action


def score(mean: float, *, harm: float = 0.0) -> CandidateScore:
    return CandidateScore(
        mean_psnr_delta=mean,
        bootstrap_p05=0.01,
        severe_loss_rate=harm,
        ssim_delta=0.001,
        lpips_delta=-0.001,
        dists_delta=-0.001,
    )


def test_highest_eligible_simple_action_wins() -> None:
    result = select_action(
        {
            "source": score(0.10),
            "scaled_source": score(0.14),
            "fixed_target": score(0.12),
        }
    )
    assert result.action == "scaled_source"


def test_missing_ridge_candidates_do_not_prevent_simple_selection() -> None:
    result = select_action(
        {
            "source": score(0.10),
            "fixed_target": score(0.14),
        }
    )
    assert result.action == "fixed_target"


def test_capped_ridge_requires_margin_and_harm_safeguard() -> None:
    rejected = select_action(
        {
            "source": score(0.20),
            "capped_ridge": score(0.23, harm=0.03),
        }
    )
    accepted = select_action(
        {
            "source": score(0.20),
            "capped_ridge": score(0.26, harm=0.03),
        }
    )
    assert rejected.action == "source"
    assert accepted.action == "capped_ridge"


def test_no_eligible_action_retains_base() -> None:
    result = select_action(
        {
            "source": score(0.01),
            "capped_ridge": score(0.03),
        }
    )
    assert result.action == "no_correction"


def test_uncapped_ridge_can_replace_capped_ridge() -> None:
    result = select_action(
        {
            "capped_ridge": score(0.20),
            "uncapped_ridge": score(0.31, harm=0.03),
        }
    )
    assert result.action == "uncapped_ridge"
