import json

from rgb_correction.cli import main


def test_demo_cli_writes_json(tmp_path, monkeypatch, capsys) -> None:
    output = tmp_path / "demo.json"
    monkeypatch.setattr(
        "sys.argv",
        ["rgb-correction", "demo", "--output", str(output)],
    )

    main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["projected_offset"]) == 3
    assert abs(payload["offset_luma"]) < 1e-12
    assert json.loads(capsys.readouterr().out) == payload


def test_select_cli_applies_risk_policy(tmp_path, monkeypatch, capsys) -> None:
    candidate_file = tmp_path / "candidates.json"
    candidate_file.write_text(
        json.dumps(
            {
                "source": {
                    "mean_psnr_delta": 0.20,
                    "bootstrap_p05": 0.10,
                    "severe_loss_rate": 0.05,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["rgb-correction", "select", "--input", str(candidate_file)],
    )

    main()

    result = json.loads(capsys.readouterr().out)
    assert result["action"] == "source"
    assert result["comparison_action"] is None
