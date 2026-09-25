"""Regression tests for the source-checkout Table 3 wrapper."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.name == "nt", reason="the wrapper is a Bash script")
def test_script_maps_reviewer_environment_variables_to_cli_arguments(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "arguments.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$ARGS_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": str(fake_python),
            "ARGS_LOG": str(log_path),
            "CHECKPOINT_DIR": "/checkpoints",
            "NOISY_DIR": "/data/noisy",
            "GT_DIR": "/data/gt",
            "OUTPUT_DIR": "/results/table3",
        }
    )
    subprocess.run(
        ["bash", "scripts/reproduce_table3.sh"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )

    assert log_path.read_text(encoding="utf-8").splitlines() == [
        str(REPOSITORY_ROOT / "scripts" / "reproduce_table3.py"),
        "--checkpoint-dir",
        "/checkpoints",
        "--noisy-dir",
        "/data/noisy",
        "--gt-dir",
        "/data/gt",
        "--output-dir",
        "/results/table3",
    ]
