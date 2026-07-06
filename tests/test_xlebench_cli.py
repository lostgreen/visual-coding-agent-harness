from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_help(*args: str) -> str:
    env = {**os.environ, "PYTHONPATH": "src:."}
    result = subprocess.run(
        [sys.executable, "main.py", *args, "--help"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_xle_cli_exposes_split_index_and_diagnose_commands() -> None:
    root_help = _run_help()
    index_help = _run_help("xle-index")
    diagnose_help = _run_help("xle-diagnose")

    assert "xle-index" in root_help
    assert "xle-diagnose" in root_help
    assert "Build an X-LeBench lifelog cold index." in index_help
    assert "--no-resume" in index_help
    assert "Run X-LeBench cold-recall diagnostics from an existing index." in diagnose_help
    assert "--build" in diagnose_help
    assert "--load-only" not in diagnose_help
