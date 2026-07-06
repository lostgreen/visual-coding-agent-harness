from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import main
from vcah.xlebench import LifeLogColdIndex, LifeLogManifest


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


def _run_tool_help(path: str) -> str:
    env = {**os.environ, "PYTHONPATH": "src:."}
    result = subprocess.run(
        [sys.executable, path, "--help"],
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
    investigate_help = _run_help("xle-investigate")

    assert "xle-index" in root_help
    assert "xle-diagnose" in root_help
    assert "xle-investigate" in root_help
    assert "Build an X-LeBench lifelog cold index." in index_help
    assert "--no-resume" in index_help
    assert "Run X-LeBench cold-recall diagnostics from an existing index." in diagnose_help
    assert "--build" in diagnose_help
    assert "--load-only" not in diagnose_help
    assert "Run a minimal X-LeBench investigator loop from an existing index." in investigate_help
    assert "--inspect-top-n" in investigate_help


def test_xle_gemini_eval_tool_exposes_investigator_mode() -> None:
    help_text = _run_tool_help("tools/xle_gemini_candidate_eval.py")

    assert "--mode" in help_text
    assert "investigator" in help_text


def test_xle_diagnose_build_path_registers_index_args(monkeypatch, tmp_path: Path, capsys) -> None:
    seen: dict[str, object] = {}
    manifest = LifeLogManifest(())

    class DummyBuilder:
        def __init__(self, loaded_manifest, config) -> None:
            seen["manifest"] = loaded_manifest
            seen["max_range_sec"] = config.max_range_sec
            seen["max_beat_sec"] = config.max_beat_sec

        def build(self, run_dir: Path, *, resume: bool = True) -> LifeLogColdIndex:
            seen["run_dir"] = Path(run_dir)
            seen["resume"] = resume
            return LifeLogColdIndex(manifest, (), Path(run_dir))

    monkeypatch.setattr(main, "load_xlebench_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(main, "LifeLogColdIndexBuilder", DummyBuilder)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "xle-diagnose",
            str(tmp_path / "xle"),
            "--run-dir",
            str(tmp_path / "run"),
            "--build",
            "--top-k",
            "5",
            "--max-range-sec",
            "12",
            "--max-beat-sec",
            "6",
            "--no-resume",
        ],
    )

    main.main()

    captured = capsys.readouterr()
    assert '"case_count": 0' in captured.out
    assert seen["max_range_sec"] == 12.0
    assert seen["max_beat_sec"] == 6.0
    assert seen["resume"] is False
