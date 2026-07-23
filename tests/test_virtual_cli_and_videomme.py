from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from vcah.videomme_virtual import build_videomme_smoke_workspaces


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


def test_virtual_video_cli_replaces_active_xle_commands() -> None:
    root_help = _run_help()

    assert "vv-build-videomme" in root_help
    assert "vv-build-mmlifelong" in root_help
    assert "vv-caption" in root_help
    assert "vv-index-caption" in root_help
    assert "vv-index" in root_help
    assert "vv-run" in root_help
    assert "vv-run-all" in root_help
    assert "xle-index" not in root_help
    assert "xle-diagnose" not in root_help
    assert "xle-investigate" not in root_help


def test_videomme_smoke_builder_writes_three_independent_virtual_workspaces(tmp_path: Path) -> None:
    dataset = tmp_path / "videomme"
    (dataset / "videomme").mkdir(parents=True)
    (dataset / "video").mkdir()
    (dataset / "subtitle").mkdir()
    records = []
    for qid, vid, question, answer in [
        ("477-2", "target_nishida", "What is the number written on the back of Nishida?", "D"),
        ("548-1", "target_glambot", "What was the person's action in number nine GlamBOT?", "A"),
        ("371-1", "target_board", "Where is the psychological tip written on the board?", "B"),
    ]:
        records.append(
            {
                "video_id": vid,
                "videoID": vid,
                "question_id": qid,
                "duration": "medium",
                "duration_sec": 600.0,
                "domain": "test",
                "sub_category": "test",
                "url": "https://example.invalid",
                "task_type": "OCR Problems",
                "question": question,
                "options": ["A. one", "B. two", "C. three", "D. four"],
                "answer": answer,
            }
        )
    for idx in range(8):
        vid = f"long_{idx}"
        records.append(
            {
                "video_id": vid,
                "videoID": vid,
                "question_id": f"9{idx:02d}-1",
                "duration": "long",
                "duration_sec": 900.0,
                "domain": "test",
                "sub_category": "test",
                "url": "https://example.invalid",
                "task_type": "Object Reasoning",
                "question": f"Distractor {idx}",
                "options": ["A. one", "B. two", "C. three", "D. four"],
                "answer": "A",
            }
        )
    pd.DataFrame(records).to_parquet(dataset / "videomme" / "test-00000-of-00001.parquet")
    for item in records:
        (dataset / "video" / f"{item['videoID']}.mp4").write_bytes(b"placeholder")
        (dataset / "subtitle" / f"{item['videoID']}.srt").write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nanchor words\n",
            encoding="utf-8",
        )

    workspaces = build_videomme_smoke_workspaces(dataset, tmp_path / "out", seed=20260707)

    assert [workspace.case.case_id for workspace in workspaces] == ["477-2", "548-1", "371-1"]
    for workspace in workspaces:
        assert (workspace.root_dir / "case.json").exists()
        assert (workspace.root_dir / "virtual_timeline.json").exists()
        assert (workspace.root_dir / "asr_virtual_cues.json").exists()
        assert len(workspace.manifest.segments) == 5
        assert workspace.case.target_segment_id in {segment.segment_id for segment in workspace.manifest.segments}
        roles = [segment.role for segment in workspace.manifest.segments]
        assert roles.count("target") == 1
        assert roles[0] != "target"
        assert roles[-1] != "target"
        cues = json.loads((workspace.root_dir / "asr_virtual_cues.json").read_text(encoding="utf-8"))
        assert cues
