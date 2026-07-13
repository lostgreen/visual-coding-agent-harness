from __future__ import annotations

import json
from pathlib import Path

from tools.select_videomme_unseen_group import _seen_cases, select_unseen_cases


def test_select_unseen_cases_excludes_seen_sources_and_stratifies_tasks() -> None:
    rows = [
        {
            "question_id": f"q{index}",
            "videoID": f"video{index}",
            "duration": "long",
            "task_type": "Counting" if index % 2 else "Spatial",
            "domain": "Sports" if index % 3 else "Knowledge",
        }
        for index in range(1, 9)
    ]

    selected = select_unseen_cases(
        rows,
        seen_case_ids={"q1"},
        seen_video_ids={"video2"},
        count=4,
        seed=7,
    )

    assert len(selected) == 4
    assert len({row["videoID"] for row in selected}) == 4
    assert {row["task_type"] for row in selected} == {"Counting", "Spatial"}
    assert "q1" not in {row["question_id"] for row in selected}
    assert "video2" not in {row["videoID"] for row in selected}


def test_seen_cases_scans_workspace_metadata_without_recursive_frame_walk(tmp_path: Path) -> None:
    case_path = tmp_path / "run-a" / "workspaces" / "q1" / "case.json"
    case_path.parent.mkdir(parents=True)
    case_path.write_text(
        json.dumps({"case_id": "q1", "metadata": {"source_video_id": "video1"}}),
        encoding="utf-8",
    )
    ignored = tmp_path / "run-a" / "workspaces" / "q1" / "observations" / "nested" / "case.json"
    ignored.parent.mkdir(parents=True)
    ignored.write_text(json.dumps({"case_id": "noise"}), encoding="utf-8")

    case_ids, video_ids = _seen_cases((tmp_path,))

    assert case_ids == {"q1"}
    assert video_ids == {"video1"}
