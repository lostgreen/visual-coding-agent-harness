from __future__ import annotations

import json
from pathlib import Path

from vcah.evals import load_videomme_case


def test_load_videomme_case_from_single_json_file(tmp_path: Path) -> None:
    root = tmp_path / "videomme"
    root.mkdir()
    (root / "cases.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "601-1",
                    "video": "videos/601.mp4",
                    "question": "What is shown?",
                    "answer": "A",
                }
            ]
        ),
        encoding="utf-8",
    )

    case = load_videomme_case(root, "601-1")

    assert case.case_id == "601-1"
    assert case.video_path == root / "videos" / "601.mp4"
    assert case.question == "What is shown?"
