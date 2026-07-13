from __future__ import annotations

from tools.select_videomme_unseen_group import select_unseen_cases


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
