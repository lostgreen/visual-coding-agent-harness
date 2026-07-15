from __future__ import annotations

import json
import zipfile
from pathlib import Path

from vcah.videomme_v2 import (
    VideoMMEV2Question,
    format_videomme_v2_question,
    load_case_group,
    load_subtitle_cues,
    options_mapping,
    score_videomme_v2_answer,
    select_video_ids,
    summarize_group_results,
)


def test_load_subtitle_cues_merges_word_level_rows(tmp_path: Path) -> None:
    subtitle_zip = tmp_path / "subtitle.zip"
    rows = [
        {"text": "Hello", "start_time": 0.0, "end_time": 0.2},
        {"text": "world", "start_time": 0.25, "end_time": 0.4},
        {"text": ".", "start_time": 0.4, "end_time": 0.45},
        {"text": "Next", "start_time": 2.0, "end_time": 2.2},
        {"text": "line", "start_time": 2.25, "end_time": 2.5},
    ]
    with zipfile.ZipFile(subtitle_zip, "w") as archive:
        archive.writestr("subtitle/042.jsonl", "\n".join(json.dumps(row) for row in rows))

    cues = load_subtitle_cues(subtitle_zip, "042", max_gap_sec=0.6, max_words=20)

    assert cues == (
        {"start": 0.0, "end": 0.45, "text": "Hello world."},
        {"start": 2.0, "end": 2.5, "text": "Next line"},
    )


def test_format_videomme_v2_question_keeps_answer_options() -> None:
    question = VideoMMEV2Question(
        video_id="042",
        question_id="042-1",
        question="What color is the car?",
        options="A. Red.\nB. Blue.",
        answer="B",
        metadata={},
    )

    formatted = format_videomme_v2_question(question)

    assert formatted == "What color is the car?\nA. Red.\nB. Blue."


def test_select_video_ids_requires_existing_mp4s_and_four_questions(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    for video_id in ("002", "003", "005"):
        (videos_dir / f"{video_id}.mp4").write_bytes(b"fake")
    questions = []
    for video_id in ("001", "002", "003", "004", "005"):
        limit = 4 if video_id != "003" else 3
        for idx in range(1, limit + 1):
            questions.append(
                VideoMMEV2Question(
                    video_id=video_id,
                    question_id=f"{video_id}-{idx}",
                    question=f"Q{idx}?",
                    options="A. One.\nB. Two.",
                    answer="A",
                    metadata={},
                )
            )

    assert select_video_ids(questions, videos_dir=videos_dir, count=2) == ("002", "005")


def test_score_videomme_v2_answer_extracts_option_letters() -> None:
    assert score_videomme_v2_answer("The answer is B.", "B") is True
    assert score_videomme_v2_answer("B. Blue.", "B") is True
    assert score_videomme_v2_answer("I choose A.", "B") is False
    assert score_videomme_v2_answer("Insufficient verified evidence.", "B") is False


def test_options_mapping_parses_multiline_v2_options() -> None:
    assert options_mapping("A. First answer.\nB. Second answer.\ncontinues here\nH. Last answer.") == {
        "A": "First answer.",
        "B": "Second answer. continues here",
        "H": "Last answer.",
    }


def test_load_case_group_and_summary_preserve_complete_video_groups(tmp_path: Path) -> None:
    path = tmp_path / "group.json"
    path.write_text(
        json.dumps(
            {
                "group_id": "v2-test",
                "cases": [
                    {"case_id": "001-1", "video_id": "001", "task_type": "Counting"},
                    {"case_id": "001-2", "video_id": "001", "task_type": "Counting"},
                    {"case_id": "002-1", "video_id": "002", "task_type": "Order"},
                ],
            }
        ),
        encoding="utf-8",
    )
    group = load_case_group(path)
    summary = summarize_group_results(
        (
            {"case_id": "001-1", "video_id": "001", "correct": True},
            {"case_id": "001-2", "video_id": "001", "correct": False},
            {"case_id": "002-1", "video_id": "002", "correct": True},
        ),
        group=group,
    )

    assert group["case_ids"] == ("001-1", "001-2", "002-1")
    assert summary["correct"] == 2
    assert summary["all_correct_group_count"] == 1
    assert summary["by_video"]["001"]["correct_prefix_length"] == 1
    assert summary["by_task_type"]["Counting"]["accuracy"] == 0.5
