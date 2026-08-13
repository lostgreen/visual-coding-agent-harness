from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.mmlifelong.adapter import build_mmlifelong_workspaces
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
    virtual_to_source_windows,
)


def _dataset(tmp_path: Path, *, rows: list[dict] | None = None) -> Path:
    root = tmp_path / "MM-Lifelong"
    videos = root / "videos" / "day" / "bilibili"
    videos.mkdir(parents=True)
    (videos / "002_second.mp4").write_bytes(b"second")
    (videos / "001_first.mp4").write_bytes(b"first")
    metadata = rows or [
        {
            "index": 0,
            "question": "What crosses the clip boundary?",
            "answer": "an event",
            "question_type": "Event Tracking",
            "clue_intervals": [[9, 11]],
            "total_intervals": [[9, 11]],
            "temporal_certificate": "Short",
        },
        {
            "index": 1,
            "question": "Which annotations need normalization?",
            "answer": "two",
            "question_type": "Counting",
            "clue_intervals": [[15, 14], [20, 20]],
            "total_intervals": [[15, 14], [20, 20]],
            "temporal_certificate": "Medium",
        },
    ]
    (root / "day").mkdir()
    (root / "day" / "test.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def _probe(path: str) -> float:
    return 10.0 if Path(path).name.startswith("001") else 20.0


def _week_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "MM-Lifelong"
    day1 = root / "videos" / "week" / "EgoLife" / "A1_JAKE" / "DAY1"
    day2 = root / "videos" / "week" / "EgoLife" / "A1_JAKE" / "DAY2"
    day1.mkdir(parents=True)
    day2.mkdir(parents=True)
    (day1 / "DAY1_A1_001.mp4").write_bytes(b"day1")
    (day2 / "DAY2_A1_001.mp4").write_bytes(b"day2")
    metadata = [
        {
            "index": 0,
            "question": "What happens on the second day?",
            "answer": "an event",
            "question_type": "Event Tracking",
            "clue_intervals": [
                {"video_id": "day2", "intervals": [[2, 4]]}
            ],
            "total_intervals": [[12, 14]],
            "temporal_certificate": "Long",
        }
    ]
    (root / "week").mkdir()
    (root / "week" / "test.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return root


def test_day_builder_writes_one_shared_asset_and_free_form_cases(tmp_path: Path) -> None:
    dataset_root = _dataset(tmp_path)
    asset_root = tmp_path / "out" / "assets" / "game"
    case_root = tmp_path / "out" / "cases" / "game" / "test"

    result = build_mmlifelong_workspaces(
        dataset_root,
        asset_root,
        case_root,
        duration_probe=_probe,
    )

    assert result.summary() == {
        "subset": "game",
        "split": "test",
        "asset_root": str(asset_root),
        "case_root": str(case_root),
        "segment_count": 2,
        "case_count": 2,
        "duration_sec": 30.0,
        "clue_interval_count": 3,
        "clue_repair_count": 2,
        "validation_status": "passed",
        "validation_path": str(asset_root / "validation.json"),
    }
    assert (asset_root / "manifest.json").exists()
    assert (asset_root / "timeline.json").exists()
    assert (asset_root / "virtual_timeline.json").exists()
    assert not (asset_root / "merged.mp4").exists()

    first, second = result.workspaces
    assert first.asset_root == asset_root
    assert first.manifest.duration_sec == 30.0
    assert [Path(segment.source_path).name for segment in first.manifest.segments] == [
        "001_first.mp4",
        "002_second.mp4",
    ]
    assert [segment.day_index for segment in first.manifest.segments] == [1, 1]
    assert first.case.options == {}
    assert first.case.gold_answer == ""
    assert first.case.gold_clue_intervals == ()
    assert result.evaluation_records[0].reference_answer == "an event"
    assert result.evaluation_records[0].clue_intervals == ((9.0, 11.0),)
    assert [window.segment_id for window in virtual_to_source_windows(first.manifest, 9.0, 11.0)] == [
        "seg_0001",
        "seg_0002",
    ]

    loaded = VirtualVideoWorkspace.load(second.root_dir)
    assert loaded.asset_root == asset_root.resolve()
    assert loaded.case.gold_clue_intervals == ()
    evaluation = json.loads(
        (second.root_dir / "evaluation_case.json").read_text(encoding="utf-8")
    )
    assert evaluation["reference_answer"] == "two"
    assert evaluation["clue_intervals"] == [[15.0, 14.0], [20.0, 20.0]]
    assert [
        repair["kind"]
        for repair in evaluation["evaluation_metadata"]["clue_repairs"]
    ] == [
        "reversed",
        "zero_length_expanded",
    ]
    assert not (second.root_dir / "virtual_timeline.json").exists()
    case_payload = json.loads((second.root_dir / "case.json").read_text(encoding="utf-8"))
    assert case_payload["asset_ref"]
    assert "gold" not in case_payload
    assert "gold_answer" not in case_payload
    assert "gold_clue_intervals" not in case_payload
    assert "target_virtual_interval" not in case_payload


def test_day_builder_rejects_unmapped_clue_and_existing_output(tmp_path: Path) -> None:
    dataset_root = _dataset(
        tmp_path,
        rows=[
            {
                "index": 5,
                "question": "Out of range?",
                "answer": "yes",
                "clue_intervals": [[29, 31]],
            }
        ],
    )
    asset_root = tmp_path / "assets"
    case_root = tmp_path / "cases"

    with pytest.raises(ValueError, match="outside"):
        build_mmlifelong_workspaces(
            dataset_root,
            asset_root,
            case_root,
            duration_probe=_probe,
        )

    valid_root = _dataset(tmp_path / "valid")
    build_mmlifelong_workspaces(valid_root, asset_root, case_root, duration_probe=_probe)
    with pytest.raises(FileExistsError, match="output already exists"):
        build_mmlifelong_workspaces(valid_root, asset_root, case_root, duration_probe=_probe)


def test_week_builder_uses_global_intervals_and_reuses_duration_cache(
    tmp_path: Path,
) -> None:
    dataset_root = _week_dataset(tmp_path)
    asset_root = tmp_path / "assets" / "week"
    case_root = tmp_path / "cases" / "week"
    probe_calls = 0

    def counted_probe(path: str) -> float:
        nonlocal probe_calls
        probe_calls += 1
        return 10.0

    result = build_mmlifelong_workspaces(
        dataset_root,
        asset_root,
        case_root,
        subset="week",
        duration_probe=counted_probe,
    )

    assert probe_calls == 2
    assert [segment.day_index for segment in result.workspaces[0].manifest.segments] == [
        1,
        2,
    ]
    assert result.evaluation_records[0].clue_intervals == ((12.0, 14.0),)
    assert result.evaluation_records[0].evaluation_metadata[
        "source_clue_intervals"
    ] == [{"video_id": "day2", "intervals": [[2, 4]]}]
    cache = json.loads(
        (asset_root / "source_durations.json").read_text(encoding="utf-8")
    )
    assert cache["entry_count"] == 2

    retry_case_root = tmp_path / "cases" / "week-retry"
    for path in (
        asset_root / "virtual_timeline.json",
        asset_root / "manifest.json",
    ):
        path.unlink()
    build_mmlifelong_workspaces(
        dataset_root,
        asset_root,
        retry_case_root,
        subset="week",
        duration_probe=counted_probe,
    )
    assert probe_calls == 2


def test_legacy_workspace_promotes_target_interval_to_one_gold_clue(tmp_path: Path) -> None:
    segment = VirtualVideoSegment("seg", "source", "source.mp4", 0.0, 5.0, 0.0, 5.0)
    manifest = VirtualVideoManifest(workspace_id="legacy", segments=(segment,))
    case = VirtualVideoCase(
        case_id="legacy",
        question="What happens?",
        options={"A": "nothing", "B": "something"},
        gold="B",
        target_segment_id="seg",
        target_virtual_interval=(1.0, 2.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "legacy", manifest=manifest, case=case)

    loaded = VirtualVideoWorkspace.load(workspace.root_dir)

    assert loaded.asset_root == workspace.root_dir
    assert loaded.case.gold_clue_intervals == ((1.0, 2.0),)
