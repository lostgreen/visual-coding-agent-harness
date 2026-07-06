from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from vcah.types import Frame
from vcah.xlebench import (
    LifeLogColdIndexBuilder,
    LifeLogIndexConfig,
    LifeLogRetriever,
    diagnose_cold_recall,
    load_xlebench_manifest,
)


class KeywordColorModel:
    embedding_dim = 3
    embed_model = "keyword-color"
    allow_placeholder_visual = True

    def embed_image(self, paths: Sequence[str]) -> np.ndarray:
        rows = []
        for path in paths:
            with Image.open(path) as image:
                pixel = image.convert("RGB").resize((1, 1)).getpixel((0, 0))
            channel = int(max(range(3), key=lambda idx: pixel[idx]))
            row = [0.0, 0.0, 0.0]
            row[channel] = 1.0
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)

    def embed_text(self, queries: Sequence[str]) -> np.ndarray:
        rows = []
        for query in queries:
            text = query.casefold()
            if "blue" in text:
                rows.append([0.0, 0.0, 1.0])
            elif "green" in text:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([1.0, 0.0, 0.0])
        return np.asarray(rows, dtype=np.float32)


def _sampler(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
    del end_sec, n_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    if "seg_b" in video_path:
        color = (20, 40, 230)
    elif start_sec >= 10:
        color = (20, 220, 30)
    else:
        color = (240, 20, 20)
    path = out_dir / f"frame_{int(start_sec):03d}.jpg"
    Image.new("RGB", (32, 18), color=color).save(path)
    return (Frame(frame_id="fr001", time_sec=start_sec, path=str(path)),)


def test_xlebench_manifest_adapter_maps_source_and_virtual_time(tmp_path: Path) -> None:
    root = tmp_path / "xle"
    root.mkdir()
    (root / "cases.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "case-1",
                        "question": "Where is the blue cup?",
                        "video_uids": ["seg_a", "seg_b"],
                        "durations": {"seg_a": 30.0, "seg_b": 20.0},
                        "query_range": {"video_uid": "seg_b", "start": 2.0, "end": 8.0},
                        "gt_interval": {"video_uid": "seg_b", "start": 4.0, "end": 6.0},
                    }
                )
            ]
        ),
        encoding="utf-8",
    )

    manifest = load_xlebench_manifest(root, video_template="/videos/{video_uid}.mp4")

    assert [segment.video_uid for segment in manifest.segments] == ["seg_a", "seg_b"]
    assert manifest.segments[1].virtual_start_sec == 30.0
    assert manifest.cases[0].scope.video_uid == "seg_b"
    assert manifest.cases[0].scope.virtual_start_sec == 32.0
    assert manifest.cases[0].gt_intervals[0].virtual_start_sec == 34.0
    assert manifest.segments[0].video_path == Path("/videos/seg_a.mp4")


def test_lifelog_builder_resumes_and_retriever_filters_by_scope(tmp_path: Path) -> None:
    root = tmp_path / "xle"
    root.mkdir()
    (root / "cases.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "case-1",
                    "query": "blue object",
                    "videos": [
                        {"video_uid": "seg_a", "duration_sec": 20.0, "video_path": "seg_a.mp4"},
                        {"video_uid": "seg_b", "duration_sec": 20.0, "video_path": "seg_b.mp4"},
                    ],
                    "query_range": [0.0, 20.0],
                    "video_uid": "seg_b",
                    "gt_interval": [0.0, 20.0],
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = load_xlebench_manifest(root, video_template=str(root / "{video_uid}.mp4"))
    calls: list[str] = []

    def range_detector(video_path: str, duration_sec: float) -> tuple[tuple[float, float], ...]:
        calls.append(Path(video_path).stem)
        return ((0.0, duration_sec),)

    builder = LifeLogColdIndexBuilder(
        manifest,
        LifeLogIndexConfig(max_range_sec=20.0, max_beat_sec=20.0),
        model=KeywordColorModel(),
        range_detector=range_detector,
        keyframe_sampler=_sampler,
    )

    first = builder.build(tmp_path / "run", resume=True)
    second = builder.build(tmp_path / "run", resume=True)
    result = LifeLogRetriever(second).retrieve("blue object", scope=manifest.cases[0].scope, top_k=5)

    assert calls == ["seg_a", "seg_b"]
    assert (tmp_path / "run" / "lifelog_index.json").exists()
    assert [segment.video_uid for segment in first.segments] == ["seg_a", "seg_b"]
    assert result.candidates
    assert {candidate.video_uid for candidate in result.candidates} == {"seg_b"}
    assert result.candidates[0].frame_refs[0].video_uid == "seg_b"


def test_xle_diagnose_reports_cold_recall_and_candidate_coverage(tmp_path: Path) -> None:
    root = tmp_path / "xle"
    root.mkdir()
    (root / "cases.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "case-blue",
                    "question": "Find the blue object",
                    "video_uid": "seg_b",
                    "videos": [
                        {"video_uid": "seg_a", "duration_sec": 10.0, "video_path": "seg_a.mp4"},
                        {"video_uid": "seg_b", "duration_sec": 10.0, "video_path": "seg_b.mp4"},
                    ],
                    "query_range": {"start_sec": 0.0, "end_sec": 10.0},
                    "gt_interval": {"start_sec": 0.0, "end_sec": 10.0},
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = load_xlebench_manifest(root, video_template=str(root / "{video_uid}.mp4"))
    index = LifeLogColdIndexBuilder(
        manifest,
        LifeLogIndexConfig(max_range_sec=10.0, max_beat_sec=10.0),
        model=KeywordColorModel(),
        range_detector=lambda _path, duration: ((0.0, duration),),
        keyframe_sampler=_sampler,
    ).build(tmp_path / "run")

    report = diagnose_cold_recall(index, manifest.cases, top_ks=(5, 20))

    assert report["case_count"] == 1
    assert report["cold_recall@5"] == 1.0
    assert report["cold_recall@20"] == 1.0
    assert report["gt_interval_candidate_coverage"] == 1.0
    assert report["per_channel_recall"]["visual@5"] == 1.0
    assert report["counts"]["segments"] == 2
