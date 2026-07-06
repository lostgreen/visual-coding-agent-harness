from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from vcah.types import Frame
from vcah.xlebench import (
    LifeLogColdIndex,
    LifeLogColdIndexBuilder,
    LifeLogIndexConfig,
    LifeLogInvestigator,
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


def _misleading_name_sampler(
    video_path: str,
    start_sec: float,
    end_sec: float,
    n_frames: int,
    out_dir: Path,
) -> tuple[Frame, ...]:
    del video_path, end_sec, n_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "frame_001.jpg"
    color = (20, 40, 230) if start_sec >= 17000.0 else (240, 20, 20)
    Image.new("RGB", (32, 18), color=color).save(path)
    return (Frame(frame_id="fr001", time_sec=start_sec + 5.0, path=str(path)),)


def _red_keyframe_blue_second_sampler(
    video_path: str,
    start_sec: float,
    end_sec: float,
    n_frames: int,
    out_dir: Path,
) -> tuple[Frame, ...]:
    del video_path, end_sec, n_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    red_path = out_dir / "frame_key_red.jpg"
    blue_path = out_dir / "frame_later_blue.jpg"
    Image.new("RGB", (32, 18), color=(240, 20, 20)).save(red_path)
    Image.new("RGB", (32, 18), color=(20, 40, 230)).save(blue_path)
    return (
        Frame(frame_id="fr001", time_sec=start_sec, path=str(red_path)),
        Frame(frame_id="fr002", time_sec=start_sec + 5.0, path=str(blue_path)),
    )


def _late_table_sampler(
    video_path: str,
    start_sec: float,
    end_sec: float,
    n_frames: int,
    out_dir: Path,
) -> tuple[Frame, ...]:
    del video_path, end_sec, n_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    early_path = out_dir / f"early_{int(start_sec):03d}.jpg"
    late_path = out_dir / f"late_{int(start_sec):03d}.jpg"
    Image.new("RGB", (32, 18), color=(240, 20, 20)).save(early_path)
    Image.new("RGB", (32, 18), color=(20, 40, 230)).save(late_path)
    return (
        Frame(frame_id="fr001", time_sec=start_sec + 5.0, path=str(early_path)),
        Frame(frame_id="fr002", time_sec=start_sec + 95.0, path=str(late_path)),
    )


class FakeXLEInvestigatorModel(KeywordColorModel):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def investigate_xle_window(self, question: str, *, candidate: object, subwindow: object, frame_refs: Sequence[object]) -> dict[str, object]:
        del candidate
        payload = {
            "question": question,
            "start_sec": getattr(subwindow, "source_start_sec"),
            "end_sec": getattr(subwindow, "source_end_sec"),
            "frame_times": [getattr(frame, "source_time_sec") for frame in frame_refs],
        }
        self.calls.append(payload)
        if any(float(time_sec) >= 90.0 for time_sec in payload["frame_times"]):
            return {
                "claim": "User put a blue cup on the table.",
                "answer": "blue cup",
                "evidence": "A late frame shows a blue cup on the table.",
                "confidence": 0.92,
            }
        return {"claim": "", "answer": "", "evidence": "No table placement is visible.", "confidence": 0.05}


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


def test_lifelog_builder_uses_segment_fingerprint_for_resume(tmp_path: Path) -> None:
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
        del duration_sec
        calls.append(Path(video_path).stem)
        return ((0.0, 20.0),)

    builder = LifeLogColdIndexBuilder(
        manifest,
        LifeLogIndexConfig(max_range_sec=20.0, max_beat_sec=20.0),
        model=KeywordColorModel(),
        range_detector=range_detector,
        keyframe_sampler=_sampler,
    )
    first = builder.build(tmp_path / "run", resume=True)
    second = builder.build(tmp_path / "run", resume=True)

    state_path = tmp_path / "run" / "segments" / "seg_b" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert calls == ["seg_a", "seg_b"]
    assert state["schema"] == "vcah.xle.segment_state.v1"
    assert state["fingerprint_inputs"]["video_uid"] == "seg_b"
    assert state["fingerprint_inputs"]["duration_sec"] == 20.0
    assert state["fingerprint_inputs"]["max_range_sec"] == 20.0
    assert first.segment("seg_a").resumed is False
    assert second.segment("seg_a").resumed is True

    changed = load_xlebench_manifest(root, video_template=str(root / "{video_uid}.mp4"))
    changed_segments = tuple(
        type(segment)(segment.video_uid, segment.video_path, 24.0 if segment.video_uid == "seg_b" else segment.duration_sec, segment.virtual_start_sec)
        for segment in changed.segments
    )
    changed_manifest = type(changed)(changed_segments, changed.cases)
    LifeLogColdIndexBuilder(
        changed_manifest,
        LifeLogIndexConfig(max_range_sec=20.0, max_beat_sec=20.0),
        model=KeywordColorModel(),
        range_detector=range_detector,
        keyframe_sampler=_sampler,
    ).build(tmp_path / "run", resume=True)

    assert calls == ["seg_a", "seg_b", "seg_b"]


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


def test_xle_frame_refs_use_sampled_frame_time_not_filename_or_beat_start(tmp_path: Path) -> None:
    root = tmp_path / "xle"
    root.mkdir()
    (root / "cases.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "case-late-blue",
                    "question": "Find the blue object",
                    "video_uid": "seg_long",
                    "videos": [{"video_uid": "seg_long", "duration_sec": 18000.0, "video_path": "seg_long.mp4"}],
                    "query_range": {"start_sec": 17000.0, "end_sec": 17010.0},
                    "gt_interval": {"start_sec": 17004.0, "end_sec": 17006.0},
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = load_xlebench_manifest(root, video_template=str(root / "{video_uid}.mp4"))
    index = LifeLogColdIndexBuilder(
        manifest,
        LifeLogIndexConfig(max_range_sec=20.0, max_beat_sec=20.0),
        model=KeywordColorModel(),
        range_detector=lambda _path, _duration: ((17000.0, 17010.0),),
        keyframe_sampler=_misleading_name_sampler,
    ).build(tmp_path / "run")

    result = LifeLogRetriever(index).retrieve("blue object", scope=manifest.cases[0].scope, top_k=1)

    assert result.candidates[0].source_start_sec == 17000.0
    assert Path(result.candidates[0].frame_refs[0].path).name == "frame_001.jpg"
    assert result.candidates[0].frame_refs[0].source_time_sec == 17005.0
    assert result.candidates[0].frame_refs[0].virtual_time_sec == 17005.0


def test_xle_retriever_uses_frame_level_visual_index_not_keyframe_only(tmp_path: Path) -> None:
    root = tmp_path / "xle"
    root.mkdir()
    (root / "cases.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "case-blue-frame",
                    "question": "Find the blue object",
                    "video_uid": "seg_a",
                    "videos": [{"video_uid": "seg_a", "duration_sec": 20.0, "video_path": "seg_a.mp4"}],
                    "query_range": {"start_sec": 0.0, "end_sec": 20.0},
                    "gt_interval": {"start_sec": 0.0, "end_sec": 20.0},
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = load_xlebench_manifest(root, video_template=str(root / "{video_uid}.mp4"))
    index = LifeLogColdIndexBuilder(
        manifest,
        LifeLogIndexConfig(max_range_sec=20.0, max_beat_sec=20.0),
        model=KeywordColorModel(),
        range_detector=lambda _path, duration: ((0.0, duration),),
        keyframe_sampler=_red_keyframe_blue_second_sampler,
    ).build(tmp_path / "run")

    result = LifeLogRetriever(index).retrieve("blue object", scope=manifest.cases[0].scope, top_k=1)
    report = diagnose_cold_recall(index, manifest.cases, top_ks=(5,))

    assert result.candidates
    assert "visual" in result.candidates[0].modalities
    assert result.candidates[0].frame_refs[1].source_time_sec == 5.0
    assert report["counts"]["frames"] == 2
    assert report["counts"]["embeddings"] == 2


def test_lifelog_load_without_model_keeps_frame_visual_search_safe(tmp_path: Path) -> None:
    root = tmp_path / "xle"
    root.mkdir()
    (root / "cases.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "case-blue-frame",
                    "question": "Find the blue object",
                    "video_uid": "seg_a",
                    "videos": [{"video_uid": "seg_a", "duration_sec": 20.0, "video_path": "seg_a.mp4"}],
                    "query_range": {"start_sec": 0.0, "end_sec": 20.0},
                    "gt_interval": {"start_sec": 0.0, "end_sec": 20.0},
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = load_xlebench_manifest(root, video_template=str(root / "{video_uid}.mp4"))
    LifeLogColdIndexBuilder(
        manifest,
        LifeLogIndexConfig(max_range_sec=20.0, max_beat_sec=20.0),
        model=KeywordColorModel(),
        range_detector=lambda _path, duration: ((0.0, duration),),
        keyframe_sampler=_red_keyframe_blue_second_sampler,
    ).build(tmp_path / "run")

    loaded = LifeLogColdIndex.load(tmp_path / "run")
    result = LifeLogRetriever(loaded).retrieve("blue object", scope=manifest.cases[0].scope, top_k=1)

    assert loaded.segment("seg_a").frame_visual_index is not None
    assert result.candidates == ()


def test_xle_retrieval_reports_minimal_hierarchical_debug(tmp_path: Path) -> None:
    root = tmp_path / "xle"
    root.mkdir()
    (root / "cases.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "case-blue-frame",
                    "question": "Find the blue object",
                    "video_uid": "seg_b",
                    "videos": [
                        {"video_uid": "seg_a", "duration_sec": 20.0, "video_path": "seg_a.mp4"},
                        {"video_uid": "seg_b", "duration_sec": 20.0, "video_path": "seg_b.mp4"},
                    ],
                    "query_range": {"start_sec": 0.0, "end_sec": 20.0},
                    "gt_interval": {"start_sec": 0.0, "end_sec": 20.0},
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = load_xlebench_manifest(root, video_template=str(root / "{video_uid}.mp4"))
    index = LifeLogColdIndexBuilder(
        manifest,
        LifeLogIndexConfig(max_range_sec=20.0, max_beat_sec=20.0),
        model=KeywordColorModel(),
        range_detector=lambda _path, duration: ((0.0, duration),),
        keyframe_sampler=_sampler,
    ).build(tmp_path / "run")

    result = LifeLogRetriever(index).retrieve("blue object", scope=manifest.cases[0].scope, top_k=5)
    report = diagnose_cold_recall(index, manifest.cases, top_ks=(5,))

    assert result.debug["retrieval_mode"] == "hierarchical-mvp"
    assert result.per_level_hits["segment"] == ("seg_b",)
    assert result.per_level_hits["beat"][0].startswith("seg_b:")
    assert result.per_level_hits["frame"][0].startswith("seg_b:")
    assert result.fusion_weights == {"text": 1.0, "visual": 1.0, "segment": 0.05}
    assert report["per_level_recall"]["segment@5"] == 1.0
    assert report["per_level_recall"]["beat@5"] == 1.0
    assert report["per_level_recall"]["frame@5"] == 1.0


def test_xle_investigator_splits_long_candidate_and_verifies_late_claim(tmp_path: Path) -> None:
    root = tmp_path / "xle"
    root.mkdir()
    (root / "cases.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "case-table",
                    "question": "What did I put on the table?",
                    "video_uid": "seg_a",
                    "videos": [{"video_uid": "seg_a", "duration_sec": 120.0, "video_path": "seg_a.mp4"}],
                    "query_range": {"start_sec": 0.0, "end_sec": 120.0},
                    "gt_interval": {"start_sec": 90.0, "end_sec": 105.0},
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = load_xlebench_manifest(root, video_template=str(root / "{video_uid}.mp4"))
    model = FakeXLEInvestigatorModel()
    index = LifeLogColdIndexBuilder(
        manifest,
        LifeLogIndexConfig(max_range_sec=120.0, max_beat_sec=120.0),
        model=model,
        range_detector=lambda _path, duration: ((0.0, duration),),
        keyframe_sampler=_late_table_sampler,
    ).build(tmp_path / "run")

    result = LifeLogInvestigator(index, model=model, max_steps=3, inspect_top_n=1, max_window_sec=30.0).answer(manifest.cases[0])

    assert result.plan.task_type == "object"
    assert len(model.calls) == 4
    assert [call["start_sec"] for call in model.calls] == [0.0, 30.0, 60.0, 90.0]
    assert result.answer == "blue cup"
    assert result.selected_interval is not None
    assert result.selected_interval.source_start_sec == 90.0
    assert result.selected_interval.source_end_sec == 120.0
    assert result.verified_claim is not None
    assert result.verified_claim.status == "supported"
    assert result.trace[0]["type"] == "plan"
    assert any(step["type"] == "inspect_window" for step in result.trace)
