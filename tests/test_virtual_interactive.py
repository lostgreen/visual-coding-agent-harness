from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from vcah.investigator import _choose_window_from_segment_packet
from vcah.multiround import InvestigationTask
from vcah.types import EvidenceRecord, Frame
from vcah import virtual_video
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
)


def _load_tool_module(name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parents[1] / "tools" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_viewer = _load_tool_module("virtual_trace_viewer", "build_virtual_trace_viewer.py")
_interactive = _load_tool_module("virtual_interactive_runner", "run_virtual_videomme_interactive.py")
AssetBundler = _viewer.AssetBundler
_render_case = _viewer._render_case
GeminiInvestigator = _interactive.GeminiInvestigator
_select_window_with_model = _interactive._select_window_with_model


def _sampler(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
    del video_path, end_sec, n_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"frame_{start_sec:.3f}.jpg"
    Image.new("RGB", (32, 18), color=(30, 90, 180)).save(path)
    return (Frame(frame_id=path.stem, time_sec=float(start_sec), path=str(path)),)


class ScriptedVisionClient:
    def __init__(self, responses: Sequence[Mapping[str, Any]]) -> None:
        self.responses = [dict(item) for item in responses]
        self.calls: list[dict[str, Any]] = []

    def chat(self, prompt: str, *, image_paths: Sequence[str] = (), max_tokens: int = 900) -> str:
        self.calls.append({"prompt": prompt, "image_paths": tuple(image_paths), "max_tokens": max_tokens})
        return json.dumps(self.responses.pop(0))


def _workspace(tmp_path: Path) -> VirtualVideoWorkspace:
    manifest = VirtualVideoManifest(
        workspace_id="interactive",
        segments=(
            VirtualVideoSegment(
                segment_id="seg_0001",
                source_video_id="source",
                source_path="source.mp4",
                source_start_sec=0.0,
                source_end_sec=180.0,
                virtual_start_sec=0.0,
                virtual_end_sec=180.0,
                role="content",
            ),
        ),
    )
    case = VirtualVideoCase(
        case_id="interactive",
        question="What number appears on the board?",
        options={"A": "7", "B": "9"},
        gold="B",
        target_segment_id="seg_0001",
        target_virtual_interval=(40.0, 60.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "workspace", manifest=manifest, case=case)
    workspace.write_asr_virtual_cues(
        (
            {"start_sec": 5.0, "end_sec": 7.0, "text": "an unrelated introduction", "segment_id": "seg_0001"},
            {"start_sec": 48.0, "end_sec": 52.0, "text": "the number on the board is discussed", "segment_id": "seg_0001"},
        )
    )
    thumbnail = workspace.root_dir / "beat.jpg"
    Image.new("RGB", (64, 36), color=(80, 80, 80)).save(thumbnail)
    (workspace.root_dir / "beat_index.json").write_text(
        json.dumps(
            {
                "beats": [
                    {
                        "beat_id": "bt0001",
                        "virtual_time_range": [0.0, 180.0],
                        "thumbnail_grid_path": str(thumbnail),
                        "thumbnail_grid_paths": [str(thumbnail)],
                        "asr_cues": workspace.read_asr_virtual_cues(),
                        "source_lineage": [
                            {
                                "segment_id": "seg_0001",
                                "source_video_id": "source",
                                "source_time_range": [0.0, 180.0],
                                "virtual_time_range": [0.0, 180.0],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return workspace


def test_uniform_item_selection_spans_the_complete_input() -> None:
    selected = virtual_video.select_uniform_items(tuple(range(64)), 16)

    assert len(selected) == 16
    assert selected[0] == 0
    assert selected[-1] == 63
    assert selected != tuple(range(16))


def test_window_choice_clusters_all_asr_hits_before_selecting() -> None:
    task = InvestigationTask(
        query_id="q1",
        goal="Find scholars who comment on Napoleon.",
        expected_evidence="scholar comments about Napoleon",
    )
    packet = {
        "virtual_time_range": [0.0, 600.0],
        "asr_cues": [
            {"start_sec": 10.0, "end_sec": 12.0, "text": "Napoleon"},
            {"start_sec": 300.0, "end_sec": 304.0, "text": "a scholar comments on Napoleon"},
            {"start_sec": 309.0, "end_sec": 313.0, "text": "another scholar discusses Napoleon"},
        ],
        "beats": [],
    }

    start, end = _choose_window_from_segment_packet(task, packet)

    assert 290.0 <= start < 305.0
    assert 310.0 < end <= 325.0


def test_invalid_model_window_uses_clustered_fallback(tmp_path: Path) -> None:
    task = InvestigationTask(
        query_id="q1",
        goal="Find scholars who comment on Napoleon.",
        expected_evidence="scholar comments about Napoleon",
    )
    packet = {
        "virtual_time_range": [0.0, 600.0],
        "asr_cues": [
            {"start_sec": 10.0, "end_sec": 12.0, "text": "Napoleon"},
            {"start_sec": 300.0, "end_sec": 304.0, "text": "a scholar comments on Napoleon"},
            {"start_sec": 309.0, "end_sec": 313.0, "text": "another scholar discusses Napoleon"},
        ],
        "beats": [],
    }
    api = ScriptedVisionClient(({},))
    trace_path = tmp_path / "trace.jsonl"

    start, end = _select_window_with_model(api, task, packet, trace_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    assert 290.0 <= start < 305.0
    assert 310.0 < end <= 325.0
    assert trace["fallback_used"] is True


def test_model_investigator_uses_preview_then_narrow_uniform_detail(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {"start_sec": 0.0, "end_sec": 120.0, "reason": "candidate region"},
            {
                "summary": "The board is present but the number is too small.",
                "confidence": 0.4,
                "need_detail": True,
                "detail_start_sec": 40.0,
                "detail_end_sec": 60.0,
                "reason": "read the board",
            },
            {
                "summary": "The board shows the number nine beside one presenter.",
                "confidence": 0.95,
                "supports_identity_anchor": False,
                "supports_answer_event": True,
                "entities": [
                    {
                        "local_id": "person_1",
                        "description": "presenter in a dark jacket",
                        "role": "presenter",
                        "question_relation": "stands beside the numbered board",
                        "supports_question_relation": True,
                    }
                ],
            },
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="r1_t1",
        goal="Read the number on the board.",
        segment_id="seg_0001",
        modality_hint=("visual", "ocr"),
        expected_evidence="number written on the board",
    )

    report = investigator.run_batch((task,))[0]

    assert len(api.calls) == 3
    preview_call = api.calls[1]
    assert "Preview window metadata" in preview_call["prompt"]
    assert "number on the board is discussed" in preview_call["prompt"]
    assert len(preview_call["image_paths"]) == 16
    assert "frame_0.000.jpg" in preview_call["image_paths"][0]
    assert "frame_120.000.jpg" in preview_call["image_paths"][-1]
    assert isinstance(report.evidence[0], EvidenceRecord)
    assert (report.evidence[0].start_sec, report.evidence[0].end_sec) == (40.0, 60.0)
    assert report.evidence[0].sampling_fps == 2.0
    assert report.evidence[0].attestation_model
    assert report.evidence[0].source_lineage[0]["source_video_id"] == "source"
    assert report.evidence[0].evidence_kind == "event_observation"
    assert report.evidence[0].operation_metadata["supports_answer_event"] is True
    assert report.evidence[0].operation_metadata["entities"][0]["local_id"] == "person_1"
    assert len(report.evidence[0].frame_refs) == 16
    assert "frame_40.000.jpg" in report.evidence[0].frame_refs[0]
    assert "frame_60.000.jpg" in report.evidence[0].frame_refs[-1]


def test_investigator_prompt_and_entity_schema_are_question_generic(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    task = InvestigationTask(
        query_id="r1_t1",
        goal="Check whether the presenter points to the diagram.",
        segment_id="seg_0001",
        modality_hint=("visual",),
        expected_evidence="presenter pointing to the diagram",
    )
    segment_packet = {
        "segment_id": "seg_0001",
        "virtual_time_range": [0.0, 180.0],
        "asr_timeline_summary": "The presenter explains a diagram.",
        "beat_count": 3,
        "thumbnail_grid_paths": [],
    }
    window = {
        "virtual_time_range": [30.0, 60.0],
        "sampling": {"fps": 0.5, "frame_count": 16},
        "asr_cues": [],
        "source_lineage": [],
    }

    preview_prompt = _interactive._preview_prompt(workspace, task, segment_packet, window)
    detail_prompt = _interactive._evidence_prompt(workspace, task, segment_packet, window)
    entities = _interactive._normalize_entities(
        [
            {
                "local_id": "person_1",
                "description": "presenter in a green jacket",
                "role": "presenter",
                "question_relation": "points to the diagram",
                "supports_question_relation": True,
            }
        ]
    )

    assert "scholar" not in preview_prompt.casefold()
    assert "comments_on_topic" not in preview_prompt
    assert "scholar" not in detail_prompt.casefold()
    assert "comments_on_topic" not in detail_prompt
    assert "supports_question_relation" in preview_prompt
    assert entities[0]["question_relation"] == "points to the diagram"
    assert entities[0]["supports_question_relation"] is True


def test_model_investigator_stops_after_sufficient_preview(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {"start_sec": 0.0, "end_sec": 120.0, "reason": "candidate region"},
            {"summary": "The visible board clearly shows nine.", "confidence": 0.95, "need_detail": False},
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="r1_t1",
        goal="Check what is visible on the board.",
        segment_id="seg_0001",
        modality_hint=("visual",),
        expected_evidence="visible board content",
    )

    report = investigator.run_batch((task,))[0]

    assert len(api.calls) == 2
    assert report.evidence[0].sampling_fps == 0.5
    assert report.cost["tool_trace"] == ("open_segment", "inspect_window:0.5")


def test_repeated_query_ids_get_distinct_observation_and_frame_ids(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {"start_sec": 0.0, "end_sec": 30.0},
            {"summary": "First pass.", "confidence": 0.8, "need_detail": False},
            {"start_sec": 0.0, "end_sec": 30.0},
            {"summary": "Second pass.", "confidence": 0.8, "need_detail": False},
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="r1_t1",
        goal="Inspect the visible board.",
        segment_id="seg_0001",
        modality_hint=("visual",),
        expected_evidence="visible board content",
    )

    reports = investigator.run_batch((task, task))
    manifest_rows = [
        json.loads(line)
        for line in (workspace.root_dir / "observations" / "window_frame_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert reports[0].evidence[0].evidence_id == "ev_r1_t1_c01_001"
    assert reports[1].evidence[0].evidence_id == reports[0].evidence[0].evidence_id
    assert reports[1].cost["reused"] is True
    assert len(api.calls) == 3
    frame_ids = [row["frame_id"] for row in manifest_rows]
    assert len(frame_ids) == len(set(frame_ids))
    ledger_rows = [
        json.loads(line)
        for line in (workspace.root_dir / "exploration_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert ledger_rows[-1]["reused_from"] == reports[0].evidence[0].evidence_id


def test_trace_viewer_renders_every_reasoner_round(tmp_path: Path) -> None:
    workspace = tmp_path / "run" / "workspaces" / "case-1"
    workspace.mkdir(parents=True)
    (workspace / "case.json").write_text(
        json.dumps({"question": "Q?", "options": {"A": "one"}, "gold": "A"}), encoding="utf-8"
    )
    (workspace / "virtual_timeline.json").write_text(
        json.dumps({"duration_sec": 120.0, "segments": []}), encoding="utf-8"
    )
    (workspace / "run_summary.json").write_text(
        json.dumps({"answer": "A. one", "correct": True, "rounds": 2, "accepted_investigations": 2, "evidence": []}),
        encoding="utf-8",
    )
    (workspace / "beat_index.json").write_text(json.dumps({"beats": []}), encoding="utf-8")
    (workspace / "evidence.jsonl").write_text(
        json.dumps({"evidence_id": "ev_r1_t1_001", "modality": "visual", "sampling_fps": 0.5}) + "\n",
        encoding="utf-8",
    )
    (workspace / "exploration_ledger.jsonl").write_text(
        json.dumps({"visit_id": "visit_0001", "status": "reused", "reused_from": "ev_r1_t1_001"}) + "\n",
        encoding="utf-8",
    )
    trace = (
        {
            "type": "reasoner_investigate",
            "round": 1,
            "prompt": "round one prompt",
            "raw": "round one raw",
            "parsed": {"tasks": [{"query_id": "r1_t1", "segment_id": "seg_0001"}]},
        },
        {"type": "reasoner_investigate", "round": 2, "prompt": "round two prompt", "raw": "round two raw", "parsed": {"tasks": [{"query_id": "r2_t1", "segment_id": "seg_0002"}] }},
        {"type": "reasoner_answer", "round": 3, "prompt": "answer prompt", "raw": "answer raw", "parsed": {"answer": "A. one"}},
    )
    (workspace / "interactions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trace), encoding="utf-8"
    )

    assets = tmp_path / "viewer" / "assets"
    bundle = AssetBundler(run_root=tmp_path / "run", assets_dir=assets, case_id="case-1")
    html, _ = _render_case(workspace, bundle)

    assert "Reasoner Round 1" in html
    assert "Reasoner Round 2" in html
    assert "r1_t1" in html
    assert "r2_t1" in html
    assert "round one raw" in html
    assert "round two raw" in html
    assert "Structured Evidence Store" in html
    assert "Exploration Ledger" in html
    assert "visit_0001" in html
    assert "reused_from" in html
