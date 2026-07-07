from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

from vcah.multiround import InvestigationTask, ReasonerDecision, VirtualVideoMultiRoundDriver
from vcah.investigator import VirtualVideoInvestigator
from vcah.types import Frame
from vcah.virtual_index import build_virtual_beat_index
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
    materialize_lowfps_frame_cache,
)


def _sampler(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
    del end_sec
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(max(1, int(n_frames))):
        time_sec = round(float(start_sec) + index * 0.5, 3)
        path = out_dir / f"{Path(video_path).stem}_{time_sec:.3f}_{index:03d}.jpg"
        Image.new("RGB", (32, 18), color=(20, 40, 230)).save(path)
        frames.append(Frame(frame_id=f"fr{index:03d}", time_sec=time_sec, path=str(path)))
    return tuple(frames)


class TinyModel:
    embedding_dim = 1
    embed_model = "tiny"
    allow_placeholder_visual = True

    def embed_image(self, paths: Sequence[str]):
        import numpy as np

        return np.ones((len(paths), 1), dtype=np.float32)

    def embed_text(self, queries: Sequence[str]):
        import numpy as np

        return np.ones((len(queries), 1), dtype=np.float32)


class ScriptedReasoner:
    def __init__(self) -> None:
        self.calls = 0
        self.kwargs: list[dict[str, object]] = []

    def decide(self, **kwargs: object) -> ReasonerDecision:
        self.calls += 1
        self.kwargs.append(dict(kwargs))
        if self.calls == 1:
            return ReasonerDecision(
                action="investigate",
                tasks=tuple(
                    InvestigationTask(
                        query_id=f"q{i}",
                        goal="Read the number on the jersey.",
                        time_range=(0.0, 2.0),
                        modality_hint=("visual", "ocr"),
                        expected_evidence="number written on jersey",
                    )
                    for i in range(6)
                ),
            )
        return ReasonerDecision(action="answer", answer="B. 11", citations=("ev_q1_001",))


def _workspace(tmp_path: Path) -> VirtualVideoWorkspace:
    manifest = VirtualVideoManifest(
        workspace_id="case-1",
        segments=(VirtualVideoSegment("seg_target", "target", "target.mp4", 10.0, 15.0, 0.0, 5.0, "target"),),
    )
    case = VirtualVideoCase(
        case_id="case-1",
        question="What number is written on the jersey?",
        options={"A": "7", "B": "11"},
        gold="B",
        target_segment_id="seg_target",
        target_virtual_interval=(0.0, 2.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "case-1", manifest=manifest, case=case)
    frames = materialize_lowfps_frame_cache(workspace, fps=1.0, sampler=_sampler)
    build_virtual_beat_index(workspace, frames, model=TinyModel(), beat_sec=3.0)
    return VirtualVideoWorkspace.load(workspace.root_dir)


def test_investigator_auto_escalates_highfps_and_reports_lineage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    investigator = VirtualVideoInvestigator(workspace, sampler=_sampler)
    task = InvestigationTask(
        query_id="q1",
        goal="Read the number on the jersey.",
        time_range=(0.0, 2.0),
        modality_hint=("visual", "ocr"),
        expected_evidence="number written on jersey",
    )

    report = investigator.inspect_window_auto(task)

    assert report.status == "satisfied"
    assert report.evidence
    assert report.evidence[0].evidence_id == "ev_q1_001"
    assert report.evidence[0].sampling["level"] == "highfps"
    assert report.evidence[0].source_lineage[0]["source_video_id"] == "target"
    assert (workspace.root_dir / "observations" / "highfps_frame_manifest.jsonl").exists()


def test_multiround_driver_caps_tasks_and_requires_cited_visual_evidence(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    reasoner = ScriptedReasoner()
    investigator = VirtualVideoInvestigator(workspace, sampler=_sampler)
    driver = VirtualVideoMultiRoundDriver(reasoner=reasoner, investigator=investigator, max_rounds=4, max_investigations=4)

    result = driver.run(workspace)

    assert result.answer == "B. 11"
    assert result.correct is True
    assert result.accepted_investigations == 4
    assert result.rounds == 2
    assert result.citations == ("ev_q1_001",)
    assert result.evidence[0].source_lineage[0]["source_time_range"] == (10.0, 12.0)


def test_reasoner_initial_context_uses_segment_overview_not_cold_candidates(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    reasoner = ScriptedReasoner()
    investigator = VirtualVideoInvestigator(workspace, sampler=_sampler)
    driver = VirtualVideoMultiRoundDriver(reasoner=reasoner, investigator=investigator, max_rounds=1, max_investigations=4)

    driver.run(workspace)

    first_call = reasoner.kwargs[0]
    assert "workspace_overview" in first_call
    assert "cold_candidates" not in first_call
    overview = first_call["workspace_overview"]
    assert isinstance(overview, dict)
    assert overview["thumbnail_count"] == 1
    assert overview["segment_overviews"][0]["segment_id"] == "seg_target"
    assert "target" not in overview["segment_overviews"][0]
