from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest
from PIL import Image

from vcah.agent import VideoAgent
from vcah.model import ScriptedModel
from vcah.types import ClaimVerdict, EvidenceRecord, Frame, InvestigatorOutputInvalid, QueryClaim


class VerifierSpyModel(ScriptedModel):
    def __init__(self) -> None:
        super().__init__(
            actions=[
                {
                    "type": "inspect_window",
                    "start_sec": 0,
                    "end_sec": 4,
                    "modalities": ["asr"],
                    "claims": [
                        {"claim_id": "cl_R1_01", "option": "A", "text": "The bridge is mentioned."},
                        {"claim_id": "cl_R1_02", "option": "B", "text": "The tower is mentioned."},
                    ],
                },
                {"type": "answer", "selected": "A", "answer": "A", "citations": ["ev_0001"]},
            ]
        )
        self.seen_claims: tuple[QueryClaim, ...] = ()
        self.seen_evidence: tuple[EvidenceRecord, ...] = ()

    def verify(self, query_claims: Sequence[QueryClaim], evidence: Sequence[EvidenceRecord]) -> tuple[ClaimVerdict, ...]:
        self.seen_claims = tuple(query_claims)
        self.seen_evidence = tuple(evidence)
        return (
            ClaimVerdict("cl_R1_01", "supported", ("ev_0001",)),
            ClaimVerdict("cl_R1_02", "unknown", ()),
        )


class BadCitationVerifierModel(VerifierSpyModel):
    def verify(self, query_claims: Sequence[QueryClaim], evidence: Sequence[EvidenceRecord]) -> tuple[ClaimVerdict, ...]:
        del query_claims, evidence
        return (ClaimVerdict("cl_R1_01", "supported", ("ev_missing",)),)


class LaterClaimVerifierModel(ScriptedModel):
    def __init__(self) -> None:
        super().__init__(
            actions=[
                {"type": "inspect_window", "start_sec": 0, "end_sec": 4, "modalities": ["asr"]},
                {
                    "type": "search_text",
                    "query": "bridge",
                    "claims": [{"claim_id": "cl_R2_01", "option": "A", "text": "The bridge is mentioned."}],
                },
            ]
        )
        self.seen_evidence: tuple[EvidenceRecord, ...] = ()

    def verify(self, query_claims: Sequence[QueryClaim], evidence: Sequence[EvidenceRecord]) -> tuple[ClaimVerdict, ...]:
        self.seen_evidence = tuple(evidence)
        if any(record.evidence_id == "ev_0001" for record in evidence):
            return (ClaimVerdict("cl_R2_01", "supported", ("ev_0001",)),)
        return (ClaimVerdict("cl_R2_01", "unknown", ()),)


def _sampler(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
    del video_path, end_sec, n_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"frame_{int(start_sec):03d}.jpg"
    Image.new("RGB", (32, 18), color=(20, 40, 230)).save(path)
    return (Frame(frame_id="fr001", time_sec=start_sec, path=str(path)),)


def test_agent_verifier_receives_query_claims_and_trace_splits_evidence_from_verdicts(tmp_path: Path) -> None:
    model = VerifierSpyModel()
    agent = VideoAgent(model=model, max_steps=3)

    answer = agent.ask(
        "/videos/demo.mp4",
        "Which statement is correct?\nA. The bridge is mentioned.\nB. The tower is mentioned.",
        run_dir=tmp_path,
        duration_sec=4.0,
        asr_cues=({"start": 0.0, "end": 4.0, "text": "The bridge is mentioned."},),
        range_detector=lambda _video_path, _duration: ((0.0, 4.0),),
        keyframe_sampler=_sampler,
    )

    trace = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]

    assert answer.citations == ("ev_0001",)
    assert model.seen_claims == (
        QueryClaim("cl_R1_01", "The bridge is mentioned."),
        QueryClaim("cl_R1_02", "The tower is mentioned."),
    )
    assert not hasattr(model.seen_claims[0], "option")
    assert model.seen_evidence[0].evidence_id == "ev_0001"
    assert trace[0]["evidence_records"][0]["evidence_id"] == "ev_0001"
    assert trace[0]["claim_verdicts"][0]["source"] == "verifier"
    assert "claim_verdicts" not in trace[0]["result"]


def test_agent_rejects_verifier_verdict_with_unknown_citation(tmp_path: Path) -> None:
    agent = VideoAgent(model=BadCitationVerifierModel(), max_steps=3)

    with pytest.raises(InvestigatorOutputInvalid):
        agent.ask(
            "/videos/demo.mp4",
            "Which statement is correct?\nA. The bridge is mentioned.\nB. The tower is mentioned.",
            run_dir=tmp_path,
            duration_sec=4.0,
            asr_cues=({"start": 0.0, "end": 4.0, "text": "The bridge is mentioned."},),
            range_detector=lambda _video_path, _duration: ((0.0, 4.0),),
            keyframe_sampler=_sampler,
        )


def test_agent_verifier_can_use_recent_evidence_when_later_claim_creates_no_new_evidence(tmp_path: Path) -> None:
    model = LaterClaimVerifierModel()
    agent = VideoAgent(model=model, max_steps=2)

    agent.ask(
        "/videos/demo.mp4",
        "Which statement is correct?\nA. The bridge is mentioned.\nB. The tower is mentioned.",
        run_dir=tmp_path,
        duration_sec=4.0,
        asr_cues=({"start": 0.0, "end": 4.0, "text": "The bridge is mentioned."},),
        range_detector=lambda _video_path, _duration: ((0.0, 4.0),),
        keyframe_sampler=_sampler,
    )

    assert any(record.evidence_id == "ev_0001" for record in model.seen_evidence)
