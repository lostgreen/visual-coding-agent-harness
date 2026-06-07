import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from visual_coding_agent_harness.agents.iterative_agent import AgentBudget, IterativeVisualAgent
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.registry import ToolRegistry, tool
from visual_coding_agent_harness.video_index import SceneIndex, VideoSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "replay"


class ReplayBackend(VisionLanguageBackend):
    def __init__(self, responses: list[Mapping[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if not self.responses:
            if request.task == "answer_from_evidence":
                return BackendResponse(
                    text=(
                        '{"answer": "need_more_evidence", "citations": [], '
                        '"missing_evidence": ["replay fixture has no scripted AnswerAgent response"], '
                        '"confidence": 0.0}'
                    )
                )
            raise AssertionError(f"Unexpected backend.generate call for task={request.task}")
        response = self.responses.pop(0)
        expected_task = str(response.get("task", ""))
        if expected_task and expected_task != request.task:
            raise AssertionError(f"Expected backend task {expected_task}, got {request.task}")
        return BackendResponse(text=str(response.get("text", "")))


@pytest.mark.parametrize("fixture_name", ["605_1.json", "611_2.json", "612_1.json"])
def test_videomme_three_case_replay_contracts(tmp_path: Path, fixture_name: str):
    fixture = _load_fixture(fixture_name)
    workspace = EvidenceWorkspace.create(tmp_path, run_id=f"replay_{fixture['case_id'].replace('-', '_')}")
    backend = ReplayBackend(list(fixture.get("backend_responses", [])))
    agent = IterativeVisualAgent(
        backend=backend,
        registry=_build_replay_registry(fixture),
        workspace=workspace,
        scene_index=_scene_index(fixture),
        budget=AgentBudget(
            max_rounds=1,
            max_tool_calls_per_round=2,
            reserve_final_round=False,
            hard_skill_runtime=True,
            default_nframes=8,
        ),
    )

    result = agent.run(question=str(fixture["question"]), video_path="/videos/replay.mp4")

    if fixture["case_id"] == "605-1":
        assert result.status != "final" or result.answer.strip().startswith("D")
        assert not (result.status == "final" and result.answer.strip().startswith("B"))
    elif fixture["case_id"] == "611-2":
        assert result.status != "max_rounds_reached"
        assert result.status == "low_confidence_final" or result.answer.strip().startswith("D")
    elif fixture["case_id"] == "612-1":
        assert result.status != "final" or result.answer.strip().startswith("B")
        assert not (result.status == "final" and result.answer.strip().startswith("D"))

    assert backend.responses == []


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _scene_index(fixture: Mapping[str, Any]) -> SceneIndex:
    segments_payload = fixture.get("segments") or [
        {"segment_id": "seg_0001", "start_sec": 0.0, "end_sec": float(fixture.get("duration_sec", 30.0))}
    ]
    segments = [
        VideoSegment(
            segment_id=str(item["segment_id"]),
            start_sec=float(item["start_sec"]),
            end_sec=float(item["end_sec"]),
        )
        for item in segments_payload
    ]
    return SceneIndex(
        video_path="/videos/replay.mp4",
        duration_sec=float(fixture.get("duration_sec", segments[-1].end_sec if segments else 0.0)),
        segments=segments,
    )


def _build_replay_registry(fixture: Mapping[str, Any]) -> ToolRegistry:
    registry = ToolRegistry()
    tools = fixture.get("tools", {}) if isinstance(fixture.get("tools", {}), Mapping) else {}

    @tool(name="global_gist", description="Replay sparse whole-video gist.")
    def global_gist(
        video_path: str,
        question: str,
        duration_sec: float,
        nframes: int = 64,
        max_pixels: int = 151200,
        seed: int = 0,
    ):
        payload = dict(_tool_payload(tools.get("global_gist", {}), str(seed)))
        return {
            "claim": payload.get("claim", ""),
            "confidence": float(payload.get("confidence", 0.0)),
            "input_artifacts": [video_path],
            "regions": [
                {
                    "start_sec": 0.0,
                    "end_sec": duration_sec,
                    "nframes": nframes,
                    "max_pixels": max_pixels,
                    "seed": seed,
                }
            ],
            "supported_option": payload.get("supported_option", ""),
            "grounding_quality": payload.get("grounding_quality", "global_sparse"),
        }

    registry.register(global_gist)

    @tool(name="caption_segment", description="Replay coarse temporal segment caption.")
    def caption_segment(
        video_path: str,
        segment_id: str,
        start_sec: float,
        end_sec: float,
        question: str = "",
        nframes: int = 8,
    ):
        claim = str(_tool_payload(tools.get("caption_segment", {}), segment_id, default="No target event appears."))
        return {
            "claim": claim,
            "confidence": 0.82,
            "input_artifacts": [video_path],
            "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec, "nframes": nframes}],
        }

    registry.register(caption_segment)

    @tool(name="ground_question", description="Replay grounding candidates.")
    def ground_question(query: str, top_k: int = 3):
        payload = dict(_lookup_by_query(tools.get("ground_question", {}), query))
        candidate = {
            "segment_id": str(payload.get("segment_id", "seg_0001")),
            "start_sec": float(payload.get("start_sec", 0.0)),
            "end_sec": float(payload.get("end_sec", 20.0)),
            "confidence": float(payload.get("confidence", 0.8)),
            "reason": "replay candidate",
        }
        return {
            "claim": f"Candidate window for {query}.",
            "confidence": candidate["confidence"],
            "regions": [candidate],
            "candidates": [candidate],
            "limitations": f"Replay grounding top_k={top_k}.",
        }

    registry.register(ground_question)

    @tool(name="vision_read", description="Replay localized visual read.")
    def vision_read(
        video_path: str,
        segment_id: str,
        start_sec: float,
        end_sec: float,
        ask_for: str,
        event_label: str = "",
        mutex_group_id: str = "",
        nframes: int = 8,
    ):
        payload = dict(_lookup_by_query(tools.get("vision_read", {}), event_label or ask_for))
        claim = str(payload.get("claim", f"{event_label or ask_for} is visible."))
        confidence = float(payload.get("confidence", 0.9))
        supported_option = str(payload.get("supported_option", ""))
        output = {
            "claim": claim,
            "confidence": confidence,
            "input_artifacts": [video_path],
            "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
            "event_label": event_label or ask_for,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "grounding_quality": payload.get("grounding_quality", "visually_confirmed"),
        }
        if "observed_at_sec" in payload:
            output["observed_at_sec"] = float(payload["observed_at_sec"])
        if supported_option:
            output["supported_option"] = supported_option
            output["candidate_option_relations"] = [
                {"option": supported_option, "relation": "support", "strength": confidence}
            ]
        if mutex_group_id:
            output["mutex_group_id"] = mutex_group_id
        return output

    registry.register(vision_read)
    return registry


def _tool_payload(source: Any, key: str, *, default: Any | None = None) -> Any:
    if isinstance(source, Mapping) and key in source:
        return source[key]
    if default is not None:
        return default
    return {}


def _lookup_by_query(source: Any, query: str) -> Mapping[str, Any]:
    if not isinstance(source, Mapping):
        return {}
    lowered = str(query).lower()
    for key, value in source.items():
        if str(key).lower() == lowered:
            return value if isinstance(value, Mapping) else {}
    for key, value in source.items():
        if str(key).lower() in lowered:
            return value if isinstance(value, Mapping) else {}
    return {}
