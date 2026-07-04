from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image

from visual_coding_agent_harness.agents.driver import MultiV3Driver
from visual_coding_agent_harness.agents.investigator import Investigator
from visual_coding_agent_harness.agents.reasoner import Reasoner
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse
from visual_coding_agent_harness.contracts.query import QueryBudget, QueryScope, ScopedQuery, VerifiableGoal
from visual_coding_agent_harness.contracts.report import CandidateShot, Finding, VerifyRequest
from visual_coding_agent_harness.tools.vlm_tools import ExploreResult
from visual_coding_agent_harness.video.index import Frame, Scene, Shot, VideoIndex
from visual_coding_agent_harness.workspace.investigator_ws import InvestigatorWorkspace


class RecordingBackend:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=self.responses.pop(0))


def _index() -> VideoIndex:
    shot = Shot(
        shot_id="sc01_sh001",
        scene_id="sc01",
        start_sec=0.0,
        end_sec=8.0,
        frames=(Frame(frame_id="fr1", time_sec=1.0, thumb_path="/thumbs/fr1.jpg"),),
        visual_caption="A red car on a street.",
        asr_text="",
        ocr_lines=(),
        entities=("car",),
        lowres_grid_path="/grids/sc01_sh001.jpg",
    )
    return VideoIndex(
        video_path="/videos/demo.mp4",
        duration_sec=8.0,
        scenes=(
            Scene(
                scene_id="sc01",
                start_sec=0.0,
                end_sec=8.0,
                title="Street",
                summary="A red car appears.",
                shots=(shot,),
                scene_thumb_path="/thumbs/sc01.jpg",
            ),
        ),
    )


def _query(query_id: str = "q1") -> ScopedQuery:
    return ScopedQuery(
        query_id=query_id,
        goal_id="g1",
        natural_query="Find the red car.",
        scope=QueryScope(scene_ids=("sc01",), entity_hints=("car",), modality_hint=("visual",)),
        expected_evidence="A red car is visible.",
        budget=QueryBudget(max_shots_to_verify=1, max_frames=16),
    )


def test_reasoner_parses_plan_action_into_goals_and_scoped_queries() -> None:
    backend = RecordingBackend(
        [
            json.dumps(
                {
                    "action": "plan",
                    "goals": [
                        {
                            "goal_id": "g1",
                            "text": "Find red car evidence.",
                            "linked_options": ["A", "C"],
                            "kind": "locate",
                            "priority": 0.9,
                        }
                    ],
                    "queries": [
                        {
                            "query_id": "q1",
                            "goal_id": "g1",
                            "natural_query": "Find the red car.",
                            "scope": {
                                "scene_ids": ["sc01"],
                                "time_range": [0, 8],
                                "entity_hints": ["car"],
                                "modality_hint": ["visual"],
                            },
                            "expected_evidence": "A verified red car sighting.",
                            "budget": {"max_shots_to_verify": 1, "max_frames": 16},
                        }
                    ],
                    "rationale": "Need visual evidence.",
                }
            )
        ]
    )

    decision = Reasoner(backend=backend).decide(
        question="Which option is supported?",
        options={"A": "red car", "C": "blue car"},
        index_context="sc01 [0-8] Street",
        overview_image_path="/overview/scene_timeline_grid.json",
        previous_digest=(),
        round_number=1,
    )

    assert decision.action == "plan"
    assert decision.goals == (
        VerifiableGoal("g1", "Find red car evidence.", ("A", "C"), "locate", 0.9),
    )
    assert decision.queries[0].scope.scene_ids == ("sc01",)
    assert backend.requests[0].task == "multi_v3_reasoner"
    assert backend.requests[0].media_type is None
    assert "PreviousDigest" in backend.requests[0].prompt


def test_reasoner_only_attaches_existing_image_overview(tmp_path: Path) -> None:
    overview = tmp_path / "scene_timeline_grid.jpg"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(overview)
    backend = RecordingBackend([json.dumps({"action": "answer", "answer": "A"})])

    Reasoner(backend=backend).decide(
        question="Which option is supported?",
        options={"A": "red car"},
        index_context="sc01",
        overview_image_path=str(overview),
        previous_digest=(),
        round_number=1,
    )

    assert backend.requests[0].media_path == str(overview)
    assert backend.requests[0].media_type == "image"


def test_reasoner_parses_answer_action_into_result() -> None:
    backend = RecordingBackend(
        [
            json.dumps(
                {
                    "action": "answer",
                    "answer": "A",
                    "confidence": "high",
                    "citations": ["ev_1"],
                    "rationale": "Evidence supports option A.",
                    "goals": [{"goal_id": "g1", "text": "Verify the red car.", "linked_options": ["A"], "kind": "locate"}],
                }
            )
        ]
    )

    decision = Reasoner(backend=backend).decide(
        question="Which option is supported?",
        options={"A": "red car"},
        index_context="sc01",
        overview_image_path="/overview.json",
        previous_digest=(),
        round_number=2,
    )

    assert decision.action == "answer"
    assert decision.answer == "A"
    assert decision.confidence == "high"
    assert decision.rationale == "Evidence supports option A."
    assert decision.goals[0].goal_id == "g1"


def test_investigator_runs_explore_then_verify_and_records_report(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_explore(*, query, index, backend, batch_size=16):
        del index, backend, batch_size
        calls.append(f"explore:{query.query_id}")
        return ExploreResult((CandidateShot("sc01_sh001", 0.95, "red car visible"),), batch_count=3)

    def fake_verify(*, query_id: str, request: VerifyRequest, frame_paths, backend):
        del backend
        calls.append(f"verify:{request.shot_id}:{len(tuple(frame_paths))}")
        return (
            Finding(
                finding_id="ev_0001",
                query_id=query_id,
                shot_id=request.shot_id,
                summary="A red car is visible.",
                supports_options=("A",),
                citation_ids=("ev_0001",),
            ),
        )

    investigator = Investigator(
        index=_index(),
        workspace=InvestigatorWorkspace(tmp_path),
        backend=object(),
        explore_fn=fake_explore,
        verify_fn=fake_verify,
        frame_sampler=lambda shot, budget: ("/frames/0001.jpg", "/frames/0002.jpg"),
    )

    report = investigator.run(_query())

    assert calls == ["explore:q1", "verify:sc01_sh001:2"]
    assert report.status == "satisfied"
    assert report.cost["explore_calls"] == 3
    assert report.verified_shots == ("sc01_sh001",)
    assert (tmp_path / "queries" / "q1" / "report.json").exists()
    ledger_rows = [json.loads(line) for line in (tmp_path / "evidence_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["finding_id"] for row in ledger_rows] == ["ev_0001"]


def test_driver_dispatches_queries_in_parallel_and_returns_reasoner_answer(tmp_path: Path) -> None:
    queries = (_query("q1"), _query("q2"))

    class FakeReasoner:
        def __init__(self) -> None:
            self.calls = 0

        def decide(self, **kwargs):
            from visual_coding_agent_harness.agents.reasoner import ReasonerDecision

            self.calls += 1
            if self.calls == 1:
                return ReasonerDecision(action="plan", goals=(), queries=queries, rationale="fan out")
            assert len(kwargs["previous_digest"]) == 2
            return ReasonerDecision(action="answer", answer="A", confidence="medium", citations=("ev_q1", "ev_q2"))

    class SlowInvestigator:
        def run(self, query: ScopedQuery):
            time.sleep(0.08)
            return type(
                "Report",
                (),
                {
                    "query_id": query.query_id,
                    "status": "satisfied",
                    "findings": (
                        Finding(
                            finding_id=f"ev_{query.query_id}",
                            query_id=query.query_id,
                            shot_id="sc01_sh001",
                            summary=f"Finding for {query.query_id}",
                            citation_ids=(f"ev_{query.query_id}",),
                        ),
                    ),
                    "explored_shots": ("sc01_sh001",),
                    "verified_shots": ("sc01_sh001",),
                    "unresolved": (),
                    "cost": {"explore_calls": 1, "verify_calls": 1, "frames_read": 1},
                    "to_dict": lambda self: {},
                },
            )()

    started = time.perf_counter()
    result = MultiV3Driver(
        reasoner=FakeReasoner(),
        investigator=SlowInvestigator(),
        workspace=InvestigatorWorkspace(tmp_path),
        max_rounds=2,
        max_concurrency=2,
    ).run(
        question="Which option is supported?",
        options={"A": "red car"},
        index_context="sc01",
        overview_image_path="/overview.json",
    )

    assert time.perf_counter() - started < 0.15
    assert result.answer == "A"
    assert result.citations == ("ev_q1", "ev_q2")
    assert result.metadata["strategy"] == "multi_v3"


def test_driver_filters_invalid_scene_ids_before_dispatch(tmp_path: Path) -> None:
    mixed_query = _query("q1")
    mixed_query = ScopedQuery(
        query_id=mixed_query.query_id,
        goal_id=mixed_query.goal_id,
        natural_query=mixed_query.natural_query,
        scope=QueryScope(scene_ids=("sc01", "missing_scene")),
        expected_evidence=mixed_query.expected_evidence,
        budget=mixed_query.budget,
    )

    class FakeReasoner:
        def decide(self, **kwargs):
            del kwargs
            from visual_coding_agent_harness.agents.reasoner import ReasonerDecision

            return ReasonerDecision(action="plan", queries=(mixed_query,), rationale="test invalid scene filtering")

    class RecordingInvestigator:
        def __init__(self) -> None:
            self.queries: list[ScopedQuery] = []

        def run(self, query: ScopedQuery):
            self.queries.append(query)
            return type(
                "Report",
                (),
                {
                    "query_id": query.query_id,
                    "status": "empty",
                    "findings": (),
                    "explored_shots": (),
                    "verified_shots": (),
                    "unresolved": (query.expected_evidence,),
                    "cost": {"explore_calls": 0, "verify_calls": 0, "frames_read": 0},
                    "to_dict": lambda self: {},
                },
            )()

    investigator = RecordingInvestigator()

    MultiV3Driver(
        reasoner=FakeReasoner(),
        investigator=investigator,
        workspace=InvestigatorWorkspace(tmp_path),
        max_rounds=1,
        max_concurrency=1,
        valid_scene_ids=("sc01",),
    ).run(
        question="Which option is supported?",
        options={"A": "red car"},
        index_context="sc01",
    )

    assert investigator.queries[0].scope.scene_ids == ("sc01",)
    trace = (tmp_path / "trace.jsonl").read_text(encoding="utf-8")
    assert "invalid_scene_ids_filtered" in trace
    assert "missing_scene" in trace
