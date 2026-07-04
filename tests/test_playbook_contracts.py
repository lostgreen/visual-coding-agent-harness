from __future__ import annotations

import json

from visual_coding_agent_harness.agents.reasoner import Reasoner
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse
from visual_coding_agent_harness.contracts.playbook import Playbook
from visual_coding_agent_harness.contracts.query import QueryBudget, QueryScope, ScopedQuery


class RecordingBackend:
    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=self.response)


def test_scoped_query_parses_playbook_and_chapter_scope_with_scene_alias() -> None:
    query = ScopedQuery.from_dict(
        {
            "query_id": "q1",
            "goal_id": "g1",
            "playbook": "locate_statement",
            "natural_query": "father occupation",
            "scope": {"scene_ids": ["ch01", "ch02"]},
            "scope_b": {"chapter_ids": ["ch07"]},
            "expected_evidence": "father worked at the shipyard",
            "budget": {"max_beats_to_verify": 4, "max_frames": 6},
        }
    )

    assert query.playbook == Playbook.LOCATE_STATEMENT
    assert query.scope.chapter_ids == ("ch01", "ch02")
    assert query.scope.scene_ids == ("ch01", "ch02")
    assert query.scope_b is not None
    assert query.scope_b.chapter_ids == ("ch07",)
    assert query.budget.max_beats_to_verify == 4
    assert query.budget.max_shots_to_verify == 4
    assert query.to_dict()["scope"] == {"chapter_ids": ["ch01", "ch02"], "time_range": None, "entity_hints": [], "modality_hint": []}


def test_reasoner_prompt_requires_playbook_and_falls_back_unknown_playbook() -> None:
    backend = RecordingBackend(
        json.dumps(
            {
                "action": "plan",
                "queries": [
                    {
                        "query_id": "q1",
                        "goal_id": "g1",
                        "playbook": "made_up",
                        "natural_query": "red car",
                        "scope": {"chapter_ids": ["ch01"]},
                        "expected_evidence": "red car",
                    }
                ],
            }
        )
    )

    decision = Reasoner(backend=backend).decide(question="What appears?", options={}, index_context="ch01")

    assert "Available playbooks" in backend.requests[0].prompt
    assert decision.queries[0].playbook == Playbook.IDENTIFY_VISUAL


def test_scoped_query_constructor_accepts_explicit_playbook() -> None:
    query = ScopedQuery(
        query_id="q1",
        goal_id="g1",
        playbook=Playbook.READ_TEXT,
        natural_query="read poster",
        scope=QueryScope(chapter_ids=("ch03",)),
        expected_evidence="poster text",
        budget=QueryBudget(max_beats_to_verify=2),
    )

    assert query.to_dict()["playbook"] == "read_text"
