from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from vcah.interactive_agents import WorkspaceReasoner
from vcah.model_client import (
    MatchedResponseCacheClient,
    MatchedResponseReplayError,
    MatchedResponseSession,
)


class StubClient:
    model = "stub-model"

    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self._last_response_metadata: dict[str, Any] = {}

    def chat(self, prompt: str, **kwargs: Any) -> str:
        del prompt, kwargs
        response = self.responses[self.calls]
        self.calls += 1
        self._last_response_metadata = {
            "finish_reason": "stop",
            "provider_request_id": f"request-{self.calls}",
        }
        return response

    @property
    def last_response_metadata(self) -> Mapping[str, Any]:
        return dict(self._last_response_metadata)

    @property
    def replay_settings(self) -> Mapping[str, Any]:
        return {"model": self.model, "requested_seed": None}

    def set_requested_seed(self, seed: int | None) -> None:
        del seed


def test_matched_response_cache_records_and_replays_by_role(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"same-frame-bytes")
    record_session = MatchedResponseSession(mode="record")
    record_reasoner_delegate = StubClient(("reasoner-pre", "reasoner-post"))
    record_investigator_delegate = StubClient(("investigator-pre",))
    record_reasoner = MatchedResponseCacheClient(
        record_reasoner_delegate,
        root=tmp_path / "fixtures",
        mode="record",
        namespace="reasoner",
        session=record_session,
    )
    record_investigator = MatchedResponseCacheClient(
        record_investigator_delegate,
        root=tmp_path / "fixtures",
        mode="record",
        namespace="investigator",
        session=record_session,
    )

    assert record_reasoner.chat("same prompt", max_tokens=20) == "reasoner-pre"
    assert (
        record_investigator.chat(
            "same visual prompt",
            image_paths=(str(image),),
            image_labels=("frame 1",),
            max_tokens=30,
        )
        == "investigator-pre"
    )
    record_session.deactivate("scoped_occurrence_resolution_persisted")
    assert record_reasoner.chat("post prompt", max_tokens=20) == "reasoner-post"
    record_summary = record_session.to_dict()
    assert record_summary["recorded_count"] == 2
    assert record_summary["live_after_treatment_count"] == 1

    replay_session = MatchedResponseSession(mode="replay")
    replay_reasoner_delegate = StubClient(("live-post",))
    replay_investigator_delegate = StubClient(())
    replay_reasoner = MatchedResponseCacheClient(
        replay_reasoner_delegate,
        root=tmp_path / "fixtures",
        mode="replay",
        namespace="reasoner",
        session=replay_session,
    )
    replay_investigator = MatchedResponseCacheClient(
        replay_investigator_delegate,
        root=tmp_path / "fixtures",
        mode="replay",
        namespace="investigator",
        session=replay_session,
    )

    assert replay_reasoner.chat("same prompt", max_tokens=20) == "reasoner-pre"
    assert (
        replay_investigator.chat(
            "same visual prompt",
            image_paths=(str(image),),
            image_labels=("frame 1",),
            max_tokens=30,
        )
        == "investigator-pre"
    )
    assert replay_reasoner_delegate.calls == 0
    assert replay_investigator_delegate.calls == 0
    replay_session.deactivate("scoped_occurrence_resolution_persisted")
    assert replay_reasoner.chat("post prompt", max_tokens=20) == "live-post"
    replay_summary = replay_session.to_dict()
    assert replay_summary["replayed_count"] == 2
    assert replay_summary["live_after_treatment_count"] == 1
    assert replay_summary["mismatch_count"] == 0


def test_matched_response_replay_rejects_request_mismatch(tmp_path: Path) -> None:
    record_session = MatchedResponseSession(mode="record")
    recorder = MatchedResponseCacheClient(
        StubClient(("recorded",)),
        root=tmp_path / "fixtures",
        mode="record",
        namespace="reasoner",
        session=record_session,
    )
    recorder.chat("expected prompt", max_tokens=20)

    replay_session = MatchedResponseSession(mode="replay")
    replayer = MatchedResponseCacheClient(
        StubClient(()),
        root=tmp_path / "fixtures",
        mode="replay",
        namespace="reasoner",
        session=replay_session,
    )
    with pytest.raises(MatchedResponseReplayError, match="request mismatch"):
        replayer.chat("different prompt", max_tokens=20)
    assert replay_session.to_dict()["mismatch_count"] == 1


def test_workspace_reasoner_deactivates_cache_after_resolution(tmp_path: Path) -> None:
    session = MatchedResponseSession(mode="record")
    delegate = StubClient((json.dumps({"action": "answer", "answer": "A"}),))
    client = MatchedResponseCacheClient(
        delegate,
        root=tmp_path / "fixtures",
        mode="record",
        namespace="reasoner",
        session=session,
    )
    reasoner = WorkspaceReasoner(
        client,
        trace_path=tmp_path / "interactions.jsonl",
        controller_mode="frozen_baseline",
        matched_response_session=session,
    )
    decision = reasoner.decide(
        question="Question?",
        options={"A": "Answer"},
        mechanical_status={
            "occurrence_resolution_state": {
                "schema_version": "OccurrenceResolutionStateV2",
                "active_resolution": "selected",
                "selection_required": False,
                "search_required": False,
            }
        },
        force_finalize=True,
    )

    assert decision.action == "answer"
    summary = session.to_dict()
    assert summary["active"] is False
    assert summary["recorded_count"] == 0
    assert summary["live_after_treatment_count"] == 1
