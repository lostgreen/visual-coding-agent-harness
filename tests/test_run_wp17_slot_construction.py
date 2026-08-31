from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from vcah.wp17_slot_memory import WP17_SLOT_TRANSACTION_CONTRACT, SlotMemoryState


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "run_mmlifelong_wp17_slot_construction.py"
)
SPEC = importlib.util.spec_from_file_location("wp17_slot_construction_runner", MODULE_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class _SequenceClient:
    model = "test-model"

    def __init__(self, responses: list[str], finish_reasons: list[str] | None = None):
        self.responses = list(responses)
        self.finish_reasons = list(finish_reasons or ["stop"] * len(responses))
        self.prompts: list[str] = []
        self.last_response_metadata: dict = {}

    def chat(self, prompt: str, **_: object) -> str:
        self.prompts.append(prompt)
        raw = self.responses.pop(0)
        finish_reason = self.finish_reasons.pop(0)
        self.last_response_metadata = {
            "finish_reason": finish_reason,
            "completion_tokens": 100,
            "reasoning_tokens": 10,
            "requested_completion_tokens": 8192,
            "content_chars": len(raw),
            "truncation_retry_count": 0,
        }
        return raw


def _payload(*operations: dict) -> dict:
    return {
        "contract": WP17_SLOT_TRANSACTION_CONTRACT,
        "observations": [
            {
                "observation_id": "obs-1",
                "kind": "event",
                "fact": "A visible action occurs.",
                "evidence_ids": ["f001"],
                "participants": [],
            }
        ],
        "slot_operations": list(operations),
        "structured_event_record": {
            "entities": [],
            "events": [],
            "state_changes": [],
            "relations": [],
            "occurrence_refs": [],
            "summary": "A visible action occurs.",
        },
    }


def _run(client: _SequenceClient, *, arm: str, state: SlotMemoryState | None):
    return RUNNER._run_one(
        client=client,
        arm=arm,
        segment_id="segment-2",
        duration_sec=120.0,
        image_paths=(),
        image_labels=(),
        frame_ids=("f001",),
        ocr_packet=(),
        asr_packet=(),
        history="",
        history_tokens=0,
        history_limit=600,
        input_digests={"frames": "digest"},
        allowed_evidence_ids=("f001",),
        evidence_id_map={"f001": "frame:segment-2:0001"},
        state=state,
        max_completion_tokens=8192,
        remaining_calls=3,
    )


def test_semantic_retry_then_transaction_abstain_preserves_state() -> None:
    state = SlotMemoryState("e1c2")
    initial = _payload(
        {
            "operation": "write",
            "slot": "current_activity",
            "expected_version": 0,
            "value": {"activity": "first"},
            "observation_ids": ["obs-1"],
        }
    )
    initial["observations"][0]["evidence_ids"] = ["frame:init"]
    state.apply(
        initial,
        segment_id="segment-1",
        allowed_evidence_ids=("frame:init",),
    )
    digest_before = state.digest()
    illegal = _payload(
        {
            "operation": "write",
            "slot": "current_activity",
            "expected_version": 1,
            "value": {"activity": "replacement"},
            "observation_ids": ["obs-1"],
        }
    )
    raw = json.dumps(illegal)
    client = _SequenceClient([raw, raw])

    result, consumed = _run(client, arm="e1c2", state=state)

    assert consumed == 2
    assert result["status"] == "success"
    assert result["slot_transaction_abstained"] is True
    assert result["ser_endpoint_eligible"] is False
    assert result["ser_trust_status"] == "untrusted_for_endpoint"
    assert result["model_output"]["slot_operations"] == []
    assert state.digest() == digest_before
    assert result["attempts"][0]["failure_code"] == "write_on_working_slot"
    assert "WP17-slot-memory-repair-v1" in client.prompts[1]
    assert "write_on_working_slot" in client.prompts[1]


def test_malformed_response_uses_serialization_retry_then_succeeds() -> None:
    valid = json.dumps(_payload())
    client = _SequenceClient(["not-json", valid])

    result, consumed = _run(client, arm="e1c0", state=None)

    assert consumed == 2
    assert result["status"] == "success"
    assert result["attempts"][0]["failure_code"] == "response_malformed"
    assert "not a valid complete JSON object" in client.prompts[1]


def test_length_finish_reason_is_classified_as_truncation() -> None:
    valid = json.dumps(_payload())
    client = _SequenceClient(["{", valid], finish_reasons=["length", "stop"])

    result, _ = _run(client, arm="e1c0", state=None)

    assert result["status"] == "success"
    assert result["attempts"][0]["failure_code"] == "response_truncated"


def test_serialization_retry_preserves_prior_semantic_repair_contract() -> None:
    state = SlotMemoryState("e1c2")
    initial = _payload(
        {
            "operation": "write",
            "slot": "current_activity",
            "expected_version": 0,
            "value": {"activity": "first"},
            "observation_ids": ["obs-1"],
        }
    )
    initial["observations"][0]["evidence_ids"] = ["frame:init"]
    state.apply(
        initial,
        segment_id="segment-1",
        allowed_evidence_ids=("frame:init",),
    )
    illegal = _payload(
        {
            "operation": "write",
            "slot": "current_activity",
            "expected_version": 1,
            "value": {"activity": "replacement"},
            "observation_ids": ["obs-1"],
        }
    )
    repaired = _payload(
        {
            "operation": "update",
            "slot": "current_activity",
            "expected_version": 1,
            "value": {"activity": "replacement"},
            "observation_ids": ["obs-1"],
        }
    )
    client = _SequenceClient(
        [json.dumps(illegal), "not-json", json.dumps(repaired)]
    )

    result, consumed = _run(client, arm="e1c2", state=state)

    assert consumed == 3
    assert result["slot_transaction_abstained"] is False
    assert result["status"] == "success"
    assert "write_on_working_slot" in client.prompts[2]
    assert "not a valid complete JSON object" in client.prompts[2]
