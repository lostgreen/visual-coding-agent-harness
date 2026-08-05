from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from runs.mger_phase5r_gate import gate_r1
from vcah.phase5r import (
    RecordedDecisionReasoner,
    mechanical_replay_audit,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_interactive_runner():
    path = Path(__file__).parents[1] / "tools" / "run_mmlifelong_interactive.py"
    spec = importlib.util.spec_from_file_location("phase5r_interactive_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture() -> dict[str, object]:
    return {
        "case_id": "case-0072",
        "runtime_summary": {"answer_present": False, "answer": ""},
        "reasoner_decisions": [
            {
                "type": "reasoner_json_repair",
                "decision_payload": {
                    "action": "investigate",
                    "tasks": [{"goal": "duplicate repair", "time_range": [1, 2]}],
                },
            },
            {
                "type": "reasoner_workspace",
                "decision_payload": {
                    "action": "investigate",
                    "tasks": [
                        {
                            "query_id": "window-1",
                            "goal": "inspect the recorded interval",
                            "inspection_mode": "window",
                            "segment_id": "seg-1",
                            "time_range": [10.0, 12.0],
                            "sampling_floor_fps": 1.0,
                        }
                    ],
                    "workspace_ops": [{"op": "add_claim", "claim_id": "historical-id"}],
                },
            },
        ],
        "frame_sampling_manifest": [
            {
                "query_id": "window-1",
                "segment_id": "seg-1",
                "source_video_id": "video-1",
                "source_time_sec": 1.0,
                "virtual_time_sec": 10.0,
                "sampling_fps": 1.0,
                "fps_level": "window",
            },
            {
                "query_id": "window-1",
                "segment_id": "seg-1",
                "source_video_id": "video-1",
                "source_time_sec": 2.0,
                "virtual_time_sec": 11.0,
                "sampling_fps": 1.0,
                "fps_level": "window",
            },
        ],
    }


def test_recorded_reasoner_skips_repair_rows_and_historical_workspace_ops(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "interactions.jsonl"
    reasoner = RecordedDecisionReasoner(_fixture(), trace_path=trace_path)

    decision = reasoner.decide(
        semantic_round=1,
        options={},
    )

    assert reasoner.decision_count == 1
    assert decision.action == "investigate"
    assert len(decision.tasks) == 1
    assert decision.tasks[0].query_id == "window-1"
    assert decision.workspace_ops == ()
    row = json.loads(trace_path.read_text(encoding="utf-8"))
    assert row["source_index"] == 1
    assert "prompt" not in row
    assert "raw" not in row


def test_interactive_runner_allows_unmaterialized_frame_manifest(
    tmp_path: Path,
) -> None:
    runner = _load_interactive_runner()

    assert runner._read_jsonl(tmp_path / "missing.jsonl") == ()


def test_mechanical_replay_requires_exact_per_task_timestamp_parity(
    tmp_path: Path,
) -> None:
    fixture = _fixture()
    workspace = tmp_path / "case-0072"
    reasoner = RecordedDecisionReasoner(
        fixture,
        trace_path=workspace / "interactions.jsonl",
    )
    decision = reasoner.decide(semantic_round=1, options={})
    frame_rows = list(fixture["frame_sampling_manifest"])
    _write_jsonl(
        workspace / "observations" / "window_frame_manifest.jsonl",
        frame_rows,
    )
    runtime_trace = [
        {
            "type": "reasoner_decision",
            "round": 1,
            "semantic_round": 1,
            "action": decision.action,
            "tasks": [
                {
                    "query_id": decision.tasks[0].query_id,
                    "goal": decision.tasks[0].goal,
                    "inspection_mode": decision.tasks[0].inspection_mode,
                    "segment_id": decision.tasks[0].segment_id,
                    "time_range": list(decision.tasks[0].time_range or ()),
                    "sampling_floor_fps": decision.tasks[0].sampling_floor_fps,
                }
            ],
        }
    ]
    observation_rows = [
        {
            "modality": "visual",
            "source_video_ids": ["video-1"],
            "sampling_config": {
                "max_frames": 2,
                "sampling_manifest": {
                    "frame_times": [10.0, 11.0],
                    "effective_fps": 1.0,
                },
            },
        }
    ]

    passed = mechanical_replay_audit(
        fixture,
        workspace_root=workspace,
        trace=runtime_trace,
        observation_rows=observation_rows,
    )
    assert passed["decision"] == "PASS"
    assert passed["cost_breakdown"]["frame_cap_hits"] == 1

    frame_rows[1] = {**frame_rows[1], "virtual_time_sec": 11.25}
    _write_jsonl(
        workspace / "observations" / "window_frame_manifest.jsonl",
        frame_rows,
    )
    stopped = mechanical_replay_audit(
        fixture,
        workspace_root=workspace,
        trace=runtime_trace,
        observation_rows=observation_rows,
    )
    assert stopped["decision"] == "STOP"
    assert "frame_timestamp_digest_exact" in stopped["failed_checks"]
    assert "per_task_timestamp_digest_exact" in stopped["failed_checks"]


def test_gate_r1_stops_when_any_case_fails() -> None:
    root = {
        "root": "/tmp/replay",
        "cases": [
            {
                "case_id": "case-a",
                "decision": "PASS",
                "failed_checks": [],
                "expected_frames": 2,
                "actual_frames": 2,
                "phase5r_mode": "recorded_replay",
                "controller_mode": "frozen_baseline",
                "recorded_fixture_digest": "digest",
            },
            {
                "case_id": "case-b",
                "decision": "STOP",
                "failed_checks": ["frame_timestamp_digest_exact"],
                "expected_frames": 2,
                "actual_frames": 2,
                "phase5r_mode": "recorded_replay",
                "controller_mode": "frozen_baseline",
                "recorded_fixture_digest": "digest",
            },
        ],
    }

    result = gate_r1(root, expected_case_ids=["case-a", "case-b"])

    assert result["decision"] == "STOP"
    assert result["failed_checks"] == ["all_case_mechanical_parity"]
    assert result["totals"]["passing_cases"] == 1
