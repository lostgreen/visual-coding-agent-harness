from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vcah.replay import (
    REPLAY_SCHEMA_VERSION,
    aggregate_seed_results,
    compare_replay_records,
    create_immutable_run,
    replay_case_metadata,
    workspace_input_checksums,
    write_immutable_summary,
)


def test_immutable_run_creation_is_exclusive_and_writes_hashed_config(tmp_path: Path) -> None:
    run = create_immutable_run(
        tmp_path / "runs",
        run_id="targeted-001",
        config={"git_commit": "abc123", "seeds": [11]},
    )

    payload = json.loads((run.root / "config.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == "targeted-001"
    assert payload["schema_version"] == REPLAY_SCHEMA_VERSION == 3
    assert payload["config_hash"] == run.config_hash
    assert (run.root / "workspaces").is_dir()
    with pytest.raises(FileExistsError):
        create_immutable_run(tmp_path / "runs", run_id="targeted-001", config={})

    summary_path = write_immutable_summary(run, {"case_count": 1})
    assert json.loads(summary_path.read_text(encoding="utf-8"))["case_count"] == 1
    with pytest.raises(FileExistsError):
        write_immutable_summary(run, {"case_count": 2})


def test_replay_metadata_and_multi_seed_aggregation_are_content_free_and_comparable(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-media")
    root = tmp_path / "workspace"
    root.mkdir()
    frame_manifest = root / "frame_manifest.jsonl"
    frame_manifest.write_text('{"frame_id":"frame_1"}\n', encoding="utf-8")
    interactions = [
        {
            "type": "reasoner_answer",
            "prompt": "Which option is correct?",
            "raw": "D. raw answer",
            "format_repaired": True,
            "api_response": {
                "provider_request_id": "request-1",
                "retry_count": 1,
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "reasoning_tokens": 3,
            },
        }
    ]
    (root / "interactions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in interactions), encoding="utf-8"
    )
    (root / "observation_log.jsonl").write_text('{"attempt_id":"attempt_a"}\n', encoding="utf-8")
    (root / "working_document.json").write_text('{"revision":1}\n', encoding="utf-8")
    (root / "workspace_ops.jsonl").write_text('{"op":"add_claim"}\n', encoding="utf-8")
    workspace = SimpleNamespace(
        root_dir=root,
        frame_manifest=frame_manifest,
        manifest=SimpleNamespace(segments=(SimpleNamespace(source_path=str(source)),)),
    )
    summary = {
        "case_id": "441-2",
        "answer": "D. raw answer",
        "correct": True,
        "reference_valid": True,
        "reference_reason": "reference_integrity_valid",
        "investigation_count": 1,
        "trace": [
            {
                "type": "reasoner_decision",
                "round": 1,
                "action": "investigate",
                "tasks": [{"query_id": "q1", "inspection_mode": "window"}],
                "remaining_budget": 4,
            },
            {
                "type": "investigator_batch",
                "round": 1,
                "requested_tasks": 1,
                "accepted_tasks": 1,
            },
            {
                "type": "answer_outcome",
                "raw_reasoner_answer": "D. raw answer",
                "answer": "D. raw answer",
                "reference_valid": True,
                "reference_reason": "reference_integrity_valid",
                "answer_owner": "reasoner",
                "framework_answer_mutation": False,
            },
        ],
    }
    record = replay_case_metadata(
        workspace_root=root,
        case_summary=summary,
        input_checksums=workspace_input_checksums(workspace),
        seed=11,
        provider_settings={"reasoner": {"requested_seed": 11}},
        gold_option="D",
    )

    assert record["source_media_checksum"]
    assert record["schema_version"] == 3
    assert record["frame_manifest_checksum"]
    assert record["provider_request_ids"] == ["request-1"]
    assert record["api_retry_count"] == 1
    assert record["investigation_ordering"][0]["tasks"][0]["query_id"] == "q1"
    assert record["reference_valid"] is True
    assert record["framework_answer_mutation"] is False
    assert record["observation_log_checksum"]
    assert record["execution_health"]["finish_reason_length_ratio"] == 0.0
    assert record["execution_health"]["format_repair_count"] == 1
    assert record["execution_health"]["reasoner_task_execution_rate"] == 1.0
    assert "Which option is correct?" not in json.dumps(record)

    current = {
        **record,
        "execution_health": {
            **record["execution_health"],
            "finish_reason_length_ratio": 0.25,
            "images_dropped": 2,
        },
    }
    comparison = compare_replay_records(record, current)
    assert comparison["format_compatible"] is True
    assert comparison["behavior_delta"]["finish_reason_length_ratio"] == 0.25
    assert comparison["behavior_delta"]["images_dropped"] == 2.0

    incompatible = dict(current)
    incompatible.pop("execution_health")
    assert compare_replay_records(record, incompatible)["format_compatible"] is False

    drifted = {
        **record,
        "seed": 12,
        "final_option": "H",
        "final_answer": "H. changed answer",
        "correct": False,
        "observation_log_checksum": record["observation_log_checksum"],
    }
    aggregate = aggregate_seed_results((record, drifted))
    case = aggregate["per_case"]["441-2"]
    assert case["final_answer_distribution"] == {"D": 1, "H": 1}
    assert case["same_observation_answer_drift"] == 1
    assert aggregate["targeted_seed_protocol"]["441-2"]["satisfied"] is False
    assert aggregate["overall"]["accuracy"] == 0.5
    assert aggregate["overall"]["reference_valid_rate"] == 1.0
    assert aggregate["overall"]["execution_health"]["reasoner_task_execution_rate"] == 1.0
