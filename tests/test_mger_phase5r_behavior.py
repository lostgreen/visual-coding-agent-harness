from __future__ import annotations

import json
from pathlib import Path

from runs.mger_phase5r_behavior import (
    _distribution,
    _provenance_summary,
    build_behavior_reference,
    collect_root,
)


def _write_root(
    root: Path,
    *,
    frame_counts: tuple[int, int],
    trace_variant: str = "shared",
) -> None:
    for index, frame_count in enumerate(frame_counts):
        case_id = f"case-{index}"
        case_dir = root / "cases" / case_id
        (case_dir / "observations").mkdir(parents=True)
        task = {
            "query_id": f"task-{index}",
            "inspection_mode": "window",
            "time_range": [0.0, 2.0],
            "sampling_floor_fps": 1.0,
            "caption_queries": [],
            "goal": trace_variant,
            "top_k": 12,
        }
        runtime = {
            "case_id": case_id,
            "answer_present": True,
            "runtime_metrics": {
                "answer_rate": 1.0,
                "observed_case_rate": 1.0,
            },
            "trace": [
                {
                    "type": "reasoner_decision",
                    "round": 1,
                    "action": "investigate",
                    "tasks": [task],
                },
                {
                    "type": "reasoner_decision",
                    "round": 2,
                    "action": "answer",
                    "tasks": [],
                },
            ],
        }
        config = {
            "answer_policy": "benchmark_best_effort",
            "max_rounds": 4,
            "max_investigations": 12,
            "max_tasks_per_round": 4,
            "caption_index_mode": "hybrid",
            "caption_query_strategy": "adaptive",
            "caption_config_digest": "caption-digest",
            "embedding": {"model": "embed", "revision": "revision"},
            "models": {"reasoner": "model", "investigator": "model"},
            "phase5r_provenance": {
                "runner_commit": "commit",
                "service_version_unpinned": True,
                "provider_request_ids": [f"request-{index}"],
                "resolved_deployment_names": [],
                "environment": {"digest": "environment"},
            },
        }
        observation = {
            "modality": "visual",
            "sampling_config": {
                "max_frames": frame_count,
                "sampling_manifest": {
                    "effective_fps": 1.0,
                    "frame_times": [float(value) for value in range(frame_count)],
                },
            },
            "source_video_ids": ["video"],
        }
        frame_rows = [
            {
                "query_id": task["query_id"],
                "source_video_id": "video",
                "source_time_sec": float(value),
                "virtual_time_sec": float(value),
            }
            for value in range(frame_count)
        ]
        (case_dir / "runtime_summary.json").write_text(
            json.dumps(runtime), encoding="utf-8"
        )
        (case_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        (case_dir / "observation_log.jsonl").write_text(
            json.dumps(observation) + "\n", encoding="utf-8"
        )
        (case_dir / "observations" / "window_frame_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in frame_rows),
            encoding="utf-8",
        )


def test_distribution_uses_interpolated_iqr() -> None:
    result = _distribution([1.0, 2.0, 3.0])

    assert result == {
        "count": 3,
        "values": [1.0, 2.0, 3.0],
        "mean": 2.0,
        "median": 2.0,
        "q1": 1.5,
        "q3": 2.5,
        "iqr": 1.0,
        "min": 1.0,
        "max": 3.0,
    }


def test_collect_root_recomputes_cost_and_decision_trace(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write_root(root, frame_counts=(2, 4))

    result = collect_root(root)

    assert result["case_count"] == 2
    assert result["audit_config_consistent"] is True
    assert result["metrics"]["visual_frames_inspected"] == 3.0
    assert result["metrics"]["visual_window_count"] == 1.0
    assert result["metrics"]["answer_rate"] == 1.0
    assert result["provenance"]["provider_request_id_count"] == 2
    assert result["provenance"]["materialized_frame_manifest_case_count"] == 2
    assert result["provenance"]["unmaterialized_frame_manifest_case_count"] == 0
    assert all(case["decision_trace_digest"] for case in result["cases"])


def test_provenance_reconstructs_historical_interaction_metadata() -> None:
    result = _provenance_summary(
        [
            {
                "models": {"reasoner": "model", "investigator": "model"},
                "caption_config_digest": "caption-digest",
                "input_digest": "input-digest",
                "embedding": {"revision": "embedding-revision"},
            }
        ],
        [
            {
                "prompt": "historical prompt",
                "api_response": {
                    "provider_request_id": "request-id",
                    "provider_reported_seed_support": "unsupported",
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
            }
        ],
        ["frame-manifest-digest"],
    )

    assert result["historical_external_reconstruction"] is True
    assert result["provider_request_ids"] == ["request-id"]
    assert result["provider_request_id_count"] == 1
    assert result["service_version_unpinned"] is True
    assert result["temperatures"] == [0.0]
    assert result["prompt_digest"]


def test_reference_reports_root_case_and_trace_distributions(tmp_path: Path) -> None:
    historical = []
    current = []
    for index, counts in enumerate(((2, 4), (4, 6), (6, 8))):
        old_root = tmp_path / f"old-{index}"
        new_root = tmp_path / f"new-{index}"
        _write_root(old_root, frame_counts=counts)
        _write_root(
            new_root,
            frame_counts=(counts[0] + 1, counts[1] + 1),
            trace_variant="shared" if index == 0 else "current",
        )
        historical.append(collect_root(old_root))
        current.append(collect_root(new_root))

    result = build_behavior_reference(
        historical,
        current,
        expected_case_count=2,
        minimum_roots_per_arm=3,
    )

    assert result["decision"] == "READY"
    assert result["checks"]["within_arm_config_consistency"] is True
    assert result["checks"]["cross_arm_audit_config_match"] is True
    frame_comparison = result["cross_arm"]["root_metric_comparison"][
        "visual_frames_inspected"
    ]
    assert frame_comparison["historical"]["median"] == 5.0
    assert frame_comparison["current"]["median"] == 6.0
    assert frame_comparison["range_overlap"] is True
    assert result["cross_arm"]["per_case_frame_comparison"]["overlap_rate"] == 1.0
    divergence = result["cross_arm"]["decision_trace_divergence"]
    assert divergence["cross_arm_pair_count"] == 18
    assert divergence["exact_cross_arm_pair_count"] == 6
    assert divergence["historical_within_arm"]["exact_pair_count"] == 6
    assert divergence["current_within_arm"]["exact_pair_count"] == 2
    assert len(divergence["cases"]["case-0"]["historical_unique_traces"]) == 1
    assert len(divergence["cases"]["case-0"]["current_unique_traces"]) == 2
