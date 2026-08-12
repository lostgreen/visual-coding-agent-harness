from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


def _load_module() -> Any:
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "audit_mmlifelong_oracle_runs.py"
    )
    spec = importlib.util.spec_from_file_location("mmlifelong_oracle_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_module()


def _write_fixture(root: Path, arm: str, case_id: str) -> None:
    run_dir = root / "cases" / case_id
    run_dir.mkdir(parents=True)
    config = {
        "case_id": case_id,
        "oracle_arm": arm,
        "models": {"reasoner": "same-model", "investigator": "same-model"},
        "caption_config_digest": "caption-digest",
        "embedding": {"model": "embedding", "revision": "frozen"},
        "controller_mode": "frozen_baseline",
        "controller_evidence_visibility": "full",
        "measurement_control": "none",
        "answer_policy": "benchmark_best_effort",
        "evidence_control_mode": "shadow",
        "evidence_state_mode": "llm_authored",
        "max_rounds": 4,
        "semantic_round_budget": 4,
        "control_retry_budget": 1,
        "max_investigations": 12,
        "max_tasks_per_round": 4,
        "caption_index_mode": "hybrid",
        "caption_query_strategy": "adaptive",
        "caption_query_policy": "requested_query",
        "effective_caption_query_strategy": "adaptive",
        "phase5r_mode": "live",
        "web_enabled": False,
        "supporting_interval_source": "explicit_support",
        "phase5r_provenance": {
            "runner_commit": "commit",
            "service_version_unpinned": True,
            "models": {
                role: {
                    "model": "same-model",
                    "temperature": None,
                    "top_p": None,
                    "requested_seed": None,
                    "provider_seed_supported": False,
                    "provider_reported_seed_support": "unknown",
                }
                for role in ("reasoner", "investigator")
            },
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    runtime: dict[str, Any] = {}
    if arm != "o0":
        o1_family = arm in {"o1", "o1.5", "o1.75", "o1.75-forced"}
        exact_family = arm in {"o2", "o2-center"}
        candidate_ids = ["o1-pool"] if o1_family else ["exact"] if exact_family else ["c0"]
        intervals = [[10.0, 12.0]] if o1_family else [[11.0, 11.5]] if exact_family else [[0.0, 1.0]]
        point_anchor = arm in {"o1.75", "o1.75-forced", "o2-center"}
        runtime["oracle_intervention_audit"] = {
            "applied": True,
            "caption_config_digest": "caption-digest",
            "intervention_digest": f"intervention-{case_id}",
            "natural_candidate_count": 12,
            "natural_clue_recall": 0.25,
            "candidate_passage_ids": candidate_ids,
            "candidate_intervals": intervals,
            "shuffle_seed_digest": f"shuffle-{case_id}",
            "guidance_type": (
                "selected_coarse_candidates_with_point_anchors"
                if arm in {"o1.75", "o1.75-forced"}
                else "exact_locators_with_point_anchors"
                if arm == "o2-center"
                else ""
            ),
            "exact_boundaries_visible": exact_family,
            "selected_candidate_ranks": [1] if point_anchor else [],
            "selected_candidate_passage_ids": candidate_ids if point_anchor else [],
            "selected_candidate_intervals": intervals if point_anchor else [],
            "anchor_count": 1 if point_anchor else 0,
            "anchor_timestamps_sec": [11.25] if point_anchor else [],
            "point_anchor_candidate_ranks": [1] if point_anchor else [],
            "point_anchor_candidate_passage_ids": candidate_ids if point_anchor else [],
        }
    (run_dir / "runtime_summary.json").write_text(
        json.dumps(runtime), encoding="utf-8"
    )


def _runs(tmp_path: Path) -> dict[str, Path]:
    roots = {arm: tmp_path / arm for arm in AUDIT.HISTORICAL_ARMS}
    for arm, root in roots.items():
        for case_id in ("case-0", "case-1"):
            _write_fixture(root, arm, case_id)
    return roots


def test_audit_confirms_frozen_same_model_stack_and_interventions(
    tmp_path: Path,
) -> None:
    report = AUDIT.build_report(
        AUDIT.collect_runs(_runs(tmp_path)),
        expected_cases=2,
        declared_reasoner_section="investigator_api",
        declared_investigator_section="investigator_api",
    )

    assert report["gate_passed"] is True
    assert report["reasoner_investigator_share_model"] is True
    assert report["recorded_section_bindings_complete"] is False
    assert report["actual_role_model_pairs"] == [
        {"reasoner": "same-model", "investigator": "same-model"}
    ]
    c0 = next(row for row in report["arms"] if row["arm"] == "c0")
    assert c0["arm_label"] == "intervention-scaffold control"
    assert report["terminology"]["oracle_gap_recovery"]["status"] == "deprecated"


def test_audit_rejects_model_drift(tmp_path: Path) -> None:
    roots = _runs(tmp_path)
    path = roots["o2"] / "cases" / "case-0" / "run_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["models"]["reasoner"] = "drifted-model"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = AUDIT.build_report(AUDIT.collect_runs(roots), expected_cases=2)

    assert report["gate_passed"] is False
    assert report["gate_checks"]["single_model_stack_per_arm"] is False
    assert report["gate_checks"]["model_stack_aligned_across_arms"] is False
