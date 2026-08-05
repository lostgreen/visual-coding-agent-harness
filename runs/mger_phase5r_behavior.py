#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from vcah.phase5r import frame_cost_breakdown, runtime_decision_trace


METRIC_KEYS = (
    "visual_frames_inspected",
    "requested_visual_window_count",
    "visual_window_count",
    "sum_requested_window_duration",
    "mean_requested_fps",
    "mean_effective_fps",
    "frame_cap_hits",
    "reinspection_count",
    "unique_visual_material_attempts",
    "caption_search_count",
    "asr_search_count",
    "semantic_rounds",
    "answer_rate",
    "observed_case_rate",
)


def collect_root(root: Path) -> dict[str, Any]:
    root = Path(root)
    cases: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    frame_manifest_digests: list[str] = []
    frame_manifest_materialized: list[bool] = []
    for case_dir in sorted((root / "cases").glob("*")):
        runtime_path = case_dir / "runtime_summary.json"
        config_path = case_dir / "run_config.json"
        if not (case_dir.is_dir() and runtime_path.is_file() and config_path.is_file()):
            continue
        runtime = _read_json(runtime_path)
        config = _read_json(config_path)
        trace = [
            dict(row)
            for row in tuple(runtime.get("trace", ()) or ())
            if isinstance(row, Mapping)
        ]
        observations = _read_jsonl(case_dir / "observation_log.jsonl")
        interactions.extend(_read_jsonl(case_dir / "interactions.jsonl"))
        frame_manifest_path = case_dir / "observations" / "window_frame_manifest.jsonl"
        frame_rows = _read_jsonl(frame_manifest_path)
        frame_manifest_digests.append(_file_sha256(frame_manifest_path))
        frame_manifest_materialized.append(frame_manifest_path.is_file())
        cost = frame_cost_breakdown(trace, observations, frame_rows)
        decision_trace = runtime_decision_trace(trace)
        runtime_metrics = _mapping(runtime.get("runtime_metrics"))
        cost["answer_rate"] = _number(
            runtime_metrics.get(
                "answer_rate", float(bool(runtime.get("answer_present")))
            )
        )
        cost["observed_case_rate"] = _number(
            runtime_metrics.get(
                "observed_case_rate",
                float(
                    _number(runtime_metrics.get("visual_interpretation_count")) > 0
                    or _number(cost.get("visual_frames_inspected")) > 0
                ),
            )
        )
        case_id = str(runtime.get("case_id", case_dir.name) or case_dir.name)
        cases.append(
            {
                "case_id": case_id,
                "metrics": {key: _number(cost.get(key)) for key in METRIC_KEYS},
                "decision_trace_digest": str(decision_trace.get("digest", "") or ""),
                "decision_trace": list(decision_trace.get("decisions", ()) or ()),
            }
        )
        configs.append(config)

    root_metrics = {
        key: _rounded_mean([case["metrics"][key] for case in cases])
        for key in METRIC_KEYS
    }
    audit_configs = [_audit_config(config) for config in configs]
    config_digests = sorted({_stable_hash(config) for config in audit_configs})
    return {
        "root": str(root),
        "case_count": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "audit_config_consistent": len(config_digests) == 1,
        "audit_config": audit_configs[0] if len(config_digests) == 1 else None,
        "audit_config_digests": config_digests,
        "metrics": root_metrics,
        "cases": cases,
        "provenance": _provenance_summary(
            configs,
            interactions,
            frame_manifest_digests,
            frame_manifest_materialized,
        ),
    }


def build_behavior_reference(
    historical_roots: Sequence[Mapping[str, Any]],
    current_roots: Sequence[Mapping[str, Any]],
    *,
    expected_case_count: int = 10,
    minimum_roots_per_arm: int = 3,
) -> dict[str, Any]:
    historical = [dict(root) for root in historical_roots]
    current = [dict(root) for root in current_roots]
    expected_ids = list(historical[0].get("case_ids", ())) if historical else []
    all_roots = [*historical, *current]
    historical_configs = {_stable_hash(root.get("audit_config")) for root in historical}
    current_configs = {_stable_hash(root.get("audit_config")) for root in current}
    checks = {
        "minimum_roots_per_arm": (
            len(historical) >= minimum_roots_per_arm
            and len(current) >= minimum_roots_per_arm
        ),
        "expected_case_count": bool(all_roots)
        and all(
            int(root.get("case_count", 0)) == expected_case_count for root in all_roots
        ),
        "case_ids_exact": bool(expected_ids)
        and all(list(root.get("case_ids", ())) == expected_ids for root in all_roots),
        "within_root_config_consistency": bool(all_roots)
        and all(bool(root.get("audit_config_consistent")) for root in all_roots),
        "within_arm_config_consistency": bool(historical and current)
        and len(historical_configs) == 1
        and len(current_configs) == 1,
        "cross_arm_audit_config_match": bool(historical and current)
        and historical_configs == current_configs,
    }
    historical_summary = _arm_summary(historical, expected_ids)
    current_summary = _arm_summary(current, expected_ids)
    return {
        "schema_version": "FrozenBehaviorReferenceV2",
        "stage": "phase5r_behavioral_distribution",
        "decision": "READY" if all(checks.values()) else "INVALID",
        "behavioral_gate_status": "descriptive_evidence_requires_report_conclusion",
        "statistical_significance_claim": False,
        "legacy_reference": {
            "mean_visual_frames": 55.1,
            "status": "historical_single_root_not_hard_truth",
        },
        "thresholds": {
            "expected_case_count": expected_case_count,
            "minimum_roots_per_arm": minimum_roots_per_arm,
        },
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "case_ids": expected_ids,
        "arms": {
            "historical_commit_current_environment": historical_summary,
            "current_frozen_compatibility": current_summary,
        },
        "cross_arm": {
            "root_metric_comparison": _metric_comparison(
                historical_summary.get("root_metric_distributions", {}),
                current_summary.get("root_metric_distributions", {}),
            ),
            "per_case_frame_comparison": _per_case_frame_comparison(
                historical_summary.get("per_case_distributions", {}),
                current_summary.get("per_case_distributions", {}),
            ),
            "decision_trace_divergence": _decision_trace_divergence(
                historical, current, expected_ids
            ),
        },
    }


def _arm_summary(
    roots: Sequence[Mapping[str, Any]], case_ids: Sequence[str]
) -> dict[str, Any]:
    root_metric_distributions = {
        key: _distribution(
            [_number(_mapping(root.get("metrics")).get(key)) for root in roots]
        )
        for key in METRIC_KEYS
    }
    per_case: dict[str, Any] = {}
    for case_id in case_ids:
        rows = [
            case
            for root in roots
            for case in tuple(root.get("cases", ()) or ())
            if isinstance(case, Mapping) and str(case.get("case_id", "")) == case_id
        ]
        per_case[case_id] = {
            key: _distribution(
                [_number(_mapping(row.get("metrics")).get(key)) for row in rows]
            )
            for key in METRIC_KEYS
        }
    return {
        "root_count": len(roots),
        "roots": [
            {
                "root": root.get("root"),
                "case_count": root.get("case_count"),
                "metrics": root.get("metrics"),
                "audit_config": root.get("audit_config"),
                "provenance": root.get("provenance"),
            }
            for root in roots
        ],
        "root_metric_distributions": root_metric_distributions,
        "per_case_distributions": per_case,
    }


def _metric_comparison(
    historical: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for key in METRIC_KEYS:
        old = _mapping(historical.get(key))
        new = _mapping(current.get(key))
        old_median = _number(old.get("median"))
        new_median = _number(new.get("median"))
        comparisons[key] = {
            "historical": dict(old),
            "current": dict(new),
            "median_delta_current_minus_historical": round(new_median - old_median, 6),
            "median_ratio_current_over_historical": (
                round(new_median / old_median, 6) if old_median else None
            ),
            "iqr_overlap": _intervals_overlap(
                _number(old.get("q1")),
                _number(old.get("q3")),
                _number(new.get("q1")),
                _number(new.get("q3")),
            ),
            "range_overlap": _intervals_overlap(
                _number(old.get("min")),
                _number(old.get("max")),
                _number(new.get("min")),
                _number(new.get("max")),
            ),
        }
    return comparisons


def _per_case_frame_comparison(
    historical: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for case_id in sorted(set(historical) | set(current)):
        old = _mapping(_mapping(historical.get(case_id)).get("visual_frames_inspected"))
        new = _mapping(_mapping(current.get(case_id)).get("visual_frames_inspected"))
        cases[case_id] = {
            "historical": dict(old),
            "current": dict(new),
            "median_delta_current_minus_historical": round(
                _number(new.get("median")) - _number(old.get("median")), 6
            ),
            "range_overlap": _intervals_overlap(
                _number(old.get("min")),
                _number(old.get("max")),
                _number(new.get("min")),
                _number(new.get("max")),
            ),
        }
    overlap_count = sum(bool(row["range_overlap"]) for row in cases.values())
    return {
        "overlap_case_count": overlap_count,
        "case_count": len(cases),
        "overlap_rate": round(overlap_count / len(cases), 6) if cases else 0.0,
        "cases": cases,
    }


def _decision_trace_divergence(
    historical_roots: Sequence[Mapping[str, Any]],
    current_roots: Sequence[Mapping[str, Any]],
    case_ids: Sequence[str],
) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    exact_pairs = 0
    total_pairs = 0
    historical_within_exact = 0
    historical_within_total = 0
    current_within_exact = 0
    current_within_total = 0
    for case_id in case_ids:
        old = _case_traces(historical_roots, case_id)
        new = _case_traces(current_roots, case_id)
        divergences: list[int | None] = []
        for old_trace in old:
            for new_trace in new:
                divergences.append(_earliest_divergence(old_trace, new_trace))
        pair_count = len(divergences)
        case_exact = sum(value is None for value in divergences)
        exact_pairs += case_exact
        total_pairs += pair_count
        historical_within = _within_arm_trace_summary(old)
        current_within = _within_arm_trace_summary(new)
        historical_within_exact += int(historical_within["exact_pair_count"])
        historical_within_total += int(historical_within["pair_count"])
        current_within_exact += int(current_within["exact_pair_count"])
        current_within_total += int(current_within["pair_count"])
        finite = [float(value) for value in divergences if value is not None]
        old_digests = {_stable_hash(trace) for trace in old}
        new_digests = {_stable_hash(trace) for trace in new}
        cases[case_id] = {
            "historical_unique_trace_count": len(old_digests),
            "current_unique_trace_count": len(new_digests),
            "shared_trace_digest_count": len(old_digests & new_digests),
            "historical_unique_traces": _unique_traces(old),
            "current_unique_traces": _unique_traces(new),
            "historical_within_arm": historical_within,
            "current_within_arm": current_within,
            "cross_arm_pair_count": pair_count,
            "exact_cross_arm_pair_count": case_exact,
            "exact_cross_arm_pair_rate": (
                round(case_exact / pair_count, 6) if pair_count else 0.0
            ),
            "earliest_divergence_round_distribution": _distribution(finite),
        }
    return {
        "cross_arm_pair_count": total_pairs,
        "exact_cross_arm_pair_count": exact_pairs,
        "exact_cross_arm_pair_rate": (
            round(exact_pairs / total_pairs, 6) if total_pairs else 0.0
        ),
        "historical_within_arm": _pair_totals(
            historical_within_exact, historical_within_total
        ),
        "current_within_arm": _pair_totals(current_within_exact, current_within_total),
        "cases": cases,
    }


def _case_traces(
    roots: Sequence[Mapping[str, Any]], case_id: str
) -> list[list[dict[str, Any]]]:
    return [
        [dict(item) for item in tuple(case.get("decision_trace", ()) or ())]
        for root in roots
        for case in tuple(root.get("cases", ()) or ())
        if isinstance(case, Mapping) and str(case.get("case_id", "")) == case_id
    ]


def _unique_traces(
    traces: Sequence[Sequence[Mapping[str, Any]]],
) -> list[list[dict[str, Any]]]:
    keyed = {
        json.dumps(trace, sort_keys=True, separators=(",", ":"), default=str): [
            dict(row) for row in trace
        ]
        for trace in traces
    }
    return [keyed[key] for key in sorted(keyed)]


def _within_arm_trace_summary(
    traces: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    divergences = [
        _earliest_divergence(traces[left], traces[right])
        for left in range(len(traces))
        for right in range(left + 1, len(traces))
    ]
    exact = sum(value is None for value in divergences)
    finite = [float(value) for value in divergences if value is not None]
    return {
        **_pair_totals(exact, len(divergences)),
        "earliest_divergence_round_distribution": _distribution(finite),
    }


def _pair_totals(exact: int, total: int) -> dict[str, Any]:
    return {
        "pair_count": total,
        "exact_pair_count": exact,
        "exact_pair_rate": round(exact / total, 6) if total else 0.0,
    }


def _earliest_divergence(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> int | None:
    for index, (left_row, right_row) in enumerate(zip(left, right), start=1):
        if dict(left_row) != dict(right_row):
            return index
    if len(left) != len(right):
        return min(len(left), len(right)) + 1
    return None


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "values": [],
            "mean": None,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "min": None,
            "max": None,
        }
    q1 = _percentile(ordered, 0.25)
    q3 = _percentile(ordered, 0.75)
    return {
        "count": len(ordered),
        "values": [round(value, 6) for value in ordered],
        "mean": round(mean(ordered), 6),
        "median": round(median(ordered), 6),
        "q1": round(q1, 6),
        "q3": round(q3, 6),
        "iqr": round(q3 - q1, 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return float(values[lower]) * (1.0 - weight) + float(values[upper]) * weight


def _audit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    embedding = _mapping(config.get("embedding"))
    return {
        "answer_policy": config.get("answer_policy"),
        "max_rounds": config.get("max_rounds"),
        "max_investigations": config.get("max_investigations"),
        "max_tasks_per_round": config.get("max_tasks_per_round"),
        "caption_index_mode": config.get("caption_index_mode"),
        "caption_query_strategy": config.get("caption_query_strategy"),
        "caption_config_digest": config.get("caption_config_digest"),
        "embedding_model": embedding.get("model", embedding.get("model_id")),
        "embedding_revision": embedding.get("revision", embedding.get("model_version")),
        "models": config.get("models"),
    }


def _provenance_summary(
    configs: Sequence[Mapping[str, Any]],
    interactions: Sequence[Mapping[str, Any]],
    frame_manifest_digests: Sequence[str],
    frame_manifest_materialized: Sequence[bool] = (),
) -> dict[str, Any]:
    embedded = [
        _mapping(config.get("phase5r_provenance"))
        for config in configs
        if isinstance(config.get("phase5r_provenance"), Mapping)
    ]
    api_rows = [
        _mapping(row.get("api_response"))
        for row in interactions
        if isinstance(row.get("api_response"), Mapping)
    ]
    provider_request_ids = {
        str(value) for row in api_rows if (value := row.get("provider_request_id"))
    } | {
        str(value)
        for row in embedded
        for value in tuple(row.get("provider_request_ids", ()) or ())
        if value
    }
    resolved_deployments = {
        str(value)
        for row in api_rows
        for key in ("resolved_deployment_name", "deployment_name", "resolved_model")
        if (value := row.get(key))
    } | {
        str(value)
        for row in embedded
        for value in tuple(row.get("resolved_deployment_names", ()) or ())
        if value
    }
    prompt_hashes = [
        hashlib.sha256(str(row.get("prompt", "")).encode("utf-8")).hexdigest()
        for row in interactions
        if row.get("prompt")
    ]
    return {
        "embedded_case_provenance_count": len(embedded),
        "historical_external_reconstruction": not bool(embedded),
        "runner_commits": sorted(
            {
                str(row.get("runner_commit", "") or "")
                for row in embedded
                if row.get("runner_commit")
            }
        ),
        "requested_models": _unique_json_values(
            [config.get("models") for config in configs if config.get("models")]
        ),
        "temperatures": sorted(
            {
                float(row["temperature"])
                for row in api_rows
                if row.get("temperature") is not None
            }
        ),
        "top_p_values": sorted(
            {float(row["top_p"]) for row in api_rows if row.get("top_p") is not None}
        ),
        "requested_seeds": sorted(
            {
                str(row.get("requested_seed"))
                for row in api_rows
                if row.get("requested_seed") is not None
            }
        ),
        "provider_seed_support": sorted(
            {
                str(row.get("provider_reported_seed_support"))
                for row in api_rows
                if row.get("provider_reported_seed_support") is not None
            }
        ),
        "provider_request_ids": sorted(provider_request_ids),
        "provider_request_id_count": len(provider_request_ids),
        "resolved_deployment_names": sorted(resolved_deployments),
        "service_version_unpinned": not bool(resolved_deployments),
        "caption_index_digests": sorted(
            {
                str(config.get("caption_config_digest", "") or "")
                for config in configs
                if config.get("caption_config_digest")
            }
        ),
        "embedding_revisions": sorted(
            {
                str(_mapping(config.get("embedding")).get("revision", "") or "")
                for config in configs
                if _mapping(config.get("embedding")).get("revision")
            }
            | {
                str(_mapping(config.get("embedding")).get("model_version", "") or "")
                for config in configs
                if _mapping(config.get("embedding")).get("model_version")
            }
        ),
        "frame_cache_digests": sorted(
            {
                str(row.get("frame_cache_digest", "") or "")
                for row in embedded
                if row.get("frame_cache_digest")
            }
        ),
        "output_frame_manifest_root_digest": _stable_hash(
            sorted(frame_manifest_digests)
        ),
        "materialized_frame_manifest_case_count": sum(frame_manifest_materialized),
        "unmaterialized_frame_manifest_case_count": len(frame_manifest_materialized)
        - sum(frame_manifest_materialized),
        "source_video_manifest_digests": sorted(
            {
                str(row.get("source_video_manifest_digest", "") or "")
                for row in embedded
                if row.get("source_video_manifest_digest")
            }
        ),
        "input_digests": sorted(
            {
                str(config.get("input_digest", "") or "")
                for config in configs
                if config.get("input_digest")
            }
        ),
        "prompt_digest": _stable_hash(prompt_hashes),
        "reasoner_system_prompt_status": "not_separate_in_client_contract",
        "environment_digests": sorted(
            {
                str(_mapping(row.get("environment")).get("digest", "") or "")
                for row in embedded
                if _mapping(row.get("environment")).get("digest")
            }
        ),
    }


def _intervals_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
    return max(a_min, b_min) <= min(a_max, b_max)


def _rounded_mean(values: Sequence[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _unique_json_values(values: Sequence[Any]) -> list[Any]:
    keyed = {
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str): value
        for value in values
    }
    return [keyed[key] for key in sorted(keyed)]


def _file_sha256(path: Path) -> str:
    if not Path(path).is_file():
        return hashlib.sha256(b"").hexdigest()
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Phase 5R multi-root behavioral reference."
    )
    parser.add_argument("--historical-root", action="append", required=True)
    parser.add_argument("--current-root", action="append", required=True)
    parser.add_argument("--expected-case-count", type=int, default=10)
    parser.add_argument("--minimum-roots-per-arm", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    historical = [collect_root(Path(path)) for path in args.historical_root]
    current = [collect_root(Path(path)) for path in args.current_root]
    result = build_behavior_reference(
        historical,
        current,
        expected_case_count=args.expected_case_count,
        minimum_roots_per_arm=args.minimum_roots_per_arm,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    old_frames = result["arms"]["historical_commit_current_environment"][
        "root_metric_distributions"
    ]["visual_frames_inspected"]
    new_frames = result["arms"]["current_frozen_compatibility"][
        "root_metric_distributions"
    ]["visual_frames_inspected"]
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "failed_checks": result["failed_checks"],
                "historical_root_count": len(historical),
                "current_root_count": len(current),
                "historical_median_mean_frames": old_frames["median"],
                "current_median_mean_frames": new_frames["median"],
                "out": str(out),
            },
            sort_keys=True,
        )
    )
    return 0 if result["decision"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
