#!/usr/bin/env python3
"""Audit frozen models, configs, and interventions across oracle runs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


HISTORICAL_ARMS = (
    "o0",
    "c0",
    "o1",
    "o1.5",
    "o1.75",
    "o1.75-forced",
    "o2",
    "o2-center",
)
INTERVENTION_ARMS = HISTORICAL_ARMS[1:]
CONTROLLER_KEYS = (
    "controller_mode",
    "controller_evidence_visibility",
    "measurement_control",
    "answer_policy",
    "evidence_control_mode",
    "evidence_state_mode",
    "max_rounds",
    "semantic_round_budget",
    "control_retry_budget",
    "max_investigations",
    "max_tasks_per_round",
    "caption_index_mode",
    "caption_query_strategy",
    "caption_query_policy",
    "effective_caption_query_strategy",
    "phase5r_mode",
    "web_enabled",
    "supporting_interval_source",
)
MODEL_SETTING_KEYS = (
    "model",
    "temperature",
    "top_p",
    "requested_seed",
    "provider_seed_supported",
    "provider_reported_seed_support",
)


def collect_runs(
    run_roots: Mapping[str, Path],
) -> dict[str, dict[str, dict[str, Any]]]:
    collected: dict[str, dict[str, dict[str, Any]]] = {}
    for declared_arm, root in run_roots.items():
        records: dict[str, dict[str, Any]] = {}
        for config_path in sorted(Path(root).glob("cases/*/run_config.json")):
            run_dir = config_path.parent
            config = _read_json(config_path)
            arm = str(config.get("oracle_arm", "o0")).casefold()
            if arm != declared_arm:
                raise ValueError(
                    f"declared arm {declared_arm} does not match {arm}: {run_dir}"
                )
            case_id = str(config.get("case_id") or run_dir.name)
            if case_id in records:
                raise ValueError(f"duplicate run record: {arm}:{case_id}")
            runtime_path = run_dir / "runtime_summary.json"
            if not runtime_path.is_file():
                raise FileNotFoundError(runtime_path)
            records[case_id] = {
                "config": config,
                "runtime": _read_json(runtime_path),
            }
        if not records:
            raise FileNotFoundError(f"no run_config.json files under {root}")
        collected[declared_arm] = records
    return collected


def build_report(
    runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    expected_cases: int,
    declared_reasoner_section: str | None = None,
    declared_investigator_section: str | None = None,
) -> dict[str, Any]:
    arm_names = tuple(arm for arm in HISTORICAL_ARMS if arm in runs)
    case_sets = {arm: set(runs[arm]) for arm in arm_names}
    common_cases = set.intersection(*(case_sets[arm] for arm in arm_names))

    arm_rows = []
    internal: dict[str, dict[str, Any]] = {}
    all_role_pairs: list[tuple[str, str]] = []
    for arm in arm_names:
        records = runs[arm]
        configs = [record["config"] for record in records.values()]
        runtimes = [record["runtime"] for record in records.values()]
        model_variants = _variants(
            configs,
            lambda config: dict(config.get("models", {}) or {}),
        )
        role_setting_variants = _variants(
            configs,
            lambda config: _safe_role_settings(
                dict(config.get("phase5r_provenance", {}) or {}).get("models")
            ),
        )
        api_binding_variants = _variants(
            configs,
            lambda config: dict(config.get("api_bindings", {}) or {}),
        )
        caption_variants = sorted(
            {str(config.get("caption_config_digest", "")) for config in configs}
        )
        embedding_variants = _variants(
            configs,
            lambda config: config.get("embedding"),
        )
        controller_variants = _variants(
            configs,
            lambda config: {key: config.get(key) for key in CONTROLLER_KEYS},
        )
        runner_commits = sorted(
            {
                str(
                    dict(config.get("phase5r_provenance", {}) or {}).get(
                        "runner_commit", ""
                    )
                )
                for config in configs
            }
        )
        service_version_unpinned = sorted(
            {
                bool(
                    dict(config.get("phase5r_provenance", {}) or {}).get(
                        "service_version_unpinned", True
                    )
                )
                for config in configs
            }
        )
        for models in model_variants:
            reasoner = str(models.get("reasoner", ""))
            investigator = str(models.get("investigator", ""))
            all_role_pairs.append((reasoner, investigator))

        audits = {
            case_id: record["runtime"].get("oracle_intervention_audit")
            for case_id, record in records.items()
        }
        candidate_signatures = {
            case_id: _candidate_signature(audit)
            for case_id, audit in audits.items()
            if isinstance(audit, Mapping)
        }
        natural_signatures = {
            case_id: _natural_signature(audit)
            for case_id, audit in audits.items()
            if isinstance(audit, Mapping)
        }
        visible_guidance_signatures = {
            case_id: _visible_guidance_signature(audit)
            for case_id, audit in audits.items()
            if isinstance(audit, Mapping)
        }
        internal[arm] = {
            "models": model_variants,
            "role_settings": role_setting_variants,
            "caption": caption_variants,
            "embedding": embedding_variants,
            "controller": controller_variants,
            "candidate": candidate_signatures,
            "natural": natural_signatures,
            "visible_guidance": visible_guidance_signatures,
            "audits": audits,
        }
        arm_rows.append(
            {
                "arm": arm,
                "arm_label": (
                    "intervention-scaffold control" if arm == "c0" else arm
                ),
                "case_count": len(records),
                "case_set_digest": _digest(sorted(records)),
                "model_variants": model_variants,
                "role_setting_variants": role_setting_variants,
                "api_binding_variants": api_binding_variants,
                "caption_config_digests": caption_variants,
                "embedding_variants": embedding_variants,
                "controller_variants": controller_variants,
                "runner_commits": runner_commits,
                "service_version_unpinned": service_version_unpinned,
                "candidate_pool_signature_digest": (
                    _digest(candidate_signatures) if candidate_signatures else None
                ),
                "natural_retrieval_signature_digest": (
                    _digest(natural_signatures) if natural_signatures else None
                ),
                "audit_applied_count": sum(
                    isinstance(audit, Mapping) and audit.get("applied") is True
                    for audit in audits.values()
                ),
            }
        )

    checks = {
        "all_historical_arms_present": set(arm_names) == set(HISTORICAL_ARMS),
        "expected_case_count_per_arm": all(
            len(runs[arm]) == int(expected_cases) for arm in arm_names
        ),
        "case_sets_aligned": all(case_sets[arm] == common_cases for arm in arm_names),
        "single_model_stack_per_arm": all(
            len(internal[arm]["models"]) == 1 for arm in arm_names
        ),
        "single_role_settings_per_arm": all(
            len(internal[arm]["role_settings"]) == 1 for arm in arm_names
        ),
        "model_stack_aligned_across_arms": _aligned_variant(
            internal, arm_names, "models"
        ),
        "role_settings_aligned_across_arms": _aligned_variant(
            internal, arm_names, "role_settings"
        ),
        "caption_digest_aligned": _aligned_variant(internal, arm_names, "caption"),
        "embedding_aligned": _aligned_variant(internal, arm_names, "embedding"),
        "controller_aligned": _aligned_variant(internal, arm_names, "controller"),
        "intervention_audits_complete": all(
            sum(
                isinstance(audit, Mapping) and audit.get("applied") is True
                for audit in internal[arm]["audits"].values()
            )
            == int(expected_cases)
            for arm in arm_names
            if arm != "o0"
        ),
        "natural_retrieval_aligned": _per_case_family_matched(
            internal, common_cases, INTERVENTION_ARMS, "natural"
        ),
        "o1_family_candidate_pools_matched": _per_case_family_matched(
            internal,
            common_cases,
            ("o1", "o1.5", "o1.75", "o1.75-forced"),
            "candidate",
        ),
        "exact_locator_pools_matched": _per_case_family_matched(
            internal, common_cases, ("o2", "o2-center"), "candidate"
        ),
        "o175_forced_visible_guidance_matched": _per_case_family_matched(
            internal,
            common_cases,
            ("o1.75", "o1.75-forced"),
            "visible_guidance",
        ),
    }
    unique_pairs = sorted(set(all_role_pairs))
    declared_bindings = {
        "reasoner_section": declared_reasoner_section,
        "investigator_section": declared_investigator_section,
        "source": "historical launch declaration; section names absent from old artifacts",
    }
    return {
        "schema_version": "MMLifelongModelAndInterventionAuditV1",
        "expected_cases": int(expected_cases),
        "common_case_count": len(common_cases),
        "gate_passed": all(checks.values()),
        "gate_checks": checks,
        "actual_role_model_pairs": [
            {"reasoner": reasoner, "investigator": investigator}
            for reasoner, investigator in unique_pairs
        ],
        "reasoner_investigator_share_model": bool(unique_pairs)
        and all(reasoner == investigator for reasoner, investigator in unique_pairs),
        "recorded_section_bindings_complete": all(
            bool(row["api_binding_variants"])
            and any(binding for binding in row["api_binding_variants"])
            for row in arm_rows
        ),
        "declared_section_bindings": declared_bindings,
        "section_audit_conclusion": (
            "Actual model provenance confirms both roles used the same model. "
            "Historical section names are supported by the declared launch command, "
            "not by the old run schema."
        ),
        "terminology": {
            "c0": "intervention-scaffold control",
            "oracle_gap_recovery": {
                "status": "deprecated",
                "reason": "O2 is not a valid empirical ceiling after later controls surpassed it.",
            },
        },
        "arms": arm_rows,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MM-Lifelong Model and Intervention Audit",
        "",
        f"Gate passed: `{str(report['gate_passed']).lower()}`",
        f"Common cases: `{report['common_case_count']}`",
        "",
        "| Arm | Label | N | Reasoner | Investigator | Caption variants | Embedding variants | Controller variants |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in report["arms"]:
        models = row["model_variants"][0] if len(row["model_variants"]) == 1 else {}
        lines.append(
            f"| {row['arm']} | {row['arm_label']} | {row['case_count']} | "
            f"{models.get('reasoner', '-')} | {models.get('investigator', '-')} | "
            f"{len(row['caption_config_digests'])} | {len(row['embedding_variants'])} | "
            f"{len(row['controller_variants'])} |"
        )
    lines.extend(("", "## Gate Checks", ""))
    lines.extend(
        f"- `{name}`: `{str(value).lower()}`"
        for name, value in report["gate_checks"].items()
    )
    bindings = report["declared_section_bindings"]
    lines.extend(
        (
            "",
            "## Section Binding",
            "",
            f"Recorded in old artifacts: `{str(report['recorded_section_bindings_complete']).lower()}`",
            f"Declared Reasoner section: `{bindings.get('reasoner_section')}`",
            f"Declared Investigator section: `{bindings.get('investigator_section')}`",
            report["section_audit_conclusion"],
            "",
            "C0 label: **intervention-scaffold control**.",
            "OGR status: **deprecated**.",
        )
    )
    return "\n".join(lines) + "\n"


def _safe_role_settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(role): {
            key: dict(settings).get(key)
            for key in MODEL_SETTING_KEYS
        }
        for role, settings in value.items()
        if isinstance(settings, Mapping)
    }


def _candidate_signature(value: Mapping[str, Any]) -> str:
    return _digest(
        {
            "passage_ids": value.get("candidate_passage_ids"),
            "intervals": value.get("candidate_intervals"),
            "shuffle_seed_digest": value.get("shuffle_seed_digest"),
        }
    )


def _natural_signature(value: Mapping[str, Any]) -> str:
    return _digest(
        {
            "caption_config_digest": value.get("caption_config_digest"),
            "intervention_digest": value.get("intervention_digest"),
            "natural_candidate_count": value.get("natural_candidate_count"),
            "natural_clue_recall": value.get("natural_clue_recall"),
        }
    )


def _visible_guidance_signature(value: Mapping[str, Any]) -> str:
    return _digest(
        {
            key: value.get(key)
            for key in (
                "guidance_type",
                "exact_boundaries_visible",
                "selected_candidate_ranks",
                "selected_candidate_passage_ids",
                "selected_candidate_intervals",
                "anchor_count",
                "anchor_timestamps_sec",
                "point_anchor_candidate_ranks",
                "point_anchor_candidate_passage_ids",
            )
        }
    )


def _variants(
    values: Sequence[Mapping[str, Any]],
    project: Callable[[Mapping[str, Any]], Any],
) -> list[Any]:
    encoded = {_canonical(project(value)) for value in values}
    return [json.loads(value) for value in sorted(encoded)]


def _aligned_variant(
    internal: Mapping[str, Mapping[str, Any]],
    arms: Sequence[str],
    key: str,
) -> bool:
    if not arms:
        return False
    return len({_canonical(internal[arm][key]) for arm in arms}) == 1


def _per_case_family_matched(
    internal: Mapping[str, Mapping[str, Any]],
    cases: set[str],
    arms: Sequence[str],
    key: str,
) -> bool:
    if not cases or any(arm not in internal for arm in arms):
        return False
    return all(
        len({internal[arm][key].get(case_id) for arm in arms}) == 1
        and all(internal[arm][key].get(case_id) is not None for arm in arms)
        for case_id in cases
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _parse_root(value: str) -> tuple[str, Path]:
    arm, separator, raw_path = value.partition("=")
    if not separator or arm not in HISTORICAL_ARMS or not raw_path:
        raise argparse.ArgumentTypeError("expected ARM=PATH for a historical arm")
    return arm, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", type=_parse_root, required=True)
    parser.add_argument("--expected-cases", type=int, default=200)
    parser.add_argument("--declared-reasoner-section")
    parser.add_argument("--declared-investigator-section")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()
    roots = dict(args.run_root)
    if len(roots) != len(args.run_root):
        raise ValueError("duplicate --run-root arm")
    report = build_report(
        collect_runs(roots),
        expected_cases=args.expected_cases,
        declared_reasoner_section=args.declared_reasoner_section,
        declared_investigator_section=args.declared_investigator_section,
    )
    _write(Path(args.out_json), json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write(Path(args.out_md), render_markdown(report))
    print(
        json.dumps(
            {
                "gate_passed": report["gate_passed"],
                "common_case_count": report["common_case_count"],
                "actual_role_model_pairs": report["actual_role_model_pairs"],
                "out_json": args.out_json,
                "out_md": args.out_md,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
