#!/usr/bin/env python3
"""Build the compact Phase 3 diagnostic fixture from a light run bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from zipfile import ZipFile


def main() -> None:
    args = _parse_args()
    bundle = Path(args.bundle).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases_dir = output / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(bundle) as archive:
        root = _bundle_root(archive)
        case_ids = _case_ids(archive, root)
        if len(case_ids) != 10:
            raise ValueError(f"expected 10 diagnostic cases, found {len(case_ids)}")
        for case_id in case_ids:
            payload = _case_fixture(archive, root, case_id)
            _write_json(cases_dir / f"{case_id}.json", payload)

    manifest = {
        "schema_version": "MGERPhase3DiagnosticFixtureV1",
        "source_bundle": bundle.name,
        "source_bundle_sha256": _file_sha256(bundle),
        "source_revision": "74f012d-r1",
        "case_ids": list(case_ids),
        "excluded_artifacts": [
            "jpg frames",
            "full reasoner prompts",
            "full judge responses",
            "host-specific frame paths",
        ],
    }
    _write_json(output / "manifest.json", manifest)


def _case_fixture(archive: ZipFile, root: str, case_id: str) -> dict[str, Any]:
    prefix = f"{root}/cases/{case_id}"
    prediction = _read_json(archive, f"{prefix}/prediction.json")
    runtime = _read_json(archive, f"{prefix}/runtime_summary.json")
    evaluation = _read_json(archive, f"{prefix}/evaluation/mmlifelong_eval.json")
    interactions = _read_jsonl(archive, f"{prefix}/interactions.jsonl")
    observations = _read_jsonl(archive, f"{prefix}/observation_log.jsonl")
    frame_manifest = _read_jsonl(
        archive,
        f"{prefix}/observations/window_frame_manifest.jsonl",
    )
    workspace_history = _read_jsonl(archive, f"{prefix}/workspace_ops.jsonl")
    run_trace = tuple(
        dict(row)
        for row in tuple(runtime.get("trace", ()) or ())
        if isinstance(row, Mapping)
    )
    caption_artifacts = _caption_artifacts(archive, prefix)

    reasoner_decisions = tuple(
        _compact_reasoner_row(row)
        for row in interactions
        if str(row.get("type", "")).startswith("reasoner_")
    )
    workspace_rejections = [
        {
            "source": "run_trace",
            "round": row.get("round"),
            "action": row.get("action"),
            "workspace_errors": list(row.get("workspace_errors", ()) or ()),
            "tasks": list(row.get("tasks", ()) or ()),
        }
        for row in run_trace
        if row.get("type") == "reasoner_decision"
        and row.get("workspace_ops_accepted") is False
    ]
    workspace_rejections.extend(
        {
            "source": "workspace_history",
            "round": row.get("round_id"),
            "operations": list(row.get("operations", ()) or ()),
            "result": dict(row.get("result", {}) or {}),
        }
        for row in workspace_history
        if isinstance(row.get("result"), Mapping)
        and row["result"].get("accepted") is False
    )

    task_requests = tuple(
        {
            "round": row.get("round"),
            "action": row.get("action"),
            "tasks": list(row.get("tasks", ()) or ()),
        }
        for row in run_trace
        if row.get("type") == "reasoner_decision" and row.get("tasks")
    )
    task_outcomes = tuple(
        _selected(row, ("type", "round", "requested_tasks", "completed_tasks", "errors", "outcomes"))
        for row in run_trace
        if row.get("type") in {"task_resolution", "investigator_batch"}
    )
    sampling_manifests = tuple(
        {
            "attempt_id": row.get("attempt_id"),
            "task_id": row.get("task_id"),
            "round_id": row.get("round_id"),
            "sampling_manifest": dict(config.get("sampling_manifest", {}) or {}),
        }
        for row in observations
        if isinstance((config := row.get("sampling_config")), Mapping)
        and isinstance(config.get("sampling_manifest"), Mapping)
    )

    answer_eval = dict(evaluation.get("answer", {}) or {})
    return {
        "schema_version": "MGERPhase3CaseDiagnosticV1",
        "case_id": case_id,
        "prediction": prediction,
        "runtime_summary": _selected(
            runtime,
            (
                "case_id",
                "answer",
                "answer_present",
                "answer_policy",
                "candidate_answer",
                "verified_answer",
                "verification_status",
                "blocking_reasons",
                "selected_option",
                "reference_valid",
                "reference_reason",
                "rounds",
                "investigation_count",
                "runtime_metrics",
                "supporting_claim_ids",
                "supporting_intervals",
                "residual_uncertainty",
            ),
        ),
        "evaluation_summary": {
            "answer": _selected(
                answer_eval,
                (
                    "judge_model",
                    "official_judge_config_match",
                    "official_judge_model_match",
                    "official_protocol",
                    "parse_status",
                    "prompt_sha256",
                    "raw_score",
                    "retry_count",
                    "score",
                    "upstream_revision",
                    "judge_response_metadata",
                ),
            ),
            "diagnostics": dict(evaluation.get("diagnostics", {}) or {}),
            "reference_grounding": dict(evaluation.get("reference_grounding", {}) or {}),
        },
        "reasoner_decisions": list(reasoner_decisions),
        "workspace_rejection_events": workspace_rejections,
        "task_requests": list(task_requests),
        "task_outcomes": list(task_outcomes),
        "caption_query_lists": [
            _selected(
                row,
                (
                    "artifact",
                    "queries",
                    "query_strategy",
                    "query_fingerprint",
                    "index_mode",
                    "top_k",
                    "time_range",
                    "segment_ids",
                    "source_video_ids",
                    "scope_empty",
                ),
            )
            for row in caption_artifacts
        ],
        "occurrence_candidates": [
            {
                "artifact": row["artifact"],
                **dict(candidate),
            }
            for row in caption_artifacts
            for candidate in tuple(
                dict(row.get("occurrence_set", {}) or {}).get("candidates", ()) or ()
            )
            if isinstance(candidate, Mapping)
        ],
        "sampling_manifests": list(sampling_manifests),
        "frame_sampling_manifest": [
            _selected(
                row,
                (
                    "frame_id",
                    "query_id",
                    "segment_id",
                    "source_video_id",
                    "source_time_sec",
                    "virtual_time_sec",
                    "sampling_fps",
                    "fps_level",
                ),
            )
            for row in frame_manifest
        ],
        "observation_interpretations": [
            _compact_observation(row) for row in observations
        ],
        "final_validation": [
            dict(row)
            for row in run_trace
            if row.get("type") in {"reference_integrity_check", "answer_outcome"}
        ],
    }


def _caption_artifacts(archive: ZipFile, prefix: str) -> tuple[dict[str, Any], ...]:
    search_prefix = f"{prefix}/caption_search/"
    rows = []
    for member in sorted(archive.namelist()):
        if not member.startswith(search_prefix) or not member.endswith(".json"):
            continue
        payload = _read_json(archive, member)
        payload["artifact"] = PurePosixPath(member).name
        rows.append(payload)
    return tuple(rows)


def _compact_reasoner_row(row: Mapping[str, Any]) -> dict[str, Any]:
    compact = _selected(
        row,
        (
            "type",
            "round",
            "semantic_round",
            "control_attempt",
            "model",
            "parsed",
            "decision_payload",
            "schema_unwrapped",
            "format_repaired",
            "repair_failed",
            "api_response",
        ),
    )
    raw = str(row.get("raw", "") or "")
    if raw:
        compact["raw_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if row.get("repair_failed") or row.get("type") == "reasoner_json_repair":
            compact["raw_excerpt"] = raw[:1200]
    return compact


def _compact_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    compact = dict(row)
    compact.pop("frame_refs", None)
    config = compact.get("sampling_config")
    if isinstance(config, Mapping):
        compact["sampling_config"] = {
            key: value
            for key, value in config.items()
            if key not in {"hits", "occurrence_set", "rendered"}
        }
    return compact


def _selected(value: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    return {key: value[key] for key in keys if key in value}


def _bundle_root(archive: ZipFile) -> str:
    roots = {
        PurePosixPath(name).parts[0]
        for name in archive.namelist()
        if name and PurePosixPath(name).parts
    }
    if len(roots) != 1:
        raise ValueError(f"expected one bundle root, found: {sorted(roots)}")
    return next(iter(roots))


def _case_ids(archive: ZipFile, root: str) -> tuple[str, ...]:
    prefix = f"{root}/cases/"
    return tuple(
        sorted(
            {
                PurePosixPath(name).parts[2]
                for name in archive.namelist()
                if name.startswith(prefix)
                and len(PurePosixPath(name).parts) > 3
            }
        )
    )


def _read_json(archive: ZipFile, member: str) -> dict[str, Any]:
    value = json.loads(archive.read(member).decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {member}")
    return dict(value)


def _read_jsonl(archive: ZipFile, member: str) -> tuple[dict[str, Any], ...]:
    rows = []
    for line in archive.read(member).decode("utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return tuple(rows)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="Light 10-case diagnostic ZIP")
    parser.add_argument("--output", required=True, help="Fixture output directory")
    return parser.parse_args()


if __name__ == "__main__":
    main()
