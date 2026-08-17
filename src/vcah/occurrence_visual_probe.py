from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from vcah.occurrence_sufficiency import REFERENT_IDENTIFYING_CONSTRAINT_TYPES
from vcah.virtual_video import VirtualVideoManifest


VISUAL_PROBE_CONTRACT = "blind_referent_visual_discriminability_v1"
VISUAL_PROBE_FRAME_CAP = 8
VISUAL_PROBE_FPS = 2.0
VISUAL_PROBE_PAIR_KINDS = ("matched", "mismatched", "null")
VISUAL_PROBE_VERDICTS = ("supported", "contradicted", "unknown")
VISUAL_PROBE_REFERENT_TYPES = frozenset(
    {"identity", "event", "action", "temporal", "order", "location"}
)
if not VISUAL_PROBE_REFERENT_TYPES.issubset(
    REFERENT_IDENTIFYING_CONSTRAINT_TYPES
):
    raise RuntimeError("visual probe constraint whitelist is not referent-only")


@dataclass(frozen=True)
class VisualProbeSource:
    case_id: str
    set_id: str
    constraints: tuple[dict[str, str], ...]
    candidates: tuple[dict[str, Any], ...]
    clue_intervals: tuple[tuple[float, float], ...]
    asset_ref: str


def load_visual_probe_source(
    positive_run_dir: Path,
    *,
    evaluation_record_path: Path,
) -> VisualProbeSource:
    run_dir = Path(positive_run_dir)
    case = _read_json(run_dir / "case.json")
    runtime = _read_json(run_dir / "runtime_summary.json")
    evaluation = _read_json(Path(evaluation_record_path))
    case_id = str(case.get("case_id", run_dir.name) or run_dir.name)
    if str(evaluation.get("case_id", "") or "") != case_id:
        raise ValueError(f"{case_id}: evaluation record case mismatch")
    trace = tuple(
        dict(row)
        for row in tuple(runtime.get("trace", ()) or ())
        if isinstance(row, Mapping)
    )
    decisions = tuple(
        row
        for row in trace
        if row.get("type")
        in {
            "occurrence_sufficiency_gate_decision",
            "occurrence_sufficiency_decision",
        }
        and str(row.get("set_id", "") or "")
    )
    if not decisions:
        raise ValueError(f"{case_id}: no frozen sufficiency decision")
    decision = decisions[-1]
    set_id = str(decision.get("set_id", "") or "")
    scope_ids = tuple(
        dict.fromkeys(
            str(value)
            for value in tuple(decision.get("scope_occurrence_ids", ()) or ())
            if str(value)
        )
    )
    constraints = _constraints_for_set(trace, set_id=set_id)
    candidates = _candidates_for_set(
        run_dir / "observation_log.jsonl",
        runtime=runtime,
        set_id=set_id,
        scope_ids=scope_ids,
    )
    clues = tuple(
        tuple(sorted((float(value[0]), float(value[1]))))
        for value in tuple(evaluation.get("clue_intervals", ()) or ())
        if _is_interval(value)
    )
    return VisualProbeSource(
        case_id=case_id,
        set_id=set_id,
        constraints=constraints,
        candidates=candidates,
        clue_intervals=clues,
        asset_ref=str(case.get("asset_ref", "") or ""),
    )


def build_case_probe_plan(
    source: VisualProbeSource,
    *,
    manifest: VirtualVideoManifest,
    seed: int,
) -> dict[str, Any]:
    constraints = tuple(
        row
        for row in source.constraints
        if row.get("constraint_type") in VISUAL_PROBE_REFERENT_TYPES
    )
    matched_candidates = tuple(
        candidate
        for candidate in source.candidates
        if _candidate_overlap(candidate, source.clue_intervals) > 0.0
    )
    mismatched_candidates = tuple(
        candidate
        for candidate in source.candidates
        if _candidate_overlap(candidate, source.clue_intervals) <= 0.0
    )
    reasons = []
    if not constraints:
        reasons.append("no_referent_constraint")
    if not matched_candidates:
        reasons.append("no_gold_overlap_candidate")
    if not mismatched_candidates:
        reasons.append("no_competing_non_gold_candidate")
    if reasons:
        return {
            "case_id": source.case_id,
            "eligible": False,
            "exclusion_reasons": reasons,
            "constraint_count": len(constraints),
            "candidate_count": len(source.candidates),
        }

    matched = sorted(
        matched_candidates,
        key=lambda row: (
            -_candidate_overlap(row, source.clue_intervals),
            -_number(row.get("max_score"), default=0.0),
            _number(row.get("rank"), default=1e9),
            str(row.get("occurrence_id", "") or ""),
        ),
    )[0]
    mismatched = sorted(
        mismatched_candidates,
        key=lambda row: (
            -_number(row.get("max_score"), default=0.0),
            _number(row.get("rank"), default=1e9),
            str(row.get("occurrence_id", "") or ""),
        ),
    )[0]
    matched_range = _candidate_range(matched)
    mismatched_range = _candidate_range(mismatched)
    null_range, null_segment = _null_window(
        manifest,
        case_id=source.case_id,
        duration_sec=max(1.0, matched_range[1] - matched_range[0]),
        excluded_ranges=(
            *source.clue_intervals,
            *tuple(_candidate_range(row) for row in source.candidates),
        ),
        excluded_source_video_ids={
            str(value)
            for row in source.candidates
            for value in tuple(row.get("source_video_ids", ()) or ())
            if str(value)
        },
        seed=seed,
    )
    null_occurrence_id = "probe_null_occ_" + _digest(
        [source.case_id, list(null_range), null_segment.source_video_id]
    )[:20]
    null_locator_id = "probe_null_locator_" + _digest(
        [source.case_id, null_occurrence_id]
    )[:20]
    windows = (
        _window_spec(
            source,
            pair_kind="matched",
            locator_id=source.set_id,
            occurrence_id=str(matched.get("occurrence_id", "") or ""),
            time_range=matched_range,
            source_kind="frozen_candidate",
        ),
        _window_spec(
            source,
            pair_kind="mismatched",
            locator_id=source.set_id,
            occurrence_id=str(mismatched.get("occurrence_id", "") or ""),
            time_range=mismatched_range,
            source_kind="frozen_candidate",
        ),
        _window_spec(
            source,
            pair_kind="null",
            locator_id=null_locator_id,
            occurrence_id=null_occurrence_id,
            time_range=null_range,
            source_kind="synthetic_null_control",
            segment_id=null_segment.segment_id,
            source_video_id=null_segment.source_video_id,
        ),
    )
    items = []
    for constraint in constraints:
        group_id = "probe_group_" + _digest(
            [source.case_id, source.set_id, constraint["constraint_id"]]
        )[:20]
        for window in windows:
            item_id = "probe_item_" + _digest(
                [group_id, window["visual_observation_id"]]
            )[:20]
            items.append(
                {
                    "item_id": item_id,
                    "pair_group_id": group_id,
                    "case_id": source.case_id,
                    "pair_kind": window["pair_kind"],
                    "visual_observation_id": window["visual_observation_id"],
                    "locator_id": window["locator_id"],
                    "occurrence_id": window["occurrence_id"],
                    "constraint_id": constraint["constraint_id"],
                    "constraint_type": constraint["constraint_type"],
                    "constraint_description": constraint["description"],
                }
            )
    candidate_registry = [
        {
            "locator_id": source.set_id,
            "occurrence_id": str(row.get("occurrence_id", "") or ""),
            "time_range": list(_candidate_range(row)),
        }
        for row in source.candidates
    ]
    candidate_registry.append(
        {
            "locator_id": null_locator_id,
            "occurrence_id": null_occurrence_id,
            "time_range": list(null_range),
            "synthetic_control": True,
        }
    )
    return {
        "case_id": source.case_id,
        "eligible": True,
        "exclusion_reasons": [],
        "set_id": source.set_id,
        "constraints": [dict(row) for row in constraints],
        "candidate_registry": candidate_registry,
        "windows": [dict(row) for row in windows],
        "items": items,
    }


def finalize_case_probe_plan(
    plan: Mapping[str, Any],
    *,
    materialized_windows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not bool(plan.get("eligible")):
        return dict(plan)
    windows = []
    for raw_window in tuple(plan.get("windows", ()) or ()):
        window = dict(raw_window)
        observation_id = str(window.get("visual_observation_id", "") or "")
        materialized = materialized_windows.get(observation_id)
        if not isinstance(materialized, Mapping):
            window.update({"executed": False, "frame_ids": [], "frames": []})
        else:
            frames = [
                dict(row)
                for row in tuple(materialized.get("frames", ()) or ())
                if isinstance(row, Mapping)
            ]
            window.update(
                {
                    "executed": bool(materialized.get("executed")) and bool(frames),
                    "frame_ids": [str(row.get("frame_id", "") or "") for row in frames],
                    "frames": frames,
                }
            )
        windows.append(window)
    return {**dict(plan), "windows": windows}


def audit_visual_probe_manifest(
    manifest: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    cases = tuple(
        row
        for row in tuple(manifest.get("cases", ()) or ())
        if isinstance(row, Mapping) and bool(row.get("eligible"))
    )
    all_items: list[Mapping[str, Any]] = []
    unbound = 0
    wrong_occurrence = 0
    invalid_constraint = 0
    incomplete_frames = 0
    silent_locator_drops = 0
    incomplete_triplets = 0
    duplicate_ids = 0
    seen_item_ids: set[str] = set()
    seen_observation_ids: set[str] = set()
    referenced_observation_ids: set[str] = set()
    for case in cases:
        constraints = {
            str(row.get("constraint_id", "") or ""): str(
                row.get("constraint_type", "") or ""
            )
            for row in tuple(case.get("constraints", ()) or ())
            if isinstance(row, Mapping)
        }
        candidates = {
            (
                str(row.get("locator_id", "") or ""),
                str(row.get("occurrence_id", "") or ""),
            ): tuple(row.get("time_range", ()) or ())
            for row in tuple(case.get("candidate_registry", ()) or ())
            if isinstance(row, Mapping)
        }
        windows = {
            str(row.get("visual_observation_id", "") or ""): row
            for row in tuple(case.get("windows", ()) or ())
            if isinstance(row, Mapping)
        }
        if len(windows) != len(tuple(case.get("windows", ()) or ())):
            duplicate_ids += 1
        duplicate_ids += len(set(windows).intersection(seen_observation_ids))
        seen_observation_ids.update(windows)
        groups: dict[str, set[str]] = {}
        for item in tuple(case.get("items", ()) or ()):
            if not isinstance(item, Mapping):
                unbound += 1
                continue
            all_items.append(item)
            item_id = str(item.get("item_id", "") or "")
            if not item_id or item_id in seen_item_ids:
                duplicate_ids += 1
            seen_item_ids.add(item_id)
            group_id = str(item.get("pair_group_id", "") or "")
            groups.setdefault(group_id, set()).add(
                str(item.get("pair_kind", "") or "")
            )
            constraint_id = str(item.get("constraint_id", "") or "")
            constraint_type = str(item.get("constraint_type", "") or "")
            if (
                constraints.get(constraint_id) != constraint_type
                or constraint_type not in VISUAL_PROBE_REFERENT_TYPES
            ):
                invalid_constraint += 1
            observation_id = str(item.get("visual_observation_id", "") or "")
            referenced_observation_ids.add(observation_id)
            window = windows.get(observation_id)
            if window is None:
                unbound += 1
                silent_locator_drops += 1
                continue
            locator_id = str(item.get("locator_id", "") or "")
            occurrence_id = str(item.get("occurrence_id", "") or "")
            if (
                str(window.get("locator_id", "") or "") != locator_id
                or str(window.get("occurrence_id", "") or "") != occurrence_id
                or (locator_id, occurrence_id) not in candidates
                or tuple(window.get("time_range", ()) or ())
                != candidates.get((locator_id, occurrence_id))
            ):
                wrong_occurrence += 1
            if not bool(window.get("executed")):
                silent_locator_drops += 1
            frames = tuple(window.get("frames", ()) or ())
            frame_ids = tuple(str(value) for value in tuple(window.get("frame_ids", ())) or ())
            if (
                not frames
                or len(frames) > VISUAL_PROBE_FRAME_CAP
                or frame_ids
                != tuple(str(row.get("frame_id", "") or "") for row in frames)
                or len(frame_ids) != len(set(frame_ids))
                or any(
                    not str(row.get("path", "") or "")
                    or not str(row.get("segment_id", "") or "")
                    or not str(row.get("source_video_id", "") or "")
                    or not isinstance(row.get("virtual_time_sec"), (int, float))
                    or not _contains(
                        tuple(window.get("time_range", ()) or ()),
                        float(row.get("virtual_time_sec", 0.0)),
                    )
                    for row in frames
                )
            ):
                incomplete_frames += 1
            if root is not None and any(
                not (Path(root) / str(row.get("path", "") or "")).is_file()
                for row in frames
            ):
                incomplete_frames += 1
        incomplete_triplets += sum(
            kinds != set(VISUAL_PROBE_PAIR_KINDS) for kinds in groups.values()
        )
    expected_item_count = sum(
        len(tuple(case.get("constraints", ()) or ()))
        * len(VISUAL_PROBE_PAIR_KINDS)
        for case in cases
    )
    unreferenced_observations = len(
        seen_observation_ids - referenced_observation_ids
    )
    counts = {
        "eligible_case_count": len(cases),
        "visual_observation_count": len(seen_observation_ids),
        "item_count": len(all_items),
        "expected_item_count": expected_item_count,
        "unbound_observation_count": unbound,
        "wrong_occurrence_binding_count": wrong_occurrence,
        "invalid_constraint_binding_count": invalid_constraint,
        "incomplete_frame_provenance_count": incomplete_frames,
        "silent_locator_drop_count": silent_locator_drops,
        "incomplete_triplet_count": incomplete_triplets,
        "unreferenced_observation_count": unreferenced_observations,
        "duplicate_id_count": duplicate_ids,
    }
    checks = {
        "contract_matches": manifest.get("contract") == VISUAL_PROBE_CONTRACT,
        "observation_provenance_coverage_complete": (
            len(all_items) == expected_item_count and unbound == 0
        ),
        "all_observations_executed": silent_locator_drops == 0,
        "locator_occurrence_bindings_valid": wrong_occurrence == 0,
        "referent_constraint_bindings_valid": invalid_constraint == 0,
        "frame_provenance_complete": incomplete_frames == 0,
        "triplets_complete": incomplete_triplets == 0,
        "all_observations_constraint_bound": unreferenced_observations == 0,
        "identifiers_unique": duplicate_ids == 0,
        "endpoint_values_not_gated": True,
    }
    return {
        "schema_version": "MMLifelongVisualProbeProvenanceAuditV1",
        "contract": VISUAL_PROBE_CONTRACT,
        "counts": counts,
        "checks": checks,
        "structural_gate_passed": bool(cases) and all(checks.values()),
    }


def visual_probe_prompt(item: Mapping[str, Any]) -> str:
    constraint_type = str(item.get("constraint_type", "") or "")
    description = str(item.get("constraint_description", "") or "")
    if constraint_type not in VISUAL_PROBE_REFERENT_TYPES or not description:
        raise ValueError("visual probe item has an invalid referent constraint")
    return (
        "Act as a blind visual referent verifier. The attached frames are ordered "
        "chronologically and come from one fixed video window. Decide only whether "
        "the visible frames establish the referent-identifying constraint below. "
        "Use supported only for direct visible support, contradicted only for direct "
        "visible incompatibility, and unknown when the frames are insufficient. Do "
        "not infer an answer, rank occurrences, or use outside knowledge. Return "
        "exactly one JSON object with key verdict and no other keys.\n"
        f"CONSTRAINT_TYPE={constraint_type}\n"
        f"CONSTRAINT={description}\n"
        'OUTPUT_SCHEMA={"verdict":"supported|contradicted|unknown"}'
    )


def parse_visual_probe_response(raw: str) -> str | None:
    text = str(raw or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping) or set(payload) != {"verdict"}:
        return None
    verdict = str(payload.get("verdict", "") or "").strip().casefold()
    return verdict if verdict in VISUAL_PROBE_VERDICTS else None


def _constraints_for_set(
    trace: Sequence[Mapping[str, Any]], *, set_id: str
) -> tuple[dict[str, str], ...]:
    declarations = tuple(
        row
        for row in trace
        if row.get("type") == "occurrence_evidence_declaration"
        and str(row.get("set_id", "") or "") == set_id
    )
    raw_constraints: Sequence[Any] = ()
    if declarations:
        raw_constraints = tuple(declarations[-1].get("constraints", ()) or ())
    if not raw_constraints:
        operation_rows = []
        for row in trace:
            if row.get("type") != "reasoner_decision":
                continue
            for operation in tuple(row.get("occurrence_ops", ()) or ()):
                if (
                    isinstance(operation, Mapping)
                    and str(operation.get("set_id", "") or "") == set_id
                    and operation.get("constraints_checked")
                ):
                    operation_rows.append(operation)
        if operation_rows:
            raw_constraints = tuple(
                operation_rows[-1].get("constraints_checked", ()) or ()
            )
    normalized = []
    seen: set[str] = set()
    for row in raw_constraints:
        if not isinstance(row, Mapping):
            continue
        constraint_id = str(row.get("constraint_id", "") or "").strip()
        constraint_type = str(row.get("constraint_type", "") or "").strip().casefold()
        description = str(row.get("description", "") or "").strip()
        if not constraint_id or constraint_id in seen or not description:
            continue
        seen.add(constraint_id)
        normalized.append(
            {
                "constraint_id": constraint_id,
                "constraint_type": constraint_type,
                "description": description,
            }
        )
    if not normalized:
        raise ValueError(f"no frozen constraints for occurrence set {set_id}")
    return tuple(normalized)


def _candidates_for_set(
    observation_path: Path,
    *,
    runtime: Mapping[str, Any],
    set_id: str,
    scope_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    candidates: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(observation_path):
        config = row.get("sampling_config")
        occurrence_set = (
            config.get("occurrence_set") if isinstance(config, Mapping) else None
        )
        if not isinstance(occurrence_set, Mapping):
            continue
        row_set_id = str(
            row.get(
                "attempt_id",
                occurrence_set.get("attempt_id", occurrence_set.get("set_id", "")),
            )
            or ""
        )
        if row_set_id != set_id:
            continue
        for candidate in tuple(occurrence_set.get("candidates", ()) or ()):
            if not isinstance(candidate, Mapping):
                continue
            occurrence_id = str(candidate.get("occurrence_id", "") or "")
            if occurrence_id:
                candidates[occurrence_id] = dict(candidate)
    if not candidates:
        state = runtime.get("occurrence_resolution_state")
        if isinstance(state, Mapping):
            for raw_set in tuple(state.get("sets", ()) or ()):
                if not isinstance(raw_set, Mapping) or str(
                    raw_set.get("set_id", "") or ""
                ) != set_id:
                    continue
                for candidate in tuple(raw_set.get("candidates", ()) or ()):
                    if isinstance(candidate, Mapping):
                        occurrence_id = str(candidate.get("occurrence_id", "") or "")
                        if occurrence_id:
                            candidates[occurrence_id] = dict(candidate)
    ordered_ids = tuple(value for value in scope_ids if value in candidates)
    if not ordered_ids:
        ordered_ids = tuple(candidates)
    result = tuple(candidates[value] for value in ordered_ids)
    if not result or any(not _valid_candidate(row) for row in result):
        raise ValueError(f"invalid frozen candidates for occurrence set {set_id}")
    return result


def _window_spec(
    source: VisualProbeSource,
    *,
    pair_kind: str,
    locator_id: str,
    occurrence_id: str,
    time_range: tuple[float, float],
    source_kind: str,
    segment_id: str = "",
    source_video_id: str = "",
) -> dict[str, Any]:
    observation_id = "visual_probe_obs_" + _digest(
        [source.case_id, pair_kind, locator_id, occurrence_id, list(time_range)]
    )[:20]
    return {
        "visual_observation_id": observation_id,
        "case_id": source.case_id,
        "pair_kind": pair_kind,
        "locator_id": locator_id,
        "occurrence_id": occurrence_id,
        "time_range": list(time_range),
        "source_kind": source_kind,
        "segment_id": segment_id,
        "source_video_id": source_video_id,
        "executed": False,
        "frame_ids": [],
        "frames": [],
    }


def _null_window(
    manifest: VirtualVideoManifest,
    *,
    case_id: str,
    duration_sec: float,
    excluded_ranges: Sequence[Sequence[float]],
    excluded_source_video_ids: set[str],
    seed: int,
):
    duration = max(1.0, float(duration_sec))
    segments = tuple(
        segment
        for segment in manifest.segments
        if segment.duration_sec >= duration + 2.0
        and segment.source_video_id not in excluded_source_video_ids
    )
    if not segments:
        segments = tuple(
            segment
            for segment in manifest.segments
            if segment.duration_sec >= duration + 2.0
        )
    if not segments:
        raise ValueError(f"{case_id}: no segment can host a null window")
    offset = int(_digest([case_id, seed])[:12], 16) % len(segments)
    ordered = segments[offset:] + segments[:offset]
    fractions = (0.17, 0.53, 0.79, 0.31, 0.67)
    for segment in ordered:
        slack = segment.duration_sec - duration
        for fraction in fractions:
            start = round(segment.virtual_start_sec + slack * fraction, 3)
            end = round(start + duration, 3)
            candidate = (start, min(end, segment.virtual_end_sec))
            if not any(_overlap(candidate, value) > 0.0 for value in excluded_ranges):
                return candidate, segment
    raise ValueError(f"{case_id}: no non-overlapping null window")


def _candidate_overlap(
    candidate: Mapping[str, Any], clues: Sequence[Sequence[float]]
) -> float:
    interval = _candidate_range(candidate)
    return sum(_overlap(interval, clue) for clue in clues)


def _candidate_range(candidate: Mapping[str, Any]) -> tuple[float, float]:
    value = tuple(candidate.get("time_range", ()) or ())
    if len(value) != 2:
        raise ValueError("candidate has no valid time range")
    return tuple(sorted((float(value[0]), float(value[1]))))


def _valid_candidate(candidate: Mapping[str, Any]) -> bool:
    try:
        interval = _candidate_range(candidate)
    except (TypeError, ValueError):
        return False
    return bool(str(candidate.get("occurrence_id", "") or "")) and interval[1] > interval[0]


def _overlap(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 2 or len(right) != 2:
        return 0.0
    return max(
        0.0,
        min(float(left[1]), float(right[1]))
        - max(float(left[0]), float(right[0])),
    )


def _contains(interval: Sequence[float], value: float) -> bool:
    return (
        len(interval) == 2
        and float(interval[0]) - 1e-6 <= value <= float(interval[1]) + 1e-6
    )


def _number(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _is_interval(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    )


def _digest(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not Path(path).is_file():
        return ()
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return tuple(rows)
