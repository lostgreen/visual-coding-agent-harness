from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from vcah.interactive_agents import _answer, _normalize_decision
from vcah.multiround import ReasonerDecision
from vcah.replay import file_checksum, git_commit, stable_hash


FRAME_TIMESTAMP_FIELDS = (
    "query_id",
    "segment_id",
    "source_video_id",
    "source_time_sec",
    "virtual_time_sec",
    "sampling_fps",
    "fps_level",
)


class MechanicalReplayClient:
    """API-shaped deterministic client used only to materialize replay frames."""

    model = "mechanical-replay-no-vlm"

    def __init__(self) -> None:
        self._last_response_metadata: dict[str, Any] = {}

    def chat(
        self,
        prompt: str,
        *,
        image_paths: Sequence[str] = (),
        max_tokens: int | None = None,
    ) -> str:
        del prompt, max_tokens
        image_count = len(tuple(image_paths))
        self._last_response_metadata = {
            "finish_reason": "mechanical_replay",
            "images_requested": image_count,
            "images_attached": image_count,
            "images_dropped": 0,
            "provider_request_id": "",
        }
        return json.dumps(
            {
                "summary": "Semantic interpretation disabled for mechanical replay.",
                "observations": [],
            },
            sort_keys=True,
        )

    @property
    def last_response_metadata(self) -> dict[str, Any]:
        return dict(self._last_response_metadata)

    @property
    def replay_settings(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": None,
            "top_p": None,
            "requested_seed": None,
            "provider_seed_supported": False,
            "provider_reported_seed_support": "not_applicable",
        }


class RecordedDecisionReasoner:
    """Replay compact historical decisions without a Reasoner API call."""

    def __init__(self, fixture: Mapping[str, Any], *, trace_path: Path) -> None:
        self.fixture = dict(fixture)
        self.trace_path = Path(trace_path)
        self.payloads = recorded_decision_payloads(self.fixture)
        self.calls = 0
        self._last_decision_metadata: dict[str, Any] = {}

    @property
    def decision_count(self) -> int:
        return len(self.payloads)

    @property
    def source_revision(self) -> str:
        return str(self.fixture.get("source_revision", "74f012d-r1") or "74f012d-r1")

    def decide(self, **kwargs: Any) -> ReasonerDecision:
        self.calls += 1
        source_index = self.calls if self.calls <= len(self.payloads) else None
        if source_index is not None:
            source_payload = dict(self.payloads[source_index - 1])
        else:
            runtime = _mapping(self.fixture.get("runtime_summary"))
            source_payload = {
                "action": "answer",
                "answer": (
                    str(runtime.get("answer", "") or "")
                    if bool(runtime.get("answer_present"))
                    else ""
                ),
            }

        # Workspace state is not part of the mechanical replay contract. Keeping
        # only actions and tasks prevents historical IDs from changing dispatch.
        payload = dict(source_payload)
        payload["workspace_ops"] = []
        payload.pop("workspace", None)
        if (
            source_index is not None
            and source_index < len(self.payloads)
            and str(payload.get("action", "") or "") == "answer"
        ):
            payload = {"action": "update_workspace", "workspace_ops": []}
        task_errors: list[dict[str, Any]] = []
        decision_errors: list[dict[str, Any]] = []
        value = _normalize_decision(
            payload,
            round_id=int(kwargs.get("semantic_round", self.calls) or self.calls),
            task_errors=task_errors,
            decision_errors=decision_errors,
        )
        value["answer"] = _answer(value["answer"], dict(kwargs.get("options") or {}))
        decision = ReasonerDecision(**value)
        fingerprint = decision_fingerprint(
            round_id=int(kwargs.get("semantic_round", self.calls) or self.calls),
            action=str(source_payload.get("action", "") or ""),
            tasks=tuple(source_payload.get("tasks", ()) or ()),
        )
        _append_jsonl(
            self.trace_path,
            {
                "type": "recorded_reasoner_decision",
                "round": self.calls,
                "semantic_round": int(
                    kwargs.get("semantic_round", self.calls) or self.calls
                ),
                "source_index": source_index,
                "source_revision": self.source_revision,
                "effective_action": decision.action,
                "decision_fingerprint": fingerprint,
            },
        )
        self._last_decision_metadata = {
            "decision_payload_valid": True,
            "decision_schema_errors": decision_errors,
            "task_resolution_errors": task_errors,
            "internal_control_retry_count": 0,
            "format_repaired": False,
            "repair_failed": False,
            "prompt_char_count": 0,
            "prompt_schema_token_cost": 0,
        }
        return decision

    def consume_decision_metadata(self) -> Mapping[str, Any]:
        metadata = dict(self._last_decision_metadata)
        self._last_decision_metadata = {}
        return metadata


def load_fixture(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Phase 5R fixture must be a JSON object: {path}")
    return dict(value)


def recorded_decision_payloads(fixture: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(row.get("decision_payload", {}) or {})
        for row in tuple(fixture.get("reasoner_decisions", ()) or ())
        if isinstance(row, Mapping)
        and str(row.get("type", "") or "") == "reasoner_workspace"
        and isinstance(row.get("decision_payload"), Mapping)
    )


def decision_fingerprint(
    *,
    round_id: int,
    action: str,
    tasks: Sequence[Any],
) -> dict[str, Any]:
    rows = [_task_fingerprint(task) for task in tasks]
    payload = {
        "round": int(round_id),
        "action": str(action or "").strip().casefold(),
        "tasks": rows,
    }
    return {**payload, "digest": stable_hash(payload)}


def runtime_decision_trace(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        decision_fingerprint(
            round_id=int(row.get("semantic_round", row.get("round", 0)) or 0),
            action=str(row.get("action", "") or ""),
            tasks=tuple(row.get("tasks", ()) or ()),
        )
        for row in trace
        if str(row.get("type", "") or "") == "reasoner_decision"
    ]
    return {
        "schema_version": "DecisionTraceDigestV1",
        "decision_count": len(rows),
        "digest": stable_hash(rows),
        "decisions": rows,
    }


def frame_cost_breakdown(
    trace: Sequence[Mapping[str, Any]],
    observation_rows: Sequence[Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    requested_tasks = [
        task
        for row in trace
        if str(row.get("type", "") or "") == "reasoner_decision"
        for task in tuple(row.get("tasks", ()) or ())
        if isinstance(task, Mapping)
    ]
    visual_tasks = [
        task
        for task in requested_tasks
        if str(task.get("inspection_mode", "") or "") == "window"
    ]
    durations = [
        float(raw[1]) - float(raw[0])
        for task in visual_tasks
        if isinstance((raw := task.get("time_range")), Sequence)
        and not isinstance(raw, (str, bytes))
        and len(raw) == 2
        and float(raw[1]) > float(raw[0])
    ]
    requested_fps = [
        float(task.get("sampling_floor_fps", 0.5) or 0.5)
        for task in visual_tasks
    ]
    visual_observations = [
        row
        for row in observation_rows
        if str(row.get("modality", "") or "") == "visual"
    ]
    manifests = [
        dict(config.get("sampling_manifest", {}) or {})
        for row in visual_observations
        if isinstance((config := row.get("sampling_config")), Mapping)
        and isinstance(config.get("sampling_manifest"), Mapping)
    ]
    effective_fps = [
        float(item.get("effective_fps", 0.0) or 0.0)
        for item in manifests
        if float(item.get("effective_fps", 0.0) or 0.0) > 0.0
    ]
    material_signatures = [
        stable_hash(
            {
                "source_video_ids": list(row.get("source_video_ids", ()) or ()),
                "frame_times": list(manifest.get("frame_times", ()) or ()),
            }
        )
        for row, manifest in zip(visual_observations, manifests)
    ]
    frame_cap_hits = sum(
        len(tuple(manifest.get("frame_times", ()) or ()))
        >= int(config.get("max_frames", 0) or 0)
        for row, manifest in zip(visual_observations, manifests)
        if isinstance((config := row.get("sampling_config")), Mapping)
        and int(config.get("max_frames", 0) or 0) > 0
    )
    semantic_rounds = {
        int(row.get("semantic_round", row.get("round", 0)) or 0)
        for row in trace
        if str(row.get("type", "") or "") == "reasoner_decision"
    }
    return {
        "schema_version": "MGERPhase5RFrameCostV1",
        "requested_visual_window_count": len(visual_tasks),
        "visual_window_count": len({str(row.get("query_id", "") or "") for row in frame_rows}),
        "sum_requested_window_duration": round(sum(durations), 6),
        "mean_requested_fps": _mean(requested_fps),
        "mean_effective_fps": _mean(effective_fps),
        "frame_cap_hits": frame_cap_hits,
        "reinspection_count": len(material_signatures) - len(set(material_signatures)),
        "unique_visual_material_attempts": len(set(material_signatures)),
        "caption_search_count": sum(
            str(row.get("modality", "") or "") == "caption_search"
            for row in observation_rows
        ),
        "asr_search_count": sum(
            str(row.get("modality", "") or "") == "asr"
            for row in observation_rows
        ),
        "semantic_rounds": len(semantic_rounds),
        "visual_frames_inspected": len(frame_rows),
    }


def mechanical_replay_audit(
    fixture: Mapping[str, Any],
    *,
    workspace_root: Path,
    trace: Sequence[Mapping[str, Any]],
    observation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_rows = _normalized_frame_rows(
        tuple(fixture.get("frame_sampling_manifest", ()) or ())
    )
    actual_rows = _normalized_frame_rows(
        _read_jsonl(Path(workspace_root) / "observations" / "window_frame_manifest.jsonl")
    )
    expected_by_task = _group_frames(expected_rows)
    actual_by_task = _group_frames(actual_rows)
    task_ids = sorted(set(expected_by_task) | set(actual_by_task))
    per_task = []
    for task_id in task_ids:
        expected = expected_by_task.get(task_id, ())
        actual = actual_by_task.get(task_id, ())
        per_task.append(
            {
                "task_id": task_id,
                "expected_frame_count": len(expected),
                "actual_frame_count": len(actual),
                "expected_timestamp_digest": stable_hash(expected),
                "actual_timestamp_digest": stable_hash(actual),
                "frame_count_exact": len(expected) == len(actual),
                "timestamp_digest_exact": stable_hash(expected) == stable_hash(actual),
            }
        )
    expected_decisions = [
        decision_fingerprint(
            round_id=index,
            action=str(payload.get("action", "") or ""),
            tasks=tuple(payload.get("tasks", ()) or ()),
        )
        for index, payload in enumerate(recorded_decision_payloads(fixture), start=1)
    ]
    recorded_rows = [
        dict(row.get("decision_fingerprint", {}) or {})
        for row in _read_jsonl(Path(workspace_root) / "interactions.jsonl")
        if str(row.get("type", "") or "") == "recorded_reasoner_decision"
        and row.get("source_index") is not None
    ]
    checks = {
        "case_id_match": str(fixture.get("case_id", "") or "")
        == Path(workspace_root).name,
        "frame_count_exact": len(expected_rows) == len(actual_rows),
        "frame_timestamp_digest_exact": stable_hash(expected_rows)
        == stable_hash(actual_rows),
        "per_task_frame_count_exact": bool(per_task)
        and all(row["frame_count_exact"] for row in per_task),
        "per_task_timestamp_digest_exact": bool(per_task)
        and all(row["timestamp_digest_exact"] for row in per_task),
        "decision_trace_exact": stable_hash(expected_decisions)
        == stable_hash(recorded_rows),
    }
    return {
        "schema_version": "MGERPhase5RMechanicalReplayV1",
        "case_id": str(fixture.get("case_id", Path(workspace_root).name) or ""),
        "source_revision": "74f012d-r1",
        "decision": "PASS" if all(checks.values()) else "STOP",
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "expected": {
            "frame_count": len(expected_rows),
            "frame_timestamp_digest": stable_hash(expected_rows),
            "decision_trace_digest": stable_hash(expected_decisions),
        },
        "actual": {
            "frame_count": len(actual_rows),
            "frame_timestamp_digest": stable_hash(actual_rows),
            "decision_trace_digest": stable_hash(recorded_rows),
        },
        "per_task": per_task,
        "cost_breakdown": frame_cost_breakdown(
            trace,
            observation_rows,
            actual_rows,
        ),
    }


def build_run_provenance(
    workspace: Any,
    *,
    interactions_path: Path,
    role_settings: Mapping[str, Mapping[str, Any]],
    caption_index_digest: str | None,
    repository_root: Path,
) -> dict[str, Any]:
    interactions = _read_jsonl(interactions_path)
    request_ids: list[str] = []
    resolved_deployments: set[str] = set()
    prompt_hashes: dict[str, list[str]] = defaultdict(list)
    for row in interactions:
        row_type = str(row.get("type", "") or "")
        role = "reasoner" if "reasoner" in row_type else "investigator"
        prompt = str(row.get("prompt", "") or "")
        if prompt:
            prompt_hashes[role].append(_text_hash(prompt))
        metadata = _mapping(row.get("api_response"))
        request_id = str(metadata.get("provider_request_id", "") or "")
        if request_id and request_id not in request_ids:
            request_ids.append(request_id)
        for key in ("resolved_deployment_name", "deployment_name", "resolved_model"):
            value = str(metadata.get(key, "") or "")
            if value:
                resolved_deployments.add(value)
    source_manifest = [
        {
            "segment_id": str(segment.segment_id),
            "source_video_id": str(segment.source_video_id),
            "source_range": [segment.source_start_sec, segment.source_end_sec],
            "virtual_range": [segment.virtual_start_sec, segment.virtual_end_sec],
            "source_file": Path(segment.source_path).name,
            "source_bytes": (
                Path(segment.source_path).stat().st_size
                if Path(segment.source_path).is_file()
                else 0
            ),
        }
        for segment in workspace.manifest.segments
    ]
    environment = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": Path(sys.executable).name,
    }
    return {
        "schema_version": "MGERPhase5RProvenanceV1",
        "runner_commit": git_commit(repository_root),
        "models": {role: dict(settings) for role, settings in role_settings.items()},
        "resolved_deployment_names": sorted(resolved_deployments),
        "provider_request_ids": request_ids,
        "service_version_unpinned": not bool(resolved_deployments),
        "caption_index_digest": str(caption_index_digest or ""),
        "frame_cache_digest": str(file_checksum(workspace.frame_manifest)["sha256"]),
        "source_video_manifest_digest": stable_hash(source_manifest),
        "prompt_digest": stable_hash(prompt_hashes),
        "reasoner_system_prompt_digest": None,
        "reasoner_system_prompt_status": "not_separate_in_client_contract",
        "environment": {**environment, "digest": stable_hash(environment)},
    }


def _task_fingerprint(task: Any) -> dict[str, Any]:
    def value(name: str, default: Any = None) -> Any:
        if isinstance(task, Mapping):
            return task.get(name, default)
        return getattr(task, name, default)

    raw_range = value("time_range")
    normalized_range = (
        [round(float(raw_range[0]), 6), round(float(raw_range[1]), 6)]
        if isinstance(raw_range, Sequence)
        and not isinstance(raw_range, (str, bytes))
        and len(raw_range) == 2
        else []
    )
    queries = tuple(value("caption_queries", value("queries", ())) or ())
    search_terms = tuple(value("search_terms", ()) or ())
    return {
        "query_id": str(value("query_id", value("id", "")) or ""),
        "action": "inspect",
        "inspection_mode": str(value("inspection_mode", "window") or "window"),
        "normalized_time_range": normalized_range,
        "query_text_hash": _text_hash("\n".join(str(item) for item in (*queries, *search_terms))),
        "goal_hash": _text_hash(str(value("goal", "") or "")),
        "fps": round(float(value("sampling_floor_fps", 0.5) or 0.5), 6),
        "top_k": int(value("top_k", 12) or 12),
    }


def _normalized_frame_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                key: (
                    round(float(row.get(key, 0.0) or 0.0), 6)
                    if key in {"source_time_sec", "virtual_time_sec", "sampling_fps"}
                    else str(row.get(key, "") or "")
                )
                for key in FRAME_TIMESTAMP_FIELDS
            }
        )
    return tuple(normalized)


def _group_frames(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("query_id", "") or "")].append(row)
    return {key: tuple(value) for key, value in grouped.items()}


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not Path(path).is_file():
        return ()
    return tuple(
        dict(value)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
        and isinstance((value := json.loads(line)), Mapping)
    )


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _text_hash(value: str) -> str:
    normalized = " ".join(str(value or "").casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
