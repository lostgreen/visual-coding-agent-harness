from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from visual_coding_agent_harness.core.contracts import CONTRACT_VERSION
from visual_coding_agent_harness.legacy.workspace_v2 import EvidenceWorkspace


class FinalDecisionOwner(str, Enum):
    MODEL = "model"
    FORMAT_REPAIR = "format_repair"
    NONE = "none"
    FRAMEWORK = "framework"


@dataclass
class TrainingTrajectory:
    run_id: str
    case_id: str
    question: str
    options: list[str]
    ground_truth: str | None
    final_decision: str
    final_decision_owner: str
    diagnostic_errors: list[str]
    framework_fallback_used: bool
    final_diagnostics: dict[str, Any]
    model_evidence_sufficiency: str
    format_repair_count: int
    no_model_final: bool
    selected_option: str | None
    is_correct: bool | None
    evidence_chain_ids: list[list[str]]
    frame_set_ids: list[str]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    planner_turns: list[dict[str, Any]]
    planner_plans: list[dict[str, Any]]
    route_repairs: list[dict[str, Any]]
    context_budget_reports: list[dict[str, Any]]
    followup_history: list[dict[str, Any]]
    contract_version: str = CONTRACT_VERSION
    schema_version: str = "TrainingTrajectoryV1"
    workspace_root: str = ""
    trajectory_path: str = ""

    @classmethod
    def from_workspace(
        cls,
        workspace: EvidenceWorkspace,
        *,
        case_id: str,
        question: str,
        options: Sequence[str] = (),
        ground_truth: str | None = None,
        final_decision: str = "",
        selected_option: str | None = None,
        is_correct: bool | None = None,
        output_path: str | Path | None = None,
    ) -> "TrainingTrajectory":
        trace_events = _read_jsonl(workspace.root / "trace.jsonl")
        chains = workspace.evidence_chain_summaries(max_chains=100)
        planner_turns = _planner_turns(trace_events, workspace=workspace)
        final_decision_owner = _final_decision_owner(trace_events)
        trajectory = cls(
            run_id=workspace.run_id,
            case_id=str(case_id),
            question=str(question),
            options=[str(item) for item in options],
            ground_truth=str(ground_truth) if ground_truth is not None else None,
            final_decision=str(final_decision),
            final_decision_owner=final_decision_owner,
            diagnostic_errors=_final_control_diagnostic_errors(trace_events),
            framework_fallback_used=_framework_fallback_used(trace_events),
            final_diagnostics=_final_diagnostics(trace_events),
            model_evidence_sufficiency=_model_evidence_sufficiency(trace_events),
            format_repair_count=_format_repair_count(trace_events),
            no_model_final=_no_model_final(trace_events, final_decision=str(final_decision)),
            selected_option=str(selected_option) if selected_option else None,
            is_correct=is_correct,
            evidence_chain_ids=[
                [str(record.get("evidence_id", "")) for record in chain.get("records", [])]
                for chain in chains
            ],
            frame_set_ids=_frame_set_ids(workspace, chains),
            tool_calls=_tool_calls(trace_events),
            tool_results=_tool_results(trace_events, workspace=workspace, planner_turns=planner_turns),
            planner_turns=planner_turns,
            planner_plans=_planner_plans(trace_events),
            route_repairs=_route_repairs(trace_events),
            context_budget_reports=_event_payloads(trace_events, "context_budget_report"),
            followup_history=_followup_events(trace_events),
            workspace_root=workspace.root.as_posix(),
        )
        if output_path is not None:
            path = Path(output_path)
            if not path.is_absolute():
                path = workspace.root / path
            path.parent.mkdir(parents=True, exist_ok=True)
            trajectory.trajectory_path = path.as_posix()
            path.write_text(json.dumps(trajectory.to_dict(), ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        return trajectory

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_REASONING_LEAK_PATTERNS = (
    re.compile(r"<think\b.*?</think>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bthe user (?:is asking|asked|wants|needs|provided)\b", re.IGNORECASE),
    re.compile(r"\b(?:i need to|i should|i will|let me|let's|we need to)\b", re.IGNORECASE),
    re.compile(r"\bcurrent state\b", re.IGNORECASE),
)


def _planner_response_excerpt(text: str) -> str:
    payload = _json_object_from_text(text)
    if payload is None:
        return "(unparsed planner response; raw artifact retained for debug only)"
    summary = _planner_action_summary(payload)
    if not summary:
        return "(planner response contained no public action fields)"
    return json.dumps(summary, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _planner_action_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("status", "skill"):
        if key in payload:
            summary[key] = payload.get(key)
    if "tool" in payload:
        summary["tool"] = payload.get("tool")
        if isinstance(payload.get("args"), Mapping):
            summary["args"] = dict(payload["args"])
        elif isinstance(payload.get("arguments"), Mapping):
            summary["args"] = dict(payload["arguments"])
    if "program" in payload:
        summary["program"] = _bounded_sequence(payload.get("program", []), max_items=8)
    if "Actions" in payload:
        summary["Actions"] = _bounded_sequence(payload.get("Actions", []), max_items=8)
    if "Finish" in payload and isinstance(payload.get("Finish"), Mapping):
        finish = dict(payload["Finish"])
        summary["Finish"] = {
            key: finish.get(key)
            for key in ("chain_complete", "completion_basis", "answer")
            if key in finish
        }
    for key in ("answer", "citations", "selected_option"):
        if key in payload:
            summary[key] = payload.get(key)
    return summary


def _json_object_from_text(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    candidates = [stripped]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(fenced)
    for candidate in candidates:
        parsed = _parse_json_object(candidate)
        if parsed is not None:
            return parsed
    for candidate in _balanced_json_candidates(stripped):
        parsed = _parse_json_object(candidate)
        if parsed is not None:
            return parsed
    return None


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _final_decision_owner(trace_events: Sequence[Mapping[str, Any]]) -> str:
    for event in reversed(trace_events):
        if _event_type(event) != "iterative_final":
            continue
        owner = str(_event_payload(event).get("final_decision_owner", "")).strip()
        if owner:
            return owner
    return ""


def _final_control_diagnostic_errors(trace_events: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    for event in trace_events:
        event_type = _event_type(event)
        payload = _event_payload(event)
        owner = str(payload.get("final_decision_owner", "")).strip()
        if event_type == "mcq_forced_fallback":
            errors.append("active trace emitted disabled mcq_forced_fallback")
        if event_type == "framework_selected_option":
            errors.append("active trace emitted disabled framework_selected_option")
        if owner == FinalDecisionOwner.FRAMEWORK.value:
            errors.append("active trace emitted framework-owned final decision")
    return sorted(set(errors))


def _framework_fallback_used(trace_events: Sequence[Mapping[str, Any]]) -> bool:
    for event in trace_events:
        event_type = _event_type(event)
        payload = _event_payload(event)
        owner = str(payload.get("final_decision_owner", "")).strip()
        if event_type in {"mcq_forced_fallback", "framework_selected_option"}:
            return True
        if owner == FinalDecisionOwner.FRAMEWORK.value:
            return True
    return False


def _final_diagnostics(trace_events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for event in reversed(trace_events):
        payload = _event_payload(event)
        diagnostics = payload.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            return dict(diagnostics)
        if _event_type(event) == "structured_final_diagnostics":
            return dict(payload)
    return {}


def _model_evidence_sufficiency(trace_events: Sequence[Mapping[str, Any]]) -> str:
    for event in reversed(trace_events):
        if _event_type(event) != "iterative_final":
            continue
        value = str(_event_payload(event).get("evidence_sufficiency", "")).strip()
        if value:
            return value
    return ""


def _format_repair_count(trace_events: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        for event in trace_events
        if _event_type(event) == "model_final_format_repair_result"
        and str(_event_payload(event).get("status", "")).strip().lower() == "final"
    )


def _no_model_final(trace_events: Sequence[Mapping[str, Any]], *, final_decision: str) -> bool:
    if str(final_decision or "").strip() == "no_model_final":
        return True
    return any(_event_type(event) == "no_model_final" for event in trace_events)


def _event_type(event: Mapping[str, Any]) -> str:
    return str(event.get("type", ""))


def _event_payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = event.get("payload", {})
    return payload if isinstance(payload, Mapping) else {}


def _balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue
        if char != "}" or depth == 0:
            continue
        depth -= 1
        if depth == 0 and start is not None:
            candidates.append(text[start : index + 1])
            start = None
    return candidates


def _sanitize_public_text(text: str, *, max_chars: int) -> tuple[str, bool]:
    original = str(text or "")
    cleaned = _REASONING_LEAK_PATTERNS[0].sub("", original)
    redacted = cleaned != original
    payload = _json_object_from_text(cleaned)
    if payload is not None:
        for key in ("claim", "summary", "description", "answer", "value"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return _limit_text(value, max_chars=max_chars), True
    lines = _public_text_units(cleaned)
    kept: list[str] = []
    for line in lines:
        if _looks_like_reasoning_leak(line):
            redacted = True
            continue
        kept.append(line)
    if not kept and lines:
        kept = lines[-1:]
        redacted = True
    sanitized = " ".join(kept)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if _looks_like_reasoning_leak(sanitized):
        return "", True
    return _limit_text(sanitized, max_chars=max_chars), redacted or sanitized != original.strip()


def _looks_like_reasoning_leak(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in _REASONING_LEAK_PATTERNS[1:])


def _public_text_units(text: str) -> list[str]:
    units: list[str] = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        pieces = re.split(r"(?<=[.!?])\s+", line)
        units.extend(piece.strip() for piece in pieces if piece.strip())
    return units


def _limit_text(text: str, *, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _planner_turns(events: Sequence[Mapping[str, Any]], *, workspace: EvidenceWorkspace) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("type", ""))
        if event_type not in {"planner_io", "workspace_plan_model_io", "workspace_final_model_io"}:
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, Mapping):
            continue
        prompt_artifact = _artifact_summary(workspace.root, _io_artifact_payload(payload, "prompt"))
        response_artifact = _artifact_summary(workspace.root, _io_artifact_payload(payload, "response"))
        prompt_text = _artifact_text(workspace.root, prompt_artifact)
        response_text = _artifact_text(workspace.root, response_artifact) or str(payload.get("response_excerpt", ""))
        evidence_section = _evidence_section(prompt_text)
        turns.append(
            {
                "round": int(payload.get("round", len(turns) + 1) or len(turns) + 1),
                "planner_input_mode": str(payload.get("planner_input_mode", "")),
                "phase": "final" if event_type == "workspace_final_model_io" else "plan",
                "prompt_artifact": prompt_artifact,
                "response_artifact": response_artifact,
                "response_excerpt": _planner_response_excerpt(response_text),
                "evidence_observation_ids": _evidence_observation_ids_from_section(evidence_section),
                "empty_evidence_claim_count": _empty_claim_line_count(evidence_section),
                "evidence_snapshot_chars": len(evidence_section),
                "created_at": str(event.get("created_at", "")),
            }
        )
    return turns


def _tool_calls(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    current_source_round = 0
    for event in events:
        event_type = str(event.get("type", ""))
        payload = event.get("payload", {})
        if event_type in {"iterative_plan", "workspace_plan_model_io", "workspace_final_model_io"} and isinstance(payload, Mapping):
            current_source_round = int(payload.get("round", current_source_round) or current_source_round)
            continue
        if event_type != "tool_use":
            continue
        if not isinstance(payload, Mapping):
            continue
        arguments = payload.get("arguments", {})
        calls.append(
            {
                "step": int(payload.get("step", len(calls) + 1) or len(calls) + 1),
                "tool": str(payload.get("tool", "")),
                "arguments": dict(arguments) if isinstance(arguments, Mapping) else {},
                "source_round": current_source_round,
                "created_at": str(event.get("created_at", "")),
            }
        )
    return calls


def _tool_results(
    events: Sequence[Mapping[str, Any]],
    *,
    workspace: EvidenceWorkspace,
    planner_turns: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    observations = _observations_by_id(workspace)
    evidence_ids = _evidence_ids_by_observation(workspace)
    visible_rounds = _visible_rounds_by_observation(planner_turns)
    results: list[dict[str, Any]] = []
    current_source_round = 0
    for event in events:
        event_type = str(event.get("type", ""))
        payload = event.get("payload", {})
        if event_type in {"iterative_plan", "workspace_plan_model_io", "workspace_final_model_io"} and isinstance(payload, Mapping):
            current_source_round = int(payload.get("round", current_source_round) or current_source_round)
            continue
        if event_type != "tool_result":
            continue
        if not isinstance(payload, Mapping):
            continue
        observation_id = str(payload.get("observation_id", ""))
        observation = observations.get(observation_id, {})
        raw_output = observation.get("raw_output", {})
        raw_mapping = raw_output if isinstance(raw_output, Mapping) else {}
        claim, claim_redacted = _sanitize_public_text(str(observation.get("claim", "")), max_chars=4000)
        limitations, limitations_redacted = _sanitize_public_text(
            str(observation.get("limitations", "")), max_chars=4000
        )
        result = {
            "step": int(payload.get("step", len(results) + 1) or len(results) + 1),
            "tool": str(payload.get("tool", "")),
            "observation_id": observation_id,
            "claim": claim,
            "confidence": float(observation.get("confidence", 0.0) or 0.0),
            "confidence_signal": str(
                observation.get("confidence_signal", "") or raw_mapping.get("confidence_signal", "")
            ),
            "grounding_quality": str(raw_mapping.get("grounding_quality", "")),
            "limitations": limitations,
            "regions": _bounded_sequence(observation.get("regions", []), max_items=8),
            "frame_set_id": str(observation.get("frame_set_id", "")),
            "input_artifacts": _bounded_sequence(observation.get("input_artifacts", []), max_items=8),
            "evidence_record_ids": evidence_ids.get(observation_id, []),
            "visible_in_planner_rounds": visible_rounds.get(observation_id, []),
            "source_round": current_source_round,
            "created_at": str(event.get("created_at", "")),
        }
        for key in (
            "mode",
            "evidence_mode",
            "time_range",
            "facts",
            "produced_anchors",
            "candidate_anchor_ids",
        ):
            if key in raw_mapping:
                result[key] = _public_bounded_value(raw_mapping.get(key), max_items=12, max_chars=4000)
        relations = raw_mapping.get("candidate_option_relations")
        if isinstance(relations, Sequence) and not isinstance(relations, (str, bytes)):
            result["candidate_option_relations"] = _bounded_sequence(relations, max_items=8)
        if claim_redacted:
            result["claim_redacted"] = True
        if limitations_redacted:
            result["limitations_redacted"] = True
        results.append(result)
    return results


def _planner_plans(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("type", ""))
        if event_type not in {"iterative_plan", "workspace_plan_model_io"}:
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, Mapping):
            continue
        program = payload.get("program", [])
        rationale_text = str(payload.get("rationale", ""))
        if event_type == "workspace_plan_model_io":
            action = _json_object_from_text(_model_io_response_text(payload))
            if action is None:
                program = []
            else:
                program = _program_from_workspace_action(action)
                rationale_text = str(action.get("rationale", ""))
        rationale, rationale_redacted = _sanitize_public_text(rationale_text, max_chars=400)
        plan = {
            "round": int(payload.get("round", len(plans) + 1) or len(plans) + 1),
            "rationale": rationale,
            "program": _bounded_sequence(program, max_items=8),
            "created_at": str(event.get("created_at", "")),
        }
        if rationale_redacted:
            plan["rationale_redacted"] = True
        plans.append(
            plan
        )
    return plans


def _route_repairs(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    current_planner_round = 0
    for event in events:
        event_type = str(event.get("type", ""))
        payload = event.get("payload", {})
        if event_type in {"planner_io", "workspace_plan_model_io", "iterative_round_start"} and isinstance(payload, Mapping):
            current_planner_round = int(payload.get("round", current_planner_round) or current_planner_round)
            continue
        if event_type != "route_tool_repaired" or not isinstance(payload, Mapping):
            continue
        repairs.append(
            {
                "round": current_planner_round,
                "skill": str(payload.get("skill", "")),
                "requested_tool": str(payload.get("requested_tool", "")),
                "resolved_tool": str(payload.get("resolved_tool", "")),
                "reason": str(payload.get("reason", "")),
                "created_at": str(event.get("created_at", "")),
            }
        )
    return repairs


def _event_payloads(events: Sequence[Mapping[str, Any]], event_type: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for event in events:
        if str(event.get("type", "")) != event_type:
            continue
        payload = event.get("payload", {})
        if isinstance(payload, Mapping):
            payloads.append(dict(payload))
    return payloads


def _followup_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("type", ""))
        if not event_type.startswith("followup") and not event_type.startswith("hard_skill_followup"):
            continue
        payload = event.get("payload", {})
        if isinstance(payload, Mapping):
            history.append({"type": event_type, **dict(payload)})
    return history


def _frame_set_ids(workspace: EvidenceWorkspace, chains: Sequence[Mapping[str, Any]]) -> list[str]:
    ids = {
        str(record.get("frame_set_id", ""))
        for chain in chains
        for record in chain.get("records", [])
        if isinstance(record, Mapping) and record.get("frame_set_id")
    }
    ids.update(str(manifest.frame_set_id) for manifest in workspace.load_all_manifests())
    return sorted(item for item in ids if item)


def _artifact_summary(root: Path, artifact: Any) -> dict[str, Any]:
    payload = artifact if isinstance(artifact, Mapping) else {}
    summary: dict[str, Any] = {
        "path": str(payload.get("path", "")),
        "chars": int(payload.get("chars", 0) or 0),
        "stored_chars": int(payload.get("stored_chars", 0) or 0),
        "truncated": bool(payload.get("truncated", False)),
    }
    path = _artifact_path(root, summary)
    if path is not None and path.exists():
        data = path.read_bytes()
        summary["sha256"] = hashlib.sha256(data).hexdigest()
        summary["bytes"] = len(data)
    else:
        summary["sha256"] = ""
        summary["bytes"] = 0
    return summary


def _io_artifact_payload(payload: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    legacy = payload.get(kind, {})
    if isinstance(legacy, Mapping) and legacy.get("path"):
        return legacy
    return {
        "path": str(payload.get(f"{kind}_path", "")),
        "chars": int(payload.get(f"{kind}_chars", 0) or 0),
    }


def _model_io_response_text(payload: Mapping[str, Any]) -> str:
    direct = str(payload.get("response", "") or "")
    if direct:
        return direct
    artifact = _io_artifact_payload(payload, "response")
    path = Path(str(artifact.get("path", "")))
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ""
    return ""


def _program_from_workspace_action(action: Mapping[str, Any]) -> list[dict[str, Any]]:
    tool = str(action.get("tool", "") or action.get("name", "")).strip()
    if not tool:
        return []
    args = action.get("args", action.get("arguments", {}))
    return [
        {
            "tool": tool,
            "args": dict(args) if isinstance(args, Mapping) else {},
        }
    ]


def _artifact_text(root: Path, artifact: Mapping[str, Any]) -> str:
    path = _artifact_path(root, artifact)
    if path is None or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""


def _artifact_path(root: Path, artifact: Mapping[str, Any]) -> Path | None:
    artifact_path = str(artifact.get("path", "")).strip()
    if not artifact_path:
        return None
    path = Path(artifact_path)
    return path if path.is_absolute() else root / path


def _evidence_observation_ids_from_prompt(prompt_text: str) -> list[str]:
    return _evidence_observation_ids_from_section(_evidence_section(prompt_text))


def _evidence_observation_ids_from_section(evidence_section: str) -> list[str]:
    return sorted(set(re.findall(r"obs_\d{4}", evidence_section)))


def _evidence_section(prompt_text: str) -> str:
    if "## Evidence" not in prompt_text:
        return ""
    section = prompt_text.split("## Evidence", 1)[1]
    for marker in ["\n## Feedback", "\n## Response Contract", "\n## Task"]:
        if marker in section:
            return section.split(marker, 1)[0]
    return section


def _empty_claim_line_count(evidence_section: str) -> int:
    count = 0
    for line in evidence_section.splitlines():
        if "claim:" not in line:
            continue
        if re.search(r"claim:\s*(?:\|\s*limitations:|$)", line):
            count += 1
    return count


def _observations_by_id(workspace: EvidenceWorkspace) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("observation_id", "")): row
        for row in _read_jsonl(workspace.root / "observations.jsonl")
        if row.get("observation_id")
    }


def _evidence_ids_by_observation(workspace: EvidenceWorkspace) -> dict[str, list[str]]:
    ids: dict[str, list[str]] = {}
    for row in _read_jsonl(workspace.root / "evidence.jsonl"):
        observation_id = str(row.get("observation_id", ""))
        evidence_id = str(row.get("evidence_id", ""))
        if observation_id and evidence_id:
            ids.setdefault(observation_id, []).append(evidence_id)
    return ids


def _visible_rounds_by_observation(planner_turns: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    visible: dict[str, list[int]] = {}
    for turn in planner_turns:
        try:
            round_number = int(turn.get("round", 0) or 0)
        except (TypeError, ValueError):
            continue
        observation_ids = turn.get("evidence_observation_ids", [])
        if not isinstance(observation_ids, Sequence) or isinstance(observation_ids, (str, bytes)):
            continue
        for observation_id in observation_ids:
            visible.setdefault(str(observation_id), []).append(round_number)
    return visible


def _bounded_sequence(value: Any, *, max_items: int) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) if isinstance(item, Mapping) else item for item in value[:max_items]]


def _public_bounded_value(value: Any, *, max_items: int, max_chars: int) -> Any:
    if isinstance(value, Mapping):
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
        if len(encoded) <= max_chars:
            return dict(value)
        return _limit_text(encoded, max_chars=max_chars)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = []
        for item in list(value)[:max_items]:
            if isinstance(item, Mapping):
                encoded = json.dumps(item, ensure_ascii=True, sort_keys=True, default=str)
                items.append(dict(item) if len(encoded) <= max_chars else _limit_text(encoded, max_chars=max_chars))
            elif isinstance(item, str):
                items.append(_limit_text(item, max_chars=max_chars))
            else:
                items.append(item)
        return items
    if isinstance(value, str):
        return _limit_text(value, max_chars=max_chars)
    return value
