from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from visual_coding_agent_harness.agents.contracts import CONTRACT_VERSION
from visual_coding_agent_harness.workspace import EvidenceWorkspace


@dataclass
class TrainingTrajectory:
    run_id: str
    case_id: str
    question: str
    options: list[str]
    ground_truth: str | None
    final_decision: str
    selected_option: str | None
    is_correct: bool | None
    evidence_chain_ids: list[list[str]]
    frame_set_ids: list[str]
    tool_calls: list[dict[str, Any]]
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
        trajectory = cls(
            run_id=workspace.run_id,
            case_id=str(case_id),
            question=str(question),
            options=[str(item) for item in options],
            ground_truth=str(ground_truth) if ground_truth is not None else None,
            final_decision=str(final_decision),
            selected_option=str(selected_option) if selected_option else None,
            is_correct=is_correct,
            evidence_chain_ids=[
                [str(record.get("evidence_id", "")) for record in chain.get("records", [])]
                for chain in chains
            ],
            frame_set_ids=_frame_set_ids(workspace, chains),
            tool_calls=_tool_calls(trace_events),
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


def _tool_calls(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for event in events:
        if str(event.get("type", "")) != "tool_use":
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, Mapping):
            continue
        arguments = payload.get("arguments", {})
        calls.append(
            {
                "step": int(payload.get("step", len(calls) + 1) or len(calls) + 1),
                "tool": str(payload.get("tool", "")),
                "arguments": dict(arguments) if isinstance(arguments, Mapping) else {},
                "created_at": str(event.get("created_at", "")),
            }
        )
    return calls


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
