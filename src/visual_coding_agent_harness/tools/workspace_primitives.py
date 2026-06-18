"""Cheap workspace read/write primitives for planner self-inspection."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

from ..registry import ToolRegistry, tool
from ..workspace import EvidenceWorkspace


PLAN_PHASE_PRIMITIVES = frozenset({"read_workspace"})
COMMIT_PHASE_PRIMITIVES = frozenset({"commit_observation", "reject_observation", "defer_observation", "no_commit_needed"})
INTERNAL_ONLY_PRIMITIVES = frozenset(
    {
        "view_observation",
        "read_observation_detail",
        "grep_evidence",
        "query_evidence_table",
        "read_timeline_sorted",
        "read_hypothesis",
        "update_hypothesis_slot",
        "write_memory",
    }
)


def build_workspace_primitives_registry(
    *,
    workspace: Optional[EvidenceWorkspace] = None,
    include: Sequence[str] = ("plan", "commit"),
) -> ToolRegistry:
    registry = ToolRegistry()

    def observation_detail_payload(obs_id: str, line_range: tuple[int, int] | None = None) -> Mapping[str, object]:
        observation = workspace.get_observation(obs_id) if workspace is not None else None
        if observation is None:
            return {
                "claim": f"Observation {obs_id} was not found.",
                "confidence": 0.0,
                "regions": [],
                "limitations": "Cheap workspace read; no video frames inspected.",
            }
        claim = _slice_lines(observation.claim, line_range)
        return {
            "claim": claim,
            "confidence": 1.0,
            "regions": [
                {
                    "obs_id": observation.observation_id,
                    "tool": observation.tool,
                    "claim": claim,
                    "confidence_signal": observation.confidence_signal,
                }
            ],
            "limitations": "Cheap workspace read; no video frames inspected.",
        }

    @tool(name="view_observation", description="Fetch one observation claim by id from the workspace.")
    def view_observation(obs_id: str, line_range: tuple[int, int] | None = None) -> Mapping[str, object]:
        return observation_detail_payload(obs_id=obs_id, line_range=line_range)

    @tool(name="read_observation_detail", description="Fetch one compact observation detail by id from the workspace.")
    def read_observation_detail(obs_id: str, line_range: tuple[int, int] | None = None) -> Mapping[str, object]:
        return observation_detail_payload(obs_id=obs_id, line_range=line_range)


    @tool(name="grep_evidence", description="Regex search over persisted observations/evidence rows.")
    def grep_evidence(pattern: str, in_field: str = "claim") -> Mapping[str, object]:
        rows = _evidence_search_rows(workspace)
        regex = re.compile(pattern, flags=re.IGNORECASE)
        obs_ids = []
        for row in rows:
            if regex.search(str(row.get(in_field, ""))):
                obs_id = str(row.get("obs_id") or row.get("observation_id") or "")
                if obs_id and obs_id not in obs_ids:
                    obs_ids.append(obs_id)
        return {
            "claim": f"grep_evidence matched {len(obs_ids)} observation(s).",
            "confidence": 1.0,
            "regions": [{"pattern": pattern, "in_field": in_field, "obs_ids": obs_ids}],
            "limitations": "Cheap regex over workspace text; no semantic matching.",
        }

    @tool(name="query_evidence_table", description="Structured exact-match query over evidence_table.jsonl.")
    def query_evidence_table(filter: Mapping[str, Any]) -> Mapping[str, object]:
        rows = workspace.read_evidence_table_v3(question="", options=[]).get("rows", []) if workspace is not None else []
        matched = [
            row
            for row in rows
            if isinstance(row, Mapping) and _row_matches_filter(row, filter)
        ]
        return {
            "claim": f"query_evidence_table matched {len(matched)} row(s).",
            "confidence": 1.0,
            "regions": [{"filter": dict(filter), "rows": matched}],
            "limitations": "Cheap structured workspace query; no video frames inspected.",
        }

    @tool(name="read_workspace", description="Read a bounded section from the durable workspace.")
    def read_workspace(section: str, filter: Mapping[str, Any] | None = None) -> Mapping[str, object]:
        rows = workspace.read_workspace_section(section, filter=filter or {}) if workspace is not None else []
        return {
            "claim": f"Read {len(rows)} row(s) from workspace section {section}.",
            "confidence": 1.0,
            "regions": [{"section": section, "rows": rows, "observations": rows if section == "observations_by_id" else []}],
            "limitations": "Cheap workspace read; no video frames inspected.",
        }

    @tool(name="read_timeline_sorted", description="Read the workspace timeline sorted by observed time.")
    def read_timeline_sorted() -> Mapping[str, object]:
        entries = workspace.read_timeline_sorted() if workspace is not None else []
        return {
            "claim": f"Read {len(entries)} timeline entr{'y' if len(entries) == 1 else 'ies'}.",
            "confidence": 1.0 if entries else 0.0,
            "regions": [{"entries": entries}],
            "limitations": "Cheap workspace read; timeline rows are only as reliable as their confidence_signal.",
        }

    @tool(name="read_hypothesis", description="Read the workspace hypothesis slots.")
    def read_hypothesis() -> Mapping[str, object]:
        slots = workspace.read_hypothesis() if workspace is not None else {}
        return {
            "claim": f"Read {len(slots)} hypothesis slot(s).",
            "confidence": 1.0,
            "regions": [{"slots": slots}],
            "limitations": "Cheap workspace read; no video frames inspected.",
        }

    @tool(name="update_hypothesis_slot", description="Update one workspace hypothesis slot status.")
    def update_hypothesis_slot(slot_name: str, status: str, evidence_obs_id: str = "") -> Mapping[str, object]:
        if workspace is None:
            return {
                "claim": "No workspace is attached; hypothesis was not updated.",
                "confidence": 0.0,
                "regions": [],
                "limitations": "Workspace-backed tool requires an EvidenceWorkspace.",
            }
        slot = workspace.update_hypothesis_slot(
            slot_name=slot_name,
            status=status,
            evidence_obs_id=evidence_obs_id,
        )
        return {
            "claim": f"Hypothesis slot {slot_name} updated to {slot['status']}.",
            "confidence": 1.0,
            "regions": [{"slot_name": slot_name, "slot": slot}],
            "limitations": "Cheap workspace write; no video frames inspected.",
        }

    @tool(name="write_memory", description="Write a concise planner memory entry backed by produced anchor ids.")
    def write_memory(
        kind: str,
        claim: str,
        anchors: Sequence[Mapping[str, Any]],
        supports_option: str = "",
        confidence: str = "medium",
        previous_memory_refs: Sequence[str] = (),
        tags: Sequence[str] = (),
        role: str = "",
        layer: str = "",
        embedding_refs: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, object]:
        if workspace is None:
            return {
                "claim": "No workspace is attached; memory was not written.",
                "confidence": 0.0,
                "regions": [],
                "limitations": "Workspace-backed tool requires an EvidenceWorkspace.",
            }
        entry = workspace.write_memory(
            kind=kind,
            claim=claim,
            anchors=anchors,
            supports_option=supports_option,
            confidence=confidence,
            previous_memory_refs=previous_memory_refs,
            tags=tags,
            role=role,
            layer=layer,
            embedding_refs=embedding_refs,
            metadata=metadata,
        )
        return {
            "claim": f"Memory {entry.entry_id} written.",
            "confidence": 1.0,
            "entry_id": entry.entry_id,
            "memory": entry.to_dict(),
            "regions": [
                {
                    "entry_id": entry.entry_id,
                    "kind": entry.kind,
                    "supports_option": entry.supports_option,
                    "anchor_ids": [anchor.anchor_id for anchor in entry.anchors],
                }
            ],
            "limitations": "Memory records planner notes; provenance is checked but semantic support is not judged.",
        }

    @tool(name="commit_observation", description="Commit one pending observation into pinned workspace memory.")
    def commit_observation(observation_id: str, writes: Mapping[str, Any]) -> Mapping[str, object]:
        if workspace is None:
            return _missing_workspace_payload("commit_observation")
        disposition = workspace.commit_observation(observation_id, writes=writes)
        return {
            "claim": f"Observation {observation_id} committed.",
            "confidence": 1.0,
            "regions": [disposition],
            "limitations": "Workspace transaction; provenance is validated, semantic entailment is not judged.",
        }

    @tool(name="reject_observation", description="Reject one pending observation so it stays out of workspace view.")
    def reject_observation(observation_id: str, reason: str) -> Mapping[str, object]:
        if workspace is None:
            return _missing_workspace_payload("reject_observation")
        disposition = workspace.reject_observation(observation_id, reason=reason)
        return {
            "claim": f"Observation {observation_id} rejected.",
            "confidence": 1.0,
            "regions": [disposition],
            "limitations": "Workspace disposition; no video frames inspected.",
        }

    @tool(name="defer_observation", description="Defer one pending observation until more context is available.")
    def defer_observation(observation_id: str, until: str = "", reason: str = "") -> Mapping[str, object]:
        if workspace is None:
            return _missing_workspace_payload("defer_observation")
        disposition = workspace.defer_observation(observation_id, until=until, reason=reason)
        return {
            "claim": f"Observation {observation_id} deferred.",
            "confidence": 1.0,
            "regions": [disposition],
            "limitations": "Workspace disposition; defer is capped by the workspace state machine.",
        }

    @tool(name="no_commit_needed", description="Acknowledge one pending observation that adds no durable evidence.")
    def no_commit_needed(observation_id: str, reason: str) -> Mapping[str, object]:
        if workspace is None:
            return _missing_workspace_payload("no_commit_needed")
        disposition = workspace.no_commit_needed(observation_id, reason=reason)
        return {
            "claim": f"Observation {observation_id} acknowledged.",
            "confidence": 1.0,
            "regions": [disposition],
            "limitations": "Workspace disposition; no video frames inspected.",
        }

    specs = {
        "view_observation": view_observation,
        "read_observation_detail": read_observation_detail,
        "grep_evidence": grep_evidence,
        "query_evidence_table": query_evidence_table,
        "read_workspace": read_workspace,
        "read_timeline_sorted": read_timeline_sorted,
        "read_hypothesis": read_hypothesis,
        "update_hypothesis_slot": update_hypothesis_slot,
        "write_memory": write_memory,
        "commit_observation": commit_observation,
        "reject_observation": reject_observation,
        "defer_observation": defer_observation,
        "no_commit_needed": no_commit_needed,
    }
    for name in _primitive_names_for(include):
        registry.register(specs[name])
    return registry


def _primitive_names_for(include: Sequence[str]) -> tuple[str, ...]:
    buckets = {
        "plan": PLAN_PHASE_PRIMITIVES,
        "commit": COMMIT_PHASE_PRIMITIVES,
        "internal": INTERNAL_ONLY_PRIMITIVES,
        "all": PLAN_PHASE_PRIMITIVES | COMMIT_PHASE_PRIMITIVES | INTERNAL_ONLY_PRIMITIVES,
    }
    selected: set[str] = set()
    for bucket_name in include:
        key = str(bucket_name).strip()
        if key not in buckets:
            raise ValueError(f"unknown_workspace_primitive_bucket: {key}")
        selected.update(buckets[key])
    ordered = (
        "view_observation",
        "read_observation_detail",
        "grep_evidence",
        "query_evidence_table",
        "read_workspace",
        "read_timeline_sorted",
        "read_hypothesis",
        "update_hypothesis_slot",
        "write_memory",
        "commit_observation",
        "reject_observation",
        "defer_observation",
        "no_commit_needed",
    )
    return tuple(name for name in ordered if name in selected)


def _slice_lines(text: str, line_range: tuple[int, int] | None) -> str:
    if line_range is None:
        return text
    start, end = line_range
    lines = text.splitlines()
    return "\n".join(lines[max(0, int(start) - 1) : max(0, int(end))])


def _evidence_search_rows(workspace: Optional[EvidenceWorkspace]) -> list[Mapping[str, Any]]:
    if workspace is None:
        return []
    rows: list[Mapping[str, Any]] = []
    table_rows = workspace.read_evidence_table_v3(question="", options=[]).get("rows", [])
    if isinstance(table_rows, list):
        rows.extend(row for row in table_rows if isinstance(row, Mapping))
    rows.extend(workspace._read_observation_dicts())
    return rows


def _row_matches_filter(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key, value in expected.items():
        if row.get(str(key)) != value:
            return False
    return True


def _missing_workspace_payload(tool_name: str) -> Mapping[str, object]:
    return {
        "claim": f"No workspace is attached; {tool_name} was not applied.",
        "confidence": 0.0,
        "regions": [],
        "limitations": "Workspace-backed tool requires an EvidenceWorkspace.",
    }
