"""Readable workspace round-log export for demo analysis."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from visual_coding_agent_harness.workspace import EvidenceWorkspace


DEFAULT_WORKSPACE_ROUND_LOG_PATH = "artifacts/workspace_round_log/workspace_round_log.md"


def export_workspace_round_log(
    workspace: EvidenceWorkspace,
    *,
    question: str,
    video_path: str,
    final: Mapping[str, Any] | None = None,
    trajectory_path: str | Path | None = None,
    evidence_chains_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_WORKSPACE_ROUND_LOG_PATH,
) -> Mapping[str, Any]:
    """Export a human-readable per-round workspace trace for demos."""

    root = workspace.root
    trace_events = _read_jsonl(root / "trace.jsonl")
    observations_by_id = {
        str(item.get("observation_id", "")): item
        for item in _read_jsonl(root / "observations.jsonl")
        if item.get("observation_id")
    }
    memory_rows = _read_jsonl(root / "memory.jsonl")
    dispositions = _read_jsonl(root / "observations" / "disposition.jsonl")
    planner_io = _planner_io_files(root)
    rounds = _events_by_round(trace_events)

    lines: list[str] = [
        "# Workspace Round Log",
        "",
        "## Summary",
        f"- workspace_root: {root.as_posix()}",
        f"- question: {_compact_text(question, limit=500)}",
        f"- video_path: {video_path}",
        f"- final_status: {_text((final or {}).get('status'))}",
        f"- final_answer: {_compact_text(_text((final or {}).get('answer')), limit=300)}",
        f"- final_citations: {', '.join(_string_items((final or {}).get('citations'))) or '(none)'}",
        f"- trajectory_json: {_relative_or_text(root, trajectory_path)}",
        f"- evidence_chains_json: {_relative_or_text(root, evidence_chains_path)}",
        f"- planner_io_dir: artifacts/planner_io",
        "",
    ]
    lines.extend(_render_first_workspace_view(root, planner_io))
    lines.extend(_render_rounds(root, rounds=rounds, observations_by_id=observations_by_id, planner_io=planner_io))
    lines.extend(_render_workspace_writes(memory_rows=memory_rows, dispositions=dispositions))

    text = "\n".join(lines).rstrip() + "\n"
    meta = workspace.write_text_artifact(output_path, text)
    workspace.write_trace_event(
        "workspace_round_log_export",
        {
            "path": meta["path"],
            "round_count": len(rounds),
            "planner_prompt_count": len([path for path in planner_io if path.endswith("_prompt.txt")]),
            "chars": meta["chars"],
        },
    )
    return {
        "path": str(root / Path(str(meta["path"]))),
        "relative_path": str(meta["path"]),
        "round_count": len(rounds),
        "planner_prompt_count": len([path for path in planner_io if path.endswith("_prompt.txt")]),
    }


def _render_first_workspace_view(root: Path, planner_io: Mapping[str, Path]) -> list[str]:
    prompt = planner_io.get("round_001_plan_prompt.txt")
    lines = [
        "## First Planner Workspace View",
        f"- source: {_relative_or_text(root, prompt)}",
        "",
    ]
    if prompt is None or not prompt.exists():
        return [*lines, "(missing first planner prompt)", ""]
    text = prompt.read_text(encoding="utf-8", errors="replace")
    workspace_view = _extract_between(text, "# Workspace", "# Last Tool Result") or text
    lines.extend(_fenced(_compact_text(workspace_view, limit=8000), info="text"))
    lines.append("")
    return lines


def _render_rounds(
    root: Path,
    *,
    rounds: Mapping[int, Sequence[Mapping[str, Any]]],
    observations_by_id: Mapping[str, Mapping[str, Any]],
    planner_io: Mapping[str, Path],
) -> list[str]:
    lines: list[str] = ["## Rounds", ""]
    if not rounds:
        return [*lines, "(no round events recorded)", ""]
    for round_number in sorted(rounds):
        lines.extend([f"### Round {round_number}", ""])
        lines.extend(_render_planner_io(root, round_number=round_number, planner_io=planner_io))
        lines.extend(_render_round_events(
            root,
            events=rounds[round_number],
            observations_by_id=observations_by_id,
            planner_io=planner_io,
        ))
    return lines


def _render_planner_io(root: Path, *, round_number: int, planner_io: Mapping[str, Path]) -> list[str]:
    lines: list[str] = []
    prompt = planner_io.get(f"round_{round_number:03d}_plan_prompt.txt")
    response = planner_io.get(f"round_{round_number:03d}_plan_response.txt")
    lines.extend(
        [
            "#### Planner IO",
            f"- prompt: {_relative_or_text(root, prompt)}",
            f"- response: {_relative_or_text(root, response)}",
        ]
    )
    if response is not None and response.exists():
        lines.extend(["", "Planner response excerpt:"])
        lines.extend(_fenced(_compact_text(response.read_text(encoding="utf-8", errors="replace"), limit=1200), info="json"))
    lines.append("")
    return lines


def _render_round_events(
    root: Path,
    *,
    events: Sequence[Mapping[str, Any]],
    observations_by_id: Mapping[str, Mapping[str, Any]],
    planner_io: Mapping[str, Path],
) -> list[str]:
    lines: list[str] = ["#### Actions And Results"]
    wrote_action = False
    for event in events:
        event_type = str(event.get("type", ""))
        payload = event.get("payload", {}) if isinstance(event.get("payload", {}), Mapping) else {}
        if event_type == "tool_use":
            wrote_action = True
            tool = _text(payload.get("tool"))
            args = _compact_json(payload.get("arguments", {}), limit=500)
            lines.append(f"- tool_call: {tool} args={args}")
        elif event_type == "tool_result":
            wrote_action = True
            observation_id = _text(payload.get("observation_id"))
            observation = observations_by_id.get(observation_id, {})
            claim = _compact_text(_text(observation.get("claim")), limit=500)
            lines.append(f"- tool_result: {_text(payload.get('tool'))} -> {observation_id} claim={claim}")
        elif event_type == "workspace_answer_rejected":
            wrote_action = True
            lines.append(f"- answer_rejected: {_compact_text(_text(payload.get('error')), limit=500)}")
    if not wrote_action:
        lines.append("(none)")
    lines.append("")

    commit_events = [event for event in events if str(event.get("type", "")) == "workspace_commit_model_io"]
    if commit_events:
        lines.extend(["#### Commit Attempts", ""])
        for event in commit_events:
            payload = event.get("payload", {}) if isinstance(event.get("payload", {}), Mapping) else {}
            attempt = int(payload.get("attempt", 1) or 1)
            prompt = planner_io.get(
                f"round_{int(payload.get('round', 0) or 0):03d}_commit_attempt_{attempt:02d}_prompt.txt"
            )
            response = planner_io.get(
                f"round_{int(payload.get('round', 0) or 0):03d}_commit_attempt_{attempt:02d}_response.txt"
            )
            lines.append(
                f"- attempt {attempt}: prompt={_relative_or_text(root, prompt)} response={_relative_or_text(root, response)}"
            )
            if response is not None and response.exists():
                lines.extend(_fenced(_compact_text(response.read_text(encoding="utf-8", errors="replace"), limit=900), info="json"))
        lines.append("")

    validation_errors = [
        event
        for event in events
        if str(event.get("type", "")) == "workspace_commit_validation_error"
    ]
    if validation_errors:
        lines.extend(["#### Validation Feedback", ""])
        for event in validation_errors:
            payload = event.get("payload", {}) if isinstance(event.get("payload", {}), Mapping) else {}
            lines.append(
                f"- attempt {payload.get('attempt', '?')}: {_compact_text(_text(payload.get('error')), limit=700)}"
            )
        lines.append("")
    return lines


def _render_workspace_writes(*, memory_rows: Sequence[Mapping[str, Any]], dispositions: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["## Workspace Writes", "", "### Memory"]
    if memory_rows:
        for row in memory_rows:
            anchors = [
                _text(anchor.get("anchor_id"))
                for anchor in row.get("anchors", [])
                if isinstance(anchor, Mapping) and _text(anchor.get("anchor_id"))
            ]
            lines.append(
                f"- {_text(row.get('entry_id'))} [{_text(row.get('kind'))}] "
                f"supports={_text(row.get('supports_option')) or '-'} anchors={', '.join(anchors) or '-'} "
                f"claim={_compact_text(_text(row.get('claim')), limit=500)}"
            )
    else:
        lines.append("(none)")

    lines.extend(["", "### Dispositions"])
    if dispositions:
        for row in dispositions:
            lines.append(
                f"- {_text(row.get('observation_id'))}: {_text(row.get('disposition'))} "
                f"reason={_compact_text(_text(row.get('reason')), limit=300)}"
            )
    else:
        lines.append("(none)")
    lines.append("")
    return lines


def _events_by_round(trace_events: Sequence[Mapping[str, Any]]) -> Mapping[int, list[Mapping[str, Any]]]:
    rounds: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    current_round = 0
    for event in trace_events:
        payload = event.get("payload", {}) if isinstance(event.get("payload", {}), Mapping) else {}
        event_type = str(event.get("type", ""))
        raw_round = payload.get("round")
        if raw_round is not None:
            try:
                current_round = int(raw_round)
            except (TypeError, ValueError):
                pass
        if event_type == "workspace_round_log_export":
            continue
        if current_round > 0:
            rounds[current_round].append(event)
    return dict(rounds)


def _planner_io_files(root: Path) -> Mapping[str, Path]:
    planner_dir = root / "artifacts" / "planner_io"
    if not planner_dir.exists():
        return {}
    return {path.name: path for path in sorted(planner_dir.glob("*.txt"))}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    end = text.find(end_marker, start)
    if end < 0:
        return text[start:]
    return text[start:end].strip()


def _fenced(text: str, *, info: str = "") -> list[str]:
    fence = "```" + info
    return [fence, text.rstrip(), "```"]


def _compact_json(value: Any, *, limit: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(value)
    return _compact_text(text, limit=limit)


def _compact_text(text: str, *, limit: int) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + f"\n...[truncated {len(normalized) - limit} chars]"


def _relative_or_text(root: Path, value: str | Path | None) -> str:
    if value is None:
        return "(none)"
    path = Path(value)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _string_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _text(value: Any) -> str:
    return str(value or "").strip()
