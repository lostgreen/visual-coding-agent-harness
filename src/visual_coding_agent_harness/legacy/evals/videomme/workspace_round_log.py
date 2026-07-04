"""Readable full workspace round-log export for demo analysis."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from visual_coding_agent_harness.legacy.workspace_v2 import EvidenceWorkspace


DEFAULT_WORKSPACE_ROUND_LOG_FILENAME = "workspace_round_log.md"


def export_workspace_round_log(
    workspace: EvidenceWorkspace,
    *,
    question: str,
    video_path: str,
    final: Mapping[str, Any] | None = None,
    trajectory_path: str | Path | None = None,
    evidence_chains_path: str | Path | None = None,
    log_root: str | Path | None = None,
    output_path: str | Path = DEFAULT_WORKSPACE_ROUND_LOG_FILENAME,
) -> Mapping[str, Any]:
    """Export a full, self-contained per-round workspace trace for demos."""

    workspace_root = workspace.root
    output_root = Path(log_root) if log_root is not None else workspace_root / "workspace_logs"
    output = Path(output_path)
    output_file = output if output.is_absolute() else output_root / output
    planner_io = _planner_io_files(output_root)

    trace_events = _read_jsonl(workspace_root / "trace.jsonl")
    observations_by_id = {
        str(item.get("observation_id", "")): item
        for item in _read_jsonl(workspace_root / "observations.jsonl")
        if item.get("observation_id")
    }
    memory_rows = _read_jsonl(workspace_root / "memory.jsonl")
    dispositions = _read_jsonl(workspace_root / "observations" / "disposition.jsonl")
    rounds = _events_by_round(trace_events)

    lines: list[str] = [
        "# Workspace Round Log",
        "",
        "## Summary",
        f"- workspace_root: {workspace_root.as_posix()}",
        f"- log_root: {output_root.as_posix()}",
        f"- question: {_text(question)}",
        f"- video_path: {video_path}",
        f"- final_status: {_text((final or {}).get('status'))}",
        f"- final_answer: {_text((final or {}).get('answer'))}",
        f"- final_citations: {', '.join(_string_items((final or {}).get('citations'))) or '(none)'}",
        f"- trajectory_json: {_relative_or_text(output_root, trajectory_path)}",
        f"- evidence_chains_json: {_relative_or_text(output_root, evidence_chains_path)}",
        f"- planner_io_dir: {output_root.as_posix()}",
        "",
    ]
    lines.extend(_render_first_workspace_view(output_root, planner_io))
    lines.extend(
        _render_rounds(
            output_root,
            rounds=rounds,
            observations_by_id=observations_by_id,
            planner_io=planner_io,
        )
    )
    lines.extend(_render_workspace_writes(memory_rows=memory_rows, dispositions=dispositions))
    lines.extend(_render_raw_trace(trace_events))

    text = "\n".join(lines).rstrip() + "\n"
    meta = _write_text_file(output_file, text)
    workspace.write_trace_event(
        "workspace_round_log_export",
        {
            "path": output_file.as_posix(),
            "round_count": len(rounds),
            "planner_prompt_count": len([path for path in planner_io if path.endswith("_prompt.txt")]),
            "chars": meta["chars"],
        },
    )
    return {
        "path": output_file.as_posix(),
        "relative_path": _relative_or_text(output_root, output_file),
        "log_root": output_root.as_posix(),
        "round_count": len(rounds),
        "planner_prompt_count": len([path for path in planner_io if path.endswith("_prompt.txt")]),
    }


def _render_first_workspace_view(log_root: Path, planner_io: Mapping[str, Path]) -> list[str]:
    prompt = planner_io.get("round_001_plan_prompt.txt")
    lines = [
        "## First Planner Workspace View",
        f"- source: {_relative_or_text(log_root, prompt)}",
        "",
    ]
    if prompt is None or not prompt.exists():
        return [*lines, "(missing first planner prompt)", ""]
    text = prompt.read_text(encoding="utf-8", errors="replace")
    workspace_view = _extract_between(text, "# Workspace", "# Last Tool Result") or text
    lines.extend(_fenced(workspace_view, info="text"))
    lines.append("")
    return lines


def _render_rounds(
    log_root: Path,
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
        lines.extend(_render_planner_io(log_root, round_number=round_number, planner_io=planner_io))
        lines.extend(
            _render_round_events(
                log_root,
                events=rounds[round_number],
                observations_by_id=observations_by_id,
                planner_io=planner_io,
            )
        )
    return lines


def _render_planner_io(log_root: Path, *, round_number: int, planner_io: Mapping[str, Path]) -> list[str]:
    prompt = planner_io.get(f"round_{round_number:03d}_plan_prompt.txt")
    response = planner_io.get(f"round_{round_number:03d}_plan_response.txt")
    lines = [
        "#### Planner IO",
        f"- prompt: {_relative_or_text(log_root, prompt)}",
        f"- response: {_relative_or_text(log_root, response)}",
        "",
    ]
    if prompt is not None and prompt.exists():
        lines.extend(["Plan prompt:", *_fenced(prompt.read_text(encoding="utf-8", errors="replace"), info="text"), ""])
    if response is not None and response.exists():
        lines.extend(
            [
                "Plan response:",
                *_fenced(response.read_text(encoding="utf-8", errors="replace"), info="json"),
                "",
            ]
        )
    return lines


def _render_round_events(
    log_root: Path,
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
            lines.extend(
                [
                    f"- tool_call: {_text(payload.get('tool'))}",
                    *_fenced(_json(payload.get("arguments", {})), info="json"),
                    "",
                ]
            )
        elif event_type == "tool_result":
            wrote_action = True
            observation_id = _text(payload.get("observation_id"))
            observation = observations_by_id.get(observation_id, {})
            lines.extend(
                [
                    f"- tool_result: {_text(payload.get('tool'))} -> {observation_id}",
                    *_fenced(_json(observation), info="json"),
                    "",
                ]
            )
        elif event_type == "workspace_answer_rejected":
            wrote_action = True
            lines.extend(["- answer_rejected", *_fenced(_json(payload), info="json"), ""])
    if not wrote_action:
        lines.append("(none)")
        lines.append("")

    commit_events = [event for event in events if str(event.get("type", "")) == "workspace_commit_model_io"]
    if commit_events:
        lines.extend(["#### Commit Attempts", ""])
        for event in commit_events:
            payload = event.get("payload", {}) if isinstance(event.get("payload", {}), Mapping) else {}
            round_number = int(payload.get("round", 0) or 0)
            attempt = int(payload.get("attempt", 1) or 1)
            prompt = planner_io.get(f"round_{round_number:03d}_commit_attempt_{attempt:02d}_prompt.txt")
            response = planner_io.get(f"round_{round_number:03d}_commit_attempt_{attempt:02d}_response.txt")
            lines.extend(
                [
                    f"- attempt {attempt}",
                    f"  - prompt: {_relative_or_text(log_root, prompt)}",
                    f"  - response: {_relative_or_text(log_root, response)}",
                    "",
                ]
            )
            if prompt is not None and prompt.exists():
                lines.extend(["Commit prompt:", *_fenced(prompt.read_text(encoding="utf-8", errors="replace"), info="text"), ""])
            if response is not None and response.exists():
                lines.extend(
                    [
                        "Commit response:",
                        *_fenced(response.read_text(encoding="utf-8", errors="replace"), info="json"),
                        "",
                    ]
                )

    validation_errors = [
        event
        for event in events
        if str(event.get("type", "")) == "workspace_commit_validation_error"
    ]
    if validation_errors:
        lines.extend(["#### Validation Feedback", ""])
        for event in validation_errors:
            payload = event.get("payload", {}) if isinstance(event.get("payload", {}), Mapping) else {}
            lines.extend([f"- attempt {payload.get('attempt', '?')}", *_fenced(_json(payload), info="json"), ""])
    final_events = [event for event in events if str(event.get("type", "")) == "workspace_final_model_io"]
    if final_events:
        lines.extend(["#### Forced Final IO", ""])
        for event in final_events:
            payload = event.get("payload", {}) if isinstance(event.get("payload", {}), Mapping) else {}
            round_number = int(payload.get("round", 0) or 0)
            prompt = planner_io.get(f"round_{round_number:03d}_final_prompt.txt")
            response = planner_io.get(f"round_{round_number:03d}_final_response.txt")
            lines.extend(
                [
                    f"- prompt: {_relative_or_text(log_root, prompt)}",
                    f"- response: {_relative_or_text(log_root, response)}",
                    "",
                ]
            )
            if prompt is not None and prompt.exists():
                lines.extend(["Final prompt:", *_fenced(prompt.read_text(encoding="utf-8", errors="replace"), info="text"), ""])
            if response is not None and response.exists():
                lines.extend(
                    [
                        "Final response:",
                        *_fenced(response.read_text(encoding="utf-8", errors="replace"), info="json"),
                        "",
                    ]
                )
    return lines


def _render_workspace_writes(*, memory_rows: Sequence[Mapping[str, Any]], dispositions: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["## Workspace Writes", "", "### Memory"]
    if memory_rows:
        for row in memory_rows:
            lines.extend([f"- {_text(row.get('entry_id'))} [{_text(row.get('kind'))}]", *_fenced(_json(row), info="json"), ""])
    else:
        lines.append("(none)")

    lines.extend(["", "### Dispositions"])
    if dispositions:
        for row in dispositions:
            lines.extend([f"- {_text(row.get('observation_id'))}: {_text(row.get('disposition'))}", *_fenced(_json(row), info="json"), ""])
    else:
        lines.append("(none)")
    lines.append("")
    return lines


def _render_raw_trace(trace_events: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["## Raw Trace Events", ""]
    if not trace_events:
        return [*lines, "(none)", ""]
    for index, event in enumerate(trace_events, start=1):
        lines.extend([f"### Trace Event {index}: {_text(event.get('type'))}", *_fenced(_json(event), info="json"), ""])
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


def _planner_io_files(log_root: Path) -> Mapping[str, Path]:
    if not log_root.exists():
        return {}
    return {path.name: path for path in sorted(log_root.glob("*.txt"))}


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
    body = str(text or "").rstrip()
    ticks = 3
    while "`" * ticks in body:
        ticks += 1
    fence = "`" * ticks
    return [fence + info, body, fence]


def _write_text_file(path: Path, text: str) -> Mapping[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {"path": path.as_posix(), "chars": len(text)}


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _relative_or_text(base: Path, value: str | Path | None) -> str:
    if value is None:
        return "(none)"
    path = Path(value)
    try:
        return path.relative_to(base).as_posix()
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
