#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence
import zipfile


def build_viewer(
    workspace: Path,
    out_dir: Path,
    *,
    max_frames: int = 24,
    title: str = "LongVideo Explorer",
    zip_path: Path | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"viewer output is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    payload = _viewer_payload(workspace, assets_dir, max_frames=max_frames, title=title)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = Path(__file__).with_name("exploration_trace_viewer_template.html").read_text(
        encoding="utf-8"
    )
    index_path = out_dir / "index.html"
    index_path.write_text(
        template.replace("__PAGE_TITLE__", html.escape(title, quote=True)).replace(
            "__TRACE_DATA__",
            serialized,
        ),
        encoding="utf-8",
    )
    manifest_path = out_dir / "viewer_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "workspace": str(workspace),
                "case_id": payload["case"]["caseId"],
                "event_count": len(payload["events"]),
                "step_count": len(payload["steps"]),
                "round_count": len(payload["rounds"]),
                "frame_count": len(payload["frames"]),
                "duration_sec": payload["durationSec"],
                "index": "index.html",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    archive_path = None
    if zip_path is not None:
        archive_path = Path(zip_path).resolve()
        if archive_path.exists():
            raise FileExistsError(f"viewer archive already exists: {archive_path}")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(out_dir.name) / path.relative_to(out_dir))
    return {
        "viewer": str(index_path),
        "manifest": str(manifest_path),
        "zip": str(archive_path) if archive_path else None,
        "case_id": payload["case"]["caseId"],
        "event_count": len(payload["events"]),
        "step_count": len(payload["steps"]),
        "round_count": len(payload["rounds"]),
        "frame_count": len(payload["frames"]),
    }


def _viewer_payload(
    workspace: Path,
    assets_dir: Path,
    *,
    max_frames: int,
    title: str,
) -> dict[str, Any]:
    case = _read_json(workspace / "case.json")
    run_summary = _read_json(workspace / "run_summary.json")
    run_config = _read_json(workspace / "run_config.json")
    asset_root = _asset_root(workspace, case)
    timeline = _read_json(
        workspace / "virtual_timeline.json"
        if (workspace / "virtual_timeline.json").is_file()
        else asset_root / "virtual_timeline.json"
    )
    observations = _read_jsonl(workspace / "observation_log.jsonl")
    ledger = _read_jsonl(workspace / "exploration_ledger.jsonl")
    interactions = _read_jsonl(workspace / "interactions.jsonl")
    workspace_ops = _read_jsonl(workspace / "workspace_ops.jsonl")
    frame_rows = _read_jsonl(workspace / "observations" / "window_frame_manifest.jsonl")
    frames = _bundle_frames(frame_rows, assets_dir, max_frames=max_frames)
    passage_ids = {
        str(hit.get("passage_id", ""))
        for row in observations
        for hit in tuple((row.get("sampling_config") or {}).get("hits", ()) or ())
        if str(hit.get("passage_id", ""))
    }
    passages = _load_caption_passages(asset_root, run_config, passage_ids)
    interaction_results = _interaction_results(interactions)
    events = _events(observations, run_summary, frames, passages, interaction_results)
    rounds = _rounds(
        run_summary,
        observations,
        frames,
        frame_rows,
        passages,
        interaction_results,
        workspace_ops,
        events,
    )
    steps = _steps(run_summary, events, rounds)
    duration = float(timeline.get("duration_sec") or 0.0)
    evaluation = dict(run_summary.get("evaluation") or {})
    score = evaluation.get("accuracy_score")
    initial_event_id = _initial_event_id(events, case)
    counts = {
        "caption": sum(event["kind"] == "caption" for event in events),
        "visual": sum(event["kind"] == "visual" for event in events),
        "evidence": sum(event["kind"] in {"evidence", "support"} for event in events),
    }
    return {
        "schemaVersion": 2,
        "title": title,
        "durationSec": duration,
        "case": {
            "caseId": str(case.get("case_id") or workspace.name),
            "question": str(case.get("question") or ""),
            "gold": str(case.get("gold_answer") or case.get("gold") or ""),
            "goldIntervals": _ranges(case.get("gold_clue_intervals") or ()),
            "answer": str(run_summary.get("answer") or ""),
            "answerPresent": bool(run_summary.get("answer_present")),
            "referenceValid": bool(run_summary.get("reference_valid")),
            "referenceReason": str(run_summary.get("reference_reason") or ""),
            "score": float(score) if isinstance(score, (int, float)) else None,
        },
        "models": dict(run_config.get("models") or {}),
        "captionConfigDigest": str(run_config.get("caption_config_digest") or ""),
        "indexDigests": list(run_config.get("caption_index_digests") or ()),
        "segments": [
            {
                "id": str(segment.get("segment_id") or ""),
                "source": str(segment.get("source_video_id") or ""),
                "start": float(segment.get("virtual_start_sec") or 0.0),
                "end": float(segment.get("virtual_end_sec") or 0.0),
            }
            for segment in tuple(timeline.get("segments") or ())
        ],
        "events": events,
        "steps": steps,
        "rounds": rounds,
        "frames": frames,
        "initialEventId": initial_event_id,
        "metrics": {
            "rounds": int(run_summary.get("rounds") or 0),
            "investigations": int(run_summary.get("investigation_count") or 0),
            "captionSearches": sum(
                (row.get("sampling_config") or {}).get("mode") == "search_caption"
                for row in observations
            ),
            "visualWindows": sum(row.get("modality") in {"visual", "ocr"} for row in observations),
            "evidenceRecords": len(tuple(run_summary.get("evidence") or ())),
            "ledgerVisits": len(ledger),
            **counts,
        },
    }


def _events(
    observations: Sequence[Mapping[str, Any]],
    run_summary: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
    passages: Mapping[str, str],
    interaction_results: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    frames_by_query: dict[str, list[str]] = {}
    for frame in frames:
        frames_by_query.setdefault(str(frame.get("queryId") or ""), []).append(str(frame["id"]))

    for row_index, row in enumerate(observations, start=1):
        config = dict(row.get("sampling_config") or {})
        mode = str(config.get("mode") or "")
        round_id = _round_number(row.get("round_id"))
        task_id = str(row.get("task_id") or "")
        if mode == "search_caption":
            queries = [str(value) for value in tuple(config.get("queries") or ()) if str(value)]
            for rank, hit in enumerate(tuple(config.get("hits") or ())[:12], start=1):
                interval = _range(hit.get("range"))
                if interval is None:
                    continue
                passage_id = str(hit.get("passage_id") or "")
                events.append(
                    {
                        "id": f"caption-{row_index}-{rank}",
                        "kind": "caption",
                        "round": round_id,
                        "start": interval[0],
                        "end": interval[1],
                        "title": f"Caption rank {rank}",
                        "subtitle": task_id or "Caption search",
                        "text": _clip(passages.get(passage_id) or row.get("raw_output") or "", 620),
                        "query": _clip(" · ".join(queries), 260),
                        "score": _number(hit.get("score")),
                        "source": passage_id,
                        "taskId": task_id,
                        "frameIds": [],
                        "observations": [],
                        "uncertainties": [],
                    }
                )
            continue
        if row.get("modality") not in {"visual", "ocr"}:
            continue
        inspected = _ranges(row.get("inspected_ranges") or ())
        if not inspected:
            requested = _range(row.get("requested_range"))
            inspected = [requested] if requested else []
        frame_ids = frames_by_query.get(task_id, [])
        parsed = dict(interaction_results.get(task_id) or _structured_payload(row.get("raw_output")))
        summary = parsed.get("summary") or row.get("raw_output") or ""
        observation_points = _observation_points(parsed.get("observations") or ())
        uncertainties = [
            _clip(value, 420) for value in tuple(parsed.get("uncertainties") or ()) if str(value).strip()
        ]
        for range_index, interval in enumerate(inspected, start=1):
            events.append(
                {
                    "id": f"visual-{row_index}-{range_index}",
                    "kind": "visual",
                    "round": round_id,
                    "start": interval[0],
                    "end": interval[1],
                    "title": "Visual inspection",
                    "subtitle": task_id or str(row.get("interpretation_id") or "window"),
                    "text": _clip(summary, 620),
                    "query": f"{config.get('fps', row.get('sampling_fps', 0))} fps · {len(frame_ids)} frames",
                    "score": None,
                    "source": str(row.get("attempt_id") or ""),
                    "taskId": task_id,
                    "frameIds": frame_ids,
                    "observations": observation_points,
                    "uncertainties": uncertainties,
                }
            )

    for index, evidence in enumerate(tuple(run_summary.get("evidence") or ()), start=1):
        interval = _range(evidence.get("virtual_time_range"))
        if interval is None:
            continue
        attempt_id = str(evidence.get("attempt_id") or "")
        matching = next(
            (
                event
                for event in events
                if event["kind"] == "visual" and event.get("source") == attempt_id
            ),
            None,
        )
        events.append(
            {
                "id": f"evidence-{index}",
                "kind": "evidence",
                "round": matching.get("round", 0) if matching else 0,
                "start": interval[0],
                "end": interval[1],
                "title": "Evidence committed",
                "subtitle": str(evidence.get("evidence_id") or "evidence"),
                "text": _clip(evidence.get("summary") or "", 620),
                "query": str(evidence.get("modality") or ""),
                "score": None,
                "source": str(evidence.get("pointer") or attempt_id),
                "taskId": str(matching.get("taskId") or "") if matching else "",
                "frameIds": list(matching.get("frameIds", ())) if matching else [],
                "observations": list(matching.get("observations", ())) if matching else [],
                "uncertainties": list(matching.get("uncertainties", ())) if matching else [],
            }
        )

    final_round = _final_round(run_summary)
    for index, interval in enumerate(_ranges(run_summary.get("supporting_intervals") or ()), start=1):
        events.append(
            {
                "id": f"support-{index}",
                "kind": "support",
                "round": final_round,
                "start": interval[0],
                "end": interval[1],
                "title": "Answer support",
                "subtitle": "Final citation",
                "text": _clip(run_summary.get("answer") or "", 620),
                "query": str(run_summary.get("reference_reason") or ""),
                "score": None,
                "source": ", ".join(str(value) for value in tuple(run_summary.get("citations") or ())),
                "taskId": "",
                "frameIds": _nearest_frame_ids(frames, interval),
                "observations": [],
                "uncertainties": [],
            }
        )
    return sorted(events, key=lambda event: (event["start"], event["kind"], event["id"]))


def _rounds(
    run_summary: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]],
    passages: Mapping[str, str],
    interaction_results: Mapping[str, Mapping[str, Any]],
    workspace_ops: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    trace = tuple(run_summary.get("trace") or ())
    batches = {
        _round_number(row.get("round")): row
        for row in trace
        if row.get("type") == "investigator_batch"
    }
    references = {
        _round_number(row.get("round")): row
        for row in trace
        if row.get("type") == "reference_integrity_check"
    }
    observations_by_task = {
        str(row.get("task_id") or ""): row for row in observations if str(row.get("task_id") or "")
    }
    frames_by_task: dict[str, list[Mapping[str, Any]]] = {}
    for frame in frames:
        frames_by_task.setdefault(str(frame.get("queryId") or ""), []).append(frame)
    source_frames_by_task: dict[str, list[Mapping[str, Any]]] = {}
    for frame in frame_rows:
        source_frames_by_task.setdefault(str(frame.get("query_id") or ""), []).append(frame)
    ops_by_round = {
        _round_number(row.get("round_id")): row for row in workspace_ops if _round_number(row.get("round_id"))
    }
    answer_outcome = next((row for row in trace if row.get("type") == "answer_outcome"), {})
    final_round = _final_round(run_summary)
    rounds: list[dict[str, Any]] = []

    for decision in (row for row in trace if row.get("type") == "reasoner_decision"):
        round_id = _round_number(decision.get("round"))
        batch = dict(batches.get(round_id) or {})
        outcomes = {
            str(outcome.get("query_id") or ""): outcome
            for outcome in tuple(batch.get("outcomes") or ())
            if isinstance(outcome, Mapping)
        }
        task_details = []
        for task_index, task_value in enumerate(tuple(decision.get("tasks") or ()), start=1):
            if not isinstance(task_value, Mapping):
                continue
            task = dict(task_value)
            query_id = str(task.get("query_id") or f"r{round_id}_t{task_index}")
            observation = dict(observations_by_task.get(query_id) or {})
            config = dict(observation.get("sampling_config") or {})
            outcome = dict(outcomes.get(query_id) or {})
            mode = str(
                task.get("inspection_mode")
                or config.get("mode")
                or observation.get("modality")
                or "investigate"
            )
            hits = [hit for hit in tuple(config.get("hits") or ()) if isinstance(hit, Mapping)]
            task_events = [event for event in events if event.get("taskId") == query_id]
            top_hits = []
            for hit in hits[:3]:
                passage_id = str(hit.get("passage_id") or "")
                matching_event = next(
                    (event for event in task_events if event.get("source") == passage_id),
                    None,
                )
                top_hits.append(
                    {
                        "passageId": passage_id,
                        "range": _range(hit.get("range")) or [],
                        "score": _number(hit.get("score")),
                        "text": _clip(passages.get(passage_id) or "", 360),
                        "eventId": str(matching_event.get("id") or "") if matching_event else "",
                        "selectedNext": False,
                    }
                )
            parsed = dict(interaction_results.get(query_id) or _structured_payload(observation.get("raw_output")))
            observation_points = _observation_points(parsed.get("observations") or ())
            result_events = _result_events(parsed.get("events") or ())
            uncertainties = [
                _clip(value, 420)
                for value in tuple(parsed.get("uncertainties") or ())
                if str(value).strip()
            ]
            inspected_ranges = _ranges(observation.get("inspected_ranges") or ())
            requested_range = _range(task.get("time_range")) or _range(config.get("time_range"))
            if not inspected_ranges and requested_range:
                inspected_ranges = [requested_range]
            task_frames = list(frames_by_task.get(query_id, ()))
            attempt_ids = [
                str(value) for value in tuple(outcome.get("attempt_ids") or ()) if str(value)
            ]
            if not attempt_ids and observation.get("attempt_id"):
                attempt_ids = [str(observation["attempt_id"])]
            evidence_ids = [
                str(value) for value in tuple(outcome.get("evidence_ids") or ()) if str(value)
            ]
            result_summary = parsed.get("summary") or (top_hits[0]["text"] if top_hits else "")
            if not result_summary and observation.get("raw_output"):
                result_summary = observation.get("raw_output")
            preferred_event = next(
                (event for event in task_events if event.get("kind") == "evidence"),
                task_events[0] if task_events else None,
            )
            action_event = next(
                (
                    event
                    for event in task_events
                    if event.get("kind") == ("caption" if mode == "search_caption" else "visual")
                    and (mode != "search_caption" or event.get("title") == "Caption rank 1")
                ),
                task_events[0] if task_events else None,
            )
            source_frames = list(source_frames_by_task.get(query_id, ()))
            task_details.append(
                {
                    "id": query_id,
                    "mode": mode,
                    "goal": _clip(task.get("goal") or "", 260),
                    "captionQueries": [
                        str(value)
                        for value in tuple(task.get("caption_queries") or config.get("queries") or ())
                        if str(value)
                    ],
                    "timeRange": requested_range or (inspected_ranges[0] if inspected_ranges else []),
                    "inspectedRanges": inspected_ranges,
                    "segmentId": str(task.get("segment_id") or ""),
                    "indexMode": str(task.get("index_mode") or config.get("index_mode") or ""),
                    "topK": int(task.get("top_k") or config.get("top_k") or len(hits) or 0),
                    "samplingFps": _number(
                        task.get("sampling_floor_fps") or config.get("fps") or observation.get("sampling_fps")
                    ),
                    "maxFrames": int(config.get("max_frames") or len(task_frames) or 0),
                    "expandNeighbors": int(task.get("expand_neighbors") or config.get("expand_neighbors") or 0),
                    "forceReinspect": bool(task.get("force_reinspect")),
                    "arbitrationAttemptId": str(task.get("arbitration_attempt_id") or ""),
                    "status": str(outcome.get("status") or ("completed" if observation else "unknown")),
                    "reused": bool(outcome.get("reused")),
                    "consumesBudget": bool(outcome.get("consumes_budget", bool(observation))),
                    "attemptIds": attempt_ids,
                    "evidenceIds": evidence_ids,
                    "eventIds": [str(event["id"]) for event in task_events],
                    "actionEventId": str(action_event.get("id") or "") if action_event else "",
                    "preferredEventId": str(preferred_event.get("id") or "") if preferred_event else "",
                    "result": {
                        "hitCount": len(hits),
                        "topHits": top_hits,
                        "frameCount": len(source_frames) or len(task_frames),
                        "bundledFrameCount": len(task_frames),
                        "summary": _clip(result_summary, 620),
                        "observations": observation_points,
                        "events": result_events,
                        "uncertainties": uncertainties,
                    },
                }
            )

        ops_row = dict(ops_by_round.get(round_id) or {})
        claims = _workspace_claims(ops_row)
        reference = dict(references.get(round_id) or {})
        round_events = [str(event["id"]) for event in events if event.get("round") == round_id]
        rounds.append(
            {
                "round": round_id,
                "decision": {
                    "action": str(decision.get("action") or "decide"),
                    "remainingBudget": int(decision.get("remaining_budget") or 0),
                    "workspaceRevision": int(decision.get("workspace_revision") or 0),
                    "supportingClaimIds": [
                        str(value) for value in tuple(decision.get("supporting_claim_ids") or ())
                    ],
                    "finalAttempt": bool(decision.get("final_attempt")),
                    "forceFinalize": bool(decision.get("force_finalize")),
                },
                "tasks": task_details,
                "claims": claims,
                "workspace": dict(ops_row.get("result") or {}),
                "reference": {
                    "passed": bool(reference.get("passed")),
                    "reason": str(reference.get("reason") or ""),
                    "errors": [str(value) for value in tuple(reference.get("errors") or ())],
                }
                if reference
                else {},
                "answer": str(answer_outcome.get("answer") or run_summary.get("answer") or "")
                if round_id == final_round
                else "",
                "eventIds": round_events,
            }
        )
    for index, round_detail in enumerate(rounds[:-1]):
        next_ranges = [
            task.get("timeRange")
            for task in tuple(rounds[index + 1].get("tasks") or ())
            if _range(task.get("timeRange"))
        ]
        for task in tuple(round_detail.get("tasks") or ()):
            for hit in tuple((task.get("result") or {}).get("topHits") or ()):
                hit_range = _range(hit.get("range"))
                hit["selectedNext"] = bool(
                    hit_range
                    and any(_intervals_overlap(hit_range, next_range) for next_range in next_ranges)
                )
    return rounds


def _steps(
    run_summary: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    rounds: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    rounds_by_id = {int(row.get("round") or 0): row for row in rounds}
    final_round = _final_round(run_summary)
    for index, row in enumerate(tuple(run_summary.get("trace") or ()), start=1):
        row_type = str(row.get("type") or "")
        round_id = _round_number(row.get("round")) or final_round
        round_detail = dict(rounds_by_id.get(round_id) or {})
        tasks = tuple(round_detail.get("tasks") or ())
        if row_type == "reasoner_decision":
            action = str(row.get("action") or "decide")
            task_ids = [str(task.get("id") or "") for task in tasks if isinstance(task, Mapping)]
            task_text = " · ".join(_task_summary(task) for task in tasks if isinstance(task, Mapping))
            steps.append(
                {
                    "id": f"step-{index}",
                    "kind": "reasoner",
                    "round": round_id,
                    "title": f"Reasoner · {', '.join(task_ids) if task_ids else action}",
                    "detail": _clip(task_text or f"Workspace revision {row.get('workspace_revision', 0)}", 300),
                    "eventIds": [event["id"] for event in events if event.get("round") == round_id],
                    "taskIds": task_ids,
                    "preferredEventId": str(tasks[0].get("actionEventId") or "") if tasks else "",
                }
            )
        elif row_type == "investigator_batch":
            task_ids = [str(task.get("id") or "") for task in tasks if isinstance(task, Mapping)]
            modes = ", ".join(_behavior_name(str(task.get("mode") or "")) for task in tasks)
            detail = " · ".join(
                f"{task.get('status', 'unknown')} · {_result_count(task)}"
                for task in tasks
                if isinstance(task, Mapping)
            )
            steps.append(
                {
                    "id": f"step-{index}",
                    "kind": "investigator",
                    "round": round_id,
                    "title": f"Investigator · {modes or 'execute'}",
                    "detail": _clip(detail or f"{row.get('completed_tasks', 0)} tasks completed", 300),
                    "eventIds": [event["id"] for event in events if event.get("round") == round_id],
                    "taskIds": task_ids,
                    "preferredEventId": str(tasks[0].get("actionEventId") or "") if tasks else "",
                }
            )
        elif row_type == "reference_integrity_check":
            steps.append(
                {
                    "id": f"step-{index}",
                    "kind": "gate",
                    "round": round_id,
                    "title": "Reference gate",
                    "detail": str(row.get("reason") or ("passed" if row.get("passed") else "rejected")),
                    "eventIds": [event["id"] for event in events if event["kind"] == "support"],
                    "taskIds": [],
                    "preferredEventId": str(
                        next((event["id"] for event in events if event["kind"] == "support"), "")
                    ),
                }
            )
        elif row_type == "answer_outcome":
            steps.append(
                {
                    "id": f"step-{index}",
                    "kind": "answer",
                    "round": round_id,
                    "title": "Answer committed",
                    "detail": _clip(row.get("answer") or "", 300),
                    "eventIds": [event["id"] for event in events if event["kind"] == "support"],
                    "taskIds": [],
                    "preferredEventId": str(
                        next((event["id"] for event in events if event["kind"] == "support"), "")
                    ),
                }
            )
    return steps


def _interaction_results(interactions: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    results = {}
    for row in interactions:
        query_id = str(row.get("query_id") or "")
        parsed = row.get("parsed")
        if query_id and isinstance(parsed, Mapping) and parsed:
            results[query_id] = dict(parsed)
    return results


def _structured_payload(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _observation_points(values: Iterable[Any]) -> list[dict[str, Any]]:
    points = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        points.append(
            {
                "time": _number(value.get("time_sec")),
                "description": _clip(value.get("description") or value.get("text") or "", 420),
            }
        )
    return points


def _result_events(values: Iterable[Any]) -> list[dict[str, Any]]:
    results = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        results.append(
            {
                "range": _range(value.get("time_range")) or [],
                "description": _clip(value.get("description") or value.get("text") or "", 420),
            }
        )
    return results


def _workspace_claims(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    claims = []
    for operation in tuple(row.get("operations") or ()):
        if not isinstance(operation, Mapping) or operation.get("op") != "add_claim":
            continue
        claim = operation.get("claim")
        if not isinstance(claim, Mapping):
            continue
        claims.append(
            {
                "id": str(claim.get("claim_id") or ""),
                "text": _clip(claim.get("text") or "", 520),
                "confidence": str(claim.get("confidence") or ""),
                "source": str(claim.get("source") or ""),
                "timeAnchor": _range(claim.get("time_anchor")) or [],
                "cites": [str(value) for value in tuple(claim.get("cites") or ())],
                "derivedFrom": [str(value) for value in tuple(claim.get("derived_from") or ())],
            }
        )
    return claims


def _final_round(run_summary: Mapping[str, Any]) -> int:
    return max(
        (_round_number(row.get("round")) for row in tuple(run_summary.get("trace") or ())),
        default=0,
    )


def _intervals_overlap(first: Sequence[float], second: Sequence[float]) -> bool:
    return float(first[1]) > float(second[0]) and float(first[0]) < float(second[1])


def _behavior_name(mode: str) -> str:
    return {
        "search_caption": "search_caption()",
        "window": "inspect_window()",
        "visual": "inspect_window()",
        "ocr": "inspect_ocr()",
    }.get(mode, f"{mode or 'investigate'}()")


def _result_count(task: Mapping[str, Any]) -> str:
    result = dict(task.get("result") or {})
    if task.get("mode") == "search_caption":
        return f"{result.get('hitCount', 0)} hits"
    if result.get("frameCount"):
        return f"{result['frameCount']} frames"
    return f"{len(tuple(task.get('evidenceIds') or ()))} evidence"


def _task_summary(task: Mapping[str, Any]) -> str:
    mode = str(task.get("mode") or "investigate")
    if mode == "search_caption":
        return f"{task.get('id')} · {mode} · {task.get('indexMode') or 'index'} · top_k={task.get('topK', 0)}"
    return f"{task.get('id')} · {mode} · {task.get('samplingFps') or 0} fps · {task.get('maxFrames', 0)} frames"


def _bundle_frames(
    rows: Sequence[Mapping[str, Any]],
    assets_dir: Path,
    *,
    max_frames: int,
) -> list[dict[str, Any]]:
    valid = [row for row in rows if Path(str(row.get("path") or "")).is_file()]
    selected = _uniform_subset(valid, max(0, int(max_frames)))
    bundled = []
    for index, row in enumerate(selected, start=1):
        source = Path(str(row["path"]))
        destination = assets_dir / f"frame-{index:03d}{source.suffix.lower() or '.jpg'}"
        shutil.copy2(source, destination)
        bundled.append(
            {
                "id": f"frame-{index:03d}",
                "src": f"assets/{destination.name}",
                "time": float(row.get("virtual_time_sec") or 0.0),
                "queryId": str(row.get("query_id") or ""),
                "source": str(row.get("source_video_id") or ""),
                "sourceTime": float(row.get("source_time_sec") or 0.0),
            }
        )
    return bundled


def _load_caption_passages(
    asset_root: Path,
    run_config: Mapping[str, Any],
    wanted_ids: set[str],
) -> dict[str, str]:
    if not wanted_ids:
        return {}
    digest = str(run_config.get("caption_config_digest") or "")
    if not digest:
        return {}
    path = asset_root / "captions" / f"passages.{digest}.jsonl"
    if not path.is_file():
        return {}
    passages: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            passage_id = str(payload.get("passage_id") or "")
            if passage_id in wanted_ids:
                passages[passage_id] = str(payload.get("text") or "")
                if len(passages) == len(wanted_ids):
                    break
    return passages


def _initial_event_id(events: Sequence[Mapping[str, Any]], case: Mapping[str, Any]) -> str:
    gold = _ranges(case.get("gold_clue_intervals") or ())
    if gold:
        gold_start, gold_end = gold[0]
        overlapping = [
            event
            for event in events
            if event["kind"] in {"visual", "evidence", "support"}
            and event["end"] > gold_start
            and event["start"] < gold_end
        ]
        if overlapping:
            return str(overlapping[-1]["id"])
    preferred = [event for event in events if event["kind"] in {"visual", "evidence", "support"}]
    return str((preferred or list(events) or [{"id": ""}])[-1]["id"])


def _nearest_frame_ids(frames: Sequence[Mapping[str, Any]], interval: Sequence[float]) -> list[str]:
    if not frames:
        return []
    midpoint = (float(interval[0]) + float(interval[1])) / 2.0
    ordered = sorted(frames, key=lambda frame: abs(float(frame["time"]) - midpoint))
    return [str(frame["id"]) for frame in ordered[:8]]


def _asset_root(workspace: Path, case: Mapping[str, Any]) -> Path:
    reference = str(case.get("asset_ref") or "").strip()
    if not reference:
        return workspace
    path = Path(reference)
    return path if path.is_absolute() else (workspace / path).resolve()


def _uniform_subset(rows: Sequence[Any], limit: int) -> tuple[Any, ...]:
    values = tuple(rows)
    count = min(len(values), max(0, int(limit)))
    if count == 0:
        return ()
    if count == len(values):
        return values
    if count == 1:
        return (values[len(values) // 2],)
    return tuple(values[round(index * (len(values) - 1) / (count - 1))] for index in range(count))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def _ranges(values: Iterable[Any]) -> list[list[float]]:
    ranges = []
    for value in values:
        interval = _range(value)
        if interval is not None:
            ranges.append(interval)
    return ranges


def _range(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return [start, end] if end > start else None


def _round_number(value: Any) -> int:
    text = str(value or "0").strip().removeprefix("round_")
    try:
        return int(text)
    except ValueError:
        return 0


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an interactive ultra-long-video exploration trace viewer.")
    parser.add_argument("--workspace", required=True, help="One completed virtual-video case workspace.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip-path")
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--title", default="LongVideo Explorer")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_viewer(
        Path(args.workspace),
        Path(args.out_dir),
        max_frames=args.max_frames,
        title=args.title,
        zip_path=Path(args.zip_path) if args.zip_path else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
