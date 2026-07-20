from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

from vcah.investigator import (
    InvestigationReport,
    ObservationAttempt,
    VirtualVideoInvestigator,
)
from vcah.model_client import ImageAttachmentError, OpenAICompatibleClient
from vcah.multiround import InvestigationTask, ReasonerDecision
from vcah.types import CoverageSegment, EvidenceRecord
from vcah.virtual_video import VirtualVideoWorkspace
from vcah.workspace import prompt_digest, stable_attempt_id


_DECISION_ACTIONS = {"investigate", "read_observations", "update_workspace", "answer"}
_DECISION_WRAPPERS = ("response", "responses", "items")


def _completion_budget(default: int) -> int:
    return max(4096, int(default))


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _parse_json(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if fenced:
        raw = fenced.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _decision_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    if str(value.get("action", "") or "").strip().casefold() in _DECISION_ACTIONS:
        return dict(value)
    for key in _DECISION_WRAPPERS:
        nested = value.get(key)
        candidates = (nested,) if isinstance(nested, Mapping) else nested
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            continue
        for candidate in candidates:
            if isinstance(candidate, Mapping) and (payload := _decision_payload(candidate)):
                return payload
    return {}


def _answer(value: Any, options: Mapping[str, str]) -> str:
    if isinstance(value, Mapping):
        nested = value.get("answer")
        if nested is not None:
            return _answer(nested, options)
        value = value.get("option") or value.get("label") or value.get("choice") or value.get("text")
    text = str(value or "").strip()
    match = re.match(r"\s*([A-H])(?:\.|\)|:|\s|$)", text.upper())
    if match and match.group(1) in options:
        label = match.group(1)
        return f"{label}. {options[label]}"
    normalized = re.sub(r"[^a-z0-9]+", "", text.casefold())
    matches = [
        (label, option)
        for label, option in options.items()
        if normalized and normalized == re.sub(r"[^a-z0-9]+", "", option.casefold())
    ]
    return f"{matches[0][0]}. {matches[0][1]}" if len(matches) == 1 else text


def _task(value: Mapping[str, Any], *, round_id: int, index: int) -> InvestigationTask | None:
    goal = str(value.get("goal") or value.get("task") or "").strip()
    if not goal:
        return None
    mode = str(value.get("inspection_mode", "window") or "window").strip().casefold()
    if mode not in {"window", "search_asr", "arbitrate_observation"}:
        return None
    raw_range = value.get("time_range")
    time_range = None
    if isinstance(raw_range, Sequence) and not isinstance(raw_range, (str, bytes)) and len(raw_range) == 2:
        try:
            start, end = float(raw_range[0]), float(raw_range[1])
            time_range = (min(start, end), max(start, end)) if start != end else None
        except (TypeError, ValueError):
            pass
    task = InvestigationTask(
        query_id=str(value.get("query_id") or f"r{round_id}_t{index}"),
        goal=goal,
        segment_id=str(value.get("segment_id", "") or ""),
        time_range=time_range,
        expected_evidence=str(value.get("expected_evidence", "") or goal),
        inspection_mode=mode,
        search_terms=tuple(value.get("search_terms", ()) or ()),
        sampling_floor_fps=value.get("sampling_floor_fps"),
        arbitration_attempt_id=str(value.get("arbitration_attempt_id", "") or ""),
    )
    if mode == "search_asr" and not task.search_terms:
        return None
    if mode == "arbitrate_observation" and not task.arbitration_attempt_id:
        return None
    if mode == "window" and not (task.segment_id or task.time_range):
        return None
    return task


def _normalize_decision(value: Mapping[str, Any], *, round_id: int) -> dict[str, Any]:
    payload = dict(value)
    raw_tasks = payload.get("tasks", ()) or ()
    tasks = []
    if isinstance(raw_tasks, Sequence) and not isinstance(raw_tasks, (str, bytes)):
        for index, row in enumerate(raw_tasks, start=1):
            if isinstance(row, Mapping):
                normalized = _task(row, round_id=round_id, index=index)
                if normalized is not None:
                    tasks.append(normalized)
    action = str(payload.get("action", "") or "").strip().casefold()
    if action not in _DECISION_ACTIONS:
        action = "update_workspace"
    return {
        "action": action,
        "tasks": tuple(tasks),
        "answer": payload.get("answer", ""),
        "citations": tuple(str(item) for item in payload.get("citations", ()) or () if str(item).strip()),
        "workspace_ops": tuple(dict(item) for item in payload.get("workspace_ops", payload.get("ops", ())) or () if isinstance(item, Mapping)),
        "supporting_claim_ids": tuple(str(item) for item in payload.get("supporting_claim_ids", ()) or () if str(item).strip()),
        "residual_uncertainty": str(payload.get("residual_uncertainty", "") or ""),
        "observation_requests": tuple(dict(item) for item in payload.get("observation_requests", ()) or () if isinstance(item, Mapping)),
    }


class WorkspaceReasoner:
    """The only component that makes semantic decisions."""

    def __init__(
        self,
        api: OpenAICompatibleClient,
        *,
        trace_path: Path,
    ) -> None:
        self.api = api
        self.trace_path = trace_path
        self.calls = 0

    def decide(self, **kwargs: Any) -> ReasonerDecision:
        self.calls += 1
        prompt = _reasoner_prompt(kwargs)
        raw = self.api.chat(prompt, max_tokens=_completion_budget(2200))
        api_response = dict(self.api.last_response_metadata)
        parsed = _parse_json(raw)
        payload = _decision_payload(parsed)
        repair_attempted = not payload
        repaired = False
        if repair_attempted:
            repair_prompt = (
                "Recover this Reasoner response as one compact Decision JSON object with exactly one valid action: "
                "investigate, read_observations, update_workspace, or answer. Preserve only content already present; "
                "do not invent observations, claims, references, support, or an answer. Return JSON only.\n"
                f"Response: {raw}"
            )
            repaired_raw = self.api.chat(repair_prompt, max_tokens=_completion_budget(1400))
            repaired_parsed = _parse_json(repaired_raw)
            payload = _decision_payload(repaired_parsed)
            repaired = bool(payload)
            _append_jsonl(
                self.trace_path,
                {
                    "type": "reasoner_json_repair",
                    "round": self.calls,
                    "prompt": repair_prompt,
                    "raw": repaired_raw,
                    "parsed": repaired_parsed,
                    "decision_payload": payload,
                    "api_response": self.api.last_response_metadata,
                    "time": time.time(),
                },
            )
        value = _normalize_decision(payload or {"action": "update_workspace"}, round_id=self.calls)
        value["answer"] = _answer(value["answer"], dict(kwargs.get("options") or {}))
        decision = ReasonerDecision(**value)
        _append_jsonl(
            self.trace_path,
            {
                "type": "reasoner_workspace",
                "round": self.calls,
                "model": self.api.model,
                "prompt": prompt,
                "raw": raw,
                "parsed": parsed,
                "decision_payload": payload,
                "schema_unwrapped": bool(payload and payload != parsed),
                "format_repaired": repaired,
                "repair_failed": repair_attempted and not payload,
                "api_response": api_response,
                "time": time.time(),
            },
        )
        return decision

class VisionInvestigator(VirtualVideoInvestigator):
    """Observation-only visual agent; it never evaluates options or claims."""

    def __init__(self, workspace: VirtualVideoWorkspace, *, api: OpenAICompatibleClient, trace_path: Path) -> None:
        super().__init__(workspace)
        self.api = api
        self.trace_path = trace_path
        self._seen_asr_attempt_ids: set[str] = set()

    def reset_run_state(self) -> None:
        super().reset_run_state()
        self._seen_asr_attempt_ids.clear()

    def _investigate_task(self, task: Any) -> InvestigationReport:
        try:
            mode = str(getattr(task, "inspection_mode", "window") or "window")
            if mode == "search_asr":
                return self._search_asr(task)
            if mode == "arbitrate_observation":
                return self._arbitrate(task)
            return self._observe_window(task)
        except ImageAttachmentError as exc:
            return self._attachment_failure(task, exc)

    def _observe_window(self, task: Any) -> InvestigationReport:
        query_id = str(getattr(task, "query_id", "") or "observation")
        segment_id = str(getattr(task, "segment_id", "") or "")
        if segment_id:
            segment_packet = self.open_segment(segment_id)
        else:
            segment = self.workspace.manifest.segments[0]
            segment_packet = self.open_segment(segment.segment_id)
        requested = getattr(task, "time_range", None)
        start_sec, end_sec = (
            (float(requested[0]), float(requested[1]))
            if requested is not None
            else tuple(float(value) for value in segment_packet["virtual_time_range"])
        )
        fps = float(getattr(task, "sampling_floor_fps", 0.5) or 0.5)
        frame_limit = min(96, max(1, int((end_sec - start_sec) * fps + 0.999)))
        window = self.inspect_window(
            start_sec,
            end_sec,
            fps=fps,
            max_frames=frame_limit,
            query_id=query_id,
        )
        frames = tuple(window["frames"])
        frame_paths = tuple(str(row["path"]) for row in frames)
        frame_times = tuple(float(row["virtual_time_sec"]) for row in frames)
        source_lineage = tuple(dict(item) for item in window["source_lineage"])
        source_video_ids = tuple(
            dict.fromkeys(str(item.get("source_video_id", "") or "") for item in source_lineage)
        )
        attempt_id = stable_attempt_id(
            source_video_ids=source_video_ids,
            frame_times=frame_times,
            inspected_ranges=((start_sec, end_sec),) if frame_paths else (),
            sampling_fps=fps,
            modality="visual",
        )
        prompt = _observation_prompt(self.workspace, task, window)
        raw = self.api.chat(prompt, image_paths=frame_paths, max_tokens=_completion_budget(1800)) if frame_paths else ""
        parsed = _parse_json(raw)
        parse_status = "parsed" if parsed else "failed"
        metadata = self.api.last_response_metadata if frame_paths else {}
        counts = _attachment_counts(metadata, frame_paths)
        evidence = _visual_evidence(
            workspace=self.workspace,
            query_id=query_id,
            attempt_id=attempt_id,
            time_range=(start_sec, end_sec),
            frame_paths=frame_paths,
            fps=fps,
            raw=raw,
            parsed=parsed,
            source_lineage=source_lineage,
            model=self.api.model,
        )
        attempt = ObservationAttempt(
            attempt_id=attempt_id,
            task_id=query_id,
            requested_range=(start_sec, end_sec),
            inspected_ranges=((start_sec, end_sec),) if counts["attached"] else (),
            attached_frame_times=frame_times if counts["attached"] else (),
            sampling_config={"fps": fps, "max_frames": frame_limit, "mode": "window", "modality": "visual"},
            images_requested=counts["requested"],
            images_attached=counts["attached"],
            images_dropped=counts["dropped"],
            parse_status=parse_status,
            execution_status="completed" if frame_paths else "failed",
            frame_refs=frame_paths,
            modality="visual",
            prompt_digest=prompt_digest(prompt),
            raw_output=raw,
            source_video_ids=source_video_ids,
        )
        _append_jsonl(
            self.trace_path,
            {
                "type": "investigator_observation",
                "query_id": query_id,
                "attempt_id": attempt_id,
                "model": self.api.model,
                "prompt": prompt,
                "frame_paths": list(frame_paths),
                "raw": raw,
                "parsed": parsed,
                "api_response": metadata,
                "time": time.time(),
            },
        )
        if frame_paths:
            self._record_visit(task, evidence, status="completed")
        return InvestigationReport(
            query_id=query_id,
            status="completed" if frame_paths else "failed",
            evidence=(evidence,) if frame_paths else (),
            attempts=(attempt,),
            cost={
                "tool_trace": ("open_segment", f"inspect_window:{fps:.1f}"),
                "frames": len(frame_paths),
                "vlm_calls": int(bool(frame_paths)),
                "reused": False,
                "consumes_budget": bool(frame_paths),
            },
            failure_reason="no frames materialized" if not frame_paths else "",
            coverage_delta=((start_sec, end_sec),) if frame_paths else (),
        )

    def _search_asr(self, task: Any) -> InvestigationReport:
        query_id = str(getattr(task, "query_id", "") or "search_asr")
        terms = tuple(getattr(task, "search_terms", ()) or ())
        segment_id = str(getattr(task, "segment_id", "") or "")
        time_range = getattr(task, "time_range", None)
        packet = self.search_asr(
            terms,
            segment_id=segment_id,
            time_range=time_range,
            max_clusters=8,
        )
        clusters = tuple(packet["clusters"])
        ranges = tuple(tuple(float(value) for value in row["virtual_time_range"]) for row in clusters)
        lineage = tuple(dict(item) for row in clusters for item in row.get("source_lineage", ()))
        source_video_ids = tuple(
            dict.fromkeys(str(item.get("source_video_id", "") or "") for item in lineage)
        )
        if not any(source_video_ids):
            source_video_ids = tuple(
                dict.fromkeys(segment.source_video_id for segment in self.workspace.manifest.segments)
            )
        attempt_id = stable_attempt_id(
            source_video_ids=source_video_ids,
            inspected_ranges=ranges,
            modality="asr",
        )
        duplicate = bool(clusters) and (
            attempt_id in self._seen_asr_attempt_ids
            or bool(_observation_rows(self.workspace.root_dir / "observation_log.jsonl", attempt_id))
        )
        if clusters:
            self._seen_asr_attempt_ids.add(attempt_id)
        if duplicate:
            _append_jsonl(
                self.trace_path,
                {
                    "type": "investigator_asr_search",
                    "query_id": query_id,
                    "attempt_id": attempt_id,
                    "search_terms": list(terms),
                    "segment_id": segment_id,
                    "time_range": list(time_range) if time_range else [],
                    "cluster_count": len(clusters),
                    "reused": True,
                    "time": time.time(),
                },
            )
            return InvestigationReport(
                query_id=query_id,
                status="completed",
                cost={
                    "tool_trace": ("search_asr", "reuse_attempt"),
                    "frames": 0,
                    "vlm_calls": 0,
                    "reused": True,
                    "consumes_budget": False,
                },
                failure_reason="duplicate_asr_material_reused",
            )
        raw = json.dumps(packet, ensure_ascii=False, sort_keys=True)
        evidence = tuple(
            EvidenceRecord(
                evidence_id=f"ev_{query_id}_{attempt_id[-8:]}_{index:02d}",
                beat_id="",
                start_sec=float(row["virtual_time_range"][0]),
                end_sec=float(row["virtual_time_range"][1]),
                modality="asr",
                pointer=f"virtual://{self.workspace.workspace_id}/asr/{attempt_id}/{index}",
                verbatim=str(row.get("excerpt", "") or "")[:1200],
                attestation_model="literal-asr-search",
                temporal_scope="window",
                evidence_kind="navigation_hint",
                observation_polarity="unknown",
                sampling_coverage="exact",
                request_ids=(query_id,),
                coverage_manifest=(CoverageSegment(query_id, *row["virtual_time_range"], "asr", 1.0),),
                task_id=query_id,
                observation_id=attempt_id,
                confidence=1.0,
                source_lineage=tuple(dict(item) for item in row.get("source_lineage", ())),
                operation_metadata={
                    "search_terms": list(terms),
                    "segment_id": segment_id,
                    "time_range": list(time_range) if time_range else [],
                    "literal_navigation_only": True,
                },
            )
            for index, row in enumerate(clusters, start=1)
        )
        attempt = ObservationAttempt(
            attempt_id=attempt_id,
            task_id=query_id,
            requested_range=time_range,
            inspected_ranges=ranges,
            sampling_config={
                "mode": "search_asr",
                "modality": "asr",
                "terms": list(terms),
                "segment_id": segment_id,
                "time_range": list(time_range) if time_range else [],
            },
            parse_status="deterministic",
            execution_status="completed",
            modality="asr",
            prompt_digest=prompt_digest("\n".join(terms)),
            raw_output=raw,
            source_video_ids=source_video_ids,
        )
        for record in evidence:
            self._record_visit(task, record, status="navigation_hint")
        _append_jsonl(
            self.trace_path,
            {
                "type": "investigator_asr_search",
                "query_id": query_id,
                "attempt_id": attempt_id,
                "search_terms": list(terms),
                "segment_id": segment_id,
                "time_range": list(time_range) if time_range else [],
                "cluster_count": len(clusters),
                "reused": False,
                "time": time.time(),
            },
        )
        return InvestigationReport(
            query_id=query_id,
            status="completed",
            evidence=evidence,
            attempts=(attempt,),
            cost={
                "tool_trace": ("search_asr",),
                "frames": 0,
                "vlm_calls": 0,
                "reused": False,
                "zero_hits": not clusters,
                "consumes_budget": bool(clusters),
            },
            failure_reason="" if clusters else "asr_zero_hits_use_visual_modality",
            coverage_delta=ranges,
        )

    def _arbitrate(self, task: Any) -> InvestigationReport:
        query_id = str(getattr(task, "query_id", "") or "arbitration")
        target = str(getattr(task, "arbitration_attempt_id", "") or "")
        rows = _observation_rows(self.workspace.root_dir / "observation_log.jsonl", target)
        if not rows:
            return InvestigationReport(query_id=query_id, status="failed", failure_reason=f"unknown attempt_id: {target}")
        first = rows[0]
        frame_paths = tuple(str(path) for path in first.get("frame_refs", ()))
        if not frame_paths:
            return InvestigationReport(query_id=query_id, status="failed", failure_reason=f"attempt has no replayable frames: {target}")
        missing_frames = tuple(path for path in frame_paths if not Path(path).is_file())
        if missing_frames:
            return InvestigationReport(
                query_id=query_id,
                status="failed",
                failure_reason=f"arbitration material incomplete: {len(missing_frames)} frame(s) missing",
            )
        prompt = _arbitration_prompt(task, rows)
        raw = self.api.chat(prompt, image_paths=frame_paths, max_tokens=_completion_budget(1600))
        parsed = _parse_json(raw)
        metadata = self.api.last_response_metadata
        counts = _attachment_counts(metadata, frame_paths)
        time_ranges = tuple(tuple(float(value) for value in item) for item in first.get("inspected_ranges", ()))
        start_sec, end_sec = time_ranges[0] if time_ranges else tuple(first.get("requested_range", (0.0, 0.0)))
        frame_times = tuple(float(value) for value in first.get("frame_times", ()))
        source_video_ids = tuple(str(value) for value in first.get("source_video_ids", ()))
        sampling_fps = float(first.get("sampling_fps", 0.0) or 0.0)
        attempt_id = stable_attempt_id(
            source_video_ids=source_video_ids,
            frame_times=frame_times,
            frame_refs=frame_paths,
            inspected_ranges=time_ranges,
            sampling_fps=sampling_fps,
            modality="visual",
        )
        if attempt_id != target:
            return InvestigationReport(
                query_id=query_id,
                status="failed",
                failure_reason="stored frame material does not match arbitration attempt_id",
            )
        evidence = _visual_evidence(
            workspace=self.workspace,
            query_id=query_id,
            attempt_id=attempt_id,
            time_range=(float(start_sec), float(end_sec)),
            frame_paths=frame_paths,
            fps=sampling_fps,
            raw=raw,
            parsed=parsed,
            source_lineage=tuple(dict(item) for item in first.get("source_lineage", ())),
            model=self.api.model,
        )
        attempt = ObservationAttempt(
            attempt_id=attempt_id,
            task_id=query_id,
            requested_range=(float(start_sec), float(end_sec)),
            inspected_ranges=time_ranges,
            attached_frame_times=frame_times,
            sampling_config={"fps": sampling_fps, "mode": "arbitrate_observation", "modality": "visual"},
            images_requested=counts["requested"],
            images_attached=counts["attached"],
            images_dropped=counts["dropped"],
            parse_status="parsed" if parsed else "failed",
            execution_status="completed",
            frame_refs=frame_paths,
            modality="visual",
            prompt_digest=prompt_digest(prompt),
            raw_output=raw,
            source_video_ids=source_video_ids,
        )
        _append_jsonl(
            self.trace_path,
            {
                "type": "investigator_same_frame_arbitration",
                "query_id": query_id,
                "attempt_id": attempt_id,
                "prompt": prompt,
                "frame_paths": list(frame_paths),
                "raw": raw,
                "parsed": parsed,
                "api_response": metadata,
                "time": time.time(),
            },
        )
        self._record_visit(task, evidence, status="completed")
        return InvestigationReport(
            query_id=query_id,
            status="completed",
            evidence=(evidence,),
            attempts=(attempt,),
            cost={"tool_trace": ("same_frame_arbitration",), "frames": len(frame_paths), "vlm_calls": 1, "consumes_budget": True},
            coverage_delta=(),
        )

    def _attachment_failure(self, task: Any, exc: ImageAttachmentError) -> InvestigationReport:
        query_id = str(getattr(task, "query_id", "") or "observation")
        metadata = dict(exc.metadata)
        frame_refs = tuple(str(path) for path in metadata.get("dropped_image_paths", ()) or ())
        segment_id = str(getattr(task, "segment_id", "") or "")
        source_video_ids = ()
        if segment_id:
            try:
                source_video_ids = (self.workspace.manifest.segment(segment_id).source_video_id,)
            except ValueError:
                pass
        attempt = ObservationAttempt(
            attempt_id=stable_attempt_id(
                source_video_ids=source_video_ids,
                frame_refs=frame_refs,
                inspected_ranges=(getattr(task, "time_range", None),) if getattr(task, "time_range", None) else (),
                sampling_fps=float(getattr(task, "sampling_floor_fps", 0.5) or 0.5),
                modality="visual",
            ),
            task_id=query_id,
            requested_range=getattr(task, "time_range", None),
            sampling_config={"mode": str(getattr(task, "inspection_mode", "window") or "window"), "modality": "visual"},
            images_requested=int(metadata.get("images_requested", 0) or 0),
            images_attached=int(metadata.get("images_attached", 0) or 0),
            images_dropped=int(metadata.get("images_dropped", 0) or 0),
            parse_status="failed",
            execution_status="failed",
            frame_refs=frame_refs,
            raw_output=str(exc),
            source_video_ids=source_video_ids,
        )
        return InvestigationReport(
            query_id=query_id,
            status="failed",
            attempts=(attempt,),
            cost={"tool_trace": ("image_attachment_failed",), "frames": 0, "vlm_calls": 0, "consumes_budget": False},
            failure_reason=str(exc),
        )


def _reasoner_prompt(kwargs: Mapping[str, Any]) -> str:
    final = bool(kwargs.get("force_finalize"))
    final_attempt = int(kwargs.get("final_attempt", 0) or 0)
    if not final:
        action_rule = "Choose exactly one action: investigate, read_observations, update_workspace, or answer."
    elif final_attempt <= 1:
        action_rule = (
            "Choose exactly one action: read_observations, update_workspace, or answer; investigate is closed. "
            "Use this call to read any needed existing observations and consolidate them with workspace_ops. "
            "Answer only with direct support; one non-investigation final call remains."
        )
    else:
        action_rule = (
            "Choose exactly one action: update_workspace or answer; investigate and read_observations are closed. "
            "This is the final call. Repair the rejected candidate against the existing Working Document without switching "
            "options merely to satisfy the gate. Answer only with direct support; otherwise use update_workspace."
        )
    return (
        "You are the sole semantic decision maker for long-video multiple-choice QA. The framework only stores observations, "
        "applies your Working Document operations, and validates references. It never judges claims, scores options, audits, "
        "or changes your answer.\n"
        f"{action_rule}\n"
        "Return one JSON object. Every action may include workspace_ops. Operation forms:\n"
        "{\"op\":\"add_claim\",\"claim\":{\"claim_id\":\"c1\",\"text\":\"...\","
        "\"source\":\"observation|derived|hypothesis\",\"cites\":[],\"derived_from\":[],"
        "\"time_anchor\":[0,1],\"status\":\"active|contested\",\"confidence\":\"high|medium|low\"}}; "
        "{\"op\":\"supersede\",\"claim_id\":\"c1\",\"superseded_by\":\"c2\"}; "
        "{\"op\":\"set_status\",\"claim_id\":\"c1\",\"status\":\"active|contested|retracted\"}; "
        "{\"op\":\"link_conflict\",\"claim_id\":\"c1\",\"other_claim_id\":\"c2\"}; "
        "{\"op\":\"note_interval\",\"time_range\":[0,1],\"label\":\"...\",\"claim_ids\":[\"c1\"]}; "
        "{\"op\":\"update_entity\",\"entity_id\":\"person_1\",\"description\":\"...\",\"aliases\":[]}.\n"
        "An observation claim must cite an attempt_id; a derived claim must name "
        "derived_from claim_ids. Keep uncertain interpretations as contested/hypothesis claims instead of deleting them.\n"
        "To fetch raw Investigator output, use action=read_observations and observation_requests with attempt_ids or time_range. "
        "To revisit exactly the same pixels, investigate with inspection_mode=arbitrate_observation and arbitration_attempt_id.\n"
        "Investigate schema: {\"action\":\"investigate\",\"tasks\":[{\"query_id\":\"r1_t1\","
        "\"goal\":\"observable question\","
        "\"segment_id\":\"seg_0001\",\"time_range\":null,\"inspection_mode\":\"window|search_asr|arbitrate_observation\","
        "\"search_terms\":[],\"arbitration_attempt_id\":\"\",\"expected_evidence\":\"direct observation\","
        "\"sampling_floor_fps\":0.5}],\"workspace_ops\":[]}. "
        "Use 0.5 fps for persistent states, 1 fps for ordinary motion, and 2 fps for brief transitions or changing text. "
        "When identity, ordering, color, text, or a brief transition remains uncertain or mismatches an option, inspect a "
        "narrow 2 fps visual window before answering; ASR cannot resolve visual attributes.\n"
        "Answer schema: {\"action\":\"answer\",\"answer\":\"A. exact option text\",\"workspace_ops\":[],"
        "\"supporting_claim_ids\":[\"c1\"],\"residual_uncertainty\":\"\"}. "
        "Never answer by closest match or add facts absent from the supporting observation lineage. If any selected-option "
        "detail is mismatched or unconfirmed, record it in residual_uncertainty instead of answering.\n"
        f"Question: {kwargs.get('question', '')}\n"
        f"Options: {json.dumps(kwargs.get('options') or {}, ensure_ascii=False)}\n"
        f"Remaining investigation budget: {int(kwargs.get('remaining_budget', 0) or 0)}\n"
        f"Mechanical status: {json.dumps(kwargs.get('mechanical_status') or {}, ensure_ascii=False)}\n"
        f"Working view:\n{kwargs.get('working_document_view', '')}\n"
        f"Workspace overview: {json.dumps(_prompt_overview(kwargs.get('workspace_overview') or {}), ensure_ascii=False)}"
    )


def _observation_prompt(
    workspace: VirtualVideoWorkspace,
    task: Any,
    window: Mapping[str, Any],
) -> str:
    frame_times = [float(row["virtual_time_sec"]) for row in window.get("frames", ())]
    metadata = {
        "virtual_time_range": window.get("virtual_time_range"),
        "frame_times_sec": frame_times,
        "sampling": window.get("sampling"),
        "asr_cues": window.get("asr_cues"),
        "source_lineage": window.get("source_lineage"),
    }
    return (
        "You are a visual Investigator. Report only what is directly visible or literally stated in the supplied local ASR. "
        "Do not select an answer option, evaluate a candidate claim, qualify an event, infer hidden intent, or decide whether "
        "the investigation succeeded. Preserve ambiguity explicitly. Return compact JSON only:\n"
        "{\"summary\":\"faithful overall description\",\"observations\":[{\"time_sec\":0.0,"
        "\"description\":\"direct observation\"}],\"entities\":[{\"name\":\"local label\","
        "\"description\":\"visible attributes\"}],\"events\":[{\"time_range\":[0.0,1.0],"
        "\"description\":\"visible change\"}],\"uncertainties\":[\"...\"]}.\n"
        "Use the supplied virtual timestamps. Keep relational context in full sentences; do not flatten ordering or roles into "
        "isolated labels. Empty arrays are valid.\n"
        f"Question context (navigation only): {workspace.case.question}\n"
        f"Observation goal: {getattr(task, 'goal', '')}\n"
        f"Expected visible material: {getattr(task, 'expected_evidence', '')}\n"
        f"Window metadata: {json.dumps(metadata, ensure_ascii=False)}"
    )


def _prompt_overview(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "workspace_id": value.get("workspace_id", ""),
        "workspace_duration_sec": value.get("workspace_duration_sec", 0.0),
        "overviews": [
            {
                "overview_id": row.get("overview_id", row.get("segment_id", "")),
                "segment_ids": list(row.get("segment_ids", ())) or [row.get("segment_id", "")],
                "virtual_time_range": row.get("virtual_time_range", ()),
                "asr_short_summary": str(row.get("asr_short_summary", "") or "")[:160],
            }
            for row in tuple(value.get("segment_overviews", ()) or ())
            if isinstance(row, Mapping)
        ],
        "available_tools": list(value.get("available_tools", ()) or ()),
        "available_navigation": list(value.get("available_navigation", ()) or ()),
    }


def _arbitration_prompt(task: Any, rows: Sequence[Mapping[str, Any]]) -> str:
    prior = [
        {
            "interpretation_id": row.get("interpretation_id"),
            "prompt_digest": row.get("prompt_digest"),
            "raw_output": row.get("raw_output"),
        }
        for row in rows[-4:]
    ]
    return (
        "You are re-inspecting exactly the same frames because prior interpretations may conflict. Compare the competing "
        "descriptions against the pixels, report the most precise direct observation, and preserve unresolved ambiguity. "
        "Do not vote between prior outputs and do not answer the multiple-choice question. Return the same observation-only "
        "JSON schema used for a normal inspection.\n"
        f"Contrastive goal: {getattr(task, 'goal', '')}\n"
        f"Prior interpretations: {json.dumps(prior, ensure_ascii=False)}"
    )


def _visual_evidence(
    *,
    workspace: VirtualVideoWorkspace,
    query_id: str,
    attempt_id: str,
    time_range: tuple[float, float],
    frame_paths: Sequence[str],
    fps: float,
    raw: str,
    parsed: Mapping[str, Any],
    source_lineage: Sequence[Mapping[str, Any]],
    model: str,
) -> EvidenceRecord:
    start_sec, end_sec = time_range
    return EvidenceRecord(
        evidence_id=f"ev_{query_id}_{attempt_id[-8:]}",
        beat_id="",
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        modality="visual",
        pointer=f"virtual://{workspace.workspace_id}/observations/{attempt_id}",
        verbatim=str(parsed.get("summary") or raw)[:1200],
        frame_refs=tuple(frame_paths),
        attestation_model=model,
        temporal_scope="window",
        evidence_kind="visual_observation",
        observation_polarity="unknown",
        sampling_coverage="sparse",
        request_ids=(query_id,),
        coverage_manifest=(CoverageSegment(query_id, float(start_sec), float(end_sec), "visual", 1.0),),
        task_id=query_id,
        observation_id=attempt_id,
        sampling_fps=float(fps),
        confidence=0.0,
        source_lineage=tuple(dict(item) for item in source_lineage),
        operation_metadata={"observation_payload": dict(parsed)},
    )


def _observation_rows(path: Path, attempt_id: str) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        return ()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, Mapping) and str(row.get("attempt_id", "")) == attempt_id:
            rows.append(dict(row))
    return tuple(rows)


def _attachment_counts(metadata: Mapping[str, Any], frame_paths: Sequence[str]) -> dict[str, int]:
    requested = int(metadata.get("images_requested", len(frame_paths)) or 0)
    attached = int(metadata.get("images_attached", len(frame_paths)) or 0)
    dropped = int(metadata.get("images_dropped", max(0, requested - attached)) or 0)
    return {"requested": requested, "attached": min(requested, attached), "dropped": dropped}
