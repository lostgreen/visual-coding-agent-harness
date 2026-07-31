from __future__ import annotations

from dataclasses import dataclass
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
_LOCATOR_PREFIX_RE = re.compile(
    r"^(?:find|locate|identify|determine|confirm|observe|verify|refine)\s+",
    re.IGNORECASE,
)
_QUERY_PART_RE = re.compile(r"\s+(?:and|then|after|before)\s+|[,;；]", re.IGNORECASE)
_TEMPORAL_CLAUSE_RE = re.compile(
    r"\b(?P<relation>after|before|when|while|until|once)\s+"
    r"(?P<clause>.+?)(?=,|\?|$)",
    re.IGNORECASE,
)
_FIRST_EVENT_RE = re.compile(
    r"^(?:the\s+)?first\s+(challenge|fight|encounter)\s+(?:against|with)\s+(.+)$",
    re.IGNORECASE,
)
_CHAPTER_BOUNDARY_RE = re.compile(r"\bchapter\b|章节|第.{0,4}[回章]", re.IGNORECASE)
_TARGET_EVENT_CUE_RE = re.compile(
    r"\b(?:challeng\w*|sparr\w*|fight\w*|battle\w*|duel\w*|faces?\s+off)\b",
    re.IGNORECASE,
)
_TEMPORAL_LOOKBACK_SEC = 7200.0


@dataclass(frozen=True)
class SearchFingerprint:
    modality: str
    normalized_terms: tuple[str, ...]
    time_range_bucket: tuple[int, int] | None
    index_version: str
    normalized_queries: tuple[str, ...] = ()
    segment_ids: tuple[str, ...] = ()
    source_video_ids: tuple[str, ...] = ()
    top_k: int = 0
    expand_neighbors: int = 0

    def similarity(self, other: "SearchFingerprint") -> float:
        if (
            self.modality != other.modality
            or self.time_range_bucket != other.time_range_bucket
            or self.index_version != other.index_version
            or self.segment_ids != other.segment_ids
            or self.source_video_ids != other.source_video_ids
            or self.top_k != other.top_k
            or self.expand_neighbors != other.expand_neighbors
        ):
            return 0.0
        if self.modality == "caption" and self.normalized_queries != other.normalized_queries:
            return 0.0
        left = set(self.normalized_terms)
        right = set(other.normalized_terms)
        union = left | right
        return len(left & right) / len(union) if union else 1.0


@dataclass(frozen=True)
class SearchOutcome:
    fingerprint: SearchFingerprint
    hit_count: int
    top_ids: tuple[str, ...]
    attempt_id: str


def _completion_budget(default: int) -> int:
    return max(4096, int(default))


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
    if mode not in {"window", "search_asr", "search_caption", "arbitrate_observation"}:
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
        coordinate_space=str(value.get("coordinate_space", "virtual") or "virtual"),
        source_video_ids=tuple(value.get("source_video_ids", ()) or ()),
        expected_evidence=str(value.get("expected_evidence", "") or goal),
        inspection_mode=mode,
        search_terms=tuple(value.get("search_terms", ()) or ()),
        caption_queries=tuple(value.get("caption_queries", value.get("queries", ())) or ()),
        top_k=int(value.get("top_k", 12) or 12),
        index_mode=str(value.get("index_mode", "lexical") or "lexical"),
        expand_neighbors=int(value.get("expand_neighbors", 0) or 0),
        sampling_floor_fps=value.get("sampling_floor_fps"),
        arbitration_attempt_id=str(value.get("arbitration_attempt_id", "") or ""),
        force_reinspect=bool(value.get("force_reinspect", False)),
    )
    if mode == "search_asr" and not task.search_terms:
        return None
    if mode == "search_caption" and not task.caption_queries:
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

    def __init__(
        self,
        workspace: VirtualVideoWorkspace,
        *,
        api: OpenAICompatibleClient,
        trace_path: Path,
        caption_embedding_adapter: Any | None = None,
        caption_index_mode: str | None = None,
        caption_config_digest: str | None = None,
        caption_query_strategy: str = "joint",
    ) -> None:
        super().__init__(
            workspace,
            caption_embedding_adapter=caption_embedding_adapter,
            caption_config_digest=caption_config_digest,
            caption_query_strategy=caption_query_strategy,
        )
        self.api = api
        self.trace_path = trace_path
        mode = str(caption_index_mode or "").strip().casefold()
        if mode and mode not in {"lexical", "dense", "hybrid"}:
            raise ValueError(f"unsupported caption_index_mode: {mode}")
        self.caption_index_mode = mode or None
        self.caption_query_strategy = self._caption_query_strategy
        self._seen_asr_attempt_ids: set[str] = set()
        self._seen_caption_attempt_ids: set[str] = set()
        self._search_outcomes: list[SearchOutcome] = []
        self._duplicate_search_count = 0
        self._empty_search_streak = 0
        self._saved_visual_calls = 0
        self._saved_visual_frames = 0
        self._visual_attempt_cache: dict[str, InvestigationReport] = {}

    def reset_run_state(self) -> None:
        super().reset_run_state()
        self._seen_asr_attempt_ids.clear()
        self._seen_caption_attempt_ids.clear()
        self._search_outcomes.clear()
        self._duplicate_search_count = 0
        self._empty_search_streak = 0
        self._saved_visual_calls = 0
        self._saved_visual_frames = 0
        self._visual_attempt_cache.clear()

    def mechanical_status(self) -> Mapping[str, Any]:
        zero_hits = [outcome for outcome in self._search_outcomes if outcome.hit_count == 0]
        return {
            "caption_query_strategy": self.caption_query_strategy,
            "empty_search_streak": self._empty_search_streak,
            "duplicate_search_count": self._duplicate_search_count,
            "previous_zero_hit_queries": [
                {
                    "modality": outcome.fingerprint.modality,
                    "terms": list(outcome.fingerprint.normalized_terms),
                    "time_range_bucket": list(outcome.fingerprint.time_range_bucket)
                    if outcome.fingerprint.time_range_bucket
                    else None,
                    "segment_ids": list(outcome.fingerprint.segment_ids),
                    "source_video_ids": list(outcome.fingerprint.source_video_ids),
                    "top_k": outcome.fingerprint.top_k,
                    "expand_neighbors": outcome.fingerprint.expand_neighbors,
                }
                for outcome in zero_hits[-6:]
            ],
            "saved_visual_calls": self._saved_visual_calls,
            "saved_visual_frames": self._saved_visual_frames,
        }

    def _investigate_task(self, task: Any) -> InvestigationReport:
        try:
            mode = str(getattr(task, "inspection_mode", "window") or "window")
            if mode == "search_asr":
                return self._search_asr(task)
            if mode == "search_caption":
                return self._search_caption(task)
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
        sampling_manifest = _sampling_manifest(
            (start_sec, end_sec),
            frame_times,
            requested_fps=fps,
        )
        observed_subranges = tuple(
            tuple(float(value) for value in interval)
            for interval in sampling_manifest["observed_subranges"]
        )
        source_lineage = tuple(dict(item) for item in window["source_lineage"])
        source_video_ids = tuple(
            dict.fromkeys(str(item.get("source_video_id", "") or "") for item in source_lineage)
        )
        attempt_id = stable_attempt_id(
            source_video_ids=source_video_ids,
            frame_times=frame_times,
            inspected_ranges=observed_subranges if frame_paths else (),
            sampling_fps=fps,
            modality="visual",
        )
        prompt = _observation_prompt(self.workspace, task, window)
        cache_key = prompt_digest(
            json.dumps(
                {
                    "model": self.api.model,
                    "prompt_digest": prompt_digest(prompt),
                    "frame_paths": list(frame_paths),
                },
                sort_keys=True,
            )
        )
        cached_report = self._visual_attempt_cache.get(cache_key)
        if cached_report is not None and not bool(getattr(task, "force_reinspect", False)):
            self._saved_visual_calls += 1
            self._saved_visual_frames += len(frame_paths)
            _append_jsonl(
                self.trace_path,
                {
                    "type": "investigator_observation_reused",
                    "query_id": query_id,
                    "attempt_ids": [attempt.attempt_id for attempt in cached_report.attempts],
                    "saved_frames": len(frame_paths),
                    "saved_calls": 1,
                    "time": time.time(),
                },
            )
            return InvestigationReport(
                query_id=query_id,
                status="completed",
                cost={
                    "tool_trace": ("inspect_window", "reuse_visual_attempt"),
                    "frames": 0,
                    "vlm_calls": 0,
                    "saved_frames": len(frame_paths),
                    "saved_calls": 1,
                    "reused": True,
                    "consumes_budget": False,
                },
                failure_reason="duplicate_visual_attempt_reused",
            )
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
            sampling_manifest=sampling_manifest,
        )
        attempt = ObservationAttempt(
            attempt_id=attempt_id,
            task_id=query_id,
            requested_range=(start_sec, end_sec),
            inspected_ranges=observed_subranges if counts["attached"] else (),
            attached_frame_times=frame_times if counts["attached"] else (),
            sampling_config={
                "fps": fps,
                "max_frames": frame_limit,
                "mode": "window",
                "modality": "visual",
                "sampling_manifest": sampling_manifest,
            },
            images_requested=counts["requested"],
            images_attached=counts["attached"],
            images_dropped=counts["dropped"],
            parse_status=parse_status,
            execution_status="completed" if frame_paths else "failed",
            frame_refs=frame_paths,
            modality="visual",
            evidence_role="candidate" if sampling_manifest["requires_refinement"] else "unclassified",
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
            self._record_visit(
                task,
                evidence,
                status="candidate_locator" if sampling_manifest["requires_refinement"] else "completed",
            )
        report = InvestigationReport(
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
                "requires_refinement": sampling_manifest["requires_refinement"],
            },
            failure_reason="no frames materialized" if not frame_paths else "",
            coverage_delta=observed_subranges if frame_paths else (),
        )
        if frame_paths:
            self._visual_attempt_cache[cache_key] = report
        return report

    def _search_asr(self, task: Any) -> InvestigationReport:
        query_id = str(getattr(task, "query_id", "") or "search_asr")
        terms = tuple(getattr(task, "search_terms", ()) or ())
        segment_id = str(getattr(task, "segment_id", "") or "")
        time_range = getattr(task, "time_range", None)
        search_fingerprint = _search_fingerprint(
            "asr",
            terms,
            time_range,
            index_version=f"literal-asr-v1:{segment_id or '*'}",
            segment_ids=(segment_id,) if segment_id else (),
            top_k=8,
        )
        cached = self._cached_search(search_fingerprint)
        if cached is not None:
            return self._reused_search_report(query_id, cached, modality="asr")
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
            self._record_search_outcome(
                SearchOutcome(
                    fingerprint=search_fingerprint,
                    hit_count=len(clusters),
                    top_ids=tuple(
                        f"{row['segment_id']}:{float(row['virtual_time_range'][0]):.3f}"
                        for row in clusters[:8]
                    ),
                    attempt_id=attempt_id,
                )
            )
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
                "hit_count": len(clusters),
            },
            parse_status="deterministic",
            execution_status="completed",
            modality="asr",
            evidence_role="candidate",
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
        self._record_search_outcome(
            SearchOutcome(
                fingerprint=search_fingerprint,
                hit_count=len(clusters),
                top_ids=tuple(
                    f"{row['segment_id']}:{float(row['virtual_time_range'][0]):.3f}"
                    for row in clusters[:8]
                ),
                attempt_id=attempt_id,
            )
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

    def _search_caption(self, task: Any) -> InvestigationReport:
        query_id = str(getattr(task, "query_id", "") or "search_caption")
        goal_query = str(getattr(task, "goal", "") or "").strip()
        requested_queries = tuple(getattr(task, "caption_queries", ()) or ())
        if self.caption_query_strategy == "rema":
            queries = _rema_caption_queries(
                goal_query,
                requested_queries,
                fallback=self.workspace.case.question,
            )
        else:
            query_candidates = (self.workspace.case.question, *requested_queries)
            queries = tuple(
                dict.fromkeys(
                    str(query).strip()
                    for query in query_candidates
                    if str(query).strip()
                )
            )[:5]
        time_range = getattr(task, "time_range", None)
        segment_id = str(getattr(task, "segment_id", "") or "")
        requested_source_video_ids = tuple(getattr(task, "source_video_ids", ()) or ())
        requested_top_k = int(getattr(task, "top_k", 12) or 12)
        top_k = requested_top_k
        if self.caption_query_strategy == "rema":
            # Round-robin fusion needs enough shared slots to reach beyond each
            # query's first two hits. Only the automatic expansion is capped.
            top_k = max(requested_top_k, min(20, len(queries) * 2 + 2))
        expand_neighbors = int(getattr(task, "expand_neighbors", 0) or 0)
        index_mode = self.caption_index_mode or str(getattr(task, "index_mode", "lexical") or "lexical")
        search_fingerprint = _search_fingerprint(
            "caption",
            queries,
            time_range,
            index_version=f"{index_mode}:{self.caption_query_strategy}",
            segment_ids=(segment_id,) if segment_id else (),
            source_video_ids=requested_source_video_ids,
            top_k=top_k,
            expand_neighbors=expand_neighbors,
        )
        cached = self._cached_search(search_fingerprint)
        if cached is not None:
            return self._reused_search_report(query_id, cached, modality="caption")
        try:
            packet = self.search_caption(
                queries,
                time_range=time_range,
                segment_ids=(segment_id,) if segment_id else (),
                source_video_ids=requested_source_video_ids,
                top_k=top_k,
                expand_neighbors=expand_neighbors,
                index_mode=index_mode,
            )
            temporal_locator = self._chapter_temporal_locator(
                packet,
                time_range=time_range,
                segment_id=segment_id,
                source_video_ids=requested_source_video_ids,
                index_mode=index_mode,
            )
            if temporal_locator:
                packet = {**packet, "temporal_locator": temporal_locator}
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return InvestigationReport(
                query_id=query_id,
                status="failed",
                cost={"tool_trace": ("search_caption",), "consumes_budget": False},
                failure_reason=str(exc),
            )
        hits = tuple(dict(hit) for hit in packet["hits"])
        query_fingerprint = str(packet["query_fingerprint"])
        pointer = (
            f"caption-search://{self.workspace.workspace_id}/"
            f"{str(packet['index_digest'])[:12]}/{query_fingerprint[:20]}"
        )
        scoped_segment_ids = set(packet.get("segment_ids", ()) or ())
        source_video_ids = tuple(packet.get("source_video_ids", ()) or ()) or tuple(
            dict.fromkeys(
                segment.source_video_id
                for segment in self.workspace.manifest.segments
                if not scoped_segment_ids or segment.segment_id in scoped_segment_ids
            )
        )
        attempt_id = stable_attempt_id(
            source_video_ids=source_video_ids,
            frame_refs=(pointer,),
            modality="caption_search",
        )
        duplicate = (
            attempt_id in self._seen_caption_attempt_ids
            or bool(_observation_rows(self.workspace.root_dir / "observation_log.jsonl", attempt_id))
        )
        self._seen_caption_attempt_ids.add(attempt_id)
        if duplicate:
            self._record_search_outcome(
                SearchOutcome(
                    fingerprint=search_fingerprint,
                    hit_count=len(hits),
                    top_ids=tuple(str(hit.get("passage_id", "")) for hit in hits[:8]),
                    attempt_id=attempt_id,
                )
            )
            return InvestigationReport(
                query_id=query_id,
                status="completed",
                cost={
                    "tool_trace": ("search_caption", "reuse_attempt"),
                    "frames": 0,
                    "vlm_calls": 0,
                    "reused": True,
                    "zero_hits": not hits,
                    "consumes_budget": False,
                },
                failure_reason="duplicate_caption_query_reused",
            )
        raw_path = self.workspace.root_dir / "caption_search" / f"{attempt_id}.json"
        _write_json(raw_path, packet)
        raw_output = json.dumps(
            {
                "raw_output_pointer": str(raw_path),
                "summary": packet["rendered"],
                "search_queries": list(queries),
                "hits": hits,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        attempt = ObservationAttempt(
            attempt_id=attempt_id,
            task_id=query_id,
            requested_range=time_range,
            inspected_ranges=(),
            sampling_config={
                "mode": "search_caption",
                "modality": "caption_search",
                "queries": list(queries),
                "top_k": top_k,
                **(
                    {"requested_top_k": requested_top_k}
                    if requested_top_k != top_k
                    else {}
                ),
                "time_range": list(time_range) if time_range else None,
                "segment_ids": list(packet.get("segment_ids", ()) or ()),
                "source_video_ids": list(source_video_ids),
                "index_mode": index_mode,
                "query_strategy": str(packet.get("query_strategy", self.caption_query_strategy)),
                "index_digest": packet["index_digest"],
                "query_fingerprint": query_fingerprint,
                "expand_neighbors": expand_neighbors,
                "hits": [
                    {
                        "passage_id": hit["passage_id"],
                        "range": [hit["virtual_start_sec"], hit["virtual_end_sec"]],
                        "score": hit["fused_score"],
                        "query_matches": list(
                            dict(hit.get("metadata") or {}).get("query_matches", ())
                            or dict(hit.get("metadata") or {}).get("matched_queries", ())
                        ),
                        "caption_excerpt": str(hit.get("text", ""))[:240],
                    }
                    for hit in hits
                ],
                **(
                    {"temporal_locator": dict(packet["temporal_locator"])}
                    if isinstance(packet.get("temporal_locator"), Mapping)
                    else {}
                ),
            },
            parse_status="deterministic",
            execution_status="completed",
            frame_refs=(pointer,),
            modality="caption_search",
            evidence_role="candidate",
            prompt_digest=prompt_digest(json.dumps(list(queries), ensure_ascii=False)),
            raw_output=raw_output,
            source_video_ids=source_video_ids,
        )
        _append_jsonl(
            self.trace_path,
            {
                "type": "investigator_caption_search",
                "query_id": query_id,
                "attempt_id": attempt_id,
                "query_fingerprint": query_fingerprint,
                "index_digest": packet["index_digest"],
                "query_strategy": str(packet.get("query_strategy", self.caption_query_strategy)),
                "hit_count": len(hits),
                "segment_ids": list(packet.get("segment_ids", ()) or ()),
                "source_video_ids": list(source_video_ids),
                "raw_output_pointer": str(raw_path),
                "reused": False,
                "time": time.time(),
            },
        )
        self._record_search_outcome(
            SearchOutcome(
                fingerprint=search_fingerprint,
                hit_count=len(hits),
                top_ids=tuple(str(hit.get("passage_id", "")) for hit in hits[:8]),
                attempt_id=attempt_id,
            )
        )
        return InvestigationReport(
            query_id=query_id,
            status="completed",
            attempts=(attempt,),
            cost={
                "tool_trace": ("search_caption",),
                "frames": 0,
                "vlm_calls": 0,
                "reused": False,
                "zero_hits": not hits,
                "consumes_budget": bool(hits),
            },
            failure_reason="" if hits else "caption_zero_hits_refine_query_or_use_visual",
        )

    def _chapter_temporal_locator(
        self,
        packet: Mapping[str, Any],
        *,
        time_range: Sequence[float] | None,
        segment_id: str,
        source_video_ids: Sequence[str],
        index_mode: str,
    ) -> dict[str, Any]:
        if self.caption_query_strategy != "rema" or time_range is not None or segment_id:
            return {}
        contract = _temporal_caption_contract(self.workspace.case.question)
        if contract is None or contract["scope_kind"] != "chapter":
            return {}
        target_queries = {
            " ".join(str(query).casefold().split())
            for query in tuple(contract["target_queries"])
        }
        target_hits = []
        for raw_hit in tuple(packet.get("hits", ()) or ()):
            if not isinstance(raw_hit, Mapping):
                continue
            metadata = raw_hit.get("metadata")
            matches = tuple(metadata.get("query_matches", ()) or ()) if isinstance(metadata, Mapping) else ()
            if any(
                " ".join(str(match.get("query", "")).casefold().split()) in target_queries
                for match in matches
                if isinstance(match, Mapping)
            ) and _caption_describes_target_event(
                str(raw_hit.get("text", "") or ""),
                str(contract["target_event_kind"]),
            ):
                target_hits.append(dict(raw_hit))
        if not target_hits:
            return {}

        anchor_queries = (str(contract["scope_query"]), "a chapter title appears")
        pairs: list[dict[str, Any]] = []
        auxiliary_search_count = 0
        for target_hit in target_hits[:8]:
            target_start = float(target_hit.get("virtual_start_sec", 0.0) or 0.0)
            if target_start <= 0.0:
                continue
            try:
                anchor_packet = self.search_caption(
                    anchor_queries,
                    time_range=(max(0.0, target_start - _TEMPORAL_LOOKBACK_SEC), target_start),
                    source_video_ids=source_video_ids,
                    top_k=6,
                    expand_neighbors=0,
                    index_mode=index_mode,
                )
            except (FileNotFoundError, RuntimeError, ValueError):
                continue
            auxiliary_search_count += 1
            anchors = [
                dict(raw_anchor)
                for raw_anchor in tuple(anchor_packet.get("hits", ()) or ())
                if isinstance(raw_anchor, Mapping)
                and float(raw_anchor.get("virtual_end_sec", 0.0) or 0.0) <= target_start
                and _CHAPTER_BOUNDARY_RE.search(str(raw_anchor.get("text", "") or ""))
            ]
            if not anchors:
                continue
            anchor = max(
                anchors,
                key=lambda item: float(item.get("virtual_end_sec", 0.0) or 0.0),
            )
            target_inspection_start = max(
                float(anchor.get("virtual_end_sec", 0.0) or 0.0),
                target_start - 90.0,
            )
            target_inspection_end = min(
                target_start,
                max(target_inspection_start + 1.0, target_start - 25.0),
            )
            if target_inspection_end <= target_inspection_start:
                continue
            pairs.append(
                {
                    "scope_anchor": _caption_locator_hit(anchor),
                    "target_event": _caption_locator_hit(target_hit),
                    "inspection_range": [target_inspection_start, target_inspection_end],
                }
            )
        if not pairs:
            return {}

        grouped: dict[str, list[dict[str, Any]]] = {}
        for pair in pairs:
            anchor = pair["scope_anchor"]
            anchor_key = str(anchor.get("passage_id", "")) or json.dumps(
                anchor.get("time_range", ()), separators=(",", ":")
            )
            grouped.setdefault(anchor_key, []).append(pair)
        candidate_groups = []
        for group_pairs in grouped.values():
            selected = min(
                group_pairs,
                key=lambda item: float(item["target_event"]["time_range"][0]),
            )
            candidate_groups.append(
                {
                    **selected,
                    "target_candidate_count": len(group_pairs),
                    "selection_reason": "earliest target candidate after the shared chapter boundary",
                }
            )
        candidate_groups.sort(
            key=lambda item: (
                -int(item["target_candidate_count"]),
                float(item["target_event"]["time_range"][0]),
            )
        )
        recommended = None
        if len(candidate_groups) == 1 or (
            int(candidate_groups[0]["target_candidate_count"])
            > int(candidate_groups[1]["target_candidate_count"])
        ):
            recommended = candidate_groups[0]
        return {
            "schema_version": "TemporalCaptionLocatorV1",
            "contract": dict(contract),
            "anchor_queries": list(anchor_queries),
            "candidate_groups": candidate_groups[:4],
            "recommended": dict(recommended) if recommended is not None else None,
            "auxiliary_search_count": auxiliary_search_count,
        }

    def _cached_search(self, fingerprint: SearchFingerprint) -> SearchOutcome | None:
        for outcome in reversed(self._search_outcomes):
            if fingerprint.similarity(outcome.fingerprint) >= 0.85:
                self._duplicate_search_count += 1
                self._empty_search_streak = (
                    self._empty_search_streak + 1 if outcome.hit_count == 0 else 0
                )
                return outcome
        return None

    def _record_search_outcome(self, outcome: SearchOutcome) -> None:
        self._search_outcomes.append(outcome)
        self._empty_search_streak = self._empty_search_streak + 1 if outcome.hit_count == 0 else 0

    def _reused_search_report(
        self,
        query_id: str,
        outcome: SearchOutcome,
        *,
        modality: str,
    ) -> InvestigationReport:
        _append_jsonl(
            self.trace_path,
            {
                "type": f"investigator_{modality}_search",
                "query_id": query_id,
                "attempt_id": outcome.attempt_id,
                "hit_count": outcome.hit_count,
                "reused": True,
                "near_duplicate": True,
                "time": time.time(),
            },
        )
        return InvestigationReport(
            query_id=query_id,
            status="completed",
            cost={
                "tool_trace": (f"search_{modality}", "reuse_search_outcome"),
                "frames": 0,
                "vlm_calls": 0,
                "reused": True,
                "zero_hits": outcome.hit_count == 0,
                "consumes_budget": False,
            },
            failure_reason=f"near_duplicate_{modality}_query_reused",
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
        raw_requested_range = tuple(first.get("requested_range", ()) or ())
        start_sec, end_sec = (
            tuple(float(value) for value in raw_requested_range)
            if len(raw_requested_range) == 2
            else time_ranges[0]
            if time_ranges
            else (0.0, 0.0)
        )
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
        stored_sampling_config = first.get("sampling_config")
        stored_sampling_manifest = (
            stored_sampling_config.get("sampling_manifest")
            if isinstance(stored_sampling_config, Mapping)
            else None
        )
        sampling_manifest = (
            dict(stored_sampling_manifest)
            if isinstance(stored_sampling_manifest, Mapping)
            else _sampling_manifest(
                (float(start_sec), float(end_sec)),
                frame_times,
                requested_fps=sampling_fps,
            )
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
            sampling_manifest=sampling_manifest,
        )
        attempt = ObservationAttempt(
            attempt_id=attempt_id,
            task_id=query_id,
            requested_range=(float(start_sec), float(end_sec)),
            inspected_ranges=time_ranges,
            attached_frame_times=frame_times,
            sampling_config={
                "fps": sampling_fps,
                "mode": "arbitrate_observation",
                "modality": "visual",
                "sampling_manifest": sampling_manifest,
            },
            images_requested=counts["requested"],
            images_attached=counts["attached"],
            images_dropped=counts["dropped"],
            parse_status="parsed" if parsed else "failed",
            execution_status="completed",
            frame_refs=frame_paths,
            modality="visual",
            evidence_role="candidate" if sampling_manifest.get("requires_refinement") else "unclassified",
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
    options = dict(kwargs.get("options") or {})
    mechanical_status = dict(kwargs.get("mechanical_status") or {})
    caption_query_strategy = str(
        mechanical_status.get("caption_query_strategy", "joint") or "joint"
    ).casefold()
    if caption_query_strategy == "rema":
        caption_search_rule = (
            "When available, search_caption is a locator using ReMA-style independent multi-query retrieval. Provide "
            "a self-contained observable event or relation as the task goal, then caption_queries as short complementary "
            "entity names or aliases; do not repeat the full question. Preserve entity spellings from the question and do "
            "not invent translations. For temporal questions, put each anchor in a separate subject-verb-object phrase. "
            "Include all scope anchors and the decisive target event in the first caption search instead of spending separate "
            "rounds on them. A chapter, location, or earlier event is usually a scope constraint; prioritize inspecting the "
            "candidate for the event or state that directly yields the requested answer. "
            "Each query receives balanced retrieval coverage within the shared top_k result budget. You may also "
            "set optional time_range, segment_id/source_video_ids, and index_mode=hybrid. Treat returned ranges as candidates "
            "and inspect decisive claims visually. "
        )
    else:
        caption_search_rule = (
            "When available, search_caption is a locator: the framework always includes the original question, and you may "
            "add caption_queries (1-4 complementary formulations), top_k, optional time_range, segment_id/source_video_ids, "
            "and index_mode=lexical, dense, or hybrid when configured. Treat returned ranges as candidates and inspect "
            "decisive claims visually. "
        )
    if options:
        task_description = "long-video multiple-choice QA"
        answer_schema = (
            'Answer schema: {"action":"answer","answer":"A. exact option text","workspace_ops":[],'
            '"supporting_claim_ids":["c1"],"residual_uncertainty":""}. '
        )
        answer_rule = (
            "Never answer by closest match or add facts absent from the supporting observation lineage. If any selected-option "
            "detail is mismatched or unconfirmed, record it in residual_uncertainty instead of answering.\n"
        )
    else:
        task_description = "long-video free-form QA"
        answer_schema = (
            'Answer schema: {"action":"answer","answer":"concise factual answer","workspace_ops":[],'
            '"supporting_claim_ids":["c1"],"residual_uncertainty":"optional concise caveat"}. '
        )
        answer_rule = (
            "Return a short direct answer grounded in the supporting observation lineage. Free-form answers may retain a "
            "concise residual_uncertainty; do not invent details absent from the cited observations.\n"
        )
    if not final:
        action_rule = "Choose exactly one action: investigate, read_observations, update_workspace, or answer."
    elif final_attempt <= 1:
        action_rule = (
            "Choose exactly one action: read_observations, update_workspace, or answer; investigate is closed. "
            "Use this call to read any needed existing observations and consolidate them with workspace_ops. "
            "Answer only with direct support; one non-investigation final call remains if validation fails."
        )
    else:
        action_rule = (
            "Return action=answer only. Tool use, observation reads, and workspace-only updates are closed. "
            "Provide the best evidence-grounded answer available and list supporting_claim_ids when valid."
        )
    return (
        f"You are the sole semantic decision maker for {task_description}. The framework only stores observations, "
        "applies your Working Document operations, and validates references. It never judges claims, scores options, audits, "
        "or changes your answer.\n"
        f"{action_rule}\n"
        "Return one JSON object. Every action may include workspace_ops. Operation forms:\n"
        "{\"op\":\"add_claim\",\"claim\":{\"claim_id\":\"c1\",\"text\":\"...\","
        "\"source\":\"observation|derived|hypothesis\",\"cites\":[],\"derived_from\":[],"
        "\"time_anchor\":[0,1],\"status\":\"active|contested\",\"confidence\":\"high|medium|low\","
        "\"entity_ids\":[],\"metadata\":{}}}; "
        "{\"op\":\"supersede\",\"claim_id\":\"c1\",\"superseded_by\":\"c2\"}; "
        "{\"op\":\"set_status\",\"claim_id\":\"c1\",\"status\":\"active|contested|retracted\"}; "
        "{\"op\":\"link_conflict\",\"claim_id\":\"c1\",\"other_claim_id\":\"c2\"}; "
        "{\"op\":\"note_interval\",\"time_range\":[0,1],\"label\":\"...\",\"claim_ids\":[\"c1\"],"
        "\"role\":\"candidate|supporting|negative\",\"metadata\":{}}; "
        "{\"op\":\"update_entity\",\"entity_id\":\"person_1\",\"description\":\"...\",\"aliases\":[]}.\n"
        "An observation claim must cite an attempt_id; a derived claim must name "
        "derived_from claim_ids. Keep uncertain interpretations as contested/hypothesis claims instead of deleting them.\n"
        "To fetch raw Investigator output, use action=read_observations and observation_requests with attempt_ids or time_range. "
        "To revisit exactly the same pixels, investigate with inspection_mode=arbitrate_observation and arbitration_attempt_id.\n"
        f"{caption_search_rule}"
        "When mechanical_status lists pending_caption_candidates, select the candidate whose query_matches and "
        "caption_excerpt best cover the unresolved condition, then inspect its time_range with inspection_mode=window; "
        "rank is retrieval priority, not proof. Caption_search and search_asr attempts cannot directly support an answer.\n"
        "When recommended_temporal_candidate is present, it is a locator-only join for an explicit after/before/first "
        "condition. Inspect its full inspection_range before unrelated Caption candidates; do not replace it with only "
        "the last seconds adjacent to the target event. Then visually verify the target identity, ordering, and requested "
        "state; the recommendation itself is not answer evidence.\n"
        "Investigate schema: {\"action\":\"investigate\",\"tasks\":[{\"query_id\":\"r1_t1\","
        "\"goal\":\"observable question\","
        "\"segment_id\":\"seg_0001\",\"time_range\":null,\"coordinate_space\":\"virtual|segment_local\","
        "\"source_video_ids\":[],"
        "\"inspection_mode\":\"window|search_asr|search_caption|arbitrate_observation\","
        "\"search_terms\":[],\"caption_queries\":[],\"top_k\":12,\"index_mode\":\"lexical|dense|hybrid\","
        "\"expand_neighbors\":0,\"arbitration_attempt_id\":\"\",\"force_reinspect\":false,"
        "\"expected_evidence\":\"direct observation\","
        "\"sampling_floor_fps\":0.5}],\"workspace_ops\":[]}. "
        "time_range defaults to virtual workspace seconds. segment_local requires a known segment_id and is converted with "
        "an explicit trace; a virtual range outside its named segment is rejected rather than remapped. "
        "Use 0.5 fps for persistent states, 1 fps for ordinary motion, and 2 fps for brief transitions or changing text. "
        "When identity, ordering, color, text, or a brief transition remains uncertain or mismatches an option, inspect a "
        "narrow 2 fps visual window before answering; ASR cannot resolve visual attributes.\n"
        f"{answer_schema}"
        f"{answer_rule}"
        f"Question: {kwargs.get('question', '')}\n"
        f"Options: {json.dumps(options, ensure_ascii=False)}\n"
        f"Remaining investigation budget: {int(kwargs.get('remaining_budget', 0) or 0)}\n"
        f"Mechanical status: {json.dumps(mechanical_status, ensure_ascii=False)}\n"
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


def _sampling_manifest(
    requested_range: tuple[float, float],
    frame_times: Sequence[float],
    *,
    requested_fps: float,
) -> dict[str, Any]:
    start_sec, end_sec = sorted((float(requested_range[0]), float(requested_range[1])))
    duration_sec = max(0.0, end_sec - start_sec)
    times = tuple(
        sorted(
            {
                min(end_sec, max(start_sec, float(value)))
                for value in frame_times
            }
        )
    )
    fps = max(1e-6, float(requested_fps or 0.0))
    half_width = 0.5 / fps
    observed = _merge_sampling_ranges(
        tuple(
            (
                max(start_sec, frame_time - half_width),
                min(end_sec, frame_time + half_width),
            )
            for frame_time in times
        )
    )
    covered_sec = sum(max(0.0, end - start) for start, end in observed)
    boundary_points = (start_sec, *times, end_sec)
    max_gap = max(
        (right - left for left, right in zip(boundary_points, boundary_points[1:])),
        default=duration_sec,
    )
    return {
        "requested_range": [round(start_sec, 6), round(end_sec, 6)],
        "frame_times": [round(value, 6) for value in times],
        "observed_subranges": [
            [round(range_start, 6), round(range_end, 6)]
            for range_start, range_end in observed
        ],
        "effective_fps": round(len(times) / duration_sec, 6) if duration_sec else 0.0,
        "max_gap": round(max_gap, 6),
        "coverage_ratio": round(covered_sec / duration_sec, 6) if duration_sec else 0.0,
        "requires_refinement": duration_sec > 120.0,
    }


def _merge_sampling_ranges(
    ranges: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    merged: list[list[float]] = []
    for start_sec, end_sec in sorted(ranges):
        if end_sec <= start_sec:
            continue
        if not merged or start_sec > merged[-1][1] + 1e-9:
            merged.append([start_sec, end_sec])
        else:
            merged[-1][1] = max(merged[-1][1], end_sec)
    return tuple((start_sec, end_sec) for start_sec, end_sec in merged)


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
    sampling_manifest: Mapping[str, Any],
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
        sampling_coverage=(
            "dense"
            if float(sampling_manifest.get("coverage_ratio", 0.0) or 0.0) >= 0.98
            and not bool(sampling_manifest.get("requires_refinement"))
            else "sparse"
        ),
        request_ids=(query_id,),
        coverage_manifest=tuple(
            CoverageSegment(
                f"{query_id}:{index:03d}",
                float(interval[0]),
                float(interval[1]),
                "visual",
                1.0,
            )
            for index, interval in enumerate(
                tuple(sampling_manifest.get("observed_subranges", ()) or ()),
                start=1,
            )
        ),
        task_id=query_id,
        observation_id=attempt_id,
        sampling_fps=float(fps),
        confidence=0.0,
        source_lineage=tuple(dict(item) for item in source_lineage),
        operation_metadata={
            "observation_payload": dict(parsed),
            "sampling_manifest": dict(sampling_manifest),
        },
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


def _rema_caption_queries(
    goal: str,
    requested_queries: Sequence[str],
    *,
    fallback: str,
) -> tuple[str, ...]:
    goal_parts = tuple(
        cleaned
        for raw in _QUERY_PART_RE.split(str(goal or ""))
        if (cleaned := _LOCATOR_PREFIX_RE.sub("", raw).strip(" .,:;"))
    )
    question_parts = tuple(
        match.group("clause").strip(" .,:;")
        for match in _TEMPORAL_CLAUSE_RE.finditer(str(fallback or ""))
        if match.group("clause").strip(" .,:;")
    )

    def relational(parts: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            part
            for part in parts
            if len(
                re.findall(
                    r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
                    part.casefold(),
                )
            )
            >= 4
        )

    candidates = (
        *_first_event_variants(question_parts),
        *relational(question_parts),
        *_first_event_variants(goal_parts),
        *relational(goal_parts),
        *(str(query).strip() for query in requested_queries),
        *goal_parts,
        _LOCATOR_PREFIX_RE.sub("", str(goal or "")).strip(" .,:;"),
    )
    normalized = tuple(dict.fromkeys(query for query in candidates if query))[:5]
    return normalized or (str(fallback or "").strip(),)


def _temporal_caption_contract(question: str) -> dict[str, Any] | None:
    clauses = tuple(
        (
            match.group("relation").casefold(),
            match.group("clause").strip(" .,:;"),
        )
        for match in _TEMPORAL_CLAUSE_RE.finditer(str(question or ""))
        if match.group("clause").strip(" .,:;")
    )
    scope_query = next((clause for relation, clause in clauses if relation == "after"), "")
    target_query = next((clause for relation, clause in clauses if relation == "before"), "")
    event_match = _FIRST_EVENT_RE.match(target_query)
    if not scope_query or event_match is None:
        return None
    scope_kind = "chapter" if "chapter" in scope_query.casefold() else "event"
    target_queries = tuple(dict.fromkeys((*_first_event_variants((target_query,)), target_query)))
    return {
        "scope_relation": "after",
        "scope_query": scope_query,
        "scope_kind": scope_kind,
        "target_relation": "before",
        "target_query": target_query,
        "target_queries": list(target_queries),
        "target_event_kind": event_match.group(1).casefold(),
        "selection": "first_target_after_scope",
    }


def _first_event_variants(parts: Sequence[str]) -> tuple[str, ...]:
    rewritten: list[str] = []
    for part in parts:
        match = _FIRST_EVENT_RE.match(part)
        if match is None:
            continue
        event, entity = match.groups()
        verb = {
            "challenge": "challenges",
            "fight": "fights",
            "encounter": "encounters",
        }[event.casefold()]
        rewritten.append(f"the player first {verb} {entity.strip(' .,:;')}")
    return tuple(rewritten)


def _caption_locator_hit(hit: Mapping[str, Any]) -> dict[str, Any]:
    metadata = hit.get("metadata")
    return {
        "passage_id": str(hit.get("passage_id", "")),
        "time_range": [
            float(hit.get("virtual_start_sec", 0.0) or 0.0),
            float(hit.get("virtual_end_sec", 0.0) or 0.0),
        ],
        "caption_excerpt": str(hit.get("text", "") or "")[:200],
        "query_matches": list(metadata.get("query_matches", ()) or ())
        if isinstance(metadata, Mapping)
        else [],
    }


def _caption_describes_target_event(text: str, event_kind: str) -> bool:
    if str(event_kind).casefold() in {"challenge", "fight"}:
        return bool(_TARGET_EVENT_CUE_RE.search(str(text or "")))
    return True


def _search_fingerprint(
    modality: str,
    queries: Sequence[str],
    time_range: Sequence[float] | None,
    *,
    index_version: str,
    segment_ids: Sequence[str] = (),
    source_video_ids: Sequence[str] = (),
    top_k: int = 0,
    expand_neighbors: int = 0,
    bucket_sec: int = 300,
) -> SearchFingerprint:
    normalized_queries = tuple(
        dict.fromkeys(" ".join(str(query).casefold().split()) for query in queries if str(query).strip())
    )
    tokens = tuple(
        sorted(
            {
                token
                for query in queries
                for token in re.findall(
                    r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
                    str(query).casefold(),
                )
                if token
            }
        )
    )
    normalized_range = None
    if time_range is not None and len(time_range) == 2:
        start, end = sorted((float(time_range[0]), float(time_range[1])))
        normalized_range = (
            int(start // max(1, int(bucket_sec))),
            int(max(start, end - 1e-9) // max(1, int(bucket_sec))),
        )
    return SearchFingerprint(
        modality=str(modality).casefold(),
        normalized_terms=tokens,
        time_range_bucket=normalized_range,
        index_version=str(index_version),
        normalized_queries=normalized_queries,
        segment_ids=tuple(sorted({str(item).strip() for item in segment_ids if str(item).strip()})),
        source_video_ids=tuple(
            sorted({str(item).strip() for item in source_video_ids if str(item).strip()})
        ),
        top_k=max(0, int(top_k)),
        expand_neighbors=max(0, int(expand_neighbors)),
    )
