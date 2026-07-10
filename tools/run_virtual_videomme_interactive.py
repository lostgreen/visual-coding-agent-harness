#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Mapping, Sequence

import requests
import yaml

from vcah.investigator import (
    InvestigationReport,
    VirtualVideoInvestigator,
    _choose_window_from_segment_packet,
    _needs_highfps,
)
from vcah.multiround import InvestigationTask, ReasonerDecision, VirtualVideoMultiRoundDriver
from vcah.types import CoverageSegment, EvidenceRecord
from vcah.video import probe_duration
from vcah.virtual_index import build_virtual_beat_index
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
    load_srt_as_virtual_cues,
    materialize_lowfps_frame_cache,
    select_uniform_items,
)


DEFAULT_CASE_IDS = ("477-2", "548-1", "371-1", "311-1", "314-3", "315-1")
LONG_INTERLEAVED_CASE_IDS = ("606-3", "698-3", "701-3", "702-1")


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    api = OpenAICompatibleVisionClient.from_yaml(Path(args.config))
    case_ids = tuple(args.case_ids or DEFAULT_CASE_IDS)
    if args.mode == "long":
        case_ids = tuple(args.case_ids or LONG_INTERLEAVED_CASE_IDS)
        if args.construction == "single_segment":
            args.construction = "interleaved_chunks"
        if float(args.min_duration_sec) == 18000.0:
            args.min_duration_sec = 21600.0
        if args.max_duration_sec is None:
            args.max_duration_sec = 25200.0
    selected = case_ids[:1] if args.mode == "smoke" else case_ids
    summaries = []
    for case_id in selected:
        workspace = build_or_load_workspace(
            dataset_root,
            out_root / "workspaces" / case_id,
            case_id=case_id,
            seed=int(args.seed),
            min_duration_sec=float(args.min_duration_sec),
            max_duration_sec=None if args.max_duration_sec is None else float(args.max_duration_sec),
            segment_sec=float(args.segment_sec),
            construction=str(args.construction),
            chunk_sec=float(args.chunk_sec),
            rebuild=bool(args.rebuild),
        )
        ensure_index(workspace, low_fps=float(args.low_fps), beat_sec=float(args.beat_sec), rebuild=bool(args.rebuild_index))
        result = run_case(workspace, api=api, max_rounds=int(args.max_rounds), max_investigations=int(args.max_investigations))
        summaries.append(
            {
                "case_id": result.case_id,
                "answer": result.answer,
                "citations": list(result.citations),
                "correct": result.correct,
                "rounds": result.rounds,
                "accepted_investigations": result.accepted_investigations,
                "workspace": str(workspace.root_dir),
                "trace": str(workspace.root_dir / "interactions.jsonl"),
            }
        )
    payload = {
        "mode": args.mode,
        "case_count": len(summaries),
        "correct": sum(1 for item in summaries if item["correct"]),
        "cases": summaries,
    }
    (out_root / f"{args.mode}_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def build_or_load_workspace(
    dataset_root: Path,
    root_dir: Path,
    *,
    case_id: str,
    seed: int,
    min_duration_sec: float,
    max_duration_sec: float | None,
    segment_sec: float,
    construction: str,
    chunk_sec: float,
    rebuild: bool,
) -> VirtualVideoWorkspace:
    if root_dir.exists() and (root_dir / "case.json").exists() and not rebuild:
        return VirtualVideoWorkspace.load(root_dir)
    rows = _load_rows(dataset_root)
    by_qid = {str(row["question_id"]): row for row in rows}
    target = by_qid[str(case_id)]
    rng = random.Random(seed + sum(ord(ch) for ch in str(case_id)))
    if construction == "source_only":
        segments = _build_source_only_segment(dataset_root, target)
    elif construction == "interleaved_chunks":
        segments = _build_interleaved_chunk_segments(
            dataset_root,
            rows,
            target,
            rng=rng,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            chunk_sec=chunk_sec,
        )
    else:
        segments = _build_segments(dataset_root, rows, target, rng=rng, min_duration_sec=min_duration_sec, segment_sec=segment_sec)
    manifest = VirtualVideoManifest(workspace_id=str(case_id), segments=tuple(segments))
    target_segments = tuple(segment for segment in segments if segment.role == "target")
    target_segment = target_segments[0]
    case = VirtualVideoCase(
        case_id=str(case_id),
        question=str(target["question"]),
        options=_options_mapping(target["options"]),
        gold=str(target["answer"]),
        target_segment_id=target_segment.segment_id,
        target_virtual_interval=(target_segments[0].virtual_start_sec, target_segments[-1].virtual_end_sec),
        metadata={
            "source_video_id": str(target["videoID"]),
            "min_duration_sec": min_duration_sec,
            "max_duration_sec": max_duration_sec,
            "seed": seed,
            "construction": construction,
            "target_segment_ids": [segment.segment_id for segment in target_segments],
            "target_virtual_intervals": [[segment.virtual_start_sec, segment.virtual_end_sec] for segment in target_segments],
        },
    )
    workspace = VirtualVideoWorkspace.create(root_dir, manifest=manifest, case=case)
    cues = []
    for segment in segments:
        cues.extend(load_srt_as_virtual_cues(dataset_root / "subtitle" / f"{segment.source_video_id}.srt", segment))
    workspace.write_asr_virtual_cues(tuple(cues))
    return workspace


def ensure_index(workspace: VirtualVideoWorkspace, *, low_fps: float, beat_sec: float, rebuild: bool) -> None:
    if (workspace.root_dir / "beat_index.json").exists() and workspace.frame_manifest.exists() and not rebuild:
        return
    frames = materialize_lowfps_frame_cache(workspace, fps=low_fps)
    build_virtual_beat_index(workspace, frames, beat_sec=beat_sec)


def run_case(
    workspace: VirtualVideoWorkspace,
    *,
    api: "OpenAICompatibleVisionClient",
    max_rounds: int,
    max_investigations: int,
) -> Any:
    trace_path = workspace.root_dir / "interactions.jsonl"
    trace_path.write_text("", encoding="utf-8")
    reasoner = GeminiReasoner(api, trace_path=trace_path)
    investigator = GeminiInvestigator(workspace, api=api, trace_path=trace_path)
    driver = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=investigator,
        max_rounds=max_rounds,
        max_investigations=max_investigations,
    )
    return driver.run(workspace)


class OpenAICompatibleVisionClient:
    def __init__(self, planner: Mapping[str, Any]) -> None:
        self.base = str(planner["base"]).rstrip("/")
        self.model = str(planner["model"])
        self.api_key = str(planner["api_key"])
        self.timeout = float(planner.get("timeout", 300))
        for key, value in (planner.get("proxy_env") or {}).items():
            os.environ[str(key)] = str(value)

    @classmethod
    def from_yaml(cls, path: Path) -> "OpenAICompatibleVisionClient":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(payload.get("planner_api") or payload)

    def chat(self, prompt: str, *, image_paths: Sequence[str] = (), max_tokens: int = 900) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            if Path(path).exists():
                content.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(path))}})
        response = requests.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": int(max_tokens),
            },
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            snippet = response.text[:500].replace(self.api_key, "<redacted>")
            raise RuntimeError(f"HTTP {response.status_code}: {snippet}")
        return str(response.json()["choices"][0]["message"]["content"])


class GeminiReasoner:
    def __init__(self, api: OpenAICompatibleVisionClient, *, trace_path: Path) -> None:
        self.api = api
        self.trace_path = trace_path
        self.calls = 0

    def decide(self, **kwargs: Any) -> ReasonerDecision:
        self.calls += 1
        overview = dict(kwargs["workspace_overview"])
        evidence_digest = tuple(kwargs.get("evidence_digest", ()) or ())
        if evidence_digest:
            image_paths = [str(row.get("overview_thumbnail_grid_path")) for row in overview.get("segment_overviews", ())]
            prompt = _followup_prompt(kwargs, evidence_digest)
            raw = self.api.chat(prompt, image_paths=image_paths)
            parsed = _parse_json(raw)
            action = str(parsed.get("action") or "answer")
            self._trace("reasoner_investigate" if action == "investigate" else "reasoner_answer", prompt, raw, parsed)
            if action == "investigate":
                tasks = parsed.get("tasks") or []
                return ReasonerDecision(action="investigate", tasks=tuple(tasks[:4]))
            return ReasonerDecision(
                action="answer",
                answer=str(parsed.get("answer") or ""),
                citations=tuple(parsed.get("citations") or (evidence_digest[-1]["evidence_id"],)),
                entity_clusters=tuple(parsed.get("entity_clusters") or ()),
            )
        image_paths = [str(row.get("overview_thumbnail_grid_path")) for row in overview.get("segment_overviews", ())]
        prompt = _investigate_prompt(kwargs)
        raw = self.api.chat(prompt, image_paths=image_paths)
        parsed = _parse_json(raw)
        self._trace("reasoner_investigate", prompt, raw, parsed)
        tasks = parsed.get("tasks") or []
        return ReasonerDecision(action="investigate", tasks=tuple(tasks[:4]))

    def _trace(self, kind: str, prompt: str, raw: str, parsed: Mapping[str, Any]) -> None:
        _append_jsonl(
            self.trace_path,
            {"type": kind, "round": self.calls, "prompt": prompt, "raw": raw, "parsed": dict(parsed), "time": time.time()},
        )


class GeminiInvestigator(VirtualVideoInvestigator):
    def __init__(self, workspace: VirtualVideoWorkspace, *, api: OpenAICompatibleVisionClient, trace_path: Path) -> None:
        super().__init__(workspace)
        self.api = api
        self.trace_path = trace_path
        self._query_calls: dict[str, int] = {}

    def _investigate_task(self, task: Any) -> InvestigationReport:
        query_id = str(getattr(task, "query_id", "") or "query")
        observation_id = self._next_observation_id(query_id)
        segment_packet = self.open_segment(str(getattr(task, "segment_id", "") or self.workspace.manifest.segments[0].segment_id))
        requested_window = getattr(task, "time_range", None)
        if requested_window is None:
            window = _select_window_with_model(self.api, task, segment_packet, self.trace_path)
        else:
            window = float(requested_window[0]), float(requested_window[1])

        required_fps = self.highfps if _needs_highfps(task) else 0.5
        cached = self._find_reusable_evidence(task, window[0], window[1], required_fps=required_fps)
        if cached is not None:
            return self._reuse_report(
                task,
                cached,
                tool_trace=("open_segment", "reuse_observation"),
                vlm_calls=0 if requested_window is not None else 1,
            )

        preview_query_id = f"{observation_id}_preview"
        low = self.inspect_window(window[0], window[1], fps=0.5, max_frames=64, query_id=preview_query_id)
        preview_frames = select_uniform_items(tuple(low["frames"]), 16)
        preview_paths = tuple(str(row["path"]) for row in preview_frames)
        preview_prompt = _preview_prompt(self.workspace, task, segment_packet, low)
        preview_raw = self.api.chat(preview_prompt, image_paths=preview_paths, max_tokens=700)
        preview = _parse_json(preview_raw)
        _append_jsonl(
            self.trace_path,
            {
                "type": "investigator_preview",
                "query_id": query_id,
                "observation_id": observation_id,
                "preview_query_id": preview_query_id,
                "window": list(window),
                "prompt": preview_prompt,
                "frame_paths": list(preview_paths),
                "raw": preview_raw,
                "parsed": preview,
                "time": time.time(),
            },
        )

        selected_window = window
        selected_packet = low
        selected_frames = preview_frames
        parsed = preview
        raw = preview_raw
        final_prompt = preview_prompt
        detail_query_id = ""
        tool_trace = ["open_segment", "inspect_window:0.5"]
        vlm_calls = 1 if requested_window is not None else 2

        if _needs_highfps(task) or _truthy(preview.get("need_detail")):
            selected_window = _select_detail_window(preview, window, task, segment_packet)
            detail_query_id = f"{observation_id}_detail"
            selected_packet = self.inspect_window(
                selected_window[0],
                selected_window[1],
                fps=self.highfps,
                max_frames=self.highfps_max_frames,
                query_id=detail_query_id,
            )
            selected_frames = select_uniform_items(tuple(selected_packet["frames"]), 16)
            detail_paths = tuple(str(row["path"]) for row in selected_frames)
            final_prompt = _evidence_prompt(self.workspace, task, segment_packet, selected_packet, preview=preview)
            raw = self.api.chat(final_prompt, image_paths=detail_paths, max_tokens=700)
            parsed = _parse_json(raw)
            tool_trace.append(f"inspect_window:{self.highfps:.1f}")
            vlm_calls += 1

        frame_paths = tuple(str(row["path"]) for row in selected_frames)
        _append_jsonl(
            self.trace_path,
            {
                "type": "investigator_evidence",
                "query_id": query_id,
                "observation_id": observation_id,
                "preview_query_id": preview_query_id,
                "detail_query_id": detail_query_id,
                "window": list(selected_window),
                "prompt": final_prompt,
                "frame_paths": list(frame_paths),
                "raw": raw,
                "parsed": parsed,
                "time": time.time(),
            },
        )
        confidence = _confidence(parsed.get("confidence"), default=0.6)
        entities = _normalize_entities(parsed.get("entities"))
        events = _normalize_events(parsed.get("events"), selected_window)
        supports_identity_anchor = _truthy(parsed.get("supports_identity_anchor"))
        supports_answer_event = bool(events) or _truthy(parsed.get("supports_answer_event"))
        if supports_answer_event:
            evidence_kind = "event_observation"
        elif supports_identity_anchor or entities:
            evidence_kind = "entity_observation"
        else:
            evidence_kind = "visual_observation"
        evidence = EvidenceRecord(
            evidence_id=f"ev_{observation_id}_001",
            beat_id="",
            start_sec=float(selected_window[0]),
            end_sec=float(selected_window[1]),
            modality="visual",
            pointer=f"virtual://{self.workspace.workspace_id}/observations/{observation_id}",
            verbatim=str(parsed.get("summary") or raw)[:1200],
            frame_refs=frame_paths,
            attestation_model=str(getattr(self.api, "model", type(self.api).__name__)),
            temporal_scope="window",
            evidence_kind=evidence_kind,
            observation_polarity="positive" if frame_paths else "unknown",
            sampling_coverage="sparse",
            request_ids=(query_id,),
            coverage_manifest=(
                CoverageSegment(query_id, float(selected_window[0]), float(selected_window[1]), "visual", 1.0),
            ),
            task_id=query_id,
            observation_id=observation_id,
            sampling_fps=float(selected_packet["sampling"]["fps"]),
            confidence=confidence,
            source_lineage=tuple(dict(item) for item in selected_packet["source_lineage"]),
            entity_ids=tuple(f"{observation_id}:{item['local_id']}" for item in entities),
            operation_metadata={
                "entities": entities,
                "events": events,
                "supports_identity_anchor": supports_identity_anchor,
                "supports_answer_event": supports_answer_event,
            },
        )
        if frame_paths:
            self._remember_evidence(task, evidence)
            self._record_visit(task, evidence, status="satisfied")
        return InvestigationReport(
            query_id=query_id,
            status="satisfied" if frame_paths else "empty",
            evidence=(evidence,) if frame_paths else (),
            cost={
                "tool_trace": tuple(tool_trace),
                "preview_frames": len(preview_paths),
                "detail_frames": len(frame_paths) if detail_query_id else 0,
                "frames": len(preview_paths) + (len(frame_paths) if detail_query_id else 0),
                "vlm_calls": vlm_calls,
                "reused": False,
            },
        )

    def _next_observation_id(self, query_id: str) -> str:
        call_index = self._query_calls.get(query_id, 0) + 1
        self._query_calls[query_id] = call_index
        return f"{query_id}_c{call_index:02d}"


def _select_window_with_model(api: OpenAICompatibleVisionClient, task: Any, segment_packet: Mapping[str, Any], trace_path: Path) -> tuple[float, float]:
    prompt = (
        "You are an Investigator. Choose the most relevant virtual-time window inside this segment.\n"
        "Return JSON only: {\"start_sec\": float, \"end_sec\": float, \"reason\": string}.\n"
        f"Task: {getattr(task, 'goal', '')}\nExpected evidence: {getattr(task, 'expected_evidence', '')}\n"
        f"Segment packet (text only): {json.dumps(_compact_segment_packet(segment_packet), ensure_ascii=False)}"
    )
    image_paths = []
    beats = select_uniform_items(tuple(segment_packet.get("beats", ()) or ()), 12)
    for beat in beats:
        image_paths.extend(str(path) for path in beat.get("thumbnail_grid_paths", ())[:1])
    raw = api.chat(prompt, image_paths=image_paths, max_tokens=400)
    parsed = _parse_json(raw)
    seg_start, seg_end = segment_packet["virtual_time_range"]
    try:
        start = max(float(seg_start), float(parsed["start_sec"]))
        end = min(float(seg_end), float(parsed["end_sec"]))
    except (KeyError, TypeError, ValueError):
        start, end = _choose_window_from_segment_packet(task, segment_packet)
        fallback_used = True
    else:
        fallback_used = end <= start
        if fallback_used:
            start, end = _choose_window_from_segment_packet(task, segment_packet)
    selected = round(float(start), 3), round(float(end), 3)
    _append_jsonl(
        trace_path,
        {
            "type": "investigator_select_window",
            "query_id": getattr(task, "query_id"),
            "raw": raw,
            "parsed": parsed,
            "selected_window": list(selected),
            "fallback_used": fallback_used,
            "time": time.time(),
        },
    )
    return selected


def _select_detail_window(
    preview: Mapping[str, Any],
    preview_window: tuple[float, float],
    task: Any,
    segment_packet: Mapping[str, Any],
    *,
    max_detail_sec: float = 60.0,
) -> tuple[float, float]:
    preview_start, preview_end = float(preview_window[0]), float(preview_window[1])
    try:
        start = max(preview_start, float(preview["detail_start_sec"]))
        end = min(preview_end, float(preview["detail_end_sec"]))
    except (KeyError, TypeError, ValueError):
        start, end = _choose_window_from_segment_packet(task, segment_packet)
        start = max(preview_start, float(start))
        end = min(preview_end, float(end))
    if end <= start:
        center = (preview_start + preview_end) / 2.0
        half = min(float(max_detail_sec), preview_end - preview_start) / 2.0
        start, end = center - half, center + half
    if end - start > float(max_detail_sec):
        center = (start + end) / 2.0
        half = float(max_detail_sec) / 2.0
        start = max(preview_start, center - half)
        end = min(preview_end, center + half)
    return round(start, 3), round(end, 3)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def _confidence(value: Any, *, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return float(default)


def _normalize_entities(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    rows = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            continue
        description = str(item.get("description", "") or "").strip()
        if not description:
            continue
        rows.append(
            {
                "local_id": str(item.get("local_id", "") or f"person_{index}"),
                "description": description,
                "role": str(item.get("role", "") or ""),
                "question_relation": str(item.get("question_relation", "") or ""),
                "supports_question_relation": _truthy(item.get("supports_question_relation")),
            }
        )
    return tuple(rows)


def _normalize_events(value: Any, window: tuple[float, float]) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    window_start, window_end = float(window[0]), float(window[1])
    rows = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            continue
        description = str(item.get("description", "") or "").strip()
        if not description or not _truthy(item.get("supports_question_event")):
            continue
        try:
            start = float(item.get("start_sec"))
            end = float(item.get("end_sec"))
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        if end < window_start or start > window_end:
            continue
        start = max(window_start, start)
        end = min(window_end, end)
        rows.append(
            {
                "local_id": str(item.get("local_id", "") or f"event_{index}"),
                "description": description,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "supports_question_event": True,
            }
        )
    return tuple(rows)


def _load_rows(dataset_root: Path) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(dataset_root / "videomme" / "test-00000-of-00001.parquet")
    return [dict(row) for row in frame.to_dict("records")]


def _build_segments(
    dataset_root: Path,
    rows: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    *,
    rng: random.Random,
    min_duration_sec: float,
    segment_sec: float,
) -> tuple[VirtualVideoSegment, ...]:
    target_video = str(target["videoID"])
    specs = [{"role": "target", "video_id": target_video, "start": 0.0, "end": min(_duration(dataset_root, target_video), segment_sec)}]
    pool = []
    seen = {target_video}
    for row in rows:
        vid = str(row["videoID"])
        if vid in seen or str(row.get("duration")) != "long":
            continue
        path = dataset_root / "video" / f"{vid}.mp4"
        if path.exists():
            dur = _duration(dataset_root, vid)
            if dur >= 600.0:
                pool.append((vid, dur))
                seen.add(vid)
    rng.shuffle(pool)
    total = specs[0]["end"] - specs[0]["start"]
    for vid, dur in pool:
        length = min(float(segment_sec), dur)
        start = 0.0 if dur <= length else rng.uniform(0.0, dur - length)
        specs.append({"role": "distractor", "video_id": vid, "start": start, "end": start + length})
        total += length
        if total >= min_duration_sec:
            break
    rng.shuffle(specs)
    if specs[0]["role"] == "target" and len(specs) > 2:
        specs[0], specs[1] = specs[1], specs[0]
    if specs[-1]["role"] == "target" and len(specs) > 2:
        specs[-1], specs[-2] = specs[-2], specs[-1]
    segments = []
    cursor = 0.0
    for idx, spec in enumerate(specs, start=1):
        dur = float(spec["end"]) - float(spec["start"])
        segments.append(
            VirtualVideoSegment(
                segment_id=f"seg_{idx:04d}",
                source_video_id=str(spec["video_id"]),
                source_path=str(dataset_root / "video" / f"{spec['video_id']}.mp4"),
                source_start_sec=round(float(spec["start"]), 3),
                source_end_sec=round(float(spec["end"]), 3),
                virtual_start_sec=round(cursor, 3),
                virtual_end_sec=round(cursor + dur, 3),
                role=str(spec["role"]),
            )
        )
        cursor += dur
    if cursor < min_duration_sec:
        raise RuntimeError(f"Only built {cursor:.1f}s, below requested {min_duration_sec:.1f}s")
    return tuple(segments)


def _build_source_only_segment(
    dataset_root: Path,
    target: Mapping[str, Any],
) -> tuple[VirtualVideoSegment, ...]:
    video_id = str(target["videoID"])
    duration = float(_duration(dataset_root, video_id))
    return (
        VirtualVideoSegment(
            segment_id="seg_0001",
            source_video_id=video_id,
            source_path=str(dataset_root / "video" / f"{video_id}.mp4"),
            source_start_sec=0.0,
            source_end_sec=duration,
            virtual_start_sec=0.0,
            virtual_end_sec=duration,
            role="target",
        ),
    )


def _build_interleaved_chunk_segments(
    dataset_root: Path,
    rows: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    *,
    rng: random.Random,
    min_duration_sec: float,
    max_duration_sec: float | None,
    chunk_sec: float,
) -> tuple[VirtualVideoSegment, ...]:
    target_video = str(target["videoID"])
    chunk_width = max(30.0, float(chunk_sec))
    target_duration = _duration(dataset_root, target_video)
    specs = list(_video_chunks("target", target_video, target_duration, chunk_width=chunk_width))
    pool = _long_video_pool(dataset_root, rows, exclude={target_video}, min_duration_sec=chunk_width)
    rng.shuffle(pool)
    total = sum(float(spec["end"]) - float(spec["start"]) for spec in specs)
    for vid, dur in pool:
        chunks = list(_video_chunks("distractor", vid, dur, chunk_width=chunk_width))
        rng.shuffle(chunks)
        for spec in chunks:
            if max_duration_sec is not None and total >= float(max_duration_sec):
                break
            specs.append(spec)
            total += float(spec["end"]) - float(spec["start"])
            if total >= float(min_duration_sec) and (max_duration_sec is None or total >= min(float(max_duration_sec), float(min_duration_sec))):
                break
        if total >= float(min_duration_sec):
            break
    if total < min_duration_sec:
        raise RuntimeError(f"Only built {total:.1f}s, below requested {min_duration_sec:.1f}s")
    specs = _interleave_specs(specs, rng=rng)
    segments = []
    cursor = 0.0
    for idx, spec in enumerate(specs, start=1):
        dur = float(spec["end"]) - float(spec["start"])
        segments.append(
            VirtualVideoSegment(
                segment_id=f"seg_{idx:04d}",
                source_video_id=str(spec["video_id"]),
                source_path=str(dataset_root / "video" / f"{spec['video_id']}.mp4"),
                source_start_sec=round(float(spec["start"]), 3),
                source_end_sec=round(float(spec["end"]), 3),
                virtual_start_sec=round(cursor, 3),
                virtual_end_sec=round(cursor + dur, 3),
                role=str(spec["role"]),
            )
        )
        cursor += dur
    return tuple(segments)


def _video_chunks(role: str, video_id: str, duration_sec: float, *, chunk_width: float) -> tuple[dict[str, Any], ...]:
    chunks = []
    start = 0.0
    while start < float(duration_sec):
        end = min(float(duration_sec), start + chunk_width)
        if end - start >= min(30.0, chunk_width):
            chunks.append({"role": role, "video_id": video_id, "start": start, "end": end})
        start = end
    return tuple(chunks)


def _long_video_pool(
    dataset_root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    exclude: set[str],
    min_duration_sec: float,
) -> list[tuple[str, float]]:
    pool = []
    seen = set(exclude)
    for row in rows:
        vid = str(row["videoID"])
        if vid in seen or str(row.get("duration")) != "long":
            continue
        path = dataset_root / "video" / f"{vid}.mp4"
        if path.exists():
            dur = _duration(dataset_root, vid)
            if dur >= min_duration_sec:
                pool.append((vid, dur))
                seen.add(vid)
    return pool


def _interleave_specs(specs: Sequence[Mapping[str, Any]], *, rng: random.Random) -> list[Mapping[str, Any]]:
    items = list(specs)
    rng.shuffle(items)
    for _ in range(4):
        changed = False
        for idx in range(1, len(items)):
            if items[idx]["video_id"] != items[idx - 1]["video_id"]:
                continue
            swap = next((pos for pos in range(idx + 1, len(items)) if items[pos]["video_id"] != items[idx]["video_id"]), None)
            if swap is not None:
                items[idx], items[swap] = items[swap], items[idx]
                changed = True
        if not changed:
            break
    if items and items[0]["role"] == "target" and len(items) > 2:
        swap = next((idx for idx, item in enumerate(items[1:], start=1) if item["role"] != "target"), None)
        if swap is not None:
            items[0], items[swap] = items[swap], items[0]
    if items and items[-1]["role"] == "target" and len(items) > 2:
        swap = next((len(items) - 1 - idx for idx, item in enumerate(reversed(items[:-1]), start=1) if item["role"] != "target"), None)
        if swap is not None:
            items[-1], items[swap] = items[swap], items[-1]
    return items


def _duration(dataset_root: Path, video_id: str) -> float:
    return probe_duration(str(dataset_root / "video" / f"{video_id}.mp4"))


def _options_mapping(value: Any) -> dict[str, str]:
    labels = "ABCDEFGH"
    values = list(value) if not isinstance(value, str) else [part.strip() for part in value.split("|")]
    result = {}
    for idx, item in enumerate(values):
        text = str(item).strip()
        label = labels[idx]
        if len(text) > 2 and text[0].upper() in labels and text[1] == ".":
            label, text = text[0].upper(), text[2:].strip()
        result[label] = text
    return result


def _investigate_prompt(kwargs: Mapping[str, Any]) -> str:
    return (
        "You are the Reasoner for a long virtual video QA agent. You only see segment overview images.\n"
        "Pick up to 4 segments to investigate. Return JSON only: "
        "{\"action\":\"investigate\",\"tasks\":[{\"query_id\":\"r1_t1\",\"goal\":\"...\",\"segment_id\":\"seg_0001\",\"time_range\":null,\"modality_hint\":[\"visual\"],\"expected_evidence\":\"...\"}]}.\n"
        "For full-video count questions, evidence from one chunk is only a source hypothesis, not complete proof.\n"
        "For total event-count questions, dispatch segment tasks that enumerate atomic event occurrences with timestamps.\n"
        "For source-relative minute questions, prioritize the supplied temporal_navigation candidate segments.\n"
        "For identity-anchor questions, first locate evidence matching every identity_anchor_term before investigating the later event.\n"
        f"Question: {kwargs['question']}\nOptions: {json.dumps(kwargs['options'], ensure_ascii=False)}\n"
        f"Query contract: {json.dumps(kwargs.get('query_contract') or {}, ensure_ascii=False)}\n"
        f"Query requirements: {json.dumps(kwargs.get('query_requirements') or {}, ensure_ascii=False)}\n"
        f"Temporal navigation: {json.dumps(kwargs.get('temporal_navigation') or {}, ensure_ascii=False)}\n"
        f"Workspace overview: {json.dumps(kwargs['workspace_overview'], ensure_ascii=False)[:6000]}"
    )


def _followup_prompt(kwargs: Mapping[str, Any], evidence_digest: Sequence[Mapping[str, Any]]) -> str:
    finalization_instruction = (
        "No investigation budget remains. Return action=answer using the best grounded evidence now; do not return tasks.\n"
        if kwargs.get("force_finalize")
        else ""
    )
    return (
        "You are the Reasoner for a long virtual video QA agent. Decide whether current evidence is enough.\n"
        f"{finalization_instruction}"
        "If enough, return JSON only: {\"action\":\"answer\", \"answer\":\"A. ...\", \"citations\":[\"ev_...\"],"
        "\"entity_clusters\":[{\"entity_id\":\"entity_1\",\"description\":\"canonical identity\","
        "\"evidence_ids\":[\"ev_...\"]}]}.\n"
        "If not enough, return JSON only: {\"action\":\"investigate\", \"tasks\":[{\"query_id\":\"r2_t1\",\"goal\":\"...\",\"segment_id\":\"seg_0001\",\"time_range\":null,\"modality_hint\":[\"visual\"],\"expected_evidence\":\"...\"}]}.\n"
        "You may request up to 4 more segments/windows. Do not answer with insufficient evidence.\n"
        "If completion_status.ready_for_answer is false, you MUST investigate its missing_segment_ids and must not answer. "
        "For a final full-video answer, cite every relevant visual evidence record from the adopted source.\n"
        "For distinct-count questions, reconcile the per-evidence entities by stable appearance. Do not add local counts. "
        "Create one entity_cluster per unique person, and make the option count equal the number of clusters.\n"
        "For total event-count questions, count only the timestamped events rows in evidence, deduplicate overlapping observations, "
        "cite every evidence record containing a positive occurrence, and never infer the count from answer options or entity clusters.\n"
        "For identity-anchor questions, do not answer while missing_identity_anchor_terms is non-empty. "
        "The final entity cluster must cite both anchor evidence and the later event evidence for the same person.\n"
        f"Question: {kwargs['question']}\nOptions: {json.dumps(kwargs['options'], ensure_ascii=False)}\n"
        f"Query contract: {json.dumps(kwargs.get('query_contract') or {}, ensure_ascii=False)}\n"
        f"Query requirements: {json.dumps(kwargs.get('query_requirements') or {}, ensure_ascii=False)}\n"
        f"Completion status: {json.dumps(kwargs.get('completion_status') or {}, ensure_ascii=False)}\n"
        f"Temporal navigation: {json.dumps(kwargs.get('temporal_navigation') or {}, ensure_ascii=False)}\n"
        f"Workspace overview: {json.dumps(kwargs['workspace_overview'], ensure_ascii=False)[:6000]}\n"
        f"Evidence so far: {json.dumps(list(evidence_digest), ensure_ascii=False)}"
    )


def _preview_prompt(workspace: VirtualVideoWorkspace, task: Any, segment_packet: Mapping[str, Any], window: Mapping[str, Any]) -> str:
    return (
        "You are the Investigator. Inspect the low-fps preview frames and local ASR without choosing an answer option. "
        "Return JSON only: {\"summary\":\"atomic observation\",\"confidence\":0.0-1.0,"
        "\"entities\":[{\"local_id\":\"person_1\",\"description\":\"stable visible attributes\","
        "\"role\":\"visible role or unknown\",\"question_relation\":\"directly observed relation or unknown\","
        "\"supports_question_relation\":true|false}],"
        "\"events\":[{\"local_id\":\"event_1\",\"description\":\"one atomic occurrence relevant to the question\","
        "\"start_sec\":float,\"end_sec\":float,\"supports_question_event\":true|false}],"
        "\"supports_identity_anchor\":true|false,\"supports_answer_event\":true|false,"
        "\"need_detail\":true|false,\"detail_start_sec\":float|null,\"detail_end_sec\":float|null,\"reason\":\"...\"}.\n"
        "List each visible person separately using stable appearance attributes. Do not estimate a segment-level or video-level count. "
        "The same person may recur in later chunks.\n"
        "Enumerate every distinct question-relevant event occurrence visible in this inspected window. "
        "Use virtual timestamps from the window metadata, one row per occurrence, and return an empty events list when none is supported.\n"
        "Set supports_identity_anchor only when one visible entity jointly matches the identifying attributes in the question. "
        "Set supports_answer_event only when the observation directly supports the event, cause, action, or state being asked about.\n"
        "Request detail only when motion, OCR, identity, or a small visual attribute remains unresolved. "
        "Any detail window must be inside the preview window and narrower than it.\n"
        f"Question: {workspace.case.question}\n"
        f"Task: {getattr(task, 'goal', '')}\nExpected evidence: {getattr(task, 'expected_evidence', '')}\n"
        f"Segment: {json.dumps(_compact_segment_packet(segment_packet), ensure_ascii=False)[:3000]}\n"
        f"Preview window metadata: {json.dumps({k: window[k] for k in ['virtual_time_range','sampling','asr_cues','source_lineage']}, ensure_ascii=False)[:5000]}"
    )


def _evidence_prompt(
    workspace: VirtualVideoWorkspace,
    task: Any,
    segment_packet: Mapping[str, Any],
    window: Mapping[str, Any],
    *,
    preview: Mapping[str, Any] | None = None,
) -> str:
    return (
        "You are the Investigator. Inspect the detail frames and local ASR. Report only an atomic observation, "
        "not an answer-option judgment. Return JSON only: "
        "{\"summary\":\"atomic visual evidence summary\", \"confidence\":0.0-1.0,"
        "\"entities\":[{\"local_id\":\"person_1\",\"description\":\"stable visible attributes\","
        "\"role\":\"visible role or unknown\",\"question_relation\":\"directly observed relation or unknown\","
        "\"supports_question_relation\":true|false}],"
        "\"events\":[{\"local_id\":\"event_1\",\"description\":\"one atomic occurrence relevant to the question\","
        "\"start_sec\":float,\"end_sec\":float,\"supports_question_event\":true|false}],"
        "\"supports_identity_anchor\":true|false,\"supports_answer_event\":true|false}.\n"
        "List visible people separately and do not infer a count across frames or chunks.\n"
        "Enumerate every distinct question-relevant event occurrence visible in this inspected window. "
        "Use virtual timestamps from the window metadata, one row per occurrence, and return an empty events list when none is supported.\n"
        "Set supports_identity_anchor only when one visible entity jointly matches the identifying attributes in the question. "
        "Set supports_answer_event only when the observation directly supports the event, cause, action, or state being asked about.\n"
        f"Question: {workspace.case.question}\n"
        f"Task: {getattr(task, 'goal', '')}\nExpected evidence: {getattr(task, 'expected_evidence', '')}\n"
        f"Preview finding: {json.dumps(dict(preview or {}), ensure_ascii=False)[:1600]}\n"
        f"Segment: {json.dumps(_compact_segment_packet(segment_packet), ensure_ascii=False)[:3000]}\n"
        f"Detail window metadata: {json.dumps({k: window[k] for k in ['virtual_time_range','sampling','asr_cues','source_lineage']}, ensure_ascii=False)[:5000]}"
    )


def _compact_segment_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": packet.get("segment_id"),
        "virtual_time_range": packet.get("virtual_time_range"),
        "asr_timeline_summary": packet.get("asr_timeline_summary"),
        "beats": [
            {"beat_id": beat.get("beat_id"), "virtual_time_range": beat.get("virtual_time_range"), "asr_excerpt": beat.get("asr_excerpt")}
            for beat in packet.get("beats", ())[:20]
        ],
    }


def _parse_json(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if match:
        raw = match.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemini-backed VirtualVideo investigation on Video-MME v1 virtual concatenations.")
    parser.add_argument("--dataset-root", default="/ytech_m2v5_hdd/workspace/kling_mm/Datasets/VLMEvalKit_Dataset_Cache/HFCache/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b")
    parser.add_argument("--out-root", default="/m2v_intern/xuboshen/zgw/VideoAgent/virtual_videomme_interactive")
    parser.add_argument("--config", required=True)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--mode", choices=("smoke", "all", "long"), default="smoke")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--min-duration-sec", type=float, default=18000.0)
    parser.add_argument("--max-duration-sec", type=float)
    parser.add_argument("--segment-sec", type=float, default=600.0)
    parser.add_argument(
        "--construction",
        choices=("source_only", "single_segment", "interleaved_chunks"),
        default="single_segment",
    )
    parser.add_argument("--chunk-sec", type=float, default=300.0)
    parser.add_argument("--low-fps", type=float, default=0.1)
    parser.add_argument("--beat-sec", type=float, default=60.0)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-investigations", type=int, default=20)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
