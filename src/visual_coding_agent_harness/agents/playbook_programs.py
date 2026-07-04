"""Fixed execution programs for playbook-guided investigation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import inspect
import re
from typing import Callable, Literal, Sequence

from visual_coding_agent_harness.backends.base import BackendRequest
from visual_coding_agent_harness.backends.base import VisionLanguageBackend
from visual_coding_agent_harness.contracts.evidence import EvidenceRecord
from visual_coding_agent_harness.contracts.playbook import Playbook
from visual_coding_agent_harness.contracts.query import ScopedQuery
from visual_coding_agent_harness.contracts.report import Finding, InvestigationReport, VerifyRequest
from visual_coding_agent_harness.tools.vlm_tools import verify_window
from visual_coding_agent_harness.workspace.investigator_ws import EvidenceRecordLedger
from visual_coding_agent_harness.workspace.memo import MemoStore, ObservationMemo
from visual_coding_agent_harness.workspace.video_workspace import Beat, VideoWorkspace
from visual_coding_agent_harness.workspace.visual_index import BeatHit


FrameSampler = Callable[..., Sequence[str]]
VerifyFn = Callable[..., Sequence[Finding]]


@dataclass(frozen=True)
class PlaybookProgram:
    search_order: tuple[Literal["text", "visual"], ...]
    top_k_candidates: int
    verify_frames_per_beat: int
    verify_resolution: Literal["low", "high"]
    stop_when_supports: bool
    dense_sampling: bool

    def execute(
        self,
        *,
        query: ScopedQuery,
        workspace: VideoWorkspace,
        backend: VisionLanguageBackend,
        frame_sampler: FrameSampler | None = None,
        verify_fn: VerifyFn = verify_window,
        memo_store: MemoStore | None = None,
        evidence_ledger: EvidenceRecordLedger | None = None,
    ) -> InvestigationReport:
        candidates = _candidate_beats(
            workspace,
            query=query,
            search_order=self.search_order,
            top_k=self.top_k_candidates,
            scope=query.scope,
        )
        if query.playbook == Playbook.COMPARE and query.scope_b is not None:
            candidates = _unique_beats(
                (
                    *candidates,
                    *_candidate_beats(
                        workspace,
                        query=query,
                        search_order=self.search_order,
                        top_k=self.top_k_candidates,
                        scope=query.scope_b,
                    ),
                )
            )
        if memo_store is not None and candidates:
            _record_observation_memos(
                query=query,
                backend=backend,
                candidates=candidates,
                memo_store=memo_store,
            )
        frame_sampler = frame_sampler or _default_frame_sampler
        findings: list[Finding] = []
        verified_shots: list[str] = []
        frames_read = 0
        for beat in candidates:
            shot_id = beat.shot_ids[0] if beat.shot_ids else beat.beat_id
            max_frames = min(query.budget.max_frames, self.verify_frames_per_beat)
            frame_paths = _sample_frames(
                frame_sampler,
                beat,
                max_frames,
                resolution=self.verify_resolution,
                dense=self.dense_sampling,
            )
            frames_read += len(frame_paths)
            request = VerifyRequest(
                shot_id=shot_id,
                time_range=(beat.start_sec, beat.end_sec),
                focus_claim=query.expected_evidence,
                sampling={
                    "max_frames": max_frames,
                    "resolution": self.verify_resolution,
                    "dense": self.dense_sampling,
                    "beat_id": beat.beat_id,
                },
                checks=({"target_id": query.goal_id, "claim": query.expected_evidence, "polarity": "presence"},),
            )
            shot_findings = tuple(verify_fn(query_id=query.query_id, request=request, frame_paths=frame_paths, backend=backend))
            if shot_findings:
                findings.extend(shot_findings)
                if evidence_ledger is not None:
                    evidence_ledger.extend(
                        _evidence_records_from_findings(
                            findings=shot_findings,
                            query=query,
                            beat=beat,
                            frame_paths=frame_paths,
                        )
                    )
                verified_shots.append(shot_id)
                if self.stop_when_supports:
                    break
        status = "satisfied" if findings else ("empty" if not candidates else "partial")
        return InvestigationReport(
            query_id=query.query_id,
            status=status,
            findings=tuple(findings),
            explored_shots=tuple(beat.shot_ids[0] if beat.shot_ids else beat.beat_id for beat in candidates),
            verified_shots=tuple(verified_shots),
            unresolved=() if findings else (query.expected_evidence,),
            cost={
                "explore_calls": len(self.search_order),
                "verify_calls": len(verified_shots) if self.stop_when_supports else len(candidates),
                "frames_read": frames_read,
            },
        )


PROGRAMS: dict[Playbook, PlaybookProgram] = {
    Playbook.LOCATE_STATEMENT: PlaybookProgram(
        search_order=("text", "visual"),
        top_k_candidates=8,
        verify_frames_per_beat=3,
        verify_resolution="low",
        stop_when_supports=True,
        dense_sampling=False,
    ),
    Playbook.READ_TEXT: PlaybookProgram(
        search_order=("text", "visual"),
        top_k_candidates=6,
        verify_frames_per_beat=4,
        verify_resolution="high",
        stop_when_supports=True,
        dense_sampling=False,
    ),
    Playbook.ORDER_ACTIONS: PlaybookProgram(
        search_order=("visual",),
        top_k_candidates=15,
        verify_frames_per_beat=6,
        verify_resolution="high",
        stop_when_supports=False,
        dense_sampling=True,
    ),
    Playbook.IDENTIFY_VISUAL: PlaybookProgram(
        search_order=("visual", "text"),
        top_k_candidates=10,
        verify_frames_per_beat=4,
        verify_resolution="high",
        stop_when_supports=True,
        dense_sampling=False,
    ),
    Playbook.COUNT: PlaybookProgram(
        search_order=("visual", "text"),
        top_k_candidates=25,
        verify_frames_per_beat=8,
        verify_resolution="high",
        stop_when_supports=False,
        dense_sampling=True,
    ),
    Playbook.COMPARE: PlaybookProgram(
        search_order=("text", "visual"),
        top_k_candidates=8,
        verify_frames_per_beat=3,
        verify_resolution="low",
        stop_when_supports=False,
        dense_sampling=False,
    ),
}


def _candidate_beats(
    workspace: VideoWorkspace,
    *,
    query: ScopedQuery,
    search_order: Sequence[str],
    top_k: int,
    scope,
) -> tuple[Beat, ...]:
    scoped = set(scope.chapter_ids)
    beat_by_id = {beat.beat_id: beat for beat in workspace.beats}
    scores: dict[str, float] = {}
    for modality in search_order:
        search_queries = query.text_queries if modality == "text" else query.visual_queries
        for search_query in search_queries:
            hits: Sequence[BeatHit]
            if modality == "text":
                hits = workspace.search_text(search_query)
            else:
                hits = workspace.search_visual(search_query, k=top_k)
            for rank, hit in enumerate(tuple(hits)[: max(0, int(top_k))], start=1):
                beat = beat_by_id.get(hit.beat_id)
                if beat is None or (scoped and beat.chapter_id not in scoped):
                    continue
                scores[hit.beat_id] = scores.get(hit.beat_id, 0.0) + 1.0 / float(60 + rank)
    ordered = sorted(scores, key=lambda beat_id: (-scores[beat_id], beat_id))
    return tuple(beat_by_id[beat_id] for beat_id in ordered[: max(0, int(top_k))])


def _unique_beats(beats: Sequence[Beat]) -> tuple[Beat, ...]:
    result: list[Beat] = []
    seen: set[str] = set()
    for beat in beats:
        if beat.beat_id in seen:
            continue
        seen.add(beat.beat_id)
        result.append(beat)
    return tuple(result)


def _default_frame_sampler(beat: Beat, max_frames: int) -> tuple[str, ...]:
    if max_frames <= 0 or not beat.keyframe_path:
        return ()
    return (beat.keyframe_path,)


def _sample_frames(
    frame_sampler: FrameSampler,
    beat: Beat,
    max_frames: int,
    *,
    resolution: str,
    dense: bool,
) -> tuple[str, ...]:
    try:
        parameters = inspect.signature(frame_sampler).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_keywords = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    if accepts_keywords or "resolution" in parameters or "dense" in parameters:
        return tuple(frame_sampler(beat, max_frames, resolution=resolution, dense=dense))
    return tuple(frame_sampler(beat, max_frames))


def _evidence_records_from_findings(
    *,
    findings: Sequence[Finding],
    query: ScopedQuery,
    beat: Beat,
    frame_paths: Sequence[str],
) -> tuple[EvidenceRecord, ...]:
    records: list[EvidenceRecord] = []
    modality, pointer, verbatim = _evidence_source(query=query, beat=beat, frame_paths=frame_paths)
    for ordinal, finding in enumerate(findings, start=1):
        summary = finding.summary.strip()
        if not (summary or verbatim):
            continue
        stance: Literal["supports", "refutes"] = "refutes" if finding.refutes_options and not finding.supports_options else "supports"
        record_verbatim = verbatim if modality in {"asr", "ocr"} and verbatim else (summary or verbatim)
        records.append(
            EvidenceRecord(
                evidence_id=_stable_evidence_id(
                    query_id=query.query_id,
                    beat_id=beat.beat_id,
                    modality=modality,
                    ordinal=ordinal,
                ),
                claim=query.expected_evidence,
                stance=stance,
                modality=modality,
                time_sec=beat.start_sec,
                pointer=pointer,
                verbatim=record_verbatim,
                query_id=query.query_id,
                beat_id=beat.beat_id,
            )
        )
    return tuple(records)


def _evidence_source(
    *,
    query: ScopedQuery,
    beat: Beat,
    frame_paths: Sequence[str],
) -> tuple[Literal["frame", "asr", "ocr"], str, str]:
    asr_text = beat.asr_verbatim.strip()
    ocr_text = " ".join(beat.ocr_verbatim).strip()
    raw_queries = (*query.text_queries, query.expected_evidence, query.natural_query)
    if query.playbook == Playbook.LOCATE_STATEMENT and asr_text and _raw_text_matches(asr_text, raw_queries):
        return "asr", beat.beat_id, asr_text
    if query.playbook == Playbook.READ_TEXT and ocr_text and _raw_text_matches(ocr_text, raw_queries):
        return "ocr", beat.beat_id, ocr_text
    if query.playbook == Playbook.LOCATE_STATEMENT and asr_text:
        return "asr", beat.beat_id, asr_text
    if query.playbook == Playbook.READ_TEXT and ocr_text:
        return "ocr", beat.beat_id, ocr_text
    if frame_paths:
        return "frame", str(frame_paths[0]), asr_text or ocr_text or "frame evidence"
    if asr_text:
        return "asr", beat.beat_id, asr_text
    return "ocr", beat.beat_id, ocr_text


def _raw_text_matches(text: str, queries: Sequence[str]) -> bool:
    text_tokens = set(_tokens(text))
    if not text_tokens:
        return False
    for query in queries:
        query_tokens = tuple(token for token in _tokens(query) if token not in {"the", "a", "an", "is", "are", "to"})
        if query_tokens and set(query_tokens).issubset(text_tokens):
            return True
    return False


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in re.finditer(r"\w+", str(text or "")))


def _stable_evidence_id(*, query_id: str, beat_id: str, modality: str, ordinal: int) -> str:
    return f"ev_{_slug(query_id)}_{_slug(beat_id)}_{_slug(modality)}_{int(ordinal):03d}"


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_")
    return text or "unknown"


def _record_observation_memos(
    *,
    query: ScopedQuery,
    backend: VisionLanguageBackend,
    candidates: Sequence[Beat],
    memo_store: MemoStore,
) -> None:
    response = backend.generate(
        BackendRequest(
            task="playbook_explore",
            prompt=_memo_prompt(query=query, candidates=candidates, memo_store=memo_store),
            frames=[beat.keyframe_path for beat in candidates if beat.keyframe_path],
            media_type="image",
            max_new_tokens=512,
            metadata={"query_id": query.query_id, "playbook": query.playbook.value},
        )
    )
    for item in _json_payload(response.text).get("observations", ()):
        if not isinstance(item, dict):
            continue
        beat_id = str(item.get("beat_id") or "").strip()
        observation = str(item.get("observation") or "").strip()
        if not beat_id or not observation:
            continue
        memo_store.append(
            ObservationMemo(
                memo_id=f"memo_{beat_id}_{query.query_id}_{len(memo_store.load()) + 1:05d}",
                beat_id=beat_id,
                observation=observation,
                source_query_id=query.query_id,
                source_playbook=query.playbook,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        )


def _memo_prompt(*, query: ScopedQuery, candidates: Sequence[Beat], memo_store: MemoStore) -> str:
    lines = [
        f"Task query: {query.natural_query}",
        f"Expected evidence: {query.expected_evidence}",
        f"Playbook: {query.playbook.value}",
        "Write observations useful for this query. Do not infer beyond visible frames or raw ASR/OCR cues.",
        "For each beat you inspect, return JSON observations only:",
        '{"observations":[{"beat_id":"bt00001","observation":"one factual visible description"}]}',
        "BeatMeta:",
    ]
    for beat in candidates:
        lines.append(f"{beat.beat_id} [{beat.start_sec:.1f}-{beat.end_sec:.1f}s] asr: {beat.asr_verbatim[:160]}")
        for memo in memo_store.get(beat.beat_id):
            lines.append(f"  previous observation: {memo.observation}")
    return "\n".join(lines)


def _json_payload(text: str) -> dict:
    try:
        payload = json.loads(str(text or ""))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
