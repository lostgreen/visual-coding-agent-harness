#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Callable, Mapping, Sequence

import requests
import yaml
from PIL import Image, ImageEnhance

from vcah.evidence_primitives import (
    ConditionResult,
    GapCondition,
    MeasurementFact,
    RelationFact,
    TargetPresenceFact,
    derive_resolution,
    extract_measurements_from_text,
    normalize_condition_results,
    normalize_measurements,
    normalize_relations,
    normalize_target_presence,
)
from vcah.investigator import (
    InvestigationReport,
    VirtualVideoInvestigator,
    _choose_window_from_segment_packet,
    _needs_highfps,
    _task_terms,
)
from vcah.multiround import InvestigationTask, ReasonerDecision, VirtualVideoMultiRoundDriver
from vcah.types import CoverageSegment, EvidenceRecord, to_jsonable
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


def _load_case_group(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = tuple(payload.get("cases", ()) or ())
    case_ids = tuple(str(row.get("case_id", "") or "").strip() for row in cases if isinstance(row, Mapping))
    if not case_ids or any(not case_id for case_id in case_ids):
        raise ValueError(f"Case group {path} must contain non-empty cases[].case_id values")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"Case group {path} contains duplicate case ids")
    construction = str(payload.get("construction", "source_only") or "source_only")
    if construction not in {"source_only", "single_segment", "interleaved_chunks"}:
        raise ValueError(f"Unsupported case-group construction: {construction}")
    return {
        **payload,
        "group_id": str(payload.get("group_id", path.stem) or path.stem),
        "construction": construction,
        "case_ids": case_ids,
    }


def _run_case_batch(
    case_ids: Sequence[str],
    run_one: Callable[[str], Mapping[str, Any]],
    *,
    workers: int,
) -> tuple[Mapping[str, Any], ...]:
    ordered = tuple(str(case_id) for case_id in case_ids)
    if not ordered:
        return ()
    worker_count = min(16, len(ordered), max(1, int(workers)))
    if worker_count == 1:
        return tuple(run_one(case_id) for case_id in ordered)
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="vv-case") as executor:
        futures = tuple(executor.submit(run_one, case_id) for case_id in ordered)
        return tuple(future.result() for future in futures)


def _load_existing_case_summary(workspace_root: Path) -> dict[str, Any] | None:
    path = Path(workspace_root) / "run_summary.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "case_id": str(payload.get("case_id", workspace_root.name) or workspace_root.name),
        "answer": str(payload.get("answer", "") or ""),
        "grounded_answer": str(payload.get("grounded_answer", "") or ""),
        "forced_answer": str(payload.get("forced_answer", "") or ""),
        "selected_option": str(payload.get("selected_option", "") or ""),
        "answer_mode": str(payload.get("answer_mode", "") or ""),
        "grounding_status": str(payload.get("grounding_status", "") or ""),
        "retrieval_status": str(payload.get("retrieval_status", "") or ""),
        "citations": list(payload.get("citations", ()) or ()),
        "correct": bool(payload.get("correct")),
        "verified": bool(payload.get("verified")),
        "verification_reason": str(payload.get("verification_reason", "") or ""),
        "rounds": int(payload.get("rounds", 0) or 0),
        "accepted_investigations": int(payload.get("accepted_investigations", 0) or 0),
        "workspace": str(workspace_root),
        "trace": str(Path(workspace_root) / "interactions.jsonl"),
        "skipped_completed": True,
        "models": dict(payload.get("models", {}) or {}),
    }


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    reasoner_api, investigator_api = load_role_clients(
        shared_config=args.config,
        reasoner_config=args.reasoner_config,
        investigator_config=args.investigator_config,
    )
    case_group = _load_case_group(Path(args.case_group)) if args.case_group else None
    if case_group is not None:
        case_ids = tuple(case_group["case_ids"])
        if args.construction == "single_segment":
            args.construction = str(case_group["construction"])
    else:
        case_ids = tuple(args.case_ids or DEFAULT_CASE_IDS)
    if args.mode == "long":
        if case_group is None:
            case_ids = tuple(args.case_ids or LONG_INTERLEAVED_CASE_IDS)
        if args.construction == "single_segment":
            args.construction = "interleaved_chunks"
        if float(args.min_duration_sec) == 18000.0:
            args.min_duration_sec = 21600.0
        if args.max_duration_sec is None:
            args.max_duration_sec = 25200.0
    selected = case_ids[:1] if args.mode == "smoke" else case_ids
    def run_one(case_id: str) -> Mapping[str, Any]:
        workspace_root = out_root / "workspaces" / case_id
        if args.skip_completed:
            existing = _load_existing_case_summary(workspace_root)
            if existing is not None:
                return existing
        workspace = build_or_load_workspace(
            dataset_root,
            workspace_root,
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
        result = run_case(
            workspace,
            reasoner_api=reasoner_api,
            investigator_api=investigator_api,
            max_rounds=int(args.max_rounds),
            max_investigations=int(args.max_investigations),
        )
        return {
            "case_id": result.case_id,
            "answer": result.answer,
            "grounded_answer": result.grounded_answer,
            "forced_answer": result.forced_answer,
            "selected_option": result.selected_option,
            "answer_mode": result.answer_mode,
            "grounding_status": result.grounding_status,
            "retrieval_status": result.retrieval_status,
            "citations": list(result.citations),
            "correct": result.correct,
            "verified": result.verified,
            "verification_reason": result.verification_reason,
            "rounds": result.rounds,
            "accepted_investigations": result.accepted_investigations,
            "workspace": str(workspace.root_dir),
            "trace": str(workspace.root_dir / "interactions.jsonl"),
            "skipped_completed": False,
            "models": {"reasoner": reasoner_api.model, "investigator": investigator_api.model},
        }

    summaries = list(_run_case_batch(selected, run_one, workers=int(args.workers)))
    payload = {
        "mode": args.mode,
        "case_group": None if case_group is None else case_group["group_id"],
        "case_count": len(summaries),
        "correct": sum(1 for item in summaries if item["correct"]),
        "models": {"reasoner": reasoner_api.model, "investigator": investigator_api.model},
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
        segments = _build_source_only_segments(dataset_root, target, chunk_sec=chunk_sec)
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
    reasoner_api: "OpenAICompatibleVisionClient" | None = None,
    investigator_api: "OpenAICompatibleVisionClient" | None = None,
    api: "OpenAICompatibleVisionClient" | None = None,
    max_rounds: int,
    max_investigations: int,
) -> Any:
    reasoner_api = reasoner_api or api
    investigator_api = investigator_api or api
    if reasoner_api is None or investigator_api is None:
        raise ValueError("run_case requires both reasoner_api and investigator_api")
    trace_path = workspace.root_dir / "interactions.jsonl"
    trace_path.write_text("", encoding="utf-8")
    reasoner = ReasonerAgent(reasoner_api, trace_path=trace_path, allow_visual_input=False)
    investigator = GeminiInvestigator(workspace, api=investigator_api, trace_path=trace_path)
    driver = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=investigator,
        max_rounds=max_rounds,
        max_investigations=max_investigations,
    )
    result = driver.run(workspace)
    _write_model_roles(workspace, reasoner_api=reasoner_api, investigator_api=investigator_api)
    return result


def load_role_clients(
    *,
    shared_config: str | Path | None,
    reasoner_config: str | Path | None,
    investigator_config: str | Path | None,
) -> tuple["OpenAICompatibleVisionClient", "OpenAICompatibleVisionClient"]:
    reasoner_value = reasoner_config or shared_config
    investigator_value = investigator_config or shared_config
    if not reasoner_value or not investigator_value:
        raise ValueError("Provide --config or both --reasoner-config and --investigator-config")
    reasoner_path = Path(reasoner_value)
    investigator_path = Path(investigator_value)
    return (
        OpenAICompatibleVisionClient.from_yaml(reasoner_path, section="reasoner_api"),
        OpenAICompatibleVisionClient.from_yaml(investigator_path, section="investigator_api"),
    )


def _write_model_roles(
    workspace: VirtualVideoWorkspace,
    *,
    reasoner_api: "OpenAICompatibleVisionClient",
    investigator_api: "OpenAICompatibleVisionClient",
) -> None:
    path = workspace.root_dir / "run_summary.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["models"] = {
        "reasoner": reasoner_api.model,
        "investigator": investigator_api.model,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


class OpenAICompatibleVisionClient:
    def __init__(self, planner: Mapping[str, Any]) -> None:
        self.base = str(planner["base"]).rstrip("/")
        self.model = str(planner["model"])
        self.api_key = str(planner["api_key"])
        self.api_type = str(planner.get("type", "openai_compatible") or "openai_compatible").strip().casefold()
        self.user_key = str(planner.get("user_key", "") or "")
        self.biz_scene = str(planner.get("biz_scene", "") or "")
        self.timeout = float(planner.get("timeout", 300))
        self.max_retries = max(0, int(planner.get("max_retries", 5)))
        self.retry_base_sec = max(0.0, float(planner.get("retry_base_sec", 1.0)))
        self.retry_max_sec = max(self.retry_base_sec, float(planner.get("retry_max_sec", 30.0)))
        self.retry_jitter = max(0.0, min(1.0, float(planner.get("retry_jitter", 0.2))))
        self.last_response_metadata: dict[str, Any] = {}
        for key, value in (planner.get("proxy_env") or {}).items():
            os.environ[str(key)] = str(value)

    @classmethod
    def from_yaml(cls, path: Path, *, section: str | None = None) -> "OpenAICompatibleVisionClient":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        planner = payload.get(section) if section else None
        return cls(planner or payload.get("planner_api") or payload)

    def chat(self, prompt: str, *, image_paths: Sequence[str] = (), max_tokens: int = 900) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            if Path(path).exists():
                content.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(path))}})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        }
        if "gpt-5" in self.model.casefold():
            body["max_completion_tokens"] = int(max_tokens)
        else:
            body["temperature"] = 0
            body["max_tokens"] = int(max_tokens)
        request = {
            "headers": self._headers(),
            "json": body,
            "timeout": self.timeout,
        }
        if self.api_type in {"gemini_gateway", "kuaishou_gateway"}:
            request["json"]["stream"] = False
        retryable_statuses = {408, 409, 429, 500, 502, 503, 504}
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(f"{self.base}/chat/completions", **request)
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"API request failed after {attempt + 1} attempts: {type(exc).__name__}") from exc
                time.sleep(self._retry_delay(attempt))
                continue
            if response.status_code < 400:
                payload = response.json()
                choice = payload["choices"][0]
                message = choice.get("message") or {}
                usage = payload.get("usage") or {}
                completion_details = usage.get("completion_tokens_details") or {}
                content = str(message.get("content") or "")
                self.last_response_metadata = {
                    "finish_reason": str(choice.get("finish_reason") or ""),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": completion_details.get("reasoning_tokens"),
                    "content_chars": len(content),
                    "requested_completion_tokens": int(max_tokens),
                }
                return content
            if response.status_code in retryable_statuses and attempt < self.max_retries:
                time.sleep(self._retry_delay(attempt, retry_after=response.headers.get("Retry-After")))
                continue
            snippet = response.text[:500].replace(self.api_key, "<redacted>")
            if self.user_key:
                snippet = snippet.replace(self.user_key, "<redacted>")
            raise RuntimeError(f"HTTP {response.status_code}: {snippet}")
        raise RuntimeError("API retry loop exhausted")

    def _headers(self) -> dict[str, str]:
        base = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_type in {"gemini_gateway", "kuaishou_gateway"}:
            return {
                **base,
                "x-api-key": self.api_key,
                "x-ks-user-key": self.user_key,
                "x-ks-llm-model": self.model,
                "x-ks-biz-scene": self.biz_scene,
            }
        return {**base, "Authorization": f"Bearer {self.api_key}"}

    def _retry_delay(self, attempt: int, *, retry_after: str | None = None) -> float:
        delay = self.retry_base_sec * (2 ** max(0, int(attempt)))
        try:
            delay = max(delay, float(retry_after)) if retry_after is not None else delay
        except (TypeError, ValueError):
            pass
        delay = min(self.retry_max_sec, delay)
        if self.retry_jitter:
            delay *= random.uniform(1.0 - self.retry_jitter, 1.0 + self.retry_jitter)
        return max(0.0, delay)


class GeminiReasoner:
    def __init__(
        self,
        api: OpenAICompatibleVisionClient,
        *,
        trace_path: Path,
        allow_visual_input: bool = True,
    ) -> None:
        self.api = api
        self.trace_path = trace_path
        self.allow_visual_input = bool(allow_visual_input)
        self.calls = 0
        self._last_candidate: ReasonerDecision | None = None
        self._last_audit_reason = ""
        self._audit_fingerprints: set[str] = set()

    def decide(self, **kwargs: Any) -> ReasonerDecision:
        self.calls += 1
        kwargs["reasoner_visual_input"] = self.allow_visual_input
        evidence_digest = tuple(kwargs.get("evidence_digest", ()) or ())
        if evidence_digest:
            image_paths, visual_manifest = self._visual_context(kwargs)
            prompt = _followup_prompt(kwargs, evidence_digest, visual_manifest=visual_manifest)
            raw = self.api.chat(prompt, image_paths=image_paths, max_tokens=1400)
            parsed = _normalize_reasoner_payload(self._parse_or_repair(raw, kwargs), round_id=self.calls)
            action = str(parsed.get("action") or "answer")
            self._trace(
                "reasoner_investigate" if action == "investigate" else "reasoner_answer",
                prompt,
                raw,
                parsed,
                image_paths=image_paths,
                visual_manifest=visual_manifest,
            )
            if action == "investigate":
                if kwargs.get("force_finalize") and self._last_candidate is not None:
                    return ReasonerDecision(
                        action="answer",
                        answer=self._last_candidate.answer,
                        citations=self._last_candidate.citations,
                        entity_clusters=self._last_candidate.entity_clusters,
                        support_status="insufficient",
                        support_reason=self._last_audit_reason
                        or "The model requested more investigation after the budget was exhausted.",
                    )
                if kwargs.get("force_finalize"):
                    return self._force_best_effort(kwargs, evidence_digest)
                tasks = parsed.get("tasks") or []
                return ReasonerDecision(
                    action="investigate",
                    tasks=tuple(tasks[:4]),
                    primary_gap=parsed.get("primary_gap"),
                )
            answer, nested_citations = _normalize_answer_payload(parsed.get("answer"), kwargs.get("options") or {})
            candidate = ReasonerDecision(
                action="answer",
                answer=answer,
                citations=tuple(parsed.get("citations") or nested_citations or (evidence_digest[-1]["evidence_id"],)),
                entity_clusters=tuple(parsed.get("entity_clusters") or ()),
            )
            if not candidate.answer:
                if kwargs.get("force_finalize") and self._last_candidate is not None:
                    return ReasonerDecision(
                        action="answer",
                        answer=self._last_candidate.answer,
                        citations=self._last_candidate.citations,
                        entity_clusters=self._last_candidate.entity_clusters,
                        support_status="insufficient",
                        support_reason=self._last_audit_reason or "The final model response omitted an answer.",
                    )
                if kwargs.get("force_finalize"):
                    return self._force_best_effort(kwargs, evidence_digest)
                return candidate
            if not _valid_option_answer(candidate.answer, kwargs.get("options") or {}):
                if self._last_candidate is not None:
                    return ReasonerDecision(
                        action="answer",
                        answer=self._last_candidate.answer,
                        citations=self._last_candidate.citations,
                        entity_clusters=self._last_candidate.entity_clusters,
                        support_status="insufficient",
                        support_reason="The latest response did not select a valid option; preserving the last valid candidate.",
                    )
                candidate = self._repair_invalid_answer(kwargs, evidence_digest, candidate)
            if not _valid_option_answer(candidate.answer, kwargs.get("options") or {}):
                if kwargs.get("force_finalize"):
                    return self._force_best_effort(kwargs, evidence_digest)
                return ReasonerDecision(
                    action="answer",
                    answer="",
                    citations=candidate.citations,
                    support_status="insufficient",
                    support_reason="The model did not select a valid answer option.",
                )
            self._last_candidate = candidate
            if not _should_audit_answer(kwargs):
                return candidate
            audit_fingerprint = _answer_audit_fingerprint(kwargs, candidate, evidence_digest)
            if audit_fingerprint in self._audit_fingerprints:
                return ReasonerDecision(
                    action="answer",
                    answer=candidate.answer,
                    citations=candidate.citations,
                    entity_clusters=candidate.entity_clusters,
                    support_status="insufficient",
                    support_reason=self._last_audit_reason or "An equivalent answer audit already found unresolved evidence.",
                )
            self._audit_fingerprints.add(audit_fingerprint)
            audit_prompt = _answer_audit_prompt(
                kwargs,
                candidate,
                evidence_digest,
                visual_manifest=visual_manifest,
            )
            audit_raw = self.api.chat(audit_prompt, image_paths=image_paths, max_tokens=1400)
            audit = _parse_answer_audit(audit_raw)
            self._trace(
                "reasoner_answer_audit",
                audit_prompt,
                audit_raw,
                audit,
                image_paths=image_paths,
                visual_manifest=visual_manifest,
            )
            verdict = str(audit.get("verdict", "unknown") or "unknown").strip().casefold()
            audit_reason = str(audit.get("reason", "") or "")
            self._last_audit_reason = audit_reason
            revised_answer, revised_nested_citations = _normalize_answer_payload(
                audit.get("revised_answer"),
                kwargs.get("options") or {},
            )
            if revised_answer and _valid_option_answer(revised_answer, kwargs.get("options") or {}):
                candidate = ReasonerDecision(
                    action="answer",
                    answer=revised_answer,
                    citations=tuple(audit.get("revised_citations") or revised_nested_citations or candidate.citations),
                    entity_clusters=tuple(audit.get("revised_entity_clusters") or ()),
                )
                self._last_candidate = candidate
                revised_status = str(audit.get("revised_support_status", "insufficient") or "insufficient").casefold()
                if revised_status == "supported":
                    return self._maybe_verify_claim(
                        kwargs,
                        evidence_digest,
                        ReasonerDecision(
                            action="answer",
                            answer=candidate.answer,
                            citations=candidate.citations,
                            entity_clusters=candidate.entity_clusters,
                            support_status="supported",
                            support_reason=audit_reason,
                        ),
                    )
            audit_tasks = tuple(audit.get("tasks") or ())[:2]
            if (
                verdict in {"insufficient", "contradicted"}
                and audit_tasks
                and int(kwargs.get("remaining_budget", 0) or 0) > 0
                and not kwargs.get("force_finalize")
            ):
                return ReasonerDecision(action="investigate", tasks=audit_tasks)
            decision = ReasonerDecision(
                action="answer",
                answer=candidate.answer,
                citations=candidate.citations,
                entity_clusters=candidate.entity_clusters,
                support_status=verdict,
                support_reason=audit_reason,
            )
            return self._maybe_verify_claim(kwargs, evidence_digest, decision) if verdict == "supported" else decision
        image_paths, visual_manifest = self._visual_context(kwargs)
        prompt = _investigate_prompt(kwargs, visual_manifest=visual_manifest)
        raw = self.api.chat(prompt, image_paths=image_paths, max_tokens=1400)
        parsed = _normalize_reasoner_payload(self._parse_or_repair(raw, kwargs), round_id=self.calls)
        self._trace(
            "reasoner_investigate",
            prompt,
            raw,
            parsed,
            image_paths=image_paths,
            visual_manifest=visual_manifest,
        )
        tasks = parsed.get("tasks") or []
        return ReasonerDecision(
            action="investigate",
            tasks=tuple(tasks[:4]),
            primary_gap=parsed.get("primary_gap"),
        )

    def _force_best_effort(
        self,
        kwargs: Mapping[str, Any],
        evidence_digest: Sequence[Mapping[str, Any]],
    ) -> ReasonerDecision:
        image_paths, visual_manifest = self._visual_context(kwargs)
        prompt = _forced_answer_prompt(kwargs, evidence_digest, visual_manifest=visual_manifest)
        raw = self.api.chat(prompt, image_paths=image_paths, max_tokens=4096)
        parsed = _parse_json(raw)
        self._trace(
            "reasoner_forced_answer",
            prompt,
            raw,
            parsed,
            image_paths=image_paths,
            visual_manifest=visual_manifest,
        )
        if not parsed:
            retry_prompt = _compact_forced_answer_prompt(kwargs, evidence_digest)
            retry_raw = self.api.chat(retry_prompt, max_tokens=4096)
            retry_parsed = _parse_json(retry_raw)
            self._trace("reasoner_forced_answer_retry", retry_prompt, retry_raw, retry_parsed)
            if retry_parsed:
                raw = retry_raw
                parsed = retry_parsed
        answer, nested_citations = _normalize_answer_payload(parsed.get("answer"), kwargs.get("options") or {})
        decision = ReasonerDecision(
            action="answer",
            answer=answer,
            citations=tuple(parsed.get("citations") or nested_citations or ()),
            entity_clusters=tuple(parsed.get("entity_clusters") or ()),
            support_status="insufficient",
            support_reason="Best-effort answer produced after the investigation budget was exhausted.",
        )
        if decision.answer and _valid_option_answer(decision.answer, kwargs.get("options") or {}):
            self._last_candidate = decision
        elif self._last_candidate is not None:
            return ReasonerDecision(
                action="answer",
                answer=self._last_candidate.answer,
                citations=self._last_candidate.citations,
                entity_clusters=self._last_candidate.entity_clusters,
                support_status="insufficient",
                support_reason="The best-effort response did not select a valid option; preserving the last valid candidate.",
            )
        else:
            decision = self._repair_invalid_answer(kwargs, evidence_digest, decision)
            if decision.answer and _valid_option_answer(decision.answer, kwargs.get("options") or {}):
                self._last_candidate = decision
        return decision

    def _parse_or_repair(self, raw: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        parsed = _parse_json(raw)
        if parsed:
            return parsed
        repair_prompt = (
            "Rewrite the truncated long-video Reasoner response as compact valid JSON. Preserve its intended action and tasks, "
            "but keep the gap description under 24 words, use at most 3 success conditions under 12 words each, and at most 4 tasks. "
            "Do not add markdown or commentary. If the original tasks were truncated, reconstruct the smallest useful tasks from the "
            "question and available segment IDs.\n"
            f"Question: {kwargs.get('question', '')}\nOptions: {json.dumps(kwargs.get('options') or {}, ensure_ascii=False)}\n"
            f"Workspace overview: {json.dumps(kwargs.get('workspace_overview') or {}, ensure_ascii=False)[:3500]}\n"
            f"Truncated response: {str(raw or '')[:6000]}"
        )
        repaired_raw = self.api.chat(repair_prompt, max_tokens=1100)
        repaired = _parse_json(repaired_raw)
        self._trace("reasoner_json_repair", repair_prompt, repaired_raw, repaired)
        return repaired

    def _repair_invalid_answer(
        self,
        kwargs: Mapping[str, Any],
        evidence_digest: Sequence[Mapping[str, Any]],
        candidate: ReasonerDecision,
    ) -> ReasonerDecision:
        del evidence_digest
        options = dict(kwargs.get("options") or {})
        normalized_answer = re.sub(r"[^a-z0-9]+", "", candidate.answer.casefold())
        matches = [
            (str(label), str(text))
            for label, text in options.items()
            if normalized_answer
            and normalized_answer == re.sub(r"[^a-z0-9]+", "", str(text).casefold())
        ]
        if len(matches) != 1:
            return candidate
        label, text = matches[0]
        return replace(candidate, answer=f"{label}. {text}")

    def _maybe_verify_claim(
        self,
        kwargs: Mapping[str, Any],
        evidence_digest: Sequence[Mapping[str, Any]],
        decision: ReasonerDecision,
    ) -> ReasonerDecision:
        if not _requires_independent_claim_verification(kwargs):
            return decision
        assessment = _matching_claim_assessment(evidence_digest, decision.answer)
        if assessment:
            verdict = str(assessment.get("verdict", "") or "").casefold()
            if verdict in {"refutes", "insufficient"}:
                return ReasonerDecision(
                    action="answer",
                    answer=decision.answer,
                    citations=decision.citations,
                    entity_clusters=decision.entity_clusters,
                    support_status="insufficient",
                    support_reason=str(
                        assessment.get("reason", "")
                        or "Independent claim verification did not support the answer."
                    ),
                )
            return decision
        if int(kwargs.get("remaining_budget", 0) or 0) <= 0 or kwargs.get("force_finalize"):
            return decision
        task = _claim_verification_task(kwargs, decision, evidence_digest, round_id=self.calls)
        return decision if task is None else ReasonerDecision(action="investigate", tasks=(task,))

    def _trace(
        self,
        kind: str,
        prompt: str,
        raw: str,
        parsed: Mapping[str, Any],
        *,
        image_paths: Sequence[str] = (),
        visual_manifest: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        _append_jsonl(
            self.trace_path,
            {
                "type": kind,
                "agent_role": "reasoner",
                "model": str(getattr(self.api, "model", type(self.api).__name__)),
                "visual_input_enabled": self.allow_visual_input,
                "round": self.calls,
                "prompt": prompt,
                "image_paths": list(image_paths),
                "visual_manifest": [dict(item) for item in visual_manifest],
                "raw": raw,
                "parsed": dict(parsed),
                "api_response": dict(getattr(self.api, "last_response_metadata", {}) or {}),
                "time": time.time(),
            },
        )

    def _visual_context(self, kwargs: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
        if not self.allow_visual_input:
            return (), ()
        return _reasoner_visual_context(kwargs)


ReasonerAgent = GeminiReasoner


class GeminiInvestigator(VirtualVideoInvestigator):
    def __init__(self, workspace: VirtualVideoWorkspace, *, api: OpenAICompatibleVisionClient, trace_path: Path) -> None:
        super().__init__(workspace)
        self.api = api
        self.trace_path = trace_path
        self._query_calls: dict[str, int] = {}

    def _parse_structured_observation(
        self,
        raw: str,
        *,
        query_id: str,
        prompt: str,
        image_paths: Sequence[str],
    ) -> tuple[dict[str, Any], str, str, int]:
        parsed = _parse_json(raw)
        if parsed:
            return parsed, "parsed", "", 0
        error = "invalid_or_truncated_json"
        repair_prompt = (
            "Recover the following investigator response as one compact valid JSON object. Preserve only facts explicitly "
            "present in the response or supplied images. Do not infer identity or causality. Return JSON only.\n"
            f"Investigation prompt: {prompt[:2400]}\nTruncated response: {str(raw or '')[:5000]}"
        )
        repaired_raw = self.api.chat(repair_prompt, image_paths=image_paths, max_tokens=900)
        repaired = _parse_json(repaired_raw)
        _append_jsonl(self.trace_path, {
            "type": "investigator_json_repair",
            "agent_role": "investigator",
            "model": str(getattr(self.api, "model", type(self.api).__name__)),
            "query_id": query_id,
            "prompt": repair_prompt,
            "frame_paths": list(image_paths),
            "raw": repaired_raw,
            "parsed": repaired,
            "time": time.time(),
        })
        if repaired:
            return repaired, "repaired", error, 1
        fallback = _recover_closed_json_fields(repaired_raw) or _recover_closed_json_fields(raw)
        return fallback, "fallback_extracted" if fallback else "failed", error, 1

    def _investigate_task(
        self,
        task: Any,
        *,
        prior_events: Sequence[Mapping[str, Any]] = (),
    ) -> InvestigationReport:
        if str(getattr(task, "inspection_mode", "window")) == "search_asr":
            return self._search_asr_task(task)
        query_id = str(getattr(task, "query_id", "") or "query")
        segment_packet = self.open_segment(str(getattr(task, "segment_id", "") or self.workspace.manifest.segments[0].segment_id))
        if str(getattr(task, "inspection_mode", "window")) == "enumerate_events":
            return self._investigate_event_beats(task, segment_packet)
        observation_id = self._next_observation_id(query_id)
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
        event_window = str(getattr(task, "inspection_mode", "window")) == "event_window"
        claim_window = str(getattr(task, "inspection_mode", "window")) == "verify_claim"
        if event_window:
            preview_prompt = _event_preview_prompt(self.workspace, task, segment_packet, low, prior_events=prior_events)
        elif claim_window:
            preview_prompt = _claim_preview_prompt(self.workspace, task, segment_packet, low)
        else:
            preview_prompt = _preview_prompt(self.workspace, task, segment_packet, low)
        preview_raw = self.api.chat(
            preview_prompt,
            image_paths=preview_paths,
            max_tokens=1400,
        )
        preview, parse_status, parse_error, parse_repair_calls = self._parse_structured_observation(
            preview_raw,
            query_id=query_id,
            prompt=preview_prompt,
            image_paths=preview_paths,
        )
        _append_jsonl(
            self.trace_path,
            {
                "type": "investigator_preview",
                "agent_role": "investigator",
                "model": str(getattr(self.api, "model", type(self.api).__name__)),
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
        model_image_paths = preview_paths
        region_rows: tuple[dict[str, Any], ...] = ()
        region_frame_paths: tuple[str, ...] = ()
        region_box = _normalized_region_box(preview.get("region_box"))
        parsed = preview
        raw = preview_raw
        final_prompt = preview_prompt
        detail_query_id = ""
        tool_trace = ["open_segment", "inspect_window:0.5"]
        vlm_calls = (1 if requested_window is not None else 2) + parse_repair_calls

        if _needs_highfps(task) or _truthy(preview.get("need_detail")) or region_box is not None:
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
            if region_box is not None:
                context_frames = select_uniform_items(selected_frames, min(4, len(selected_frames)))
                context_paths = tuple(str(row["path"]) for row in context_frames)
                model_box_rows = _materialize_region_crops(
                    self.workspace,
                    observation_id,
                    context_frames,
                    region_box,
                    prefix="model_box",
                    region_kind="model_box",
                )
                coarse_rows = []
                representative_frames = select_uniform_items(context_frames, min(2, len(context_frames)))
                coarse_boxes = (
                    (0.0, 0.0, 0.55, 0.55),
                    (0.45, 0.0, 1.0, 0.55),
                    (0.0, 0.45, 0.55, 1.0),
                    (0.45, 0.45, 1.0, 1.0),
                )
                for tile_index, coarse_box in enumerate(coarse_boxes, start=1):
                    coarse_rows.extend(
                        _materialize_region_crops(
                            self.workspace,
                            observation_id,
                            representative_frames,
                            coarse_box,
                            prefix=f"coarse_{tile_index}",
                            region_kind="coarse_tile",
                        )
                    )
                region_rows = (*model_box_rows, *coarse_rows)
                region_frame_paths = tuple(str(row["path"]) for row in region_rows)
                model_image_paths = (*context_paths, *region_frame_paths)
                selected_packet = {
                    **selected_packet,
                    "region_observation": {
                        "hint": str(getattr(task, "region_hint", "") or preview.get("region_hint", "") or ""),
                        "normalized_box": list(region_box),
                        "crop_count": len(region_frame_paths),
                        "ordered_image_groups": [
                            {"kind": "full_context", "count": len(context_paths)},
                            {"kind": "model_box", "count": len(model_box_rows)},
                            {"kind": "coarse_tile", "count": len(coarse_rows)},
                        ],
                    },
                }
            else:
                model_image_paths = detail_paths
            if event_window:
                final_prompt = _event_evidence_prompt(
                    self.workspace,
                    task,
                    segment_packet,
                    selected_packet,
                    preview=preview,
                    prior_events=prior_events,
                )
            elif claim_window:
                final_prompt = _claim_evidence_prompt(
                    self.workspace,
                    task,
                    segment_packet,
                    selected_packet,
                    preview=preview,
                )
            else:
                final_prompt = _evidence_prompt(self.workspace, task, segment_packet, selected_packet, preview=preview)
            raw = self.api.chat(
                final_prompt,
                image_paths=model_image_paths,
                max_tokens=1400,
            )
            parsed, parse_status, parse_error, detail_repair_calls = self._parse_structured_observation(
                raw,
                query_id=query_id,
                prompt=final_prompt,
                image_paths=model_image_paths,
            )
            tool_trace.append(f"inspect_window:{self.highfps:.1f}")
            vlm_calls += 1 + detail_repair_calls

        parsed, fallback_used = _with_explicit_measurement_fallback(
            parsed,
            raw,
            task=task,
            question=self.workspace.case.question,
        )
        if fallback_used:
            parse_status = "fallback_extracted"

        frame_paths = tuple(model_image_paths)
        frame_time_by_path = {
            str(row.get("path", "")): float(row["virtual_time_sec"])
            for row in (*selected_frames, *region_rows)
            if str(row.get("path", "")) and row.get("virtual_time_sec") is not None
        }
        frame_times = tuple(frame_time_by_path.get(path) for path in frame_paths)
        _append_jsonl(
            self.trace_path,
            {
                "type": "investigator_evidence",
                "agent_role": "investigator",
                "model": str(getattr(self.api, "model", type(self.api).__name__)),
                "query_id": query_id,
                "observation_id": observation_id,
                "preview_query_id": preview_query_id,
                "detail_query_id": detail_query_id,
                "window": list(selected_window),
                "prompt": final_prompt,
                "frame_paths": list(frame_paths),
                "region_box": list(region_box) if region_box is not None else None,
                "region_frame_paths": list(region_frame_paths),
                "raw": raw,
                "parsed": parsed,
                "structured_parse_status": parse_status,
                "structured_parse_error": parse_error,
                "time": time.time(),
            },
        )
        evidence_id = f"ev_{observation_id}_001"
        confidence = _confidence(parsed.get("confidence"), default=0.6)
        entities = _normalize_entities(
            parsed.get("entities"),
            frame_paths=frame_paths,
            frame_times=frame_times,
            observation_id=observation_id,
            window_duration_sec=max(0.0, float(selected_window[1]) - float(selected_window[0])),
        )
        events = _normalize_events(parsed.get("events"), selected_window)
        claim_assessment = _normalize_claim_assessment(parsed, task) if claim_window else {}
        target_presence = normalize_target_presence(parsed.get("target_presence"), evidence_id=evidence_id)
        measurements = normalize_measurements(parsed.get("measurements"), evidence_id=evidence_id)
        relations = normalize_relations(parsed.get("relations"), evidence_id=evidence_id)
        supports_identity_anchor = _truthy(parsed.get("supports_identity_anchor")) and any(
            bool(item.get("countable")) for item in entities
        )
        supports_answer_event = bool(events) or _truthy(parsed.get("supports_answer_event"))
        outcome = _normalize_investigation_outcome(
            parsed,
            task,
            evidence_id=evidence_id,
            has_observation=bool(frame_paths),
            claim_assessment=claim_assessment,
            entities=entities,
            events=events,
            target_presence=target_presence,
            measurements=measurements,
            relations=relations,
            region_used=bool(region_frame_paths),
            selected_window=selected_window,
        )
        if claim_assessment:
            evidence_kind = "claim_verification"
        elif supports_answer_event:
            evidence_kind = "event_observation"
        elif supports_identity_anchor or entities:
            evidence_kind = "entity_observation"
        else:
            evidence_kind = "visual_observation"
        evidence = EvidenceRecord(
            evidence_id=evidence_id,
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
                "claim_assessment": claim_assessment,
                "target_presence": to_jsonable(target_presence),
                "measurements": to_jsonable(measurements),
                "relations": to_jsonable(relations),
                "structured_parse_status": parse_status,
                "structured_parse_error": parse_error,
                "source_candidate_ids": list(getattr(task, "source_candidate_ids", ()) or ()),
                "inspection_intent": str(getattr(task, "inspection_intent", "") or ""),
                "supports_identity_anchor": supports_identity_anchor,
                "supports_answer_event": supports_answer_event,
                "investigation": outcome,
                "region_observation": {
                    "hint": str(getattr(task, "region_hint", "") or preview.get("region_hint", "") or ""),
                    "normalized_box": list(region_box) if region_box is not None else [],
                    "frame_paths": list(region_frame_paths),
                    "frames": [dict(row) for row in region_rows],
                },
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
                "region_frames": len(region_frame_paths),
                "vlm_calls": vlm_calls,
                "reused": False,
            },
            gap_id=str(getattr(task, "gap_id", "") or ""),
            resolution=str(outcome["resolution"]),
            resolved_conditions=tuple(outcome["resolved_conditions"]),
            unresolved_conditions=tuple(outcome["unresolved_conditions"]),
            failure_reason=str(outcome["failure_reason"]),
            progress_flags=tuple(outcome["progress_flags"]),
            coverage_delta=(tuple(outcome["coverage_delta"]),) if outcome["coverage_delta"] else (),
            condition_results=tuple(outcome["condition_results"]),
            goal_progress=tuple(outcome["goal_progress"]),
        )

    def _search_asr_task(self, task: Any) -> InvestigationReport:
        query_id = str(getattr(task, "query_id", "") or "search_asr")
        terms = tuple(getattr(task, "search_terms", ()) or ()) or _task_terms(task)
        packet = self.search_asr(terms, max_clusters=8)
        evidence = []
        for index, cluster in enumerate(packet["clusters"], start=1):
            start_sec, end_sec = cluster["virtual_time_range"]
            record = EvidenceRecord(
                evidence_id=f"ev_{query_id}_asr_{index:03d}",
                beat_id="",
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                modality="asr",
                pointer=f"virtual://{self.workspace.workspace_id}/asr-search/{query_id}/{index}",
                verbatim=str(cluster.get("excerpt", "") or "")[:1200],
                attestation_model="literal-asr-search",
                temporal_scope="window",
                evidence_kind="navigation_hint",
                observation_polarity="positive",
                sampling_coverage="exact",
                request_ids=(query_id,),
                coverage_manifest=(
                    CoverageSegment(query_id, float(start_sec), float(end_sec), "asr", 1.0),
                ),
                task_id=query_id,
                observation_id=f"{query_id}_asr_{index:03d}",
                confidence=min(1.0, 0.5 + 0.1 * len(cluster.get("matched_terms", ()))),
                source_lineage=tuple(dict(item) for item in cluster.get("source_lineage", ())),
                operation_metadata={
                    "navigation_only": True,
                    "search_terms": list(packet["terms"]),
                    "matched_terms": list(cluster.get("matched_terms", ())),
                    "hit_count": int(cluster.get("hit_count", 0) or 0),
                },
            )
            evidence.append(record)
            self._record_visit(task, record, status="navigation_hint")
        if not evidence:
            record = EvidenceRecord(
                evidence_id=f"ev_{query_id}_asr_empty",
                beat_id="",
                start_sec=None,
                end_sec=None,
                modality="asr",
                pointer=f"virtual://{self.workspace.workspace_id}/asr-search/{query_id}/empty",
                verbatim=f"No literal ASR matches found for terms: {', '.join(packet['terms'])}.",
                attestation_model="literal-asr-search",
                temporal_scope="workspace",
                evidence_kind="navigation_hint",
                observation_polarity="negative",
                sampling_coverage="exact",
                request_ids=(query_id,),
                task_id=query_id,
                observation_id=f"{query_id}_asr_empty",
                confidence=1.0,
                operation_metadata={
                    "navigation_only": True,
                    "matched_terms": [],
                    "search_terms": list(packet["terms"]),
                    "hit_count": 0,
                },
            )
            evidence.append(record)
            self._record_visit(task, record, status="navigation_hint")
        _append_jsonl(
            self.trace_path,
            {
                "type": "investigator_asr_search",
                "agent_role": "investigator",
                "model": str(getattr(self.api, "model", type(self.api).__name__)),
                "query_id": query_id,
                "search_terms": list(packet["terms"]),
                "clusters": list(packet["clusters"]),
                "time": time.time(),
            },
        )
        conditions = tuple(getattr(task, "conditions", ()) or ())
        has_hits = bool(packet["clusters"])
        evidence_ids = tuple(record.evidence_id for record in evidence if record.observation_polarity == "positive")
        condition_results = tuple(
            ConditionResult(
                condition.condition_id,
                "satisfied"
                if has_hits and condition.condition_type == "lexical_navigation"
                else "unknown",
                "Literal ASR matches were returned." if has_hits else "No literal ASR matches were returned.",
                evidence_ids=evidence_ids if has_hits else (),
            )
            for condition in conditions
        )
        resolution = derive_resolution(conditions, condition_results) if conditions else "resolved"
        resolved = tuple(
            condition.description
            for condition, result in zip(conditions, condition_results)
            if result.status == "satisfied"
        )
        unresolved = tuple(
            condition.description
            for condition, result in zip(conditions, condition_results)
            if result.status != "satisfied"
        )
        goal_progress = tuple(
            f"condition_satisfied:{result.condition_id}"
            for result in condition_results
            if result.status == "satisfied"
        )
        return InvestigationReport(
            query_id=query_id,
            status="satisfied",
            evidence=tuple(evidence),
            cost={
                "tool_trace": ("search_asr",),
                "frames": 0,
                "vlm_calls": 0,
                "reused": False,
                "hit_clusters": len(evidence),
            },
            gap_id=str(getattr(task, "gap_id", "") or ""),
            resolution=resolution,
            resolved_conditions=resolved,
            unresolved_conditions=unresolved,
            progress_flags=("lexical_navigation_completed", *goal_progress),
            coverage_delta=tuple(
                (float(record.start_sec), float(record.end_sec))
                for record in evidence
                if record.start_sec is not None and record.end_sec is not None
            ),
            condition_results=condition_results,
            goal_progress=goal_progress,
        )

    def _investigate_event_beats(self, task: Any, segment_packet: Mapping[str, Any]) -> InvestigationReport:
        windows = _event_enumeration_windows(segment_packet)
        reports = []
        prior_events: tuple[Mapping[str, Any], ...] = ()
        for window in windows:
            beat_task = InvestigationTask(
                query_id=str(getattr(task, "query_id", "") or "query"),
                goal=str(getattr(task, "goal", "") or ""),
                segment_id=str(getattr(task, "segment_id", "") or ""),
                time_range=window,
                modality_hint=tuple(getattr(task, "modality_hint", ()) or ()),
                expected_evidence=str(getattr(task, "expected_evidence", "") or ""),
                inspection_mode="event_window",
                priority=float(getattr(task, "priority", 0.0) or 0.0),
                gap_id=str(getattr(task, "gap_id", "") or ""),
                success_conditions=tuple(getattr(task, "success_conditions", ()) or ()),
                conditions=tuple(getattr(task, "conditions", ()) or ()),
                source_candidate_ids=tuple(getattr(task, "source_candidate_ids", ()) or ()),
                inspection_intent=str(getattr(task, "inspection_intent", "") or ""),
                direction=str(getattr(task, "direction", "") or ""),
                preferred_ranges=tuple(getattr(task, "preferred_ranges", ()) or ()),
                excluded_ranges=tuple(getattr(task, "excluded_ranges", ()) or ()),
            )
            report = self._investigate_task(beat_task, prior_events=prior_events)
            reports.append(report)
            prior_events = _ending_event_context(report.evidence, window)
        evidence = tuple(record for report in reports for record in report.evidence)
        condition_results = _merge_condition_results(reports)
        goal_progress = tuple(
            f"condition_{result.status}:{result.condition_id}"
            for result in condition_results
            if result.status in {"satisfied", "contradicted"}
        )
        return InvestigationReport(
            query_id=str(getattr(task, "query_id", "") or ""),
            status="satisfied" if evidence else "empty",
            evidence=evidence,
            cost={
                "beat_windows": len(windows),
                "tool_trace": tuple(step for report in reports for step in report.cost.get("tool_trace", ())),
                "preview_frames": sum(int(report.cost.get("preview_frames", 0) or 0) for report in reports),
                "detail_frames": sum(int(report.cost.get("detail_frames", 0) or 0) for report in reports),
                "frames": sum(int(report.cost.get("frames", 0) or 0) for report in reports),
                "vlm_calls": sum(int(report.cost.get("vlm_calls", 0) or 0) for report in reports),
                "reused": bool(reports) and all(bool(report.cost.get("reused")) for report in reports),
            },
            gap_id=str(getattr(task, "gap_id", "") or ""),
            resolution=(
                "resolved"
                if evidence and all(report.resolution == "resolved" for report in reports)
                else "partial"
                if evidence
                else "unresolved"
            ),
            resolved_conditions=tuple(
                dict.fromkeys(condition for report in reports for condition in report.resolved_conditions)
            ),
            unresolved_conditions=tuple(
                dict.fromkeys(condition for report in reports for condition in report.unresolved_conditions)
            ),
            failure_reason="; ".join(
                dict.fromkeys(report.failure_reason for report in reports if report.failure_reason)
            ),
            progress_flags=tuple(
                dict.fromkeys(flag for report in reports for flag in report.progress_flags)
            ),
            coverage_delta=tuple(
                interval for report in reports for interval in report.coverage_delta
            ),
            condition_results=condition_results,
            goal_progress=goal_progress,
        )

    def _next_observation_id(self, query_id: str) -> str:
        call_index = self._query_calls.get(query_id, 0) + 1
        self._query_calls[query_id] = call_index
        return f"{query_id}_c{call_index:02d}"


def _select_window_with_model(api: OpenAICompatibleVisionClient, task: Any, segment_packet: Mapping[str, Any], trace_path: Path) -> tuple[float, float]:
    prompt = (
        "You are an Investigator. Choose the most relevant virtual-time window inside this segment.\n"
        "Times or numbers mentioned in the question/options may be visible content such as a game clock; do not reinterpret them "
        "as virtual timestamps unless the task explicitly supplies a virtual time_range. Follow the requested direction and avoid excluded ranges.\n"
        "Return JSON only: {\"start_sec\": float, \"end_sec\": float, \"reason\": string}.\n"
        f"Task: {getattr(task, 'goal', '')}\nExpected evidence: {getattr(task, 'expected_evidence', '')}\n"
        f"Success conditions: {json.dumps(list(getattr(task, 'success_conditions', ()) or ()), ensure_ascii=False)}\n"
        f"Direction: {getattr(task, 'direction', '')}\nExcluded ranges: {json.dumps(list(getattr(task, 'excluded_ranges', ()) or ()), ensure_ascii=False)}\n"
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
            "agent_role": "investigator",
            "model": str(getattr(api, "model", type(api).__name__)),
            "query_id": getattr(task, "query_id"),
            "raw": raw,
            "parsed": parsed,
            "selected_window": list(selected),
            "fallback_used": fallback_used,
            "time": time.time(),
        },
    )
    return selected


def _event_enumeration_windows(
    segment_packet: Mapping[str, Any],
    *,
    max_windows: int = 12,
) -> tuple[tuple[float, float], ...]:
    beats = tuple(segment_packet.get("beats", ()) or ())
    if not beats:
        start, end = segment_packet["virtual_time_range"]
        return ((float(start), float(end)),)
    limit = max(1, int(max_windows))
    group_size = max(1, (len(beats) + limit - 1) // limit)
    windows = []
    for index in range(0, len(beats), group_size):
        group = beats[index : index + group_size]
        start = float(group[0]["virtual_time_range"][0])
        end = float(group[-1]["virtual_time_range"][1])
        if end > start:
            windows.append((start, end))
    return tuple(windows)


def _ending_event_context(
    evidence: Sequence[EvidenceRecord],
    window: tuple[float, float],
) -> tuple[Mapping[str, Any], ...]:
    window_end = float(window[1])
    rows = []
    for record in evidence:
        for event in record.operation_metadata.get("events", ()) or ():
            if not isinstance(event, Mapping):
                continue
            try:
                event_end = float(event.get("end_sec"))
            except (TypeError, ValueError):
                continue
            if not (_truthy(event.get("continues_to_next")) or abs(event_end - window_end) <= 2.0):
                continue
            rows.append(
                {
                    "event_key": str(event.get("event_key", "") or ""),
                    "description": str(event.get("description", "") or ""),
                    "start_sec": event.get("start_sec"),
                    "end_sec": event.get("end_sec"),
                    "continues_to_next": _truthy(event.get("continues_to_next")),
                }
            )
    return tuple(rows[-4:])


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
    minimum = _minimum_detail_duration(task, preview_end - preview_start)
    if end - start < minimum:
        center = (start + end) / 2.0
        start = max(preview_start, center - minimum / 2.0)
        end = min(preview_end, start + minimum)
        if end - start < minimum:
            start = max(preview_start, end - minimum)
    return round(start, 3), round(end, 3)


def _normalized_region_box(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, Mapping):
        raw = (value.get("x1"), value.get("y1"), value.get("x2"), value.get("y2"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        raw = tuple(value)
    else:
        return None
    try:
        x1, y1, x2, y2 = (max(0.0, min(1.0, float(item))) for item in raw)
    except (TypeError, ValueError):
        return None
    if x2 - x1 < 0.03 or y2 - y1 < 0.03:
        return None
    return x1, y1, x2, y2


def _materialize_region_crops(
    workspace: VirtualVideoWorkspace,
    observation_id: str,
    frames: Sequence[Mapping[str, Any]],
    region_box: tuple[float, float, float, float],
    *,
    prefix: str = "region",
    region_kind: str = "model_box",
) -> tuple[dict[str, Any], ...]:
    out_dir = workspace.root_dir / "observations" / "regions" / observation_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    x1, y1, x2, y2 = region_box
    for index, frame in enumerate(frames, start=1):
        source = Path(str(frame.get("path", "") or ""))
        if not source.exists():
            continue
        try:
            with Image.open(source) as opened:
                image = opened.convert("RGB")
                left = max(0, min(image.width - 1, int(round(x1 * image.width))))
                top = max(0, min(image.height - 1, int(round(y1 * image.height))))
                right = max(left + 1, min(image.width, int(round(x2 * image.width))))
                bottom = max(top + 1, min(image.height, int(round(y2 * image.height))))
                crop = image.crop((left, top, right, bottom))
                crop = ImageEnhance.Contrast(crop).enhance(1.35)
                target_long_edge = min(1024, max(512, max(crop.size)))
                scale = min(4.0, target_long_edge / max(1, max(crop.size)))
                if scale > 1.01:
                    crop = crop.resize(
                        (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))),
                        Image.Resampling.LANCZOS,
                    )
                path = out_dir / f"{prefix}_{index:03d}.jpg"
                crop.save(path, quality=94)
        except OSError:
            continue
        rows.append(
            {
                **dict(frame),
                "path": str(path),
                "parent_path": str(source),
                "region_box": list(region_box),
                "region_kind": str(region_kind),
            }
        )
    return tuple(rows)


def _normalize_investigation_outcome(
    parsed: Mapping[str, Any],
    task: Any,
    *,
    evidence_id: str,
    has_observation: bool,
    claim_assessment: Mapping[str, Any],
    entities: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    target_presence: TargetPresenceFact,
    measurements: Sequence[MeasurementFact],
    relations: Sequence[RelationFact],
    region_used: bool,
    selected_window: tuple[float, float],
) -> dict[str, Any]:
    conditions = tuple(
        item if isinstance(item, GapCondition) else GapCondition(**dict(item))
        for item in tuple(getattr(task, "conditions", ()) or ())
        if isinstance(item, (GapCondition, Mapping))
    )
    condition_results = normalize_condition_results(
        parsed.get("condition_results"),
        conditions,
        evidence_id=evidence_id,
        target_presence=target_presence,
        measurements=measurements,
        relations=relations,
    )
    if conditions:
        resolution = derive_resolution(conditions, condition_results)
    elif claim_assessment and str(claim_assessment.get("verdict", "") or "") in {"supports", "refutes"}:
        resolution = "resolved"
    else:
        resolution = "resolved" if has_observation and bool(str(parsed.get("summary", "") or "").strip()) else "unresolved"
    by_id = {condition.condition_id: condition for condition in conditions}
    resolved = [
        by_id[result.condition_id].description
        for result in condition_results
        if result.status == "satisfied" and result.condition_id in by_id
    ]
    unresolved = [
        condition.description
        for condition in conditions
        if not any(
            result.condition_id == condition.condition_id and result.status == "satisfied"
            for result in condition_results
        )
    ]
    failure_reason = str(parsed.get("failure_reason", "") or "").strip()
    if resolution != "resolved" and not failure_reason:
        failure_reason = str(parsed.get("reason", "") or "semantic success conditions remain unresolved").strip()
    progress = list(_string_items(parsed.get("progress_flags")))
    goal_progress = [
        f"condition_{result.status}:{result.condition_id}"
        for result in condition_results
        if result.status in {"satisfied", "contradicted"}
    ]
    progress.extend(goal_progress)
    if entities:
        progress.append("entity_observed")
    if events:
        progress.append("event_observed")
    if region_used:
        progress.append("region_observed")
    return {
        "gap_id": str(getattr(task, "gap_id", "") or ""),
        "resolution": resolution,
        "resolved_conditions": list(dict.fromkeys(resolved)),
        "unresolved_conditions": list(dict.fromkeys(unresolved)),
        "failure_reason": failure_reason,
        "progress_flags": list(dict.fromkeys(progress)),
        "goal_progress": list(dict.fromkeys(goal_progress)),
        "condition_results": condition_results,
        "coverage_delta": [float(selected_window[0]), float(selected_window[1])] if has_observation else [],
    }


def _merge_condition_results(reports: Sequence[InvestigationReport]) -> tuple[ConditionResult, ...]:
    by_id: dict[str, list[ConditionResult]] = {}
    for report in reports:
        for result in report.condition_results:
            by_id.setdefault(result.condition_id, []).append(result)
    merged = []
    for condition_id, rows in by_id.items():
        if rows and all(row.status == "satisfied" for row in rows):
            status = "satisfied"
        elif any(row.status == "contradicted" for row in rows):
            status = "contradicted"
        else:
            status = "unknown"
        merged.append(
            ConditionResult(
                condition_id=condition_id,
                status=status,
                observation="; ".join(dict.fromkeys(row.observation for row in rows if row.observation))[:800],
                evidence_ids=tuple(
                    dict.fromkeys(evidence_id for row in rows for evidence_id in row.evidence_ids)
                ),
            )
        )
    return tuple(merged)


def _string_items(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, Sequence):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _minimum_detail_duration(task: Any, preview_duration: float) -> float:
    text = " ".join(
        [
            str(getattr(task, "goal", "") or ""),
            str(getattr(task, "expected_evidence", "") or ""),
            str(getattr(task, "claim_relation", "") or ""),
            " ".join(str(item) for item in getattr(task, "modality_hint", ()) or ()),
        ]
    ).casefold()
    requires_context = any(
        marker in text
        for marker in (
            "temporal context",
            "causal",
            "cause",
            "action",
            "motion",
            "sequence",
            "before",
            "after",
            "injur",
            "sustain",
            "drag",
            "fall",
        )
    )
    return min(30.0, max(0.0, float(preview_duration))) if requires_context else 0.0


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def _confidence(value: Any, *, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return float(default)


ENTITY_WITNESS_MAX_WINDOW_SEC = 120.0


def _normalize_entities(
    value: Any,
    *,
    frame_paths: Sequence[str] = (),
    frame_times: Sequence[float | None] = (),
    observation_id: str = "",
    window_duration_sec: float | None = None,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    available_frames = tuple(str(path) for path in frame_paths if str(path).strip())
    available_times = tuple(frame_times)
    rows = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            continue
        description = str(item.get("description", "") or "").strip()
        if not description:
            continue
        local_id = str(item.get("local_id", "") or f"person_{index}")
        visual_signature = str(item.get("visual_signature", "") or description).strip()
        raw_indices = item.get("frame_indices", ())
        if not isinstance(raw_indices, Sequence) or isinstance(raw_indices, (str, bytes)):
            raw_indices = ()
        frame_indices = []
        for raw_index in raw_indices:
            try:
                frame_index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if 0 <= frame_index < len(available_frames) and frame_index not in frame_indices:
                frame_indices.append(frame_index)
        witness_frame_refs = tuple(available_frames[frame_index] for frame_index in frame_indices)
        witness_virtual_times_sec = tuple(
            float(available_times[frame_index])
            for frame_index in frame_indices
            if frame_index < len(available_times) and available_times[frame_index] is not None
        )
        supports_question_relation = _truthy(item.get("supports_question_relation"))
        candidate_reason = ""
        if not witness_frame_refs:
            candidate_reason = "missing_frame_witness"
        elif not visual_signature:
            candidate_reason = "missing_visual_signature"
        elif not supports_question_relation:
            candidate_reason = "question_relation_unverified"
        elif window_duration_sec is not None and float(window_duration_sec) > ENTITY_WITNESS_MAX_WINDOW_SEC:
            candidate_reason = "coarse_window_candidate"
        entity_observation_id = f"{observation_id}:{local_id}" if observation_id else local_id
        rows.append(
            {
                "local_id": local_id,
                "entity_observation_id": entity_observation_id,
                "description": description,
                "visual_signature": visual_signature,
                "role": str(item.get("role", "") or ""),
                "question_relation": str(item.get("question_relation", "") or ""),
                "supports_question_relation": supports_question_relation,
                "frame_indices": frame_indices,
                "witness_frame_refs": list(witness_frame_refs),
                "witness_virtual_times_sec": list(witness_virtual_times_sec),
                "witness_count": len(witness_frame_refs),
                "countable": not candidate_reason,
                "candidate_only": bool(candidate_reason),
                "candidate_reason": candidate_reason,
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
                "event_key": " ".join(str(item.get("event_key", "") or "").strip().casefold().split()),
                "description": description,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "supports_question_event": True,
                "continues_from_previous": _truthy(item.get("continues_from_previous")),
                "continues_to_next": _truthy(item.get("continues_to_next")),
            }
        )
    return tuple(rows)


def _normalize_claim_assessment(value: Mapping[str, Any], task: Any) -> dict[str, Any]:
    verdict = str(value.get("claim_verdict", "insufficient") or "insufficient").strip().casefold()
    if verdict not in {"supports", "refutes", "insufficient"}:
        verdict = "insufficient"
    claim_relation = str(getattr(task, "claim_relation", "") or "").strip().casefold()
    candidate_role = str(value.get("candidate_role", "unclear") or "unclear").strip().casefold()
    reason = str(value.get("reason", "") or "")
    if verdict == "supports" and not _claim_role_satisfies(claim_relation, candidate_role):
        verdict = "insufficient"
        mismatch = f"Candidate role {candidate_role or 'unclear'} does not satisfy required claim relation {claim_relation}."
        reason = f"{mismatch} {reason}".strip()
    return {
        "candidate_answer": str(getattr(task, "claim_to_verify", "") or ""),
        "alternative_answers": tuple(str(item) for item in getattr(task, "alternative_answers", ()) or ()),
        "verdict": verdict,
        "claim_relation": claim_relation,
        "relation_type": str(value.get("relation_type", "unclear") or "unclear"),
        "candidate_role": candidate_role,
        "strongest_alternative": str(value.get("strongest_alternative", "") or ""),
        "reason": reason,
    }


def _claim_role_satisfies(claim_relation: str, candidate_role: str) -> bool:
    allowed = {
        "decision_motive": {"decision_motive", "initiating_cause"},
        "initiating_cause_or_mechanism": {"initiating_cause", "mechanism"},
        "identity_linked_cause": {"initiating_cause", "mechanism"},
        "opinion": {"opinion_statement"},
        "identity_link": {"identity_match"},
    }
    return not claim_relation or candidate_role in allowed.get(claim_relation, {candidate_role})


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


def _build_source_only_segments(
    dataset_root: Path,
    target: Mapping[str, Any],
    *,
    chunk_sec: float,
) -> tuple[VirtualVideoSegment, ...]:
    video_id = str(target["videoID"])
    duration = float(_duration(dataset_root, video_id))
    chunk = max(1.0, float(chunk_sec))
    segments = []
    start = 0.0
    index = 1
    while start < duration:
        end = min(duration, start + chunk)
        segments.append(
            VirtualVideoSegment(
                segment_id=f"seg_{index:04d}",
                source_video_id=video_id,
                source_path=str(dataset_root / "video" / f"{video_id}.mp4"),
                source_start_sec=round(start, 3),
                source_end_sec=round(end, 3),
                virtual_start_sec=round(start, 3),
                virtual_end_sec=round(end, 3),
                role="target",
            )
        )
        start = end
        index += 1
    return tuple(segments)


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


def _reasoner_visual_context(
    kwargs: Mapping[str, Any],
    *,
    max_images: int = 40,
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    budget = max(1, int(max_images))
    overview_rows = tuple(dict(row) for row in kwargs.get("workspace_overview", {}).get("segment_overviews", ()) or ())
    evidence = tuple(kwargs.get("evidence", ()) or ())
    aggregation = str(dict(kwargs.get("query_contract") or {}).get("aggregation", "none") or "none")
    if aggregation == "count":
        evidence_budget = 0
    elif aggregation == "deduplicate":
        evidence_budget = min(16, budget)
    else:
        evidence_budget = min(32, budget)
    evidence_paths: list[str] = []
    evidence_manifest: list[dict[str, Any]] = []
    if evidence and evidence_budget:
        selected_evidence = select_uniform_items(evidence, min(16, len(evidence), evidence_budget))
        per_record = 1 if aggregation == "deduplicate" else min(4, max(1, evidence_budget // len(selected_evidence)))
        for record in selected_evidence:
            refs = tuple(str(path) for path in getattr(record, "frame_refs", ()) if Path(str(path)).exists())
            selected_refs = select_uniform_items(refs, min(per_record, len(refs))) if refs else ()
            if not selected_refs:
                continue
            start_position = len(evidence_paths) + 1
            evidence_paths.extend(selected_refs)
            evidence_manifest.append(
                {
                    "kind": "evidence",
                    "evidence_id": str(getattr(record, "evidence_id", "") or ""),
                    "virtual_time_range": [
                        float(getattr(record, "start_sec", 0.0) or 0.0),
                        float(getattr(record, "end_sec", 0.0) or 0.0),
                    ],
                    "local_image_positions": [start_position, len(evidence_paths)],
                }
            )
    evidence_paths = evidence_paths[:evidence_budget]
    overview_budget = min(8 if evidence_paths else budget, max(0, budget - len(evidence_paths)))
    valid_overviews = tuple(
        row
        for row in overview_rows
        if str(row.get("overview_thumbnail_grid_path", "") or "")
        and Path(str(row.get("overview_thumbnail_grid_path"))).exists()
    )
    selected_overviews = select_uniform_items(valid_overviews, min(overview_budget, len(valid_overviews))) if valid_overviews else ()
    overview_paths = tuple(str(row["overview_thumbnail_grid_path"]) for row in selected_overviews)
    manifest: list[dict[str, Any]] = []
    for index, row in enumerate(selected_overviews, start=1):
        manifest.append(
            {
                "kind": "overview",
                "segment_id": str(row.get("segment_id", "") or ""),
                "image_positions": [index, index],
            }
        )
    evidence_offset = len(overview_paths)
    for item in evidence_manifest:
        local_start, local_end = item.pop("local_image_positions")
        if local_start > len(evidence_paths):
            continue
        manifest.append(
            {
                **item,
                "image_positions": [evidence_offset + local_start, evidence_offset + min(local_end, len(evidence_paths))],
            }
        )
    return (*overview_paths, *evidence_paths), tuple(manifest)


def _visual_manifest_prompt(visual_manifest: Sequence[Mapping[str, Any]]) -> str:
    if not visual_manifest:
        return ""
    return (
        "\nRe-check replayed evidence images directly; Investigator summaries are fallible claims, not ground truth. "
        "Visual input manifest (1-indexed supplied-image positions; overview images are coarse and evidence images are "
        "Investigator observations): "
        f"{json.dumps([dict(item) for item in visual_manifest], ensure_ascii=False)}"
    )


def _investigate_prompt(
    kwargs: Mapping[str, Any],
    *,
    visual_manifest: Sequence[Mapping[str, Any]] = (),
) -> str:
    context_description = (
        "You receive segment overview images plus metadata."
        if kwargs.get("reasoner_visual_input", True)
        else "You are text-only. You receive segment/ASR metadata and structured Investigator evidence; image paths are provenance, not visible evidence."
    )
    return (
        f"You are the Reasoner for a long virtual video QA agent. {context_description}\n"
        "Identify one primary observable evidence gap, then dispatch up to 4 tasks that directly test its success conditions. "
        "One gap may use several parallel windows. Return JSON only: "
        "{\"action\":\"investigate\",\"primary_gap\":{\"gap_id\":\"gap_r1\",\"description\":\"observable unknown\","
        "\"success_conditions\":[\"condition 1\"],\"falsification_conditions\":[]},\"tasks\":[{\"query_id\":\"r1_t1\",\"goal\":\"...\","
        "\"segment_id\":\"seg_0001\",\"time_range\":null,\"inspection_mode\":\"window|search_asr\","
        "\"search_terms\":[],\"modality_hint\":[\"visual\"],\"expected_evidence\":\"...\","
        "\"gap_id\":\"gap_r1\",\"success_conditions\":[\"condition 1\"],\"direction\":\"local|forward|backward|global\","
        "\"region_hint\":\"optional visible region\"}]}.\n"
        "The gap must be a neutral fact that can be observed, not an answer option or a request for more related frames. "
        "Keep the gap description under 24 words, use at most 3 success conditions under 12 words each, and keep each task goal under 30 words. "
        "Clock-like or numeric text in answer options is visual content to read, not a virtual timestamp, unless temporal_navigation explicitly maps it.\n"
        "search_asr is optional literal grep over raw timestamped ASR. Use 1-5 distinctive exact terms from the question or options "
        "when the overview cannot locate a dialogue/fact. Its hits are navigation only: after a hit, inspect that segment/time window "
        "visually before answering, and never cite an ASR search hint as final visual evidence.\n"
        "For identity or cause questions with answer options, prefer one contrastive search_asr task containing one distinctive term "
        "from every materially different competing option, up to 5 total terms. This preserves task slots and prevents committing to "
        "the first lexical hit. Do not repeat an identical lexical search after any navigation result: inspect positive hit windows "
        "visually, and change terms after a negative result.\n"
        "For full-video count questions, evidence from one chunk is only a source hypothesis, not complete proof. For distinct-person "
        "counts, a window longer than 120 seconds is candidate discovery only; follow it with narrower windows that can bind each "
        "person to explicit frame witnesses and stable appearance attributes.\n"
        "For total event-count questions, dispatch segment tasks that enumerate atomic event occurrences with timestamps.\n"
        "For scalar_quantity questions, inspect each relevant displayed operand or checkpoint and ask for its unit, delta or cumulative "
        "semantics, and relation to the stated boundary. Do not infer a quantity from answer options.\n"
        "For source-relative minute questions, prioritize the supplied temporal_navigation candidate segments.\n"
        "For identity-anchor questions, first locate evidence matching every identity_anchor_term before investigating the later event.\n"
        f"Question: {kwargs['question']}\nOptions: {json.dumps(kwargs['options'], ensure_ascii=False)}\n"
        f"Query contract: {json.dumps(kwargs.get('query_contract') or {}, ensure_ascii=False)}\n"
        f"Query requirements: {json.dumps(kwargs.get('query_requirements') or {}, ensure_ascii=False)}\n"
        f"Temporal navigation: {json.dumps(kwargs.get('temporal_navigation') or {}, ensure_ascii=False)}\n"
        f"Workspace overview: {json.dumps(kwargs['workspace_overview'], ensure_ascii=False)[:6000]}"
        f"{_visual_manifest_prompt(visual_manifest)}"
    )


def _followup_prompt(
    kwargs: Mapping[str, Any],
    evidence_digest: Sequence[Mapping[str, Any]],
    *,
    visual_manifest: Sequence[Mapping[str, Any]] = (),
) -> str:
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
        "\"evidence_ids\":[\"ev_...\"],\"entity_observation_ids\":[\"observation:person_1\"]}]}.\n"
        "If not enough, identify the single primary observable evidence gap and return JSON only: "
        "{\"action\":\"investigate\",\"primary_gap\":{\"gap_id\":\"gap_r2\",\"description\":\"observable unknown\","
        "\"success_conditions\":[\"condition 1\"],\"falsification_conditions\":[]},\"tasks\":[{\"query_id\":\"r2_t1\","
        "\"goal\":\"...\",\"segment_id\":\"seg_0001\",\"time_range\":null,"
        "\"inspection_mode\":\"window|search_asr\",\"search_terms\":[],\"modality_hint\":[\"visual\"],"
        "\"expected_evidence\":\"...\",\"gap_id\":\"gap_r2\",\"success_conditions\":[\"condition 1\"],"
        "\"direction\":\"local|forward|backward|global\",\"region_hint\":\"optional visible region\"}]}.\n"
        "You may request up to 4 more segments/windows. Do not answer with insufficient evidence.\n"
        "Keep the gap description under 24 words, use at most 3 success conditions under 12 words each, and keep each task goal under 30 words.\n"
        "A returned frame is not proof that a gap was resolved. Use the investigation outcomes below: partial/unresolved results must change "
        "range, direction, modality, or region focus rather than repeat the same request. Clock-like or numeric option text is content to read, "
        "not a virtual timestamp, unless temporal_navigation explicitly maps it.\n"
        "Evidence with evidence_kind=navigation_hint comes from literal ASR grep and is navigation only. Dispatch a visual window "
        "inspection at its segment/time range before using that clue in an answer.\n"
        "For identity or cause questions, use one contrastive search_asr task with one distinctive term from every materially different "
        "competing option, up to 5 terms, then visually inspect unresolved positive clusters before committing. Do not repeat an "
        "identical lexical search after any navigation result: inspect positive hit windows, and change terms after a negative result.\n"
        "If completion_status.grounded_ready is false, prefer a targeted repair for missing segments, unresolved critical conditions, "
        "unsupported claim atoms, or high-value unseen candidates. Do not claim grounded support merely because a best choice exists; "
        "when no actionable repair remains, preserve the best choice honestly rather than looping. "
        "For a final full-video answer, cite every relevant visual evidence record from the adopted source.\n"
        "For distinct-count questions, reconcile the per-evidence entities by stable appearance. Do not add local counts. "
        "Create one entity_cluster per unique person, list every adopted entity_observation_id in that cluster, and make the "
        "option count equal the number of clusters. Only entities with countable=true are admissible. A summary-level person, "
        "candidate_only entity, parse-failed observation, or broad-window count without a frame witness is navigation context, "
        "not count evidence. When Latest answer-gate feedback reports entity_cluster_witness_missing, investigate narrower repair "
        "windows instead of repeating the answer. Candidate entities include witness_virtual_times_sec; inspect a narrow window "
        "around those timestamps to promote or reject them.\n"
        "For total event-count questions, count only the timestamped events rows in evidence, deduplicate overlapping observations, "
        "cite every evidence record containing a positive occurrence, and never infer the count from answer options or entity clusters.\n"
        "For scalar_quantity questions, use only structured measurement facts with the requested unit. Distinguish delta or cumulative "
        "measurements, exclude observations after the stated boundary, and cite every operand used in the derivation.\n"
        "For identity-anchor questions, do not answer while missing_identity_anchor_terms is non-empty. "
        "The final entity cluster must cite both anchor evidence and the later event evidence for the same person.\n"
        "Treat independent claim_assessment evidence as a direct check of the proposed relation. If it refutes a candidate, "
        "revise the answer or investigate the strongest alternative; do not relabel relevance as support. If candidate_role "
        "does not satisfy claim_relation, investigate the missing relation or choose a better-supported option; a stated use, "
        "downstream consequence, or after-state is not a decision motive.\n"
        f"Question: {kwargs['question']}\nOptions: {json.dumps(kwargs['options'], ensure_ascii=False)}\n"
        f"Query contract: {json.dumps(kwargs.get('query_contract') or {}, ensure_ascii=False)}\n"
        f"Query requirements: {json.dumps(kwargs.get('query_requirements') or {}, ensure_ascii=False)}\n"
        f"Completion status: {json.dumps(kwargs.get('completion_status') or {}, ensure_ascii=False)}\n"
        f"Temporal navigation: {json.dumps(kwargs.get('temporal_navigation') or {}, ensure_ascii=False)}\n"
        f"Workspace overview: {json.dumps(kwargs['workspace_overview'], ensure_ascii=False)[:6000]}\n"
        f"Evidence so far: {json.dumps(list(evidence_digest), ensure_ascii=False)}\n"
        f"Investigation outcomes: {json.dumps(list(kwargs.get('investigation_outcomes') or ()), ensure_ascii=False)}\n"
        f"Navigation candidates: {json.dumps(list(kwargs.get('navigation_candidates') or ()), ensure_ascii=False)}\n"
        f"Stagnation status: {json.dumps(kwargs.get('stagnation_status') or {}, ensure_ascii=False)}\n"
        f"Latest answer-gate feedback: {json.dumps(kwargs.get('answer_gate_feedback') or {}, ensure_ascii=False)}"
        f"{_visual_manifest_prompt(visual_manifest)}"
    )


def _should_audit_answer(kwargs: Mapping[str, Any]) -> bool:
    question = str(kwargs.get("question", "") or "").casefold()
    contract = dict(kwargs.get("query_contract") or {})
    requirements = dict(kwargs.get("query_requirements") or {})
    if requirements.get("requires_identity_link"):
        return True
    if str(contract.get("required_scope", "") or "") == "full_video":
        return True
    relation_patterns = (
        r"\bwhy\b",
        r"\bhow\s+(?:did|does|do|was|were)\b",
        r"\b(?:before|after)\b",
        r"\bnot\s+(?:true|correct)\b",
        r"\brelationship\b",
        r"\b(?:view|opinion|believ\w*)\b",
    )
    return any(re.search(pattern, question) for pattern in relation_patterns)


def _requires_independent_claim_verification(kwargs: Mapping[str, Any]) -> bool:
    question = str(kwargs.get("question", "") or "").casefold()
    requirements = dict(kwargs.get("query_requirements") or {})
    if requirements.get("requires_identity_link"):
        return True
    relation_patterns = (
        r"\bwhy\b",
        r"\b(?:cause|reason|motive|because)\b",
        r"\brelationship\b",
        r"\b(?:view|opinion|believ\w*)\b",
    )
    return any(re.search(pattern, question) for pattern in relation_patterns)


def _matching_claim_assessment(
    evidence_digest: Sequence[Mapping[str, Any]],
    candidate_answer: str,
) -> Mapping[str, Any] | None:
    candidate_key = _answer_candidate_key(candidate_answer)
    if not candidate_key:
        return None
    for row in reversed(tuple(evidence_digest)):
        assessment = row.get("claim_assessment")
        if not isinstance(assessment, Mapping):
            continue
        assessed_key = _answer_candidate_key(str(assessment.get("candidate_answer", "") or ""))
        if assessed_key == candidate_key:
            return assessment
    return None


def _answer_candidate_key(answer: str) -> str:
    label = re.match(r"\s*([A-H])(?:\.|\)|:|\s|$)", str(answer or "").upper())
    if label:
        return f"option:{label.group(1)}"
    return re.sub(r"[^a-z0-9]+", "", str(answer or "").casefold())


def _normalize_answer_payload(value: Any, options: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return str(value or "").strip(), ()
    citations = tuple(str(item) for item in value.get("citations", ()) or () if str(item).strip())
    label_value = value.get("option") or value.get("label") or value.get("choice")
    label_match = re.match(r"\s*([A-H])(?:\.|\)|:|\s|$)", str(label_value or "").upper())
    if label_match:
        label = label_match.group(1)
        text = str(options.get(label) or value.get("text") or "").strip()
        return (f"{label}. {text}" if text else label), citations
    nested = value.get("answer")
    if nested is not None and nested is not value:
        answer, nested_citations = _normalize_answer_payload(nested, options)
        return answer, citations or nested_citations
    return str(value.get("text") or "").strip(), citations


def _normalize_reasoner_payload(value: Mapping[str, Any], *, round_id: int = 0) -> dict[str, Any]:
    payload = dict(value)
    payload["tasks"] = list(_normalize_reasoner_tasks(payload.get("tasks"), round_id=round_id))
    action = str(payload.get("action", "") or "").strip().casefold()
    if action not in {"investigate", "answer"}:
        if payload.get("tasks"):
            action = "investigate"
        elif payload.get("answer"):
            action = "answer"
    if action:
        payload["action"] = action
    if not isinstance(payload.get("primary_gap"), Mapping) and isinstance(payload.get("gap"), Mapping):
        payload["primary_gap"] = dict(payload["gap"])
    return payload


def _normalize_reasoner_tasks(value: Any, *, round_id: int) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    normalized = []
    for task_index, raw_task in enumerate(value, start=1):
        if not isinstance(raw_task, Mapping):
            continue
        task = dict(raw_task)
        goal = str(task.get("goal") or task.get("task") or "").strip()
        if not goal or not _is_observation_task_goal(goal):
            continue
        segment_values = task.get("segment_ids") or task.get("segments") or ()
        if isinstance(segment_values, str):
            segment_values = (segment_values,)
        if isinstance(segment_values, Sequence) and not isinstance(segment_values, (str, bytes)):
            segments = tuple(str(item).strip() for item in segment_values if str(item).strip())
        else:
            segments = ()
        if segments and not task.get("segment_id"):
            for segment_index, segment_id in enumerate(segments, start=1):
                normalized.append(
                    {
                        "query_id": f"auto_r{round_id}_t{task_index}_{segment_index}",
                        "goal": goal,
                        "segment_id": segment_id,
                        "time_range": None,
                        "inspection_mode": "window",
                        "search_terms": [],
                        "modality_hint": list(task.get("modality_hint") or ("visual", "asr")),
                        "expected_evidence": str(task.get("expected_evidence") or goal),
                    }
                )
            continue
        segment_id = str(task.get("segment_id", "") or "").strip()
        time_range = task.get("time_range")
        inspection_mode = str(task.get("inspection_mode", "window") or "window")
        search_terms = list(task.get("search_terms") or ())
        executable = bool(segment_id or time_range is not None)
        if inspection_mode == "search_asr":
            executable = bool(search_terms)
        if not executable:
            continue
        task["query_id"] = str(task.get("query_id") or f"auto_r{round_id}_t{task_index}")
        task["goal"] = goal
        normalized.append(task)
    return tuple(normalized[:4])


def _is_observation_task_goal(goal: str) -> bool:
    text = str(goal or "").casefold()
    return any(
        marker in text
        for marker in (
            "inspect",
            "find",
            "scan",
            "check",
            "locate",
            "verify",
            "identify",
            "confirm",
            "observe",
            "examine",
            "read",
            "search",
        )
    )


def _valid_option_answer(answer: str, options: Mapping[str, Any]) -> bool:
    label_match = re.match(r"\s*([A-H])(?:\.|\)|:|\s|$)", str(answer or "").upper())
    if label_match and label_match.group(1) in {str(label).upper() for label in options}:
        return True
    normalized_answer = re.sub(r"[^a-z0-9]+", "", str(answer or "").casefold())
    if not normalized_answer:
        return False
    matches = 0
    for text in options.values():
        normalized_option = re.sub(r"[^a-z0-9]+", "", str(text or "").casefold())
        if normalized_option and (
            normalized_option in normalized_answer or normalized_answer in normalized_option
        ):
            matches += 1
    return matches == 1


def _claim_verification_task(
    kwargs: Mapping[str, Any],
    candidate: ReasonerDecision,
    evidence_digest: Sequence[Mapping[str, Any]],
    *,
    round_id: int,
) -> InvestigationTask | None:
    cited_ids = set(candidate.citations)
    cited = [row for row in evidence_digest if str(row.get("evidence_id", "")) in cited_ids]
    if not cited:
        cited = list(evidence_digest[-2:])
    ranges = [tuple(row.get("virtual_time_range", ()) or ()) for row in cited]
    ranges = [row for row in ranges if len(row) == 2 and row[0] is not None and row[1] is not None]
    if not ranges:
        return None
    start = min(float(row[0]) for row in ranges) - 180.0
    end = max(float(row[1]) for row in ranges) + 180.0
    duration = max(0.0, float(kwargs.get("workspace_duration_sec", end) or end))
    start = max(0.0, start)
    end = min(duration, end)
    if end - start > 360.0:
        center = (start + end) / 2.0
        start, end = max(0.0, center - 180.0), min(duration, center + 180.0)
    lineage = next(
        (
            item
            for row in cited
            for item in (row.get("source_lineage", ()) or ())
            if isinstance(item, Mapping) and str(item.get("segment_id", "") or "")
        ),
        {},
    )
    segment_id = str(lineage.get("segment_id", "") or "")
    if not segment_id:
        return None
    selected = re.match(r"\s*([A-H])(?:\.|\)|:|\s|$)", candidate.answer.upper())
    selected_label = selected.group(1) if selected else ""
    alternatives = tuple(
        f"{label}. {text}"
        for label, text in dict(kwargs.get("options") or {}).items()
        if str(label).upper() != selected_label
    )
    return InvestigationTask(
        query_id=f"verify_r{int(round_id)}_candidate",
        goal="Independently verify or refute the proposed answer using the cited scene and adjacent context.",
        segment_id=segment_id,
        time_range=(round(start, 3), round(end, 3)),
        modality_hint=("visual", "asr"),
        expected_evidence="direct evidence that distinguishes the proposed answer from the strongest alternative",
        inspection_mode="verify_claim",
        priority=1.0,
        claim_to_verify=candidate.answer,
        claim_relation=_claim_relation_for_question(kwargs),
        alternative_answers=alternatives,
    )


def _claim_relation_for_question(kwargs: Mapping[str, Any]) -> str:
    question = str(kwargs.get("question", "") or "").casefold()
    requires_identity = bool(dict(kwargs.get("query_requirements") or {}).get("requires_identity_link"))
    if re.search(r"\bwhy\b", question):
        return "decision_motive"
    if re.search(r"\bhow\s+(?:did|does|do|was|were)\b", question):
        return "identity_linked_cause" if requires_identity else "initiating_cause_or_mechanism"
    if re.search(r"\b(?:view|opinion|believ\w*)\b", question):
        return "opinion"
    if requires_identity:
        return "identity_link"
    return "direct_relation"


def _answer_audit_prompt(
    kwargs: Mapping[str, Any],
    candidate: ReasonerDecision,
    evidence_digest: Sequence[Mapping[str, Any]],
    *,
    visual_manifest: Sequence[Mapping[str, Any]] = (),
) -> str:
    cited_ids = set(candidate.citations)
    cited = [dict(row) for row in evidence_digest if str(row.get("evidence_id", "")) in cited_ids]
    surrounding = [
        dict(row)
        for row in evidence_digest
        if str(row.get("evidence_id", "")) not in cited_ids
    ][-8:]
    context = (cited + surrounding)[:12]
    budget = int(kwargs.get("remaining_budget", 0) or 0)
    task_instruction = (
        "If evidence is insufficient or contradictory, return 1-2 targeted investigation tasks."
        if budget > 0 and not kwargs.get("force_finalize")
        else "No investigation budget remains; return no tasks and assess the best-effort answer honestly."
    )
    claim_contract = _compile_option_claim_contract(
        str(kwargs.get("question", "") or ""),
        dict(kwargs.get("options") or {}),
        candidate.answer,
    )
    return (
        "Audit a proposed answer for a long-video QA agent using only the supplied evidence. Start by assuming the "
        "proposed answer may be wrong and compare every option against the evidence. "
        "Do not reward citation relevance alone. The exact selected option must follow from the observations at the "
        "causal, temporal, identity, count, and attribute granularity asked by the question. Audit only the required atoms in "
        "OptionClaimContract; do not require non-discriminative background from the question. For why questions, distinguish "
        "an underlying motive or observed cause from a downstream benefit, an after-state, "
        "or mere co-occurrence. For identity-linked questions, require evidence that links the same visible entity across the "
        "anchor and answer event. For distinct counts, audit every entity cluster against its entity_observation_ids: each adopted "
        "observation must have countable=true, a direct frame witness, a stable visual signature, and a verified question relation. "
        "Free-text local counts and candidate-only entities are not count evidence. Do not use answer-option plausibility as evidence. "
        f"{task_instruction}\n"
        "Identify the single strongest_alternative after comparing every option internally. If that alternative is directly "
        "supported, provide revised_answer and its citations. Do not revise based only on plausibility or elimination. Keep the "
        "reason under 100 words and each task goal under 40 words. Return compact JSON only: "
        "{\"verdict\":\"supported|insufficient|contradicted\",\"reason\":\"...\","
        "\"evidence_relation\":\"direct|causal_chain|consequence_only|cooccurrence_only|unclear\","
        "\"strongest_alternative\":{\"option\":\"A\",\"support\":\"direct|indirect|contradicted|missing\","
        "\"evidence_ids\":[\"ev_...\"],\"reason\":\"...\"},"
        "\"revised_answer\":null,\"revised_citations\":[],\"revised_entity_clusters\":[{\"entity_id\":\"entity_1\","
        "\"description\":\"...\",\"evidence_ids\":[\"ev_...\"],\"entity_observation_ids\":[\"obs:person_1\"]}],"
        "\"revised_support_status\":\"supported|insufficient\",\"tasks\":[{\"query_id\":\"audit_r2_t1\","
        "\"goal\":\"...\",\"segment_id\":\"seg_0001\",\"time_range\":[0.0,60.0],"
        "\"modality_hint\":[\"visual\",\"asr\"],\"expected_evidence\":\"...\"}]}.\n"
        f"Question: {kwargs['question']}\nOptions: {json.dumps(kwargs['options'], ensure_ascii=False)}\n"
        f"Proposed answer: {candidate.answer}\nCitations: {json.dumps(list(candidate.citations), ensure_ascii=False)}\n"
        f"OptionClaimContract: {json.dumps(claim_contract, ensure_ascii=False)}\n"
        f"Evidence context: {json.dumps(context, ensure_ascii=False)}\n"
        f"Temporal navigation: {json.dumps(kwargs.get('temporal_navigation') or {}, ensure_ascii=False)}"
        f"{_visual_manifest_prompt(visual_manifest)}"
    )


def _compile_option_claim_contract(question: str, options: Mapping[str, Any], answer: str) -> dict[str, Any]:
    del question
    label_match = re.match(r"\s*([A-H])(?:\.|\)|:|\s|$)", str(answer or "").upper())
    option_id = label_match.group(1) if label_match else ""
    option_text = str(options.get(option_id, "") or answer).strip()
    atoms = tuple(
        fragment.strip(" ,.;")
        for fragment in re.split(r"\s*(?:,|;|\bthen\b|\band\b)\s*", option_text, flags=re.IGNORECASE)
        if len(fragment.strip(" ,.;")) >= 3
    )
    return {
        "option_id": option_id,
        "atoms": list(atoms or (option_text,)),
        "excluded_context_atoms": [],
        "compiler_source": "selected_option_text",
        "compiler_version": "v1",
    }


def _answer_audit_fingerprint(
    kwargs: Mapping[str, Any],
    candidate: ReasonerDecision,
    evidence_digest: Sequence[Mapping[str, Any]],
) -> str:
    cited = set(candidate.citations)
    ranges = [row.get("virtual_time_range") for row in evidence_digest if str(row.get("evidence_id", "") or "") in cited]
    contract = _compile_option_claim_contract(
        str(kwargs.get("question", "") or ""),
        dict(kwargs.get("options") or {}),
        candidate.answer,
    )
    return json.dumps({
        "option": contract["option_id"],
        "atoms": contract["atoms"],
        "citations": sorted(cited),
        "ranges": ranges,
        "method": "answer_audit_v1",
    }, sort_keys=True)


def _forced_answer_prompt(
    kwargs: Mapping[str, Any],
    evidence_digest: Sequence[Mapping[str, Any]],
    *,
    visual_manifest: Sequence[Mapping[str, Any]] = (),
) -> str:
    rows = select_uniform_items(tuple(evidence_digest), 20)
    dashboard = [
        {
            "evidence_id": row.get("evidence_id"),
            "summary": str(row.get("summary", "") or "")[:320],
            "virtual_time_range": row.get("virtual_time_range"),
            "modality": row.get("modality"),
            "evidence_kind": row.get("evidence_kind"),
            "entities": list(row.get("entities", ()) or ())[:4],
            "events": list(row.get("events", ()) or ())[:4],
            "claim_assessment": dict(row.get("claim_assessment", {}) or {}),
            "source_lineage": list(row.get("source_lineage", ()) or ())[:2],
        }
        for row in rows
    ]
    return (
        "The investigation budget is exhausted. You must choose one best option using the evidence dashboard; "
        "do not return investigate, abstain, or an empty answer. This is a best-effort answer and may remain unverified. "
        "Return JSON only: {\"answer\":\"A. option text\",\"citations\":[\"ev_...\"],"
        "\"entity_clusters\":[{\"entity_id\":\"entity_1\",\"description\":\"...\","
        "\"evidence_ids\":[\"ev_...\"],\"entity_observation_ids\":[\"observation:person_1\"]}]}. "
        "For distinct counts, use only countable entity observations; candidate-only summaries may inform the best effort but "
        "must not be presented as grounded identity evidence.\n"
        f"Question: {kwargs['question']}\nOptions: {json.dumps(kwargs['options'], ensure_ascii=False)}\n"
        f"Completion status: {json.dumps(kwargs.get('completion_status') or {}, ensure_ascii=False)}\n"
        f"Evidence dashboard: {json.dumps(dashboard, ensure_ascii=False)}"
        f"{_visual_manifest_prompt(visual_manifest)}"
    )


def _compact_forced_answer_prompt(
    kwargs: Mapping[str, Any],
    evidence_digest: Sequence[Mapping[str, Any]],
) -> str:
    rows = []
    for row in select_uniform_items(tuple(evidence_digest), 12):
        entities = [
            {
                "id": entity.get("entity_observation_id"),
                "signature": str(entity.get("visual_signature", "") or entity.get("description", ""))[:140],
                "countable": bool(entity.get("countable")),
            }
            for entity in tuple(row.get("entities", ()) or ())[:4]
        ]
        rows.append(
            {
                "evidence_id": row.get("evidence_id"),
                "summary": str(row.get("summary", "") or "")[:220],
                "entities": entities,
            }
        )
    return (
        "Choose the single best multiple-choice answer from the compact verified evidence. Do not request more investigation. "
        "For distinct counts, merge repeated people by visual signature and count only countable entities. Return compact JSON "
        "only: {\"answer\":\"A. option text\",\"citations\":[\"ev_...\"],\"entity_clusters\":[{\"entity_id\":"
        "\"entity_1\",\"description\":\"...\",\"evidence_ids\":[\"ev_...\"],\"entity_observation_ids\":[\"obs:person_1\"]}]}.\n"
        f"Question: {kwargs['question']}\nOptions: {json.dumps(kwargs['options'], ensure_ascii=False)}\n"
        f"Evidence: {json.dumps(rows, ensure_ascii=False)}"
    )


def _resolution_prompt(task: Any) -> str:
    conditions = tuple(getattr(task, "conditions", ()) or ())
    return (
        "Evaluate only what is directly observable, not whether frames were returned. In the same JSON include "
        "\"target_presence\":{\"target\":\"...\",\"status\":\"present|absent|uncertain\",\"confidence\":0.0}, "
        "\"measurements\":[{\"value\":0.0,\"unit\":\"...\",\"relation\":\"exact|approx|greater_than|less_than\","
        "\"measurement_semantics\":\"delta|cumulative|unknown\",\"subject_id\":\"\",\"source_time_sec\":null,"
        "\"boundary_relation\":\"before|at|after|unknown\",\"raw_text\":\"\"}], "
        "\"relations\":[{\"relation_type\":\"identity|temporal|causal|transition\",\"subject_id\":\"\","
        "\"object_id\":\"\",\"status\":\"supported|contradicted|unknown\",\"description\":\"\"}], and "
        "\"condition_results\":[{\"condition_id\":\"...\",\"status\":\"satisfied|unknown|contradicted\","
        "\"observation\":\"direct observation\"}]. Use only the stable condition_id values below. "
        "For a crop, mark target_presence present only if the requested target is actually inside that crop; otherwise use absent or uncertain. "
        "The driver derives overall resolution, so do not self-declare it. Return empty measurement/relation arrays when unsupported. "
        f"Stable conditions: {json.dumps(to_jsonable(conditions), ensure_ascii=False)}\n"
    )


def _preview_prompt(workspace: VirtualVideoWorkspace, task: Any, segment_packet: Mapping[str, Any], window: Mapping[str, Any]) -> str:
    return (
        "You are the Investigator. Inspect the low-fps preview frames and local ASR without choosing an answer option. "
        "Return JSON only: {\"summary\":\"atomic observation\",\"confidence\":0.0-1.0,"
        "\"entities\":[{\"local_id\":\"person_1\",\"description\":\"atomic visible observation\","
        "\"visual_signature\":\"stable face, hair, clothing, and accessories\",\"frame_indices\":[0],"
        "\"role\":\"visible role or unknown\",\"question_relation\":\"directly observed relation or unknown\","
        "\"supports_question_relation\":true|false}],"
        "\"events\":[{\"local_id\":\"event_1\",\"description\":\"one atomic occurrence relevant to the question\","
        "\"start_sec\":float,\"end_sec\":float,\"supports_question_event\":true|false}],"
        "\"supports_identity_anchor\":true|false,\"supports_answer_event\":true|false,"
        "\"need_detail\":true|false,\"detail_start_sec\":float|null,\"detail_end_sec\":float|null,\"reason\":\"...\","
        "\"region_hint\":\"scoreboard/text/object or empty\",\"region_box\":[x1,y1,x2,y2]|null}. "
        "Region coordinates are normalized 0-1.\n"
        "List each visible person separately using stable appearance attributes. Every entity must cite one or more 0-based "
        "frame_indices from the supplied images; omit people who are inferred from ASR or summary text but are not visible in "
        "those frames. The summary must not introduce a person absent from entities. Do not estimate a segment-level or "
        "video-level count. The same person may recur in later chunks. A window longer than 120 seconds is candidate discovery "
        "only: request a narrower detail window before treating any identity as countable.\n"
        "Enumerate every distinct question-relevant event occurrence visible in this inspected window. "
        "Use virtual timestamps from the window metadata, one row per occurrence, and return an empty events list when none is supported.\n"
        "Set supports_identity_anchor only when one visible entity jointly matches the identifying attributes in the question. "
        "Set supports_answer_event only when the observation directly supports the event, cause, action, or state being asked about.\n"
        "Request detail only when motion, OCR, identity, or a small visual attribute remains unresolved. "
        "Any detail window must be inside the preview window and narrower than it.\n"
        f"{_resolution_prompt(task)}"
        f"Question: {workspace.case.question}\n"
        f"Task: {getattr(task, 'goal', '')}\nExpected evidence: {getattr(task, 'expected_evidence', '')}\n"
        f"Segment: {json.dumps(_compact_segment_packet(segment_packet), ensure_ascii=False)[:3000]}\n"
        f"Preview window metadata: {json.dumps({k: window[k] for k in ['virtual_time_range','sampling','asr_cues','source_lineage']}, ensure_ascii=False)[:5000]}"
    )


def _claim_preview_prompt(
    workspace: VirtualVideoWorkspace,
    task: Any,
    segment_packet: Mapping[str, Any],
    window: Mapping[str, Any],
) -> str:
    return (
        "You are an independent claim verifier, not the Reasoner who proposed the answer. Re-observe the frames and local ASR, "
        "actively look for disconfirming evidence, and distinguish direct support from correlation, downstream consequence, "
        "after-state, or identity mismatch. For why/how questions, separate the decision motive or initiating cause from the "
        "action's implementation, stated use, and downstream consequence. A stated benefit is not automatically the decision motive. "
        "You must identify the strongest alternative from the supplied list unless every alternative is contradicted. "
        "Return JSON only: {\"summary\":\"atomic verification observation\","
        "\"confidence\":0.0-1.0,\"claim_verdict\":\"supports|refutes|insufficient\","
        "\"relation_type\":\"direct|causal_chain|consequence_only|cooccurrence_only|identity_mismatch|unclear\","
        "\"candidate_role\":\"decision_motive|initiating_cause|mechanism|stated_use|downstream_consequence|after_state|unclear\","
        "\"strongest_alternative\":\"B. ...|none\",\"reason\":\"...\",\"need_detail\":true|false,"
        "\"detail_start_sec\":float|null,\"detail_end_sec\":float|null}.\n"
        "Use supports only when the exact candidate relation is directly established. The fact that candidate-related objects, "
        "people, or words appear is not enough. Request a narrower detail window when motion or text remains unresolved.\n"
        f"{_resolution_prompt(task)}"
        f"Question: {workspace.case.question}\nOptions: {json.dumps(dict(workspace.case.options), ensure_ascii=False)}\n"
        f"Candidate claim: {getattr(task, 'claim_to_verify', '')}\n"
        f"Required claim relation: {getattr(task, 'claim_relation', '')}\n"
        f"Alternative answers: {json.dumps(list(getattr(task, 'alternative_answers', ()) or ()), ensure_ascii=False)}\n"
        f"Segment: {json.dumps(_compact_segment_packet(segment_packet), ensure_ascii=False)[:3000]}\n"
        f"Window metadata: {json.dumps({k: window[k] for k in ['virtual_time_range','sampling','asr_cues','source_lineage']}, ensure_ascii=False)[:6000]}"
    )


def _event_preview_prompt(
    workspace: VirtualVideoWorkspace,
    task: Any,
    segment_packet: Mapping[str, Any],
    window: Mapping[str, Any],
    *,
    prior_events: Sequence[Mapping[str, Any]] = (),
) -> str:
    return (
        "You are the Investigator. Enumerate atomic question-relevant event occurrences in this low-fps window. "
        "Return concise JSON only: {\"summary\":\"brief window observation\",\"confidence\":0.0-1.0,"
        "\"events\":[{\"local_id\":\"event_1\",\"event_key\":\"stable topic/title signature\","
        "\"description\":\"one occurrence\",\"start_sec\":float,\"end_sec\":float,"
        "\"supports_question_event\":true|false,\"continues_from_previous\":true|false,"
        "\"continues_to_next\":true|false}],"
        "\"supports_answer_event\":true|false,\"need_detail\":true|false,"
        "\"detail_start_sec\":float|null,\"detail_end_sec\":float|null,\"reason\":\"...\"}.\n"
        "List every distinct supported occurrence, use virtual timestamps inside this window, and return an empty events list when none. "
        "event_key must identify this occurrence by topic, title, or visible anchor; never use only the generic event class. "
        "Compare against the prior adjacent-window events below. Reuse an exact event_key and set continues_from_previous=true "
        "only when the same occurrence visibly continues across the boundary. "
        "Do not list people or infer a video-level count. Request a narrower detail window only when an occurrence boundary is unclear.\n"
        f"{_resolution_prompt(task)}"
        f"Question: {workspace.case.question}\n"
        f"Task: {getattr(task, 'goal', '')}\nExpected evidence: {getattr(task, 'expected_evidence', '')}\n"
        f"Prior adjacent-window ending events: {json.dumps(list(prior_events), ensure_ascii=False)[:1800]}\n"
        f"Segment: {json.dumps(_compact_segment_packet(segment_packet), ensure_ascii=False)[:3000]}\n"
        f"Window metadata: {json.dumps({k: window[k] for k in ['virtual_time_range','sampling','asr_cues','source_lineage']}, ensure_ascii=False)[:5000]}"
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
        "\"entities\":[{\"local_id\":\"person_1\",\"description\":\"atomic visible observation\","
        "\"visual_signature\":\"stable face, hair, clothing, and accessories\",\"frame_indices\":[0],"
        "\"role\":\"visible role or unknown\",\"question_relation\":\"directly observed relation or unknown\","
        "\"supports_question_relation\":true|false}],"
        "\"events\":[{\"local_id\":\"event_1\",\"description\":\"one atomic occurrence relevant to the question\","
        "\"start_sec\":float,\"end_sec\":float,\"supports_question_event\":true|false}],"
        "\"supports_identity_anchor\":true|false,\"supports_answer_event\":true|false}.\n"
        "List visible people separately and give every entity one or more 0-based frame_indices from the supplied images. "
        "Omit any person not directly visible in a cited frame, and never introduce additional people only in summary text. "
        "Do not infer a count across frames or chunks.\n"
        "Enumerate every distinct question-relevant event occurrence visible in this inspected window. "
        "Use virtual timestamps from the window metadata, one row per occurrence, and return an empty events list when none is supported.\n"
        "Set supports_identity_anchor only when one visible entity jointly matches the identifying attributes in the question. "
        "Set supports_answer_event only when the observation directly supports the event, cause, action, or state being asked about.\n"
        f"{_resolution_prompt(task)}"
        f"Question: {workspace.case.question}\n"
        f"Task: {getattr(task, 'goal', '')}\nExpected evidence: {getattr(task, 'expected_evidence', '')}\n"
        f"Preview finding: {json.dumps(dict(preview or {}), ensure_ascii=False)[:1600]}\n"
        f"Segment: {json.dumps(_compact_segment_packet(segment_packet), ensure_ascii=False)[:3000]}\n"
        f"Detail window metadata: {json.dumps(_window_prompt_metadata(window), ensure_ascii=False)[:5000]}"
    )


def _event_evidence_prompt(
    workspace: VirtualVideoWorkspace,
    task: Any,
    segment_packet: Mapping[str, Any],
    window: Mapping[str, Any],
    *,
    preview: Mapping[str, Any] | None = None,
    prior_events: Sequence[Mapping[str, Any]] = (),
) -> str:
    return (
        "You are the Investigator. Verify atomic question-relevant event occurrences in this detail window. "
        "Return concise JSON only: {\"summary\":\"brief verified observation\",\"confidence\":0.0-1.0,"
        "\"events\":[{\"local_id\":\"event_1\",\"event_key\":\"stable topic/title signature\","
        "\"description\":\"one occurrence\",\"start_sec\":float,\"end_sec\":float,"
        "\"supports_question_event\":true|false,\"continues_from_previous\":true|false,"
        "\"continues_to_next\":true|false}],"
        "\"supports_answer_event\":true|false}.\n"
        "List every distinct supported occurrence with virtual timestamps. event_key must identify the occurrence by topic, "
        "title, or visible anchor, not only its generic class. Compare against the prior adjacent-window events below; reuse "
        "an exact event_key and set continues_from_previous=true only for the same continuing occurrence. "
        "Do not list people or infer a video-level count.\n"
        f"{_resolution_prompt(task)}"
        f"Question: {workspace.case.question}\n"
        f"Task: {getattr(task, 'goal', '')}\nExpected evidence: {getattr(task, 'expected_evidence', '')}\n"
        f"Prior adjacent-window ending events: {json.dumps(list(prior_events), ensure_ascii=False)[:1800]}\n"
        f"Preview finding: {json.dumps(dict(preview or {}), ensure_ascii=False)[:1600]}\n"
        f"Segment: {json.dumps(_compact_segment_packet(segment_packet), ensure_ascii=False)[:3000]}\n"
        f"Detail window metadata: {json.dumps(_window_prompt_metadata(window), ensure_ascii=False)[:5000]}"
    )


def _claim_evidence_prompt(
    workspace: VirtualVideoWorkspace,
    task: Any,
    segment_packet: Mapping[str, Any],
    window: Mapping[str, Any],
    *,
    preview: Mapping[str, Any] | None = None,
) -> str:
    return (
        "You are an independent claim verifier. Use the detail frames and local ASR to verify or refute the exact candidate, "
        "not merely whether the scene is relevant. For why/how questions, separate the decision motive or initiating cause from "
        "implementation, stated use, and downstream consequence, then compare the candidate against the strongest alternative. "
        "Return JSON only: {\"summary\":\"verified atomic observation\","
        "\"confidence\":0.0-1.0,\"claim_verdict\":\"supports|refutes|insufficient\","
        "\"relation_type\":\"direct|causal_chain|consequence_only|cooccurrence_only|identity_mismatch|unclear\","
        "\"candidate_role\":\"decision_motive|initiating_cause|mechanism|stated_use|downstream_consequence|after_state|unclear\","
        "\"strongest_alternative\":\"B. ...|none\",\"reason\":\"...\"}.\n"
        f"{_resolution_prompt(task)}"
        f"Question: {workspace.case.question}\nOptions: {json.dumps(dict(workspace.case.options), ensure_ascii=False)}\n"
        f"Candidate claim: {getattr(task, 'claim_to_verify', '')}\n"
        f"Required claim relation: {getattr(task, 'claim_relation', '')}\n"
        f"Alternative answers: {json.dumps(list(getattr(task, 'alternative_answers', ()) or ()), ensure_ascii=False)}\n"
        f"Preview finding: {json.dumps(dict(preview or {}), ensure_ascii=False)[:1800]}\n"
        f"Segment: {json.dumps(_compact_segment_packet(segment_packet), ensure_ascii=False)[:3000]}\n"
        f"Detail window metadata: {json.dumps(_window_prompt_metadata(window), ensure_ascii=False)[:6000]}"
    )


def _window_prompt_metadata(window: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("virtual_time_range", "sampling", "asr_cues", "source_lineage", "region_observation")
    return {key: window[key] for key in keys if key in window}


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


def _recover_closed_json_fields(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("summary", "failure_reason"):
        match = re.search(rf'"{field}"\s*:\s*"((?:\\.|[^"\\])*)"', str(text or ""), re.S)
        if not match:
            continue
        try:
            result[field] = json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            continue
    return result


def _with_explicit_measurement_fallback(
    parsed: Mapping[str, Any],
    raw: str,
    *,
    task: Any,
    question: str,
) -> tuple[dict[str, Any], bool]:
    result = dict(parsed)
    if tuple(result.get("measurements", ()) or ()):
        return result, False
    source_text = " ".join((str(result.get("summary", "") or ""), str(raw or "")))
    context = " ".join((
        str(question or ""),
        str(getattr(task, "goal", "") or ""),
        str(getattr(task, "expected_evidence", "") or ""),
    )).casefold()
    combined = f"{source_text} {context}".casefold()
    if any(term in combined for term in ("diameter", "wide", "width", "extent")):
        quantity_type = "diameter"
    elif "calorie" in combined:
        quantity_type = "calorie"
    elif any(term in combined for term in ("game clock", "scoreboard clock", "countdown")):
        quantity_type = "countdown_clock"
    elif any(term in combined for term in ("distance", "meters", "metres", "kilometers", "kilometres")):
        quantity_type = "distance"
    else:
        quantity_type = ""
    explicit = bool(quantity_type and quantity_type.replace("_", " ") in source_text.casefold())
    if quantity_type == "diameter":
        explicit = any(term in source_text.casefold() for term in ("diameter", "wide", "width", "extent"))
    measurements = extract_measurements_from_text(
        source_text,
        quantity_type=quantity_type,
        binding_status="explicit" if explicit else "contextual" if quantity_type else "unbound",
    )
    if not measurements:
        return result, False
    result["measurements"] = [to_jsonable(item) for item in measurements]
    return result, True


def _parse_answer_audit(text: str) -> dict[str, Any]:
    parsed = _parse_json(text)
    if parsed:
        return parsed
    verdict_match = re.search(
        r'\"verdict\"\s*:\s*\"(supported|insufficient|contradicted)\"',
        str(text or ""),
        flags=re.IGNORECASE,
    )
    verdict = verdict_match.group(1).casefold() if verdict_match else "unknown"
    return {
        "verdict": verdict,
        "reason": (
            "Answer audit output was truncated; only its explicit leading verdict could be retained."
            if verdict_match
            else "Answer audit returned no parseable verdict; preserve the independent completion gate result."
        ),
        "tasks": [],
    }


def _image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dual-model Reasoner/Investigator evaluation on Video-MME virtual videos.")
    parser.add_argument("--dataset-root", default="/ytech_m2v5_hdd/workspace/kling_mm/Datasets/VLMEvalKit_Dataset_Cache/HFCache/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b")
    parser.add_argument("--out-root", default="/m2v_intern/xuboshen/zgw/VideoAgent/virtual_videomme_interactive")
    parser.add_argument("--config", help="Legacy shared API config used for both roles.")
    parser.add_argument("--reasoner-config", help="Text-only planning/reasoning API config.")
    parser.add_argument("--investigator-config", help="Multimodal observation API config.")
    cases = parser.add_mutually_exclusive_group()
    cases.add_argument("--case-ids", nargs="*")
    cases.add_argument("--case-group", help="JSON manifest containing an ordered cases[] list and default construction.")
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
    parser.add_argument("--workers", type=int, default=1, help="Parallel case workers, clamped to 1-16.")
    parser.add_argument("--skip-completed", action="store_true", help="Reuse cases with an existing run_summary.json.")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
