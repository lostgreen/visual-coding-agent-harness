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
import threading
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
    _task_terms,
)
from vcah.multiround import (
    InvestigationTask,
    ReasonerDecision,
    VirtualVideoMultiRoundDriver,
    requires_option_audit,
)
from vcah.replay import (
    aggregate_seed_results,
    create_immutable_run,
    default_run_id,
    file_checksum,
    git_commit,
    replay_case_metadata,
    workspace_input_checksums,
    write_immutable_summary,
)
from vcah.semantic_evidence import qualify_absence
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
        "grounding_level": str(payload.get("grounding_level", "none") or "none"),
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
    if args.skip_completed:
        raise ValueError("--skip-completed is incompatible with immutable replay runs; create a new --run-id instead")
    seeds = tuple(dict.fromkeys(int(seed) for seed in (args.seeds or (args.seed,))))
    if not seeds:
        raise ValueError("Provide at least one seed.")
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
    run = create_immutable_run(
        Path(args.out_root),
        run_id=args.run_id or default_run_id(),
        config=_replay_run_config(
            args,
            selected=selected,
            seeds=seeds,
            reasoner_api=reasoner_api,
            investigator_api=investigator_api,
        ),
    )

    def run_one(case_id: str, seed: int) -> Mapping[str, Any]:
        reasoner_api.set_requested_seed(seed)
        investigator_api.set_requested_seed(seed)
        workspace_root = run.root / "workspaces" / case_id
        if len(seeds) > 1:
            workspace_root = workspace_root / f"seed-{seed}"
        workspace = build_or_load_workspace(
            dataset_root,
            workspace_root,
            case_id=case_id,
            seed=seed,
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
        case_summary = json.loads((workspace.root_dir / "run_summary.json").read_text(encoding="utf-8"))
        replay = replay_case_metadata(
            workspace_root=workspace.root_dir,
            case_summary=case_summary,
            input_checksums=workspace_input_checksums(workspace),
            seed=seed,
            provider_settings={
                "reasoner": {**reasoner_api.replay_settings, "requested_seed": seed},
                "investigator": {**investigator_api.replay_settings, "requested_seed": seed},
            },
            gold_option=workspace.case.gold,
        )
        return {
            "case_id": result.case_id,
            "seed": seed,
            "answer": result.answer,
            "grounded_answer": result.grounded_answer,
            "forced_answer": result.forced_answer,
            "selected_option": result.selected_option,
            "answer_mode": result.answer_mode,
            "grounding_status": result.grounding_status,
            "grounding_level": result.grounding_level,
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
            "replay": replay,
        }

    summaries: list[Mapping[str, Any]] = []
    for seed in seeds:
        summaries.extend(
            _run_case_batch(
                selected,
                lambda case_id, current_seed=seed: run_one(case_id, current_seed),
                workers=int(args.workers),
            )
        )
    payload = {
        "mode": args.mode,
        "case_group": None if case_group is None else case_group["group_id"],
        "case_count": len(summaries),
        "seeds": list(seeds),
        "correct": sum(1 for item in summaries if item["correct"]),
        "models": {"reasoner": reasoner_api.model, "investigator": investigator_api.model},
        "cases": summaries,
        "multi_seed_report": aggregate_seed_results(
            [dict(item.get("replay", {}) or {}) for item in summaries]
        ),
    }
    summary_path = write_immutable_summary(run, payload)
    print(json.dumps({
        "run_id": run.run_id,
        "summary": str(summary_path),
        "case_count": len(summaries),
        "correct": payload["correct"],
        "seeds": list(seeds),
    }, ensure_ascii=False, sort_keys=True))


def _replay_run_config(
    args: argparse.Namespace,
    *,
    selected: Sequence[str],
    seeds: Sequence[int],
    reasoner_api: "OpenAICompatibleVisionClient",
    investigator_api: "OpenAICompatibleVisionClient",
) -> dict[str, Any]:
    source_paths = {
        "shared_config": args.config,
        "reasoner_config": args.reasoner_config,
        "investigator_config": args.investigator_config,
        "case_group": args.case_group,
    }
    return {
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "models": {"reasoner": reasoner_api.model, "investigator": investigator_api.model},
        "provider_settings": {
            "reasoner": reasoner_api.replay_settings,
            "investigator": investigator_api.replay_settings,
        },
        "seeds": [int(seed) for seed in seeds],
        "case_ids": [str(case_id) for case_id in selected],
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"config", "reasoner_config", "investigator_config", "case_group", "run_id"}
        },
        "input_config_checksums": {
            key: file_checksum(Path(value))
            for key, value in source_paths.items()
            if value
        },
    }


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


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on", "supported"}
    return bool(value)


def _seed_support_status(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    if _as_bool(value):
        return "supported"
    if isinstance(value, str) and value.strip().casefold() in {"unknown", "unreported", "not_reported"}:
        return "unknown"
    return "unsupported"


def _provider_request_id(response: Any, payload: Mapping[str, Any]) -> str:
    headers = getattr(response, "headers", {}) or {}
    for key in ("x-request-id", "request-id", "x-amzn-requestid"):
        value = headers.get(key) or headers.get(key.title())
        if value:
            return str(value)
    return str(dict(payload).get("id", "") or "")


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
        self.temperature = _optional_float(planner.get("temperature"))
        self.top_p = _optional_float(planner.get("top_p"))
        self.provider_reported_seed_support = _seed_support_status(
            planner.get(
                "provider_reported_seed_support",
                planner.get("provider_seed_supported", planner.get("supports_seed")),
            )
        )
        self.provider_seed_supported = self.provider_reported_seed_support == "supported"
        self._configured_seed = _optional_int(planner.get("seed"))
        self._thread_state = threading.local()
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
            if self.temperature is not None:
                body["temperature"] = self.temperature
            if self.top_p is not None:
                body["top_p"] = self.top_p
        else:
            body["temperature"] = 0 if self.temperature is None else self.temperature
            if self.top_p is not None:
                body["top_p"] = self.top_p
            body["max_tokens"] = int(max_tokens)
        requested_seed = self.requested_seed
        if self.provider_seed_supported and requested_seed is not None:
            body["seed"] = requested_seed
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
                    "provider_request_id": _provider_request_id(response, payload),
                    "retry_count": attempt,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "requested_seed": requested_seed,
                    "provider_seed_supported": self.provider_seed_supported,
                    "provider_reported_seed_support": self.provider_reported_seed_support,
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

    @property
    def last_response_metadata(self) -> dict[str, Any]:
        return dict(getattr(self._thread_state, "last_response_metadata", {}) or {})

    @last_response_metadata.setter
    def last_response_metadata(self, value: Mapping[str, Any]) -> None:
        self._thread_state.last_response_metadata = dict(value)

    @property
    def requested_seed(self) -> int | None:
        return getattr(self._thread_state, "requested_seed", self._configured_seed)

    def set_requested_seed(self, seed: int | None) -> None:
        self._thread_state.requested_seed = None if seed is None else int(seed)

    @property
    def replay_settings(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "requested_seed": self.requested_seed,
            "provider_seed_supported": self.provider_seed_supported,
            "provider_reported_seed_support": self.provider_reported_seed_support,
        }

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
            raw = self.api.chat(prompt, image_paths=image_paths, max_tokens=self._completion_budget(1400))
            api_response = self._response_metadata()
            parsed = _normalize_reasoner_payload(self._parse_or_repair(raw, kwargs), round_id=self.calls)
            action = str(parsed.get("action") or "answer")
            self._trace(
                "reasoner_investigate" if action == "investigate" else "reasoner_answer",
                prompt,
                raw,
                parsed,
                image_paths=image_paths,
                visual_manifest=visual_manifest,
                api_response=api_response,
            )
            if action == "investigate":
                if kwargs.get("force_finalize") and self._last_candidate is not None:
                    preserved = ReasonerDecision(
                        action="answer",
                        answer=self._last_candidate.answer,
                        citations=self._last_candidate.citations,
                        entity_clusters=self._last_candidate.entity_clusters,
                        support_status="insufficient",
                        support_reason=self._last_audit_reason
                        or "The model requested more investigation after the budget was exhausted.",
                    )
                    return preserved
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
                option_verdicts=dict(parsed.get("option_verdicts") or {}),
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
            audit_raw = self.api.chat(
                audit_prompt,
                image_paths=image_paths,
                max_tokens=self._completion_budget(1400),
            )
            audit_api_response = self._response_metadata()
            audit = _parse_answer_audit(audit_raw)
            self._trace(
                "reasoner_answer_audit",
                audit_prompt,
                audit_raw,
                audit,
                image_paths=image_paths,
                visual_manifest=visual_manifest,
                api_response=audit_api_response,
            )
            audit_parseable = bool(audit.get("_parseable", False))
            verdict = (
                str(audit.get("verdict", "unknown") or "unknown").strip().casefold()
                if audit_parseable else "unknown"
            )
            audit_reason = str(audit.get("reason", "") or "")
            selected_match = re.match(r"\s*([A-H])(?:\.|\)|:|\s|$)", candidate.answer.upper())
            selected_option = selected_match.group(1) if selected_match else ""
            strongest = dict(audit.get("strongest_alternative") or {})
            if selected_option and str(strongest.get("option", "") or "").strip().upper() == selected_option:
                verdict = "insufficient"
                audit_reason = (
                    f"{audit_reason} The strongest alternative duplicates the selected option, so the audit is invalid."
                ).strip()
            audit_verdicts = {
                str(option).strip().upper(): dict(row)
                for option, row in dict(audit.get("option_verdicts") or {}).items()
                if str(option).strip().upper() in dict(kwargs.get("options") or {}) and isinstance(row, Mapping)
            }
            audit_flags = []
            if not audit_parseable:
                audit_flags.append("audit_parse_failed")
            if set(audit_verdicts) != set(dict(kwargs.get("options") or {})):
                audit_flags.append("all_option_verdicts_incomplete")
            if selected_option and str(strongest.get("option", "") or "").strip().upper() == selected_option:
                audit_flags.append("strongest_alternative_duplicates_selected")
            audit_record = {
                "audit_status": "complete" if not audit_flags else "invalid" if not audit_parseable else "partial",
                "invalidity_flags": audit_flags,
                "source_revision_context": dict(
                    dict(kwargs.get("completion_status") or {}).get("revision_context") or {}
                ),
                "audit_reason": audit_reason,
            }
            self._last_audit_reason = audit_reason
            revised_answer, revised_nested_citations = _normalize_answer_payload(
                audit.get("revised_answer"),
                kwargs.get("options") or {},
            )
            if revised_answer and _valid_option_answer(revised_answer, kwargs.get("options") or {}):
                challenge = {
                    "original_answer": candidate.answer,
                    "proposed_answer": revised_answer,
                    "proposed_citations": list(
                        audit.get("revised_citations") or revised_nested_citations or candidate.citations
                    ),
                    "proposed_support_status": str(
                        audit.get("revised_support_status", "insufficient") or "insufficient"
                    ).casefold(),
                    "reason": audit_reason,
                    "adopted": False,
                }
                self._trace(
                    "reasoner_answer_challenge",
                    audit_prompt,
                    audit_raw,
                    challenge,
                    image_paths=image_paths,
                    visual_manifest=visual_manifest,
                    api_response=audit_api_response,
                )
                audit_reason = (
                    f"{audit_reason} Audit proposed {revised_answer}, but audit revisions are advisory and were not "
                    "adopted without a new Reasoner decision."
                ).strip()
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
                support_status="insufficient" if revised_answer else verdict,
                support_reason=audit_reason,
                option_verdicts=audit_verdicts,
                audit_record=audit_record,
            )
            return self._maybe_verify_claim(kwargs, evidence_digest, decision) if verdict == "supported" else decision
        image_paths, visual_manifest = self._visual_context(kwargs)
        prompt = _investigate_prompt(kwargs, visual_manifest=visual_manifest)
        raw = self.api.chat(prompt, image_paths=image_paths, max_tokens=self._completion_budget(1400))
        api_response = self._response_metadata()
        parsed = _normalize_reasoner_payload(self._parse_or_repair(raw, kwargs), round_id=self.calls)
        self._trace(
            "reasoner_investigate",
            prompt,
            raw,
            parsed,
            image_paths=image_paths,
            visual_manifest=visual_manifest,
            api_response=api_response,
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
        api_response = self._response_metadata()
        parsed = _parse_json(raw)
        self._trace(
            "reasoner_forced_answer",
            prompt,
            raw,
            parsed,
            image_paths=image_paths,
            visual_manifest=visual_manifest,
            api_response=api_response,
        )
        if not parsed:
            retry_prompt = _compact_forced_answer_prompt(kwargs, evidence_digest)
            retry_raw = self.api.chat(retry_prompt, max_tokens=4096)
            retry_api_response = self._response_metadata()
            retry_parsed = _parse_json(retry_raw)
            self._trace(
                "reasoner_forced_answer_retry",
                retry_prompt,
                retry_raw,
                retry_parsed,
                api_response=retry_api_response,
            )
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
        repaired_raw = self.api.chat(repair_prompt, max_tokens=self._completion_budget(1100))
        repaired_api_response = self._response_metadata()
        repaired = _parse_json(repaired_raw)
        self._trace(
            "reasoner_json_repair",
            repair_prompt,
            repaired_raw,
            repaired,
            api_response=repaired_api_response,
        )
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

    def _completion_budget(self, default: int) -> int:
        model = str(getattr(self.api, "model", "") or "").casefold()
        return max(int(default), 4096) if "gpt-5" in model else int(default)

    def _response_metadata(self) -> dict[str, Any]:
        return dict(getattr(self.api, "last_response_metadata", {}) or {})

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
                    option_verdicts=decision.option_verdicts,
                    audit_record=decision.audit_record,
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
        api_response: Mapping[str, Any] | None = None,
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
                "api_response": dict(api_response) if api_response is not None else self._response_metadata(),
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
        self._event_segment_cache: dict[tuple[str, str, float], InvestigationReport] = {}
        self._window_observation_history: dict[tuple[str, float, float, str, str], list[dict[str, Any]]] = {}
        self._terminal_window_evidence: dict[tuple[str, float, float, str, str], EvidenceRecord] = {}

    def reset_run_state(self) -> None:
        super().reset_run_state()
        self._query_calls.clear()
        self._event_segment_cache.clear()
        self._window_observation_history.clear()
        self._terminal_window_evidence.clear()

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
            "present in the response. Do not infer identity or causality. Return JSON only.\n"
            f"Investigation prompt: {prompt[:2400]}\nTruncated response: {str(raw or '')[:5000]}"
        )
        repaired_raw = self.api.chat(repair_prompt, image_paths=(), max_tokens=1800)
        repaired_api_response = dict(getattr(self.api, "last_response_metadata", {}) or {})
        repaired = _parse_json(repaired_raw)
        _append_jsonl(self.trace_path, {
            "type": "investigator_json_repair",
            "agent_role": "investigator",
            "model": str(getattr(self.api, "model", type(self.api).__name__)),
            "query_id": query_id,
            "prompt": repair_prompt,
            "frame_paths": [],
            "raw": repaired_raw,
            "parsed": repaired,
            "api_response": repaired_api_response,
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

        window_key = (
            str(getattr(task, "segment_id", "") or ""),
            round(float(window[0]), 1),
            round(float(window[1]), 1),
            str(getattr(task, "inspection_mode", "window") or "window"),
            _sampling_goal_key(task),
        )
        prior_window_observations = tuple(self._window_observation_history.get(window_key, ()))
        terminal_evidence = self._terminal_window_evidence.get(window_key)
        if terminal_evidence is not None and len(prior_window_observations) >= 3:
            return self._reuse_report(
                task,
                terminal_evidence,
                tool_trace=("open_segment", "adaptive_attempt_cap"),
                vlm_calls=0 if requested_window is not None else 1,
            )
        requested_floor = float(getattr(task, "sampling_floor_fps", 0.5) or 0.5)
        required_fps = requested_floor
        phase_offset_sec = 0.0
        cached = self._find_reusable_evidence(
            task,
            window[0],
            window[1],
            required_fps=required_fps,
        )
        if cached is not None:
            return self._reuse_report(
                task,
                cached,
                tool_trace=("open_segment", "reuse_observation"),
                vlm_calls=0 if requested_window is not None else 1,
            )

        event_window = str(getattr(task, "inspection_mode", "window")) == "event_window"
        anchor_event_window = bool(
            event_window
            and str(getattr(task, "inspection_intent", "") or "")
            == "event_participant_anchor_discovery"
        )
        claim_window = str(getattr(task, "inspection_mode", "window")) == "verify_claim"
        association_window = str(getattr(task, "inspection_mode", "window")) == "entity_association"
        narrative_window = str(getattr(task, "inspection_mode", "window")) == "narrative_bridge"
        adaptive_rows: list[dict[str, Any]] = []
        adaptive_triggers: list[str] = []
        preview_packets: list[Mapping[str, Any]] = []
        preview_frames_by_attempt: list[tuple[Mapping[str, Any], ...]] = []
        preview_paths_by_attempt: list[tuple[str, ...]] = []
        preview_query_ids: list[str] = []
        preview_prompts: list[str] = []
        preview_raws: list[str] = []
        preview_payloads: list[dict[str, Any]] = []
        preview_api_responses: list[dict[str, Any]] = []
        parse_status = "failed"
        parse_error = ""
        parse_repair_calls = 0
        adaptive_extra_frames = 0

        max_adaptive_attempts = max(1, 3 - min(2, len(prior_window_observations)))
        for adaptive_index in range(max_adaptive_attempts):
            preview_query_id = f"{observation_id}_preview_a{adaptive_index + 1}"
            preview_max_frames = min(
                512 if adaptive_index == 0 else max(1, 154 - adaptive_extra_frames),
                max(64, int(max(0.0, float(window[1]) - float(window[0])) * required_fps + 0.999)),
            )
            low = self.inspect_window(
                window[0],
                window[1],
                fps=required_fps,
                max_frames=preview_max_frames,
                query_id=preview_query_id,
                phase_offset_sec=phase_offset_sec,
            )
            preview_frame_limit = 16 if required_fps <= 0.5 else 32 if required_fps <= 1.0 else 64
            preview_frames = select_uniform_items(tuple(low["frames"]), preview_frame_limit)
            preview_paths = tuple(str(row["path"]) for row in preview_frames)
            arbitration_context = (*prior_window_observations, *adaptive_rows)
            if event_window:
                preview_prompt = _event_preview_prompt(
                    self.workspace, task, segment_packet, low,
                    prior_events=prior_events, prior_observations=arbitration_context,
                )
            elif association_window or narrative_window:
                preview_prompt = _preview_prompt(
                    self.workspace, task, segment_packet, low,
                    prior_observations=arbitration_context,
                )
            elif claim_window:
                preview_prompt = _claim_preview_prompt(
                    self.workspace, task, segment_packet, low,
                    prior_observations=arbitration_context,
                )
            else:
                preview_prompt = _preview_prompt(
                    self.workspace, task, segment_packet, low,
                    prior_observations=arbitration_context,
                )
            preview_raw = self.api.chat(preview_prompt, image_paths=preview_paths, max_tokens=1400)
            preview_api_response = dict(getattr(self.api, "last_response_metadata", {}) or {})
            preview, attempt_parse_status, attempt_parse_error, attempt_repair_calls = self._parse_structured_observation(
                preview_raw, query_id=query_id, prompt=preview_prompt, image_paths=preview_paths,
            )
            parse_status = attempt_parse_status
            parse_error = attempt_parse_error
            parse_repair_calls += attempt_repair_calls
            row = _observation_governance_row(preview, required_fps, phase_offset_sec)
            trigger_reasons = _adaptive_sampling_triggers(row, arbitration_context)
            adaptive_rows.append(row)
            adaptive_triggers.extend(trigger_reasons)
            preview_packets.append(low)
            preview_frames_by_attempt.append(tuple(preview_frames))
            preview_paths_by_attempt.append(preview_paths)
            preview_query_ids.append(preview_query_id)
            preview_prompts.append(preview_prompt)
            preview_raws.append(preview_raw)
            preview_payloads.append(preview)
            preview_api_responses.append(preview_api_response)
            _append_jsonl(
                self.trace_path,
                {
                    "type": "investigator_preview",
                    "agent_role": "investigator",
                    "model": str(getattr(self.api, "model", type(self.api).__name__)),
                    "query_id": query_id,
                    "observation_id": observation_id,
                    "preview_query_id": preview_query_id,
                    "adaptive_attempt": adaptive_index + 1,
                    "adaptive_trigger_reasons": trigger_reasons,
                    "window": list(window),
                    "prompt": preview_prompt,
                    "frame_paths": list(preview_paths),
                    "raw": preview_raw,
                    "parsed": preview,
                    "api_response": preview_api_response,
                    "time": time.time(),
                },
            )
            if not trigger_reasons or adaptive_index + 1 >= max_adaptive_attempts:
                break
            if adaptive_index > 0:
                adaptive_extra_frames += len(preview_frames)
            if adaptive_extra_frames >= 154:
                adaptive_triggers.append("adaptive_frame_budget_exhausted")
                break
            required_fps = min(2.0, max(required_fps * 2.0, 1.0))
            phase_offset_sec = 1.0 / (2.0 * required_fps) if adaptive_index == 0 else 2.0 / (3.0 * required_fps)

        adaptive_extra_frames = sum(len(items) for items in preview_frames_by_attempt[1:])
        low = preview_packets[-1]
        preview_frames = preview_frames_by_attempt[-1]
        preview_paths = preview_paths_by_attempt[-1]
        preview_query_id = preview_query_ids[-1]
        preview_prompt = preview_prompts[-1]
        preview_raw = preview_raws[-1]
        preview = preview_payloads[-1]
        final_api_response = preview_api_responses[-1]

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
        tool_trace = ["open_segment"]
        for row in adaptive_rows:
            inspection_trace = f"inspect_window:{float(row['sampling_fps']):.1f}"
            if float(row["phase_offset_sec"]):
                inspection_trace += f":phase={float(row['phase_offset_sec']):.3f}"
            tool_trace.append(inspection_trace)
        vlm_calls = len(preview_payloads) + (0 if requested_window is not None else 1) + parse_repair_calls

        requires_specialized_evidence = anchor_event_window or association_window or narrative_window
        if (
            _truthy(preview.get("need_detail"))
            or region_box is not None
            or requires_specialized_evidence
        ) and len(adaptive_rows) < 3:
            detail_fps, detail_max_frames = _detail_sampling_request(
                preview,
                default_fps=self.highfps,
                default_max_frames=self.highfps_max_frames,
            )
            detail_fps = max(detail_fps, required_fps)
            selected_window = _select_detail_window(
                preview,
                window,
                task,
                segment_packet,
                max_detail_sec=detail_max_frames / detail_fps,
            )
            detail_query_id = f"{observation_id}_detail"
            detail_max_frames = min(
                512,
                max(
                    detail_max_frames,
                    int(max(0.0, selected_window[1] - selected_window[0]) * detail_fps + 0.999),
                ),
            )
            selected_packet = self.inspect_window(
                selected_window[0],
                selected_window[1],
                fps=detail_fps,
                max_frames=detail_max_frames,
                query_id=detail_query_id,
                phase_offset_sec=phase_offset_sec,
            )
            source_detail_frames = tuple(selected_packet["frames"])
            selected_frames = _materialize_model_frames(
                self.workspace,
                observation_id,
                source_detail_frames,
            )
            detail_paths = tuple(str(row["path"]) for row in selected_frames)
            selected_packet = {
                **selected_packet,
                "frames": [dict(row) for row in selected_frames],
                "model_frame_preprocessing": {
                    "frame_count": len(selected_frames),
                    "total_bytes": sum(Path(str(row["path"])).stat().st_size for row in selected_frames),
                    "max_total_bytes": 18_000_000,
                },
            }
            if region_box is not None:
                context_frames = select_uniform_items(source_detail_frames, min(4, len(source_detail_frames)))
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
                selected_frames = context_frames
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
                if anchor_event_window:
                    final_prompt = _anchor_event_evidence_prompt(
                        self.workspace, task, segment_packet, selected_packet, preview=preview,
                    )
                else:
                    final_prompt = _event_evidence_prompt(
                        self.workspace,
                        task,
                        segment_packet,
                        selected_packet,
                        preview=preview,
                        prior_events=prior_events,
                    )
            elif association_window:
                final_prompt = _entity_association_prompt(
                    self.workspace, task, segment_packet, selected_packet, preview=preview,
                )
            elif narrative_window:
                final_prompt = _narrative_bridge_prompt(
                    self.workspace, task, segment_packet, selected_packet, preview=preview,
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
                max_tokens=2200 if anchor_event_window else 1400,
            )
            final_api_response = dict(getattr(self.api, "last_response_metadata", {}) or {})
            parsed, parse_status, parse_error, detail_repair_calls = self._parse_structured_observation(
                raw,
                query_id=query_id,
                prompt=final_prompt,
                image_paths=model_image_paths,
            )
            tool_trace.append(f"inspect_window:{detail_fps:.1f}:{len(selected_frames)}")
            vlm_calls += 1 + detail_repair_calls

        parsed, fallback_used = _with_explicit_measurement_fallback(
            parsed,
            raw,
            task=task,
            question=self.workspace.case.question,
        )
        if fallback_used:
            parse_status = "fallback_extracted"
        governance_row = _observation_governance_row(parsed, float(selected_packet["sampling"]["fps"]), phase_offset_sec)
        governance_row["summary"] = str(parsed.get("summary") or raw or "").strip()[:800]
        conflicted_slot_ids = _conflicted_slot_ids((*prior_window_observations, *adaptive_rows, governance_row))
        history = self._window_observation_history.setdefault(window_key, [])
        history.extend(dict(row) for row in adaptive_rows)
        if detail_query_id and governance_row != adaptive_rows[-1]:
            history.append(governance_row)
        window_attempt_count = len(history)

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
                "api_response": final_api_response,
                "structured_parse_status": parse_status,
                "structured_parse_error": parse_error,
                "time": time.time(),
            },
        )
        evidence_id = f"ev_{observation_id}_001"
        confidence = _confidence(parsed.get("confidence"), default=0.6)
        finding = _normalize_finding(parsed)
        entities = _normalize_entities(
            parsed.get("entities"),
            frame_paths=frame_paths,
            frame_times=frame_times,
            observation_id=observation_id,
            window_duration_sec=max(0.0, float(selected_window[1]) - float(selected_window[0])),
        )
        events = _normalize_events(
            parsed.get("events"),
            selected_window,
            anchor_discovery=(
                str(getattr(task, "inspection_intent", "") or "")
                == "event_participant_anchor_discovery"
            ),
        )
        entity_associations = _normalize_entity_associations(
            parsed.get("entity_associations"),
            reference_entities=tuple(getattr(task, "reference_entities", ()) or ()),
            entities=entities,
        )
        narrative_facts = _normalize_narrative_facts(
            parsed.get("narrative_facts"),
            options=self.workspace.case.options,
            default_episode_id=(
                str(getattr(task, "episode_id", "") or getattr(task, "boundary_episode_id", "") or "")
                or f"narrative:{segment_packet.get('segment_id', 'unknown')}"
            ),
        )
        claim_assessment = _normalize_claim_assessment(parsed, task) if claim_window else {}
        target_presence = normalize_target_presence(parsed.get("target_presence"), evidence_id=evidence_id)
        measurements = normalize_measurements(parsed.get("measurements"), evidence_id=evidence_id)
        relations = normalize_relations(parsed.get("relations"), evidence_id=evidence_id)
        state_transitions = _normalize_state_transitions(parsed.get("state_transitions"))
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
        if entity_associations:
            evidence_kind = "entity_association"
        elif narrative_facts:
            evidence_kind = "narrative_inference"
        elif claim_assessment:
            evidence_kind = "claim_verification"
        elif supports_answer_event:
            evidence_kind = "event_observation"
        elif supports_identity_anchor or entities:
            evidence_kind = "entity_observation"
        else:
            evidence_kind = "visual_observation"
        effective_fps = float(selected_packet["sampling"]["fps"])
        absence = qualify_absence(
            (float(selected_window[0]), float(selected_window[1])),
            ((float(selected_window[0]), float(selected_window[1])),),
            1.0 / effective_fps if effective_fps > 0 else None,
            getattr(task, "expected_event_dwell_sec", None),
            str(parsed.get("visibility_status", "unknown") or "unknown"),
            targeted_inspection_count=window_attempt_count,
        )
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
            observation_polarity=(
                "positive" if finding == "found" else "negative" if finding == "not_found" else "unknown"
            ),
            sampling_coverage="sparse",
            request_ids=(query_id,),
            coverage_manifest=(
                CoverageSegment(query_id, float(selected_window[0]), float(selected_window[1]), "visual", 1.0),
            ),
            task_id=query_id,
            observation_id=observation_id,
            sampling_fps=effective_fps,
            confidence=confidence,
            source_lineage=tuple(dict(item) for item in selected_packet["source_lineage"]),
            entity_ids=tuple(f"{observation_id}:{item['local_id']}" for item in entities),
            operation_metadata={
                "entities": entities,
                "events": events,
                "entity_associations": entity_associations,
                "narrative_facts": narrative_facts,
                "claim_assessment": claim_assessment,
                "target_presence": to_jsonable(target_presence),
                "measurements": to_jsonable(measurements),
                "relations": to_jsonable(relations),
                "state_transitions": state_transitions,
                "structured_parse_status": parse_status,
                "structured_parse_error": parse_error,
                "source_candidate_ids": list(getattr(task, "source_candidate_ids", ()) or ()),
                "inspection_mode": str(getattr(task, "inspection_mode", "") or ""),
                "sampling_floor_fps": getattr(task, "sampling_floor_fps", None),
                "expected_event_dwell_sec": getattr(task, "expected_event_dwell_sec", None),
                "inspection_intent": str(getattr(task, "inspection_intent", "") or ""),
                "origin_gap_id": str(getattr(task, "origin_gap_id", "") or ""),
                "target_condition_ids": list(getattr(task, "target_condition_ids", ()) or ()),
                "boundary_episode_id": str(getattr(task, "boundary_episode_id", "") or ""),
                "target_option_predicates": list(getattr(task, "target_option_predicates", ()) or ()),
                "target_requirement_ids": list(getattr(task, "target_requirement_ids", ()) or ()),
                "candidate_id": str(getattr(task, "candidate_id", "") or ""),
                "episode_id": str(getattr(task, "episode_id", "") or ""),
                "entity_hypothesis_id": str(getattr(task, "entity_hypothesis_id", "") or ""),
                "target_option_predicate_ids": list(getattr(task, "target_option_predicate_ids", ()) or ()),
                "supports_identity_anchor": supports_identity_anchor,
                "supports_answer_event": supports_answer_event,
                "investigation": outcome,
                "sampling_policy": {
                    "floor_fps": requested_floor,
                    "floor_specified": bool(getattr(task, "sampling_floor_specified", False)),
                    "floor_unspecified": not bool(getattr(task, "sampling_floor_specified", False)),
                    "temporal_resolution_rationale": str(
                        getattr(task, "temporal_resolution_rationale", "") or ""
                    ),
                    "effective_fps": float(selected_packet["sampling"]["fps"]),
                    "phase_offset_sec": phase_offset_sec,
                    "adaptive_attempt_count": len(adaptive_rows) + (1 if detail_query_id else 0),
                    "window_attempt_count": window_attempt_count,
                    "adaptive_trigger_reasons": list(dict.fromkeys(adaptive_triggers)),
                    "adaptive_frame_budget": 154,
                    "adaptive_extra_frames": adaptive_extra_frames,
                    "arbitration_prior_observations": list(prior_window_observations[-3:]),
                },
                "finding": finding,
                "structured_slots": governance_row["structured_slots"],
                "conflicted_slot_ids": list(conflicted_slot_ids),
                "evidence_state": "conflicted" if conflicted_slot_ids else "active",
                "absence_resolution_fps": (
                    effective_fps if finding == "not_found" else None
                ),
                "absence_status": absence.status if finding == "not_found" else "not_observed",
                "absence_qualification": to_jsonable(absence),
                "phase_coverage": sorted(
                    {round(float(row["phase_offset_sec"]), 6) for row in adaptive_rows}
                ),
                "qualified_absence": bool(
                    finding == "not_found" and absence.status == "qualified_absence"
                ),
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
        if window_attempt_count >= 3 and (
            evidence.observation_polarity != "positive" or conflicted_slot_ids
        ):
            self._terminal_window_evidence[window_key] = evidence
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
        segment_id = str(getattr(task, "segment_id", "") or "")
        cache_key = (
            segment_id,
            _sampling_goal_key(task),
            float(getattr(task, "sampling_floor_fps", 0.5) or 0.5),
        )
        cached = self._event_segment_cache.get(cache_key)
        if cached is not None and any(
            record.observation_polarity != "positive"
            or str(record.operation_metadata.get("evidence_state", "active") or "active") != "active"
            for record in cached.evidence
        ):
            cached = None
        if cached is not None:
            return replace(
                cached,
                query_id=str(getattr(task, "query_id", "") or cached.query_id),
                cost={
                    "beat_windows": 0,
                    "tool_trace": ("reuse_segment_event_observation",),
                    "preview_frames": 0,
                    "detail_frames": 0,
                    "frames": 0,
                    "vlm_calls": 0,
                    "reused": True,
                },
                progress_flags=tuple(dict.fromkeys((*cached.progress_flags, "segment_event_observation_reused"))),
                coverage_delta=(),
            )
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
                origin_gap_id=str(getattr(task, "origin_gap_id", "") or ""),
                target_condition_ids=tuple(getattr(task, "target_condition_ids", ()) or ()),
                boundary_episode_id=str(getattr(task, "boundary_episode_id", "") or ""),
                target_option_predicates=tuple(getattr(task, "target_option_predicates", ()) or ()),
                target_requirement_ids=tuple(getattr(task, "target_requirement_ids", ()) or ()),
                candidate_id=str(getattr(task, "candidate_id", "") or ""),
                episode_id=str(getattr(task, "episode_id", "") or ""),
                entity_hypothesis_id=str(getattr(task, "entity_hypothesis_id", "") or ""),
                target_option_predicate_ids=tuple(getattr(task, "target_option_predicate_ids", ()) or ()),
                sampling_floor_fps=(
                    float(getattr(task, "sampling_floor_fps", 0.5) or 0.5)
                    if bool(getattr(task, "sampling_floor_specified", False))
                    else None
                ),
                temporal_resolution_rationale=str(
                    getattr(task, "temporal_resolution_rationale", "") or ""
                ),
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
        combined = InvestigationReport(
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
        self._event_segment_cache[cache_key] = combined
        return combined

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
    api_response = dict(getattr(api, "last_response_metadata", {}) or {})
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
            "api_response": api_response,
            "selected_window": list(selected),
            "fallback_used": fallback_used,
            "time": time.time(),
        },
    )
    return selected


def _event_enumeration_windows(
    segment_packet: Mapping[str, Any],
) -> tuple[tuple[float, float], ...]:
    start, end = segment_packet["virtual_time_range"]
    return ((float(start), float(end)),) if float(end) > float(start) else ()


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


def _detail_sampling_request(
    preview: Mapping[str, Any],
    *,
    default_fps: float = 2.0,
    default_max_frames: int = 64,
) -> tuple[float, int]:
    allowed_fps = (0.5, 1.0, 2.0)
    try:
        requested_fps = float(preview.get("requested_fps", default_fps))
    except (TypeError, ValueError):
        requested_fps = float(default_fps)
    fps = min(allowed_fps, key=lambda candidate: abs(candidate - requested_fps))
    try:
        requested_max_frames = int(preview.get("requested_max_frames", default_max_frames))
    except (TypeError, ValueError):
        requested_max_frames = int(default_max_frames)
    return float(fps), max(1, min(512, requested_max_frames))


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


def _materialize_model_frames(
    workspace: VirtualVideoWorkspace,
    observation_id: str,
    frames: Sequence[Mapping[str, Any]],
    *,
    max_total_bytes: int = 18_000_000,
) -> tuple[dict[str, Any], ...]:
    source_rows = tuple(dict(frame) for frame in frames)
    if not source_rows:
        return ()
    output_dir = workspace.root_dir / "observations" / "model_frames" / observation_id
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = ((512, 82), (448, 76), (384, 70), (320, 64))
    rendered: tuple[dict[str, Any], ...] = ()
    for max_edge, quality in profiles:
        rows = []
        for index, frame in enumerate(source_rows, start=1):
            source = Path(str(frame.get("path", "") or ""))
            if not source.exists():
                continue
            output_path = output_dir / f"frame_{index:04d}.jpg"
            with Image.open(source) as opened:
                image = opened.convert("RGB")
                scale = min(1.0, float(max_edge) / max(1, max(image.size)))
                if scale < 1.0:
                    image = image.resize(
                        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
                        Image.Resampling.LANCZOS,
                    )
                image.save(output_path, format="JPEG", quality=quality, optimize=True)
            rows.append({**frame, "path": str(output_path), "parent_path": str(source)})
        rendered = tuple(rows)
        if sum(Path(str(row["path"])).stat().st_size for row in rendered) <= int(max_total_bytes):
            return rendered
    return rendered


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


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


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
        entity_hypothesis_id = re.sub(
            r"[^a-z0-9_]+",
            "_",
            str(item.get("entity_hypothesis_id", "") or "").strip().casefold(),
        ).strip("_")
        association_confidence = _confidence(item.get("association_confidence"), default=0.0)
        rows.append(
            {
                "local_id": local_id,
                "entity_observation_id": entity_observation_id,
                "entity_hypothesis_id": entity_hypothesis_id,
                "association_confidence": association_confidence,
                "description": description,
                "visual_signature": visual_signature,
                "attributes": dict(item.get("attributes", {}) or {}) if isinstance(item.get("attributes"), Mapping) else {},
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


def _normalize_events(
    value: Any,
    window: tuple[float, float],
    *,
    anchor_discovery: bool = False,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    window_start, window_end = float(window[0]), float(window[1])
    rows = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            continue
        description = str(item.get("description", "") or "").strip()
        supports_question_event = _truthy(item.get("supports_question_event"))
        supports_anchor_event = bool(anchor_discovery and _truthy(item.get("supports_anchor_event")))
        qualification_status = str(item.get("qualification_status", "") or "").strip().casefold()
        preconditions_failed = "preconditions_met" in item and not _truthy(item.get("preconditions_met"))
        if preconditions_failed:
            supports_question_event = False
            qualification_status = qualification_status or "unqualified_precondition"
        if not description or not (
            supports_question_event
            or supports_anchor_event
            or preconditions_failed
            or qualification_status in {"observed_candidate", "unqualified_precondition", "conflicted"}
        ):
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
        subject_id = " ".join(str(item.get("subject_id", "") or "").strip().casefold().split())
        object_id = " ".join(str(item.get("object_id", "") or "").strip().casefold().split())
        participant_roles = []
        for raw_participant in tuple(item.get("participants", ()) or ()):
            if not isinstance(raw_participant, Mapping):
                continue
            hypothesis_id = re.sub(
                r"[^a-z0-9_]+",
                "_",
                str(raw_participant.get("entity_hypothesis_id", "") or "").strip().casefold(),
            ).strip("_")
            participant_id = " ".join(
                str(raw_participant.get("participant_id", "") or hypothesis_id).strip().casefold().split()
            )
            if not participant_id:
                continue
            participant_roles.append(
                {
                    "participant_id": participant_id,
                    "entity_hypothesis_id": hypothesis_id,
                    "role": str(raw_participant.get("role", "") or "").strip().casefold(),
                    "visual_signature": str(raw_participant.get("visual_signature", "") or "").strip(),
                    "attributes": (
                        dict(raw_participant.get("attributes", {}) or {})
                        if isinstance(raw_participant.get("attributes"), Mapping)
                        else {}
                    ),
                    "association_confidence": _confidence(
                        raw_participant.get("association_confidence"), default=0.0
                    ),
                    "ordinal": _positive_int(raw_participant.get("ordinal")),
                }
            )
        participant_ids = list(
            dict.fromkeys(
                value
                for value in (
                    *(
                        " ".join(str(value or "").strip().casefold().split())
                        for value in tuple(item.get("participant_ids", ()) or ())
                    ),
                    subject_id,
                    object_id,
                    *(participant["participant_id"] for participant in participant_roles),
                )
                if value
            )
        )[:8]
        rows.append(
            {
                "local_id": str(item.get("local_id", "") or f"event_{index}"),
                "event_key": " ".join(str(item.get("event_key", "") or "").strip().casefold().split()),
                "event_class": " ".join(str(item.get("event_class", "") or "").strip().casefold().split()),
                "counting_unit": " ".join(str(item.get("counting_unit", "") or "").strip().casefold().split()),
                "participant_ids": participant_ids,
                "participants": participant_roles,
                "subject_id": subject_id,
                "object_id": object_id,
                "state_before": str(item.get("state_before", "") or "").strip(),
                "transition": str(item.get("transition", "") or "").strip(),
                "state_after": str(item.get("state_after", "") or "").strip(),
                "preconditions_met": _truthy(item.get("preconditions_met")) if "preconditions_met" in item else None,
                "qualification_status": qualification_status,
                "qualification": (
                    dict(item.get("qualification", {}) or {})
                    if isinstance(item.get("qualification"), Mapping)
                    else {}
                ),
                "phase": " ".join(str(item.get("phase", "unknown") or "unknown").strip().casefold().split()),
                "description": description,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "supports_question_event": bool(supports_question_event or supports_anchor_event),
                "continues_from_previous": _truthy(item.get("continues_from_previous")),
                "continues_to_next": _truthy(item.get("continues_to_next")),
            }
        )
    return tuple(rows)


def _normalize_entity_associations(
    value: Any,
    *,
    reference_entities: Sequence[Mapping[str, Any]],
    entities: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    references = {
        str(item.get("participant_id", "") or "").strip().casefold(): dict(item)
        for item in reference_entities
        if str(item.get("participant_id", "") or "").strip()
    }
    targets = {str(item.get("local_id", "") or ""): item for item in entities}
    rows = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            continue
        participant_id = str(item.get("source_participant_id", "") or "").strip().casefold()
        reference = references.get(participant_id)
        target_local_id = str(item.get("target_local_id", "") or "").strip()
        target = targets.get(target_local_id)
        if reference is None or target is None:
            continue
        expected_hypothesis = str(reference.get("entity_hypothesis_id", "") or "").strip().casefold()
        hypothesis_id = re.sub(
            r"[^a-z0-9_]+", "_", str(item.get("entity_hypothesis_id", "") or "").strip().casefold()
        ).strip("_")
        status = str(item.get("status", "unknown") or "unknown").strip().casefold()
        if status not in {"supported", "refuted", "unknown"}:
            status = "unknown"
        confidence = _confidence(item.get("confidence"), default=0.0)
        shared = dict(item.get("shared_attributes", {}) or {}) if isinstance(item.get("shared_attributes"), Mapping) else {}
        distinguishing = (
            dict(item.get("distinguishing_attributes", {}) or {})
            if isinstance(item.get("distinguishing_attributes"), Mapping)
            else {}
        )
        if hypothesis_id != expected_hypothesis:
            status, confidence = "unknown", min(confidence, 0.5)
            hypothesis_id = expected_hypothesis
        if status == "supported" and len([value for value in shared.values() if str(value).strip()]) < 2:
            status, confidence = "unknown", min(confidence, 0.59)
        rows.append(
            {
                "association_id": str(item.get("association_id", "") or f"association_{index}"),
                "source_participant_id": str(reference.get("participant_id", "") or participant_id),
                "source_event_key": str(reference.get("source_event_key", "") or ""),
                "source_episode_id": str(reference.get("source_episode_id", "") or reference.get("source_event_key", "") or ""),
                "source_event_role": str(reference.get("role", "") or ""),
                "ordinal": _positive_int(reference.get("ordinal")),
                "target_entity_observation_id": str(target.get("entity_observation_id", "") or ""),
                "entity_hypothesis_id": hypothesis_id,
                "status": status,
                "confidence": confidence,
                "shared_attributes": shared,
                "distinguishing_attributes": distinguishing,
                "reason": str(item.get("reason", "") or "")[:400],
            }
        )
    return tuple(rows)


def _normalize_state_transitions(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    rows = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        object_id = str(item.get("object_hypothesis_id", item.get("object_id", "")) or "").strip()
        attribute_type = str(item.get("attribute_type", "") or "").strip().casefold()
        before = item.get("raw_value_before", item.get("value_before"))
        after = item.get("raw_value_after", item.get("value_after"))
        same_object = str(item.get("same_object_relation", "unknown") or "unknown").strip().casefold()
        if same_object not in {"supported", "contradicted", "unknown"}:
            same_object = "unknown"
        rows.append(
            {
                "object_hypothesis_id": object_id,
                "attribute_type": attribute_type,
                "raw_value_before": before,
                "raw_value_after": after,
                "before_witness": list(item.get("before_witness", item.get("before_witness_range", ())) or ()),
                "after_witness": list(item.get("after_witness", item.get("after_witness_range", ())) or ()),
                "same_object_relation": same_object,
                "coverage_occlusion_status": str(
                    item.get("coverage_occlusion_status", item.get("occlusion_status", "unknown")) or "unknown"
                ).strip().casefold(),
            }
        )
    return tuple(rows)


def _normalize_narrative_facts(
    value: Any,
    *,
    options: Mapping[str, Any],
    default_episode_id: str = "",
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    valid_options = {str(option).upper() for option in options}
    rows = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            continue
        setup = str(item.get("setup_state", "") or "").strip()
        outcome = str(item.get("outcome_state", "") or "").strip()
        inference = str(item.get("inference", "") or "").strip()
        if not (setup or outcome or inference):
            continue
        complete = bool(setup and outcome and inference)
        episode_id = str(item.get("episode_id", "") or default_episode_id).strip()
        relation_type = str(item.get("relation_type", "") or "").strip().casefold()
        assessments = []
        for assessment in tuple(item.get("hypothesis_assessments", ()) or ()):
            if not isinstance(assessment, Mapping):
                continue
            option_id = str(assessment.get("option_id", "") or "").strip().upper()
            if option_id not in valid_options:
                continue
            status = str(assessment.get("status", "unknown") or "unknown").strip().casefold()
            if status not in {"supported", "contradicted", "unknown"}:
                status = "unknown"
            if not complete:
                status = "unknown"
            if relation_type == "temporal_cooccurrence" and status == "supported":
                status = "unknown"
            assessments.append(
                {
                    "option_id": option_id,
                    "status": status,
                    "reason": str(assessment.get("reason", "") or "")[:400],
                }
            )
        counterevidence = [
            {
                "option_id": str(row.get("option_id", "") or "").strip().upper(),
                "observation": str(row.get("observation", "") or "")[:400],
            }
            for row in tuple(item.get("alternative_counterevidence", ()) or ())
            if isinstance(row, Mapping)
            and str(row.get("option_id", "") or "").strip().upper() in valid_options
            and str(row.get("observation", "") or "").strip()
        ]
        confidence = _confidence(item.get("confidence"), default=0.0)
        rows.append(
            {
                "fact_id": str(item.get("fact_id", "") or f"narrative_{index}"),
                "episode_id": episode_id,
                "timeline_phase": str(item.get("timeline_phase", "unknown") or "unknown").strip().casefold(),
                "temporal_role": str(item.get("temporal_role", item.get("timeline_phase", "unknown")) or "unknown").strip().casefold(),
                "anchor_event_id": str(item.get("anchor_event_id", "") or "").strip(),
                "anchor_match": _truthy(item.get("anchor_match")),
                "subject_id": str(item.get("subject_id", "") or "").strip(),
                "relation_type": relation_type,
                "predicate": str(item.get("predicate", "") or "").strip().casefold(),
                "setup_state": setup,
                "observed_bridge": str(item.get("observed_bridge", "") or "").strip(),
                "outcome_state": outcome,
                "inference": inference,
                "inference_basis": str(item.get("inference_basis", "") or "").strip().casefold(),
                "confidence": confidence if complete else min(confidence, 0.59),
                "hypothesis_assessments": assessments,
                "alternative_counterevidence": counterevidence,
                "supporting_observation_ids": [
                    str(value) for value in tuple(item.get("supporting_observation_ids", ()) or ()) if str(value)
                ],
                "agent_witness_type": str(item.get("agent_witness_type", "") or "").strip().casefold(),
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


def _sampling_plan_instruction() -> str:
    return (
        "Every visual task must declare sampling_floor_fps as 0.5, 1.0, or 2.0, expected_event_dwell_sec as a positive "
        "number when absence may matter, and temporal_resolution_rationale as one short sentence about dwell time or transition speed. "
        "Choose from evidence dynamics, not the question category: use 0.5 for persistent scenes or states, 1.0 for "
        "ordinary appearance and motion, and 2.0 for brief transitions, fast relative motion, small changing text, or exact "
        "temporal boundaries. Estimate target dwell time d when possible and keep the sampling interval <= d/2. "
        "Use inspection_mode=enumerate_events when the task must exhaustively enumerate timestamped occurrences, "
        "event_window for one atomic occurrence, verify_claim for an independent contrastive check, entity_association for "
        "cross-window participant re-identification, narrative_bridge for setup-to-outcome inference, search_asr for literal "
        "navigation, and window otherwise. When validating a change, request ordered before/during/after evidence with timestamps.\n"
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
        "\"segment_id\":\"seg_0001\",\"time_range\":null,\"inspection_mode\":\"window|event_window|enumerate_events|verify_claim|search_asr|entity_association|narrative_bridge\","
        "\"search_terms\":[],\"modality_hint\":[\"visual\"],\"expected_evidence\":\"...\","
        "\"gap_id\":\"gap_r1\",\"success_conditions\":[\"condition 1\"],\"direction\":\"local|forward|backward|global\","
        "\"region_hint\":\"optional visible region\",\"sampling_floor_fps\":1.0,"
        "\"temporal_resolution_rationale\":\"Expected visible dwell time and required interval.\"}]}.\n"
        f"{_sampling_plan_instruction()}"
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
        "semantics, subject binding, and relation to the stated boundary. If the question contrasts one participant with the other, "
        "the counter must be visibly bound to the requested participant. Do not infer a quantity from answer options.\n"
        "For summarize questions, inspect opening/closing framing plus representative early, middle, and late segments. A visually "
        "salient local process is not the full-video topic unless the narration and repeated structure support it.\n"
        "For temporal-label questions, first witness the target event, then bind it to the requested episode, day/meal, period, or "
        "relative video position using nearby captions, narration, or timeline context. A virtual timestamp alone is not an option label.\n"
        "For front/middle/back options, compare the witnessed event time with workspace_duration_sec; do not infer position from the "
        "segment number or from how late the event was discovered.\n"
        "For causal, agent-relation, and event-outcome questions, request evidence for the exact predicate in each plausible option, "
        "including the transition or interaction that distinguishes outcome from a nearby but non-answer event.\n"
        "For source-relative minute questions, prioritize the supplied temporal_navigation candidate segments.\n"
        "For identity-anchor questions, first locate evidence matching every identity_anchor_term before investigating the later event.\n"
        f"Question: {kwargs['question']}\nOptions: {json.dumps(kwargs['options'], ensure_ascii=False)}\n"
        f"Query contract: {json.dumps(kwargs.get('query_contract') or {}, ensure_ascii=False)}\n"
        f"Query requirements: {json.dumps(kwargs.get('query_requirements') or {}, ensure_ascii=False)}\n"
        f"workspace_duration_sec: {float(kwargs.get('workspace_duration_sec', 0.0) or 0.0)}\n"
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
        else "This is the last repairable round. Propose the best candidate now when possible so the answer audit can issue a targeted repair; avoid broad exploration.\n"
        if kwargs.get("pre_final_checkpoint")
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
        "\"inspection_mode\":\"window|event_window|enumerate_events|verify_claim|search_asr|entity_association|narrative_bridge\",\"search_terms\":[],\"modality_hint\":[\"visual\"],"
        "\"expected_evidence\":\"...\",\"gap_id\":\"gap_r2\",\"success_conditions\":[\"condition 1\"],"
        "\"direction\":\"local|forward|backward|global\",\"region_hint\":\"optional visible region\","
        "\"sampling_floor_fps\":1.0,\"temporal_resolution_rationale\":\"Expected dwell time and required interval.\"}]}.\n"
        f"{_sampling_plan_instruction()}"
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
        "For total event-count questions, use confirmed_event_candidates in completion_status and matching event_candidate evidence rows. "
        "Count one per stable candidate_id, cite at least one supporting visual evidence_id per adopted candidate, investigate every "
        "unresolved_event_window before claiming grounded completion, and never use entity_clusters for events. The completion_status "
        "event ledger is the only count state: do not independently recount a sampled prose digest. If it appears over-split or "
        "over-merged, request candidate reconciliation instead of revising to a prose-derived total.\n"
        "For scalar_quantity questions, use only structured measurement facts with the requested unit. Distinguish delta or cumulative "
        "measurements, bind the value to the requested subject, exclude observations after the stated boundary, and cite every operand "
        "used in the derivation. An unbound overlay cannot answer a one-team-versus-other-team question.\n"
        "For summarize questions, require representative full-video coverage and explicit narrative framing; do not promote the best "
        "documented local subtopic into the global title. For temporal-label questions, require both the target event and its label/context "
        "binding. For front/middle/back labels, normalize the event time by workspace_duration_sec rather than segment order. For causal, "
        "agent-relation, and event-outcome questions, compare the exact option predicates rather than topical overlap.\n"
        "For identity-anchor questions, do not answer while missing_identity_anchor_terms is non-empty. "
        "The final entity cluster must cite both anchor evidence and the later event evidence for the same person.\n"
        "Treat independent claim_assessment evidence as a direct check of the proposed relation. If it refutes a candidate, "
        "revise the answer or investigate the strongest alternative; do not relabel relevance as support. If candidate_role "
        "does not satisfy claim_relation, investigate the missing relation or choose a better-supported option; a stated use, "
        "downstream consequence, or after-state is not a decision motive.\n"
        f"Question: {kwargs['question']}\nOptions: {json.dumps(kwargs['options'], ensure_ascii=False)}\n"
        f"Query contract: {json.dumps(kwargs.get('query_contract') or {}, ensure_ascii=False)}\n"
        f"Query requirements: {json.dumps(kwargs.get('query_requirements') or {}, ensure_ascii=False)}\n"
        f"workspace_duration_sec: {float(kwargs.get('workspace_duration_sec', 0.0) or 0.0)}\n"
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
    contract = dict(kwargs.get("query_contract") or {})
    requirements = dict(kwargs.get("query_requirements") or {})
    return requires_option_audit(contract, requirements)


def _requires_independent_claim_verification(kwargs: Mapping[str, Any]) -> bool:
    question = str(kwargs.get("question", "") or "").casefold()
    requirements = dict(kwargs.get("query_requirements") or {})
    contract = dict(kwargs.get("query_contract") or {})
    if (
        requirements.get("requires_identity_link")
        or requirements.get("requires_temporal_sequence")
        or requirements.get("requires_state_tracking")
        or str(contract.get("aggregation", "") or "") in {"compare", "order"}
        or str(contract.get("quantifier", "") or "") in {"comparison", "order"}
    ):
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
                        "inspection_mode": str(task.get("inspection_mode", "window") or "window"),
                        "search_terms": list(task.get("search_terms") or ()),
                        "modality_hint": list(task.get("modality_hint") or ("visual", "asr")),
                        "expected_evidence": str(task.get("expected_evidence") or goal),
                        "temporal_resolution_rationale": str(
                            task.get("temporal_resolution_rationale", "") or ""
                        ),
                        **(
                            {"sampling_floor_fps": task.get("sampling_floor_fps")}
                            if "sampling_floor_fps" in task
                            else {}
                        ),
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
            "enumerate",
            "list",
            "catalog",
            "track",
            "compare",
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
        "Free-text local counts and candidate-only entities are not count evidence. For summarize claims, require representative "
        "full-video coverage and narrative framing; a locally dominant visual process is not enough. For temporal labels, require a "
        "witnessed target event that satisfies every identifying qualifier plus explicit nearby episode/day/meal/period or relative-position binding; normalize front/middle/back "
        "against workspace_duration_sec. For event outcomes and agent "
        "relations, require the exact option predicate, not a nearby action involving the same objects. Do not use answer-option "
        "plausibility as evidence. For contrastive measurements, require the value to be explicitly bound to the participant "
        "named by the question; a visible but subject-unbound counter is insufficient. For event counts, use "
        "completion_status.confirmed_event_candidates as the canonical ledger. Do not replace its count by recounting a sampled "
        "evidence digest; when candidate identity is disputed, return insufficient with a reconciliation task. "
        f"{task_instruction}\n"
        "Identify the single strongest_alternative after comparing every option internally. If that alternative is directly "
        "supported, provide revised_answer and its citations. Do not revise based only on plausibility or elimination. Keep the "
        "reason under 100 words and each task goal under 40 words. Every option key must appear exactly once in option_verdicts; "
        "use unknown for an undecidable option rather than omitting it. "
        f"{_sampling_plan_instruction()}"
        "Return compact JSON only: "
        "{\"verdict\":\"supported|insufficient|contradicted\",\"reason\":\"...\","
        "\"option_verdicts\":{\"A\":{\"status\":\"supported|contradicted|unknown\","
        "\"support_level\":\"direct|inferred|uncorroborated_summary|none\",\"evidence_ids\":[\"ev_...\"],"
        "\"canonical_fact_ids\":[\"event_candidate_001\"],\"reason\":\"...\"}},"
        "\"evidence_relation\":\"direct|causal_chain|consequence_only|cooccurrence_only|unclear\","
        "\"strongest_alternative\":{\"option\":\"A\",\"support\":\"direct|indirect|contradicted|missing\","
        "\"evidence_ids\":[\"ev_...\"],\"reason\":\"...\"},"
        "\"revised_answer\":null,\"revised_citations\":[],\"revised_entity_clusters\":[{\"entity_id\":\"entity_1\","
        "\"description\":\"...\",\"evidence_ids\":[\"ev_...\"],\"entity_observation_ids\":[\"obs:person_1\"]}],"
        "\"revised_support_status\":\"supported|insufficient\",\"tasks\":[{\"query_id\":\"audit_r2_t1\","
        "\"goal\":\"...\",\"segment_id\":\"seg_0001\",\"time_range\":[0.0,60.0],"
        "\"modality_hint\":[\"visual\",\"asr\"],\"expected_evidence\":\"...\","
        "\"sampling_floor_fps\":1.0,\"temporal_resolution_rationale\":\"Expected dwell time and interval.\"}]}.\n"
        f"Question: {kwargs['question']}\nOptions: {json.dumps(kwargs['options'], ensure_ascii=False)}\n"
        f"Query contract: {json.dumps(kwargs.get('query_contract') or {}, ensure_ascii=False)}\n"
        f"workspace_duration_sec: {float(kwargs.get('workspace_duration_sec', 0.0) or 0.0)}\n"
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
        "revision_context": dict(
            dict(kwargs.get("completion_status") or {}).get("revision_context") or {}
        ),
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
    option_support = _forced_option_support_dashboard(kwargs, evidence_digest)
    return (
        "The investigation budget is exhausted. You must choose one best option using the evidence dashboard; "
        "do not return investigate, abstain, or an empty answer. This is a best-effort answer and may remain unverified. "
        "Return JSON only: {\"answer\":\"A. option text\",\"citations\":[\"ev_...\"],"
        "\"entity_clusters\":[{\"entity_id\":\"entity_1\",\"description\":\"...\","
        "\"evidence_ids\":[\"ev_...\"],\"entity_observation_ids\":[\"observation:person_1\"]}]}. "
        "Canonical facts and option_support are status context only. Evidence summaries are supporting context only and must "
        "never be recounted or used to override a canonical count, identity, order, or transition. Return your own best-effort "
        "raw answer; final selection is performed separately by the VCAH adjudicator.\n"
        f"Question: {kwargs['question']}\nOptions: {json.dumps(kwargs['options'], ensure_ascii=False)}\n"
        f"Completion status: {json.dumps(kwargs.get('completion_status') or {}, ensure_ascii=False)}\n"
        f"Option support: {json.dumps(option_support, ensure_ascii=False)}\n"
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
    option_support = _forced_option_support_dashboard(kwargs, evidence_digest)
    return (
        "Choose the single best multiple-choice answer from the compact verified evidence. Do not request more investigation. "
        "For distinct counts, merge repeated people by visual signature and count only countable entities. Return compact JSON "
        "only: {\"answer\":\"A. option text\",\"citations\":[\"ev_...\"],\"entity_clusters\":[{\"entity_id\":"
        "\"entity_1\",\"description\":\"...\",\"evidence_ids\":[\"ev_...\"],\"entity_observation_ids\":[\"obs:person_1\"]}]}.\n"
        f"Question: {kwargs['question']}\nOptions: {json.dumps(kwargs['options'], ensure_ascii=False)}\n"
        f"Option support: {json.dumps(option_support, ensure_ascii=False)}\n"
        f"Evidence: {json.dumps(rows, ensure_ascii=False)}"
    )


def _forced_option_support_dashboard(
    kwargs: Mapping[str, Any],
    evidence_digest: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    del evidence_digest
    options = {str(key): str(value) for key, value in dict(kwargs.get("options") or {}).items()}
    completion = dict(kwargs.get("completion_status") or {})
    table = dict(completion.get("option_verdict_table") or {})
    verdicts = dict(table.get("option_verdicts") or {})
    scores: dict[str, int] = {}
    reasons: dict[str, list[str]] = {}
    status_score = {"supported": 100, "unknown": 0, "contradicted": -100}
    for key in options:
        row = dict(verdicts.get(key, {}) or {})
        status = str(row.get("status", "unknown") or "unknown").casefold()
        scores[key] = status_score.get(status, 0)
        reasons[key] = [str(row.get("reason", "") or "canonical_verdict_unavailable")]
    ranked = sorted(options, key=lambda key: (-scores[key], key))
    supported = [key for key in ranked if str(dict(verdicts.get(key, {}) or {}).get("status", "")) == "supported"]
    recommended = str(table.get("best_option", "") or "")
    return {
        "policy": "canonical_option_verdict_table",
        "selection_authority": "none",
        "recommended_option": recommended if recommended in options else ranked[0] if ranked else "",
        "positive_signal": bool(supported),
        "fact_source": "completion_status.canonical_fact_snapshot",
        "audit_status": str(table.get("audit_status", "missing") or "missing"),
        "options": [
            {"option": key, "score": scores[key], "reasons": reasons[key]}
            for key in ranked
        ],
    }


def _forced_option_count(text: str) -> int | None:
    match = re.search(r"\b(\d+)\b", str(text or ""))
    if match:
        return int(match.group(1))
    words = {
        "zero": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    normalized = str(text or "").casefold()
    return next((value for word, value in words.items() if re.search(rf"\b{word}\b", normalized)), None)


def _sampling_goal_key(task: Any) -> str:
    text = " ".join(
        (
            str(getattr(task, "goal", "") or ""),
            str(getattr(task, "expected_evidence", "") or ""),
        )
    ).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()[:240]


_SEMANTIC_EMPTY_SLOT_VALUES = {
    "unknown", "uncertain", "not visible", "observable unknown", "unknown observable",
    "template example", "placeholder", "n/a", "na", "tbd",
}


def _normalize_structured_slots(value: Any) -> dict[str, dict[str, Any]]:
    rows = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()
    result: dict[str, dict[str, Any]] = {}
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        slot_id = re.sub(r"[^a-z0-9_]+", "_", str(item.get("slot_id", "") or "").casefold()).strip("_")
        slot_value = re.sub(r"\s+", " ", str(item.get("value", "") or "").strip().casefold())
        if not slot_id or not slot_value or slot_value in _SEMANTIC_EMPTY_SLOT_VALUES:
            continue
        attribute = re.sub(
            r"[^a-z0-9_]+", "_", str(item.get("attribute", "") or slot_id).casefold()
        ).strip("_")
        scope = str(item.get("slot_type", item.get("scope", "")) or "").strip().casefold()
        if scope not in {
            "entity_identity", "persistent_attribute", "observation_attribute", "event_role", "state_at_time",
        }:
            scope = (
                "observation_attribute"
                if any(token in attribute for token in ("pose", "position", "location", "near", "beside"))
                else "event_role" if attribute.endswith("role")
                else "persistent_attribute"
            )
        raw_time = tuple(item.get("time_scope", ()) or ())
        time_scope = (
            [float(raw_time[0]), float(raw_time[1])]
            if len(raw_time) == 2 and all(isinstance(part, (int, float)) for part in raw_time)
            else []
        )
        result[slot_id] = {
            "slot_type": scope,
            "entity_id": str(item.get("entity_id", "") or "").strip().casefold(),
            "attribute": attribute,
            "value": slot_value,
            "time_scope": time_scope,
            "observation_id": str(item.get("observation_id", "") or ""),
            "confidence": _confidence(item.get("confidence"), default=0.7),
        }
    return result


def _normalize_finding(value: Mapping[str, Any]) -> str:
    finding = str(value.get("finding", "") or "").strip().casefold()
    if finding in {"found", "not_found", "uncertain"}:
        return finding
    presence = dict(value.get("target_presence", {}) or {})
    status = str(presence.get("status", "") or "").casefold()
    if status == "present":
        return "found"
    if status == "absent":
        return "not_found"
    if value.get("entities") or value.get("events") or _truthy(value.get("supports_answer_event")):
        return "found"
    return "uncertain"


def _observation_governance_row(
    value: Mapping[str, Any],
    sampling_fps: float,
    phase_offset_sec: float,
) -> dict[str, Any]:
    return {
        "summary": str(value.get("summary", "") or "")[:500],
        "finding": _normalize_finding(value),
        "confidence": _confidence(value.get("confidence"), default=0.6),
        "confidence_explicit": "confidence" in value,
        "structured_slots": _normalize_structured_slots(value.get("structured_slots")),
        "slot_arbitration": [
            dict(item)
            for item in tuple(value.get("slot_arbitration", ()) or ())
            if isinstance(item, Mapping)
        ],
        "need_detail": _truthy(value.get("need_detail")),
        "sampling_fps": float(sampling_fps),
        "phase_offset_sec": float(phase_offset_sec),
    }


def _conflicted_slot_ids(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    observations: dict[str, list[dict[str, Any]]] = {}
    resolved: set[str] = set()
    for row in rows:
        for slot_id, raw_slot in dict(row.get("structured_slots", {}) or {}).items():
            slot = dict(raw_slot) if isinstance(raw_slot, Mapping) else {
                "slot_type": "persistent_attribute", "entity_id": "", "attribute": str(slot_id),
                "value": str(raw_slot), "time_scope": [], "confidence": 0.7,
            }
            observations.setdefault(str(slot_id), []).append(slot)
        for verdict in tuple(row.get("slot_arbitration", ()) or ()):
            if not isinstance(verdict, Mapping):
                continue
            if str(verdict.get("verdict", "") or "") in {"confirm_latest", "confirm_prior"}:
                resolved.add(str(verdict.get("slot_id", "") or ""))
    conflicted = []
    for slot_id, slots in observations.items():
        if slot_id in resolved:
            continue
        for index, left in enumerate(slots):
            for right in slots[index + 1:]:
                if _typed_slots_conflict(left, right):
                    conflicted.append(slot_id)
                    break
            if slot_id in conflicted:
                break
    return tuple(sorted(set(conflicted)))


def _typed_slots_conflict(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if min(float(left.get("confidence", 0.0) or 0.0), float(right.get("confidence", 0.0) or 0.0)) < 0.6:
        return False
    if str(left.get("entity_id", "") or "") != str(right.get("entity_id", "") or ""):
        return False
    if str(left.get("attribute", "") or "") != str(right.get("attribute", "") or ""):
        return False
    if str(left.get("value", "") or "") == str(right.get("value", "") or ""):
        return False
    scopes = {str(left.get("slot_type", "") or ""), str(right.get("slot_type", "") or "")}
    if scopes.intersection({"entity_identity", "persistent_attribute"}):
        return True
    left_time = tuple(left.get("time_scope", ()) or ())
    right_time = tuple(right.get("time_scope", ()) or ())
    return bool(
        len(left_time) == 2 and len(right_time) == 2
        and min(float(left_time[1]), float(right_time[1])) >= max(float(left_time[0]), float(right_time[0]))
    )


def _adaptive_sampling_triggers(
    current: Mapping[str, Any],
    prior: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    reasons = []
    if bool(current.get("need_detail")):
        return ()
    if str(current.get("finding", "") or "") == "not_found":
        reasons.append("not_found")
    if bool(current.get("confidence_explicit")) and float(current.get("confidence", 0.0) or 0.0) < 0.7:
        reasons.append("low_confidence")
    if _conflicted_slot_ids((*prior, current)):
        reasons.append("structured_slot_conflict")
    return tuple(reasons)


def _resolution_prompt(task: Any, *, question: str = "") -> str:
    conditions = tuple(getattr(task, "conditions", ()) or ())
    return (
        "Evaluate only what is directly observable, not whether frames were returned. In the same JSON include "
        "\"finding\":\"found|not_found|uncertain\", where finding refers only to the task's expected evidence, "
        "\"visibility_status\":\"clear|occluded|unknown\", "
        "\"structured_slots\":[{\"slot_id\":\"stable task-local name\","
        "\"slot_type\":\"entity_identity|persistent_attribute|observation_attribute|event_role|state_at_time\","
        "\"entity_id\":\"canonical entity if known\",\"attribute\":\"typed attribute\","
        "\"value\":\"directly observed value\",\"time_scope\":[0.0,1.0],\"confidence\":0.8,"
        "\"frame_indices\":[0]}], and "
        "\"target_presence\":{\"target\":\"...\",\"status\":\"present|absent|uncertain\",\"confidence\":0.0}, "
        "\"measurements\":[{\"value\":0.0,\"unit\":\"...\",\"relation\":\"exact|approx|greater_than|less_than\","
        "\"measurement_semantics\":\"delta|cumulative|unknown\",\"subject_id\":\"\",\"source_time_sec\":null,"
        "\"boundary_relation\":\"before|at|after|unknown\",\"quantity_type\":\"score|clock|amount|\","
        "\"event_id\":\"halftime|period_end|event identifier|\",\"binding_status\":\"explicit|contextual|ambiguous|unbound\","
        "\"raw_text\":\"\"}], "
        "\"relations\":[{\"relation_type\":\"identity|temporal|causal|transition|spatial|relative_bearing|relative_facing\","
        "\"subject_id\":\"\",\"object_id\":\"\",\"value\":\"left_front|right_front|front|behind|left|right|\","
        "\"reference_frame\":\"viewer|subject_egocentric|object_egocentric|scene\",\"same_frame\":true|false,"
        "\"witness_frame_indices\":[0],"
        "\"status\":\"supported|contradicted|unknown\",\"description\":\"\"}], and "
        "\"state_transitions\":[{\"object_hypothesis_id\":\"stable object ID\",\"attribute_type\":\"surface_color|shape|state\","
        "\"raw_value_before\":\"\",\"raw_value_after\":\"\",\"before_witness\":[0.0,1.0],\"after_witness\":[2.0,3.0],"
        "\"same_object_relation\":\"supported|contradicted|unknown\",\"coverage_occlusion_status\":\"clear|occluded|unknown\"}], and "
        "\"condition_results\":[{\"condition_id\":\"...\",\"status\":\"satisfied|unknown|contradicted\","
        "\"observation\":\"direct observation\"}]. Use only the stable condition_id values below. "
        "For a crop, mark target_presence present only if the requested target is actually inside that crop; otherwise use absent or uncertain. "
        "For a scoreboard boundary question, compare every answer-option score pair rather than anchoring on the first readable score. "
        "Emit both team scores as separate measurements from the same frame, use unit=point, "
        "quantity_type=score, the same event_id, boundary_relation=at, and binding_status=explicit only when the requested phase "
        "or boundary is visible in that frame. For relative spatial questions, bind subject and object, state the relation value and "
        "reference_frame explicitly, and set same_frame=true only when both are jointly visible. "
        "For every same_frame relation, list one or more 0-based witness_frame_indices where both bound entities are visible in that "
        "single supplied image; if no such image exists, set same_frame=false and status=unknown. "
        "For a state transition, keep raw before/after values, identify the same object explicitly, and provide one temporal witness "
        "range for each side. Do not claim a transition when the object identity is unbound or either side is occluded. "
        "For a completed-task progress checkpoint, emit the visible completed value as unit=task, measurement_semantics=cumulative, "
        "boundary_relation=at, and binding_status=explicit only when the checkpoint event and counter belong to the same scene. "
        f"{_measurement_subject_semantics(question)}"
        f"{_spatial_reference_semantics(question)}"
        "The driver derives overall resolution, so do not self-declare it. Return empty measurement/relation arrays when unsupported. "
        f"Stable conditions: {json.dumps(to_jsonable(conditions), ensure_ascii=False)}\n"
    )


def _spatial_reference_semantics(question: str) -> str:
    text = str(question or "").casefold()
    if not any(term in text for term in ("in relation to", "relative to")):
        return ""
    predicate = (
        "The question asks for the subject's forward-facing direction expressed in the object's frame; emit "
        "relation_type=relative_facing. Do not substitute the subject's relative position. "
        if "facing" in text
        else "The question asks for the subject's position in the object's frame; emit relation_type=relative_bearing. "
    )
    return (
        "For wording 'subject ... in relation to object', subject_id is the first named entity and object_id is the reference entity. "
        f"{predicate}"
        "Interpret left/right/front/behind in the object's intrinsic forward-facing frame and emit reference_frame=object_egocentric. "
        "Viewer-relative or subject-egocentric facts are auxiliary only. If the object's forward "
        "direction is not visually established, emit status=unknown instead of guessing. "
    )


def _measurement_subject_semantics(question: str) -> str:
    text = str(question or "").casefold()
    if re.search(r"\bone\s+(?:team|group|person)\b[^?]*\bthe\s+other\s+(?:team|group|person)\b", text):
        return (
            "For a one-participant-versus-other-participant measurement, set subject_id=other_team (or other_subject) only when "
            "the counter is visibly or explicitly linked to the requested counterpart. A counter overlaid while the boundary team is "
            "on screen remains unbound unless the video establishes whose progress it represents; do not assign it by proximity. "
        )
    if re.search(r"\bwho\s+(?:consum\w*|ate|eaten|eating|bought|purchased)\b", text):
        return (
            "The measured subject is defined by an earlier identity anchor in the question. Set subject_id=anchored_subject only "
            "when the observation links the measured person to that earlier anchor; otherwise leave subject_id empty and "
            "binding_status=unbound. The cumulative value must also be at or before the named boundary event. "
        )
    return ""


def _summary_observation_semantics(question: str) -> str:
    text = str(question or "").casefold()
    if not (
        re.search(r"\b(?:title|heading)\b.*\b(?:summari[sz]\w*|best)\b", text)
        or re.search(r"\b(?:main|overall|central)\s+(?:topic|theme|idea|message)\b", text)
        or "mainly about" in text
    ):
        return ""
    return (
        "For a title or synopsis question, preserve the local narration even when it is in another language. Format summary as "
        "'Narrative thesis: ...; Segment role: setup|example|process detail|comparison|conclusion; Visual example: ...'. "
        "Translate the ASR meaning semantically when needed. Do not let a long factory, cooking, sports, or demonstration sequence "
        "replace the broader narrated topic merely because it occupies more frames. "
    )


def _prior_observation_arbitration(rows: Sequence[Any]) -> str:
    prior = tuple(dict(item) if isinstance(item, Mapping) else {"summary": str(item)} for item in rows)[-3:]
    if not prior:
        return ""
    return (
        "This is a denser, phase-shifted arbitration pass over a previously inspected window. "
        "Do not repeat the most recent description by default. Compare the new frames against every prior observation, "
        "state which claims remain visually supported, and mark unresolved when they conflict. If structured_slots disagree, "
        "include slot_arbitration [{\"slot_id\":\"...\",\"verdict\":\"confirm_latest|confirm_prior|cannot_determine\","
        "\"frame_indices\":[0]}]. "
        f"Prior observations: {json.dumps(prior, ensure_ascii=False)}\n"
    )


def _preview_prompt(
    workspace: VirtualVideoWorkspace,
    task: Any,
    segment_packet: Mapping[str, Any],
    window: Mapping[str, Any],
    *,
    prior_observations: Sequence[str] = (),
) -> str:
    return (
        "You are the Investigator. Inspect the low-fps preview frames and local ASR without choosing an answer option. "
        "Return JSON only: {\"summary\":\"atomic observation\",\"confidence\":0.0-1.0,"
        "\"entities\":[{\"local_id\":\"person_1\",\"entity_hypothesis_id\":\"stable task-level entity ID or empty\","
        "\"association_confidence\":0.0-1.0,\"description\":\"atomic visible observation\","
        "\"visual_signature\":\"stable face, hair, clothing, and accessories\",\"frame_indices\":[0],"
        "\"attributes\":{\"jacket_color\":\"green\",\"helmet_color\":\"black\"},"
        "\"role\":\"visible role or unknown\",\"question_relation\":\"directly observed relation or unknown\","
        "\"supports_question_relation\":true|false}],"
        "\"events\":[{\"local_id\":\"event_1\",\"event_key\":\"stable occurrence identity\","
        "\"subject_id\":\"tracked subject\",\"object_id\":\"other participant or object\","
        "\"state_before\":\"observable state before\",\"transition\":\"observable change/action\","
        "\"state_after\":\"observable state after\",\"preconditions_met\":true|false,"
        "\"qualification_status\":\"qualified|observed_candidate|unqualified_precondition|conflicted\","
        "\"qualification\":{\"required_prior_state\":\"supported|refuted|unknown\","
        "\"transition\":\"supported|refuted|unknown\",\"same_subject\":\"supported|refuted|unknown\","
        "\"episode_boundary\":\"supported|refuted|unknown\"},"
        "\"description\":\"one atomic occurrence relevant to the question\","
        "\"start_sec\":float,\"end_sec\":float,\"supports_question_event\":true|false}],"
        "\"supports_identity_anchor\":true|false,\"supports_answer_event\":true|false,"
        "\"need_detail\":true|false,\"detail_start_sec\":float|null,\"detail_end_sec\":float|null,"
        "\"requested_fps\":0.5|1.0|2.0,\"requested_max_frames\":1-512,\"reason\":\"...\","
        "\"region_hint\":\"scoreboard/text/object or empty\",\"region_box\":[x1,y1,x2,y2]|null}. "
        "Region coordinates are normalized 0-1.\n"
        "List each visible person separately using stable appearance attributes. Every entity must cite one or more 0-based "
        "frame_indices from the supplied images; omit people who are inferred from ASR or summary text but are not visible in "
        "those frames. The summary must not introduce a person absent from entities. Do not estimate a segment-level or "
        "video-level count. The same person may recur in later chunks. A window longer than 120 seconds is candidate discovery "
        "only: request a narrower detail window before treating any identity as countable.\n"
        "Enumerate every distinct question-relevant event occurrence visible in this inspected window. For a conditional event "
        "such as an action after a prior state, supports_question_event may be true only when the named subject, required prior "
        "state, transition, and after-state are all established; otherwise set preconditions_met=false. "
        "Use virtual timestamps from the window metadata, one row per occurrence, and return an empty events list when none is supported.\n"
        "Set supports_identity_anchor only when one visible entity jointly matches the identifying attributes in the question. "
        "Set supports_answer_event only when the observation directly supports the event, cause, action, or state being asked about.\n"
        "Request detail only when motion, OCR, identity, or a small visual attribute remains unresolved. "
        "Choose the smallest sufficient sampling request: 0.5 fps for scene/state inspection, 1 fps for ordinary actions, "
        "and 2 fps for fast motion, transitions, OCR, or fine temporal order. requested_max_frames is a hard cost budget up "
        "to 512. A detail request may cover the full preview window when complete segment coverage is required. Ensure "
        "duration * requested_fps does not exceed the requested frame budget.\n"
        f"{_summary_observation_semantics(workspace.case.question)}"
        f"{_prior_observation_arbitration(prior_observations)}"
        f"{_resolution_prompt(task, question=workspace.case.question)}"
        f"Question: {workspace.case.question}\n"
        "Candidate option predicates are observation targets, not an invitation to choose an answer. Test and report the exact "
        f"visible predicates when relevant: {json.dumps(dict(workspace.case.options), ensure_ascii=False)}\n"
        f"Task: {getattr(task, 'goal', '')}\nExpected evidence: {getattr(task, 'expected_evidence', '')}\n"
        f"Segment: {json.dumps(_compact_segment_packet(segment_packet), ensure_ascii=False)[:3000]}\n"
        f"Preview window metadata: {json.dumps({k: window[k] for k in ['virtual_time_range','sampling','asr_cues','source_lineage']}, ensure_ascii=False)[:5000]}"
    )


def _claim_preview_prompt(
    workspace: VirtualVideoWorkspace,
    task: Any,
    segment_packet: Mapping[str, Any],
    window: Mapping[str, Any],
    *,
    prior_observations: Sequence[str] = (),
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
        "\"detail_start_sec\":float|null,\"detail_end_sec\":float|null,\"requested_fps\":0.5|1.0|2.0,"
        "\"requested_max_frames\":1-512}.\n"
        "Use supports only when the exact candidate relation is directly established. The fact that candidate-related objects, "
        "people, or words appear is not enough. Request a narrower detail window when motion or text remains unresolved. "
        "Use 0.5 fps for stable scenes, 1 fps for ordinary actions, and 2 fps for fast motion, OCR, or exact temporal order, "
        "with at most 512 frames.\n"
        f"{_prior_observation_arbitration(prior_observations)}"
        f"{_resolution_prompt(task, question=workspace.case.question)}"
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
    prior_observations: Sequence[str] = (),
) -> str:
    return (
        "You are the Investigator. Enumerate atomic question-relevant event occurrences in this low-fps window. "
        "Return concise JSON only: {\"summary\":\"brief window observation\",\"confidence\":0.0-1.0,"
        "\"events\":[{\"local_id\":\"event_1\",\"event_key\":\"stable occurrence identity\","
        "\"event_class\":\"audition|news_segment|other\",\"counting_unit\":\"question-defined unit\","
        "\"participant_ids\":[\"stable named group or participant\"],\"phase\":\"intro|main|judging|result|replay|unknown\","
        "\"participants\":[{\"participant_id\":\"visible participant\",\"entity_hypothesis_id\":\"stable task-level ID or empty\","
        "\"role\":\"overtaker|overtaken|actor|object\",\"visual_signature\":\"multi-attribute signature\","
        "\"attributes\":{\"jacket_color\":\"green\",\"helmet_color\":\"black\"},\"association_confidence\":0.0-1.0}],"
        "\"subject_id\":\"tracked subject\",\"object_id\":\"other participant or object\","
        "\"state_before\":\"observable state before\",\"transition\":\"observable change/action\","
        "\"state_after\":\"observable state after\",\"preconditions_met\":true|false,"
        "\"description\":\"one occurrence\",\"start_sec\":float,\"end_sec\":float,"
        "\"supports_question_event\":true|false,\"supports_anchor_event\":true|false,"
        "\"continues_from_previous\":true|false,"
        "\"continues_to_next\":true|false}],"
        "\"supports_answer_event\":true|false,\"need_detail\":true|false,"
        "\"detail_start_sec\":float|null,\"detail_end_sec\":float|null,\"requested_fps\":0.5|1.0|2.0,"
        "\"requested_max_frames\":1-512,\"reason\":\"...\"}.\n"
        "List every distinct supported occurrence, use virtual timestamps inside this window, and return an empty events list when none. "
        "For conditional counts, count only events whose named subject and required prior state are visibly established; set "
        "preconditions_met=false and supports_question_event=false for visually similar events that fail the condition. "
        "For recorder position-loss events, qualification_status=qualified requires all four qualification fields supported "
        "by the inspected frames or linked boundary evidence; never infer them from a descriptive summary alone. "
        "event_key must identify this occurrence by topic, title, or visible anchor; never use only the generic event class. "
        "The counted unit is defined by the question. For audition counts, a named group's introduction, performance, judging, "
        "and result are phases of one audition: use counting_unit=audition_group, preserve the same participant_ids and exact "
        "group-based event_key, and do not emit feedback, applause, "
        "or a buzzer as additional occurrences. If no stable title, name, or visual signature distinguishes a candidate, set "
        "supports_question_event=false instead of using event_key='one occurrence'. "
        "For news-segment appearance counts, use counting_unit=news_broadcast_appearance. A teaser, replay, phone feed, or headline "
        "is not a counted news segment unless it visibly functions as a distinct broadcast segment. "
        "Compare against the prior adjacent-window events below. Reuse an exact event_key and set continues_from_previous=true "
        "only when the same occurrence visibly continues across the boundary. "
        "For identity-linked questions, assign participant roles and reuse an entity_hypothesis_id only when multiple visible "
        "attributes support the association; otherwise leave it empty. "
        "Do not list people or infer a video-level count. The preview covers one complete segment exactly once. Choose a detail "
        "fps that covers the full segment when exhaustive enumeration is required, or request one narrower interval when only a "
        "specific fast boundary is unresolved. Use 0.5 fps for stable scenes, 1 fps for ordinary actions, and 2 fps for fast "
        "boundaries, with at most 512 frames.\n"
        f"{_prior_observation_arbitration(prior_observations)}"
        f"{_resolution_prompt(task, question=workspace.case.question)}"
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
        "\"entities\":[{\"local_id\":\"person_1\",\"entity_hypothesis_id\":\"stable task-level entity ID or empty\","
        "\"association_confidence\":0.0-1.0,\"description\":\"atomic visible observation\","
        "\"visual_signature\":\"stable face, hair, clothing, and accessories\",\"frame_indices\":[0],"
        "\"attributes\":{\"jacket_color\":\"green\",\"helmet_color\":\"black\"},"
        "\"role\":\"visible role or unknown\",\"question_relation\":\"directly observed relation or unknown\","
        "\"supports_question_relation\":true|false}],"
        "\"events\":[{\"local_id\":\"event_1\",\"event_key\":\"stable occurrence identity\","
        "\"subject_id\":\"tracked subject\",\"object_id\":\"other participant or object\","
        "\"state_before\":\"observable state before\",\"transition\":\"observable change/action\","
        "\"state_after\":\"observable state after\",\"preconditions_met\":true|false,"
        "\"qualification_status\":\"qualified|observed_candidate|unqualified_precondition|conflicted\","
        "\"qualification\":{\"required_prior_state\":\"supported|refuted|unknown\","
        "\"transition\":\"supported|refuted|unknown\",\"same_subject\":\"supported|refuted|unknown\","
        "\"episode_boundary\":\"supported|refuted|unknown\"},"
        "\"description\":\"one atomic occurrence relevant to the question\","
        "\"start_sec\":float,\"end_sec\":float,\"supports_question_event\":true|false}],"
        "\"supports_identity_anchor\":true|false,\"supports_answer_event\":true|false}.\n"
        "List visible people separately and give every entity one or more 0-based frame_indices from the supplied images. "
        "Reuse entity_hypothesis_id only when a prior event participant or observation has a matching multi-attribute visual "
        "signature; otherwise leave it empty. Clothing color alone is insufficient when multiple candidates share it. "
        "Omit any person not directly visible in a cited frame, and never introduce additional people only in summary text. "
        "Do not infer a count across frames or chunks.\n"
        "Enumerate every distinct question-relevant event occurrence visible in this inspected window. Conditional events count "
        "only when the named subject and required prior state are established; otherwise set preconditions_met=false and "
        "supports_question_event=false. "
        "For recorder position-loss events, emit qualified only when required_prior_state, transition, same_subject, and "
        "episode_boundary are each explicitly supported; missing fields make the row observed_candidate. "
        "Use virtual timestamps from the window metadata, one row per occurrence, and return an empty events list when none is supported.\n"
        "Set supports_identity_anchor only when one visible entity jointly matches the identifying attributes in the question. "
        "Set supports_answer_event only when the observation directly supports the event, cause, action, or state being asked about.\n"
        f"{_summary_observation_semantics(workspace.case.question)}"
        f"{_resolution_prompt(task, question=workspace.case.question)}"
        f"Question: {workspace.case.question}\n"
        "Candidate option predicates are observation targets, not an invitation to choose an answer. Test and report the exact "
        f"visible predicates when relevant: {json.dumps(dict(workspace.case.options), ensure_ascii=False)}\n"
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
    anchor_instruction = (
        "This is participant anchor discovery: mark an event supports_anchor_event=true when it is the overtaking, meeting, "
        "or other referenced anchor that identifies the later participant, even though the final question asks a later "
        "attribute rather than the event itself. Include every visible participant and explicit role. "
        if str(getattr(task, "inspection_intent", "") or "") == "event_participant_anchor_discovery"
        else ""
    )
    return (
        "You are the Investigator. Verify atomic question-relevant event occurrences in this detail window. "
        "Return concise JSON only: {\"summary\":\"brief verified observation\",\"confidence\":0.0-1.0,"
        "\"events\":[{\"local_id\":\"event_1\",\"event_key\":\"stable occurrence identity\","
        "\"event_class\":\"audition|news_segment|other\",\"counting_unit\":\"question-defined unit\","
        "\"participant_ids\":[\"stable named group or participant\"],\"phase\":\"intro|main|judging|result|replay|unknown\","
        "\"participants\":[{\"participant_id\":\"visible participant\",\"entity_hypothesis_id\":\"stable task-level ID or empty\","
        "\"role\":\"overtaker|overtaken|actor|object\",\"visual_signature\":\"multi-attribute signature\","
        "\"attributes\":{\"jacket_color\":\"green\",\"helmet_color\":\"black\"},\"association_confidence\":0.0-1.0}],"
        "\"subject_id\":\"tracked subject\",\"object_id\":\"other participant or object\","
        "\"state_before\":\"observable state before\",\"transition\":\"observable change/action\","
        "\"state_after\":\"observable state after\",\"preconditions_met\":true|false,"
        "\"qualification_status\":\"qualified|observed_candidate|unqualified_precondition|conflicted\","
        "\"qualification\":{\"required_prior_state\":\"supported|refuted|unknown\","
        "\"transition\":\"supported|refuted|unknown\",\"same_subject\":\"supported|refuted|unknown\","
        "\"episode_boundary\":\"supported|refuted|unknown\"},"
        "\"description\":\"one occurrence\",\"start_sec\":float,\"end_sec\":float,"
        "\"supports_question_event\":true|false,\"continues_from_previous\":true|false,"
        "\"continues_to_next\":true|false}],"
        "\"supports_answer_event\":true|false}.\n"
        "List every distinct supported occurrence with virtual timestamps. For conditional event counts, supports_question_event "
        "may be true only when the named subject, required state_before, transition, and state_after are all established. "
        "For recorder position-loss events, qualification_status=qualified additionally requires explicit support for "
        "required_prior_state, transition, same_subject, and episode_boundary; otherwise preserve it as a candidate. "
        "event_key must identify the occurrence by topic, "
        "title, or visible anchor, not only its generic class. Preserve the question's counted unit across phases. For audition "
        "counts, introduction, performance, judging, and result for one named group use counting_unit=audition_group, the same "
        "participant_ids, and one exact group-based event_key; applause, feedback, and buzzer moments are not extra "
        "auditions. Mark an unidentifiable candidate supports_question_event=false rather than keying it as 'one occurrence'. Reuse "
        "an exact event_key and set continues_from_previous=true only for the same continuing occurrence. "
        "For news counts use counting_unit=news_broadcast_appearance and exclude teasers, replays, phone feeds, and incidental headlines. "
        "For identity-linked questions, assign participant roles and reuse an entity_hypothesis_id only when multiple visible "
        "attributes support the association; otherwise leave it empty. "
        + anchor_instruction
        +
        "Do not list people or infer a video-level count.\n"
        f"{_resolution_prompt(task, question=workspace.case.question)}"
        f"Question: {workspace.case.question}\n"
        f"Task: {getattr(task, 'goal', '')}\nExpected evidence: {getattr(task, 'expected_evidence', '')}\n"
        f"Prior adjacent-window ending events: {json.dumps(list(prior_events), ensure_ascii=False)[:1800]}\n"
        f"Preview finding: {json.dumps(dict(preview or {}), ensure_ascii=False)[:1600]}\n"
        f"Segment: {json.dumps(_compact_segment_packet(segment_packet), ensure_ascii=False)[:3000]}\n"
        f"Detail window metadata: {json.dumps(_window_prompt_metadata(window), ensure_ascii=False)[:5000]}"
    )


def _anchor_event_evidence_prompt(
    workspace: VirtualVideoWorkspace,
    task: Any,
    segment_packet: Mapping[str, Any],
    window: Mapping[str, Any],
    *,
    preview: Mapping[str, Any] | None = None,
) -> str:
    return (
        "You are locating the event that identifies a participant for later re-identification. Return compact JSON only: "
        "{\"summary\":\"brief direct observation\",\"confidence\":0.0-1.0,\"events\":[{"
        "\"event_key\":\"stable occurrence key\",\"description\":\"one visible anchor event\","
        "\"start_sec\":float,\"end_sec\":float,\"supports_anchor_event\":true|false,"
        "\"participant_ids\":[\"recorder\",\"visible participant\"],\"participants\":[{"
        "\"participant_id\":\"visible participant\",\"role\":\"overtaker|overtaken|actor|object\","
        "\"visual_signature\":\"at least two visible appearance attributes\","
        "\"attributes\":{\"clothing_color\":\"green\",\"helmet_color\":\"black\"}}]}]}\n"
        "List only events directly visible in this window. Mark supports_anchor_event=true when the event is the overtaking, "
        "meeting, or other referenced occurrence used to identify the later participant. The final question may ask a later "
        "attribute; that does not make the anchor event irrelevant. Include explicit participant roles and an empty events list "
        "when no matching anchor is visible. Do not answer the multiple-choice question and do not emit other schema fields.\n"
        f"Question: {workspace.case.question}\nTask: {getattr(task, 'goal', '')}\n"
        f"Preview finding: {json.dumps(dict(preview or {}), ensure_ascii=False)[:1000]}\n"
        f"Segment: {json.dumps(_compact_segment_packet(segment_packet), ensure_ascii=False)[:2200]}\n"
        f"Detail window metadata: {json.dumps(_window_prompt_metadata(window), ensure_ascii=False)[:3800]}"
    )


def _entity_association_prompt(
    workspace: VirtualVideoWorkspace,
    task: Any,
    segment_packet: Mapping[str, Any],
    window: Mapping[str, Any],
    *,
    preview: Mapping[str, Any] | None = None,
) -> str:
    references = [dict(item) for item in tuple(getattr(task, "reference_entities", ()) or ())]
    return (
        "You are performing a targeted cross-window entity association. Compare visible people in the supplied frames "
        "with the reference event participant. Do not choose an answer option. Return compact JSON only: "
        "{\"summary\":\"direct comparison\",\"confidence\":0.0-1.0,"
        "\"entities\":[{\"local_id\":\"target_1\",\"entity_hypothesis_id\":\"reference hypothesis ID only when supported\","
        "\"association_confidence\":0.0-1.0,\"description\":\"visible target\","
        "\"visual_signature\":\"face, hair, clothing, helmet, accessories\",\"frame_indices\":[0],"
        "\"attributes\":{\"clothing_color\":\"green\",\"finish_position\":\"third\"},"
        "\"role\":\"later appearance\",\"question_relation\":\"same event participant\","
        "\"supports_question_relation\":true}],"
        "\"entity_associations\":[{\"association_id\":\"assoc_1\","
        "\"source_participant_id\":\"exact reference participant_id\",\"source_event_key\":\"exact reference event key\","
        "\"target_local_id\":\"target_1\",\"entity_hypothesis_id\":\"exact reference hypothesis ID\","
        "\"status\":\"supported|refuted|unknown\",\"confidence\":0.0-1.0,"
        "\"shared_attributes\":{\"helmet_color\":\"black\"},"
        "\"distinguishing_attributes\":{\"clothing_color\":\"green\"},\"reason\":\"brief comparison\"}]}\n"
        "Use supported only when at least two compatible appearance attributes or a distinctive identity cue match and no "
        "visible cue conflicts. Clothing color alone is insufficient. Use refuted for a clear conflict and unknown when coverage "
        "or visibility cannot decide. Every target must cite 0-based frame_indices; do not create an association without a visible target.\n"
        f"Reference entities: {json.dumps(references, ensure_ascii=False)[:2400]}\n"
        f"Question: {workspace.case.question}\nTask: {getattr(task, 'goal', '')}\n"
        f"Preview: {json.dumps(dict(preview or {}), ensure_ascii=False)[:1200]}\n"
        f"Segment: {json.dumps(_compact_segment_packet(segment_packet), ensure_ascii=False)[:2400]}\n"
        f"Detail window metadata: {json.dumps(_window_prompt_metadata(window), ensure_ascii=False)[:4200]}"
    )


def _narrative_bridge_prompt(
    workspace: VirtualVideoWorkspace,
    task: Any,
    segment_packet: Mapping[str, Any],
    window: Mapping[str, Any],
    *,
    preview: Mapping[str, Any] | None = None,
) -> str:
    prior_facts = [dict(item) for item in tuple(getattr(task, "reference_facts", ()) or ())]
    return (
        "You are extracting a canonical narrative bridge from visible scenes and local ASR. Do not select a final answer. "
        "Return compact JSON only: {\"summary\":\"setup and outcome\",\"confidence\":0.0-1.0,"
        "\"narrative_facts\":[{\"fact_id\":\"narrative_1\",\"episode_id\":\"stable anchor episode\","
        "\"timeline_phase\":\"setup|transition|outcome|final\",\"anchor_match\":true|false,"
        "\"subject_id\":\"stable character or group\",\"relation_type\":\"observed_action|spoken_intention|"
        "observed_outcome|temporal_cooccurrence|agent_causation|inferred_intention|final_decision\","
        "\"predicate\":\"normalized relation predicate\","
        "\"setup_state\":\"directly observed initial intention/situation\","
        "\"observed_bridge\":\"direct transition, dialogue, or explicitly unshown gap\","
        "\"outcome_state\":\"directly observed later state\",\"inference\":\"minimal implication linking setup to outcome\","
        "\"inference_basis\":\"direct_transition|setup_outcome_inference\",\"confidence\":0.0-1.0,"
        "\"hypothesis_assessments\":[{\"option_id\":\"A\",\"status\":\"supported|contradicted|unknown\","
        "\"reason\":\"which observed predicate agrees or conflicts\"}],"
        "\"alternative_counterevidence\":[{\"option_id\":\"B\",\"observation\":\"specific conflicting fact\"}]}]}\n"
        "A supported assessment requires both setup_state and outcome_state. Treat thoughts or motives as inferred unless spoken. "
        "Bind every fact to the question's anchor episode. Do not mark out-of-episode actions as anchor_match. "
        "Temporal co-occurrence such as a siren near a character never establishes that character caused it. "
        "Use unknown when the inspected window contains only one side of the gap. Assess every option predicate independently; "
        "do not force exactly one supported option and do not use general story plausibility.\n"
        "Prior incomplete facts are navigation hypotheses only. Re-observe their missing side before completing them: "
        f"{json.dumps(prior_facts, ensure_ascii=False)[:2400]}\n"
        f"Question: {workspace.case.question}\n"
        f"Option predicates: {json.dumps(dict(workspace.case.options), ensure_ascii=False)}\n"
        f"Task: {getattr(task, 'goal', '')}\nExpected evidence: {getattr(task, 'expected_evidence', '')}\n"
        f"Preview: {json.dumps(dict(preview or {}), ensure_ascii=False)[:1200]}\n"
        f"Segment: {json.dumps(_compact_segment_packet(segment_packet), ensure_ascii=False)[:2400]}\n"
        f"Detail window metadata: {json.dumps(_window_prompt_metadata(window), ensure_ascii=False)[:4200]}"
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
        f"{_resolution_prompt(task, question=workspace.case.question)}"
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
        return {**parsed, "_parseable": True}
    verdict_match = re.search(
        r'\"verdict\"\s*:\s*\"(supported|insufficient|contradicted)\"',
        str(text or ""),
        flags=re.IGNORECASE,
    )
    verdict = verdict_match.group(1).casefold() if verdict_match else "unknown"
    return {
        "verdict": verdict,
        "_parseable": False,
        "reason": (
            "Answer audit output was truncated; its leading verdict is telemetry only and cannot preserve strict grounding."
            if verdict_match
            else "Answer audit returned no parseable verdict; strict grounding must be downgraded."
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
    parser.add_argument(
        "--out-root",
        default="/m2v_intern/xuboshen/zgw/VideoAgent/virtual_videomme_interactive/runs",
        help="Parent directory for create-exclusive replay runs.",
    )
    parser.add_argument("--run-id", help="Create exactly this immutable run ID; an existing ID is rejected.")
    parser.add_argument("--config", help="Legacy shared API config used for both roles.")
    parser.add_argument("--reasoner-config", help="Text-only planning/reasoning API config.")
    parser.add_argument("--investigator-config", help="Multimodal observation API config.")
    cases = parser.add_mutually_exclusive_group()
    cases.add_argument("--case-ids", nargs="*")
    cases.add_argument("--case-group", help="JSON manifest containing an ordered cases[] list and default construction.")
    parser.add_argument("--mode", choices=("smoke", "all", "long"), default="smoke")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--seeds", type=int, nargs="+", help="Run every selected case once per seed in one immutable suite.")
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
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Deprecated: immutable replay runs never reuse an existing workspace.",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
