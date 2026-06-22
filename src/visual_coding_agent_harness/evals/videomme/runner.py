from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from visual_coding_agent_harness.core.budget import AgentBudget
from visual_coding_agent_harness.workspace.context_budget import parse_budget_ratios
from visual_coding_agent_harness.agents.workspace_agent import WorkspaceVisualAgent
from visual_coding_agent_harness.evals.videomme.scene_index_builder import RootIndexPolicy, SceneIndexBuilder, SubtitleCue
from visual_coding_agent_harness.evals.videomme.scene_index_cache import SceneIndexCache
from visual_coding_agent_harness.tools.frame_cache import FrameSampler, build_frame_cache_for_video
from visual_coding_agent_harness.tools.workspace_v2 import build_workspace_v2_registry
from visual_coding_agent_harness.video.index import SceneIndex
from visual_coding_agent_harness.video.map import IndexRefiner, VideoMap, VideoMapStore
from visual_coding_agent_harness.workspace import EvidenceWorkspace

from .summary_schema import RunSummary, validate as validate_run_summary
from .training_trajectory import TrainingTrajectory
from .trajectory_markdown import write_trajectory_markdown
from .workspace_round_log import export_workspace_round_log


REMOTE_PYTHON = "/home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python"
KML_MANAGED_ROOT = Path("/m2v_intern/xuboshen/zgw/visual-coding-agent-harness")
MODEL_PATH = "/home/xuboshen/models/Qwen3-VL-4B-Instruct"
PLANNER_MODEL_PATH = ""
DATA_ROOT = Path(
    "/ytech_m2v5_hdd/workspace/kling_mm/Datasets/VLMEvalKit_Dataset_Cache/HFCache/"
    "datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b"
)
DEFAULT_PARQUET_PATH = DATA_ROOT / "videomme/test-00000-of-00001.parquet"
DEFAULT_VIDEO_DIR = DATA_ROOT / "video"
DEFAULT_SUBTITLE_DIR = DATA_ROOT / "subtitle"
DEFAULT_RUN_ROOT = KML_MANAGED_ROOT / "runs" / "videomme_agent_eval"
DEFAULT_SCENE_INDEX_CACHE_DIR = KML_MANAGED_ROOT / "scene_index_cache"
DEFAULT_CASES = ("605-1", "611-2", "612-1")
WORKSPACE_V2_STRATEGY = "workspace_v2"
DEFAULT_STRATEGIES = (WORKSPACE_V2_STRATEGY,)
STRATEGIES = (WORKSPACE_V2_STRATEGY,)
WINDOW_SEC = 300.0
SEGMENT_NFRAMES = 8
FRAME_CACHE_FPS = 2.0
TOOL_FRAME_CACHE_MAX_FPS = 2.0
ROOT_DVC_FRAME_FPS = 0.5


@dataclass(frozen=True)
class EvalConfig:
    run_root: Path
    workspace_root: Path
    model_path: str
    data_root: Path
    parquet_path: Path
    video_dir: Path
    subtitle_dir: Path
    cases: Sequence[str]
    strategies: Sequence[str]
    planner_model_path: str = PLANNER_MODEL_PATH
    planner_api_type: str = "openai_compatible"
    planner_api_base: str = ""
    planner_api_model: str = ""
    planner_api_key: str = "EMPTY"
    planner_api_user_key: str = ""
    planner_api_biz_scene: str = ""
    planner_api_version: str = ""
    planner_api_base_env: str = ""
    planner_api_model_env: str = ""
    planner_api_key_env: str = ""
    planner_api_version_env: str = ""
    planner_api_user_key_env: str = ""
    planner_api_biz_scene_env: str = ""
    planner_api_use_for_tools: bool = False
    planner_api_proxy_env: Mapping[str, str] | None = None
    planner_api_timeout: float = 180.0
    planner_thinking_token_budget: int | None = None
    planner_enable_thinking: bool | None = None
    window_sec: float = WINDOW_SEC
    scene_index_cache_dir: Path = DEFAULT_SCENE_INDEX_CACHE_DIR
    scene_index_cache_enabled: bool = True
    scene_caption_nframes: int = SEGMENT_NFRAMES
    scene_index_concurrency: int = 1
    scene_index_frame_fps: float = ROOT_DVC_FRAME_FPS
    scene_index_max_beats_per_root: int = RootIndexPolicy().max_beats_per_root
    scene_index_max_new_tokens: int = RootIndexPolicy().max_new_tokens
    frame_cache_fps: float = FRAME_CACHE_FPS
    frame_cache_root: Path | None = None
    budget: AgentBudget = AgentBudget()
    export_training: bool = False
    ablation_flags: Mapping[str, Any] | None = None
    source_config_path: Path | None = None


def validate_python(*, expected: str = REMOTE_PYTHON, allow_any_python: bool = False) -> None:
    print(f"PYTHON_EXECUTABLE {sys.executable}", flush=True)
    if not allow_any_python and expected and sys.executable != expected:
        raise SystemExit(f"Expected {expected}, got {sys.executable}. Pass --allow-any-python for local dry runs.")


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        timeout=20,
    ).strip()
    return float(out)


def normalize_options(options: Any) -> list[str]:
    if hasattr(options, "tolist"):
        options = options.tolist()
    return [str(item) for item in options]


def row_get(row: Any, key: str, default: Any = "") -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    try:
        return row[key]
    except Exception:
        return default


def make_question(row: Any) -> str:
    options = normalize_options(row_get(row, "options", []))
    return (
        f"Question: {row_get(row, 'question')}\n"
        "Options:\n"
        + "\n".join(options)
    )


def extract_choice(text: str) -> str:
    if not text:
        return ""
    upper = text.upper()
    patterns = [
        r"\b(?:ANSWER|CHOICE|OPTION|FINAL)\s*(?:IS|:)?\s*([ABCD])\b",
        r"^\s*([ABCD])\b",
        r"\b([ABCD])\s*[\).:-]",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return ""


def compact_text(text: str, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def parse_time(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def clean_subtitle_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def parse_srt_cues(path: Path) -> list[SubtitleCue]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    cues: list[SubtitleCue] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        time_index = next((idx for idx, line in enumerate(lines) if "-->" in line), None)
        if time_index is None:
            continue
        time_parts = lines[time_index].split("-->")
        start = time_parts[0].strip()
        end = time_parts[1].strip() if len(time_parts) > 1 else start
        body = " ".join(lines[time_index + 1 :])
        cleaned = clean_subtitle_text(body)
        if cleaned:
            cue_id = lines[0] if time_index > 0 else f"cue_{len(cues) + 1}"
            cues.append(
                SubtitleCue(
                    start_sec=parse_time(start),
                    end_sec=max(parse_time(end), parse_time(start) + 0.001),
                    text=cleaned,
                    cue_id=str(cue_id),
                )
            )
    return cues


def run_loop(
    backend: Any,
    *,
    video_path: str,
    question: str,
    duration_sec: float,
    run_id: str,
    scene_index: SceneIndex,
    workspace_root: Path,
    budget: AgentBudget,
    extract_clips: bool = True,
    frame_sampler: FrameSampler | None = None,
    strategy: str = WORKSPACE_V2_STRATEGY,
) -> dict[str, Any]:
    start = time.perf_counter()
    workspace = EvidenceWorkspace.create(base_dir=workspace_root, run_id=run_id)
    workspace_log_dir = workspace_root.parent / "workspace_logs" / run_id
    video_map = VideoMap.from_scene_index(scene_index)
    video_map_store = VideoMapStore(video_map)
    if strategy == WORKSPACE_V2_STRATEGY:
        index_refiner = IndexRefiner(
            backend=backend,
            frame_sampler=frame_sampler,
            artifact_root=workspace.root / "artifacts" / "index_refinement",
        )
        agent = WorkspaceVisualAgent(
            backend=backend,
            registry=build_workspace_v2_registry(
                video_map=video_map_store,
                backend=backend,
                workspace=workspace,
                index_refiner=index_refiner,
                frame_sampler=frame_sampler,
            ),
            workspace=workspace,
            max_rounds=budget.max_rounds,
            video_path=video_path,
            video_map=video_map_store,
            log_root=workspace_log_dir,
        )
        result = agent.run(question)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    seconds = time.perf_counter() - start
    status = _result_status(result)
    answer = str(getattr(result, "answer", ""))
    citations = _result_citations(result)
    confidence = getattr(result, "confidence", "")
    reward_tags = _reward_tags_for_result(workspace=workspace, status=status, citations=citations)
    trajectory_payload = workspace.export_longvideoagent_trajectory(
        question=question,
        video_path=video_path,
        final={
            "answer": answer,
            "status": status,
            "citations": list(citations),
            "confidence": confidence,
        },
        verifier_result={"status": status},
        reward_tags=reward_tags,
    )
    evidence_chains_payload = workspace.export_evidence_chains()
    trajectory_path = workspace.root / "artifacts" / "trajectories" / "longvideoagent_trajectory.json"
    evidence_chains_path = workspace.root / "artifacts" / "evidence_chains" / "evidence_chains.json"
    workspace_round_log = export_workspace_round_log(
        workspace,
        question=question,
        video_path=video_path,
        final={
            "answer": answer,
            "status": status,
            "citations": list(citations),
            "confidence": confidence,
        },
        trajectory_path=trajectory_path,
        evidence_chains_path=evidence_chains_path,
        log_root=workspace_log_dir,
    )
    planner_io_dir = workspace_log_dir
    tools, segments = _result_tools_and_segments(result, workspace=workspace)
    backend_call_counters = _backend_call_counters(workspace=workspace, scene_index=scene_index)
    return {
        "answer": answer,
        "choice": extract_choice(answer),
        "status": status,
        "confidence": confidence,
        "citations": list(citations),
        "rounds": _result_round_count(result),
        "tools": tools,
        "segments": segments,
        "seconds": round(seconds, 3),
        "trajectory_path": str(trajectory_path),
        "trajectory_action_count": len(trajectory_payload.get("actions", [])),
        "evidence_chains_path": str(evidence_chains_path),
        "evidence_chain_count": int(evidence_chains_payload.get("chain_count", 0) or 0),
        "workspace_log_dir": str(workspace_log_dir),
        "workspace_round_log_path": workspace_round_log["path"],
        "workspace_round_log_round_count": workspace_round_log["round_count"],
        "planner_io_dir": str(planner_io_dir),
        "planner_prompt_count": len(list(planner_io_dir.glob("*_prompt.txt"))) if planner_io_dir.exists() else 0,
        "backend_call_counters": backend_call_counters,
        **backend_call_counters,
        "reward_tags": reward_tags,
    }


def _result_status(result: Any) -> str:
    status = getattr(result, "status", "")
    if status:
        return str(status)
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, Mapping) and metadata.get("status"):
        return str(metadata["status"])
    answer = str(getattr(result, "answer", ""))
    if answer and answer != "need_more_evidence":
        return "final"
    return "need_more_evidence"


def _result_citations(result: Any) -> tuple[str, ...]:
    citations = getattr(result, "citations", ())
    if isinstance(citations, str):
        return (citations,) if citations else ()
    if isinstance(citations, Sequence):
        return tuple(str(item) for item in citations if str(item))
    return ()


def _result_round_count(result: Any) -> int:
    rounds = getattr(result, "rounds", ())
    if isinstance(rounds, int):
        return rounds
    if isinstance(rounds, Sequence):
        return len(rounds)
    return 0


def _result_tools_and_segments(result: Any, *, workspace: EvidenceWorkspace | None = None) -> tuple[list[str], list[str]]:
    rounds = getattr(result, "rounds", ())
    tools = []
    segments = []
    if not isinstance(rounds, int) and isinstance(rounds, Sequence):
        for round_item in rounds:
            program = getattr(round_item, "program", ())
            for step in program:
                if not isinstance(step, Mapping):
                    continue
                tools.append(str(step.get("tool", "")))
                args = step.get("args", {}) if isinstance(step.get("args", {}), Mapping) else {}
                if args.get("segment_id"):
                    segments.append(str(args["segment_id"]))
    if (not tools and not segments) and workspace is not None:
        for event in _load_trace_events(workspace):
            if _event_type(event) != "tool_use":
                continue
            payload = event.get("payload", {}) if isinstance(event.get("payload", {}), Mapping) else {}
            tool_name = str(payload.get("tool") or "").strip()
            if tool_name:
                tools.append(tool_name)
            arguments = payload.get("arguments", {}) if isinstance(payload.get("arguments", {}), Mapping) else {}
            segment_id = _segment_id_from_tool_arguments(arguments)
            if segment_id:
                segments.append(segment_id)
    return tools, segments


def _segment_id_from_tool_arguments(arguments: Mapping[str, Any]) -> str:
    if arguments.get("segment_id"):
        return str(arguments["segment_id"])
    scope = arguments.get("scope")
    if isinstance(scope, Mapping) and scope.get("segment_id"):
        return str(scope["segment_id"])
    return ""


def _backend_call_counters(*, workspace: EvidenceWorkspace, scene_index: SceneIndex) -> dict[str, int]:
    events = _load_trace_events(workspace)
    return {
        "root_index_backend_calls": sum(
            1 for segment in scene_index.segments if getattr(segment, "index_level", "root") == "root"
        ),
        "refinement_backend_calls": sum(1 for event in events if _event_type(event) == "index_refinement_created"),
        "verify_backend_calls": sum(1 for event in events if _event_type(event) == "segment_verify_dispatched"),
    }


def summarize_strategy(raw: Mapping[str, Any], gt: str) -> dict[str, Any]:
    summary = {
        "choice": raw.get("choice", ""),
        "correct": raw.get("choice", "") == gt,
        "seconds": raw.get("seconds"),
        "status": raw.get("status", "ok"),
        "rounds": raw.get("rounds"),
        "tools": raw.get("tools", []),
        "segments": raw.get("segments", []),
        "citation_count": len(raw.get("citations", [])),
        "answer_excerpt": compact_text(str(raw.get("answer", "")), limit=240),
    }
    for key in [
        "trajectory_path",
        "trajectory_action_count",
        "evidence_chains_path",
        "evidence_chain_count",
        "workspace_log_dir",
        "workspace_round_log_path",
        "workspace_round_log_round_count",
        "planner_io_dir",
        "planner_prompt_count",
        "backend_call_counters",
        "root_index_backend_calls",
        "refinement_backend_calls",
        "verify_backend_calls",
        "reward_tags",
    ]:
        if key in raw:
            summary[key] = raw[key]
    return summary


def _reward_tags_for_result(*, workspace: EvidenceWorkspace, status: str, citations: Sequence[str]) -> list[str]:
    tags = []
    if status == "final":
        tags.append("final")
    elif status:
        tags.append(str(status))
    if citations:
        tags.append("has_citations")
    else:
        tags.append("missing_citations")
    if workspace.has_non_navigation_visual_citation(citations):
        tags.append("non_navigation_visual_citation")
    else:
        tags.append("no_non_navigation_visual_citation")
    return tags


def load_rows_by_id(parquet_path: Path, cases: Sequence[str]) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    rows = {}
    for qid in cases:
        matches = df[df["question_id"].astype(str).eq(str(qid))]
        if matches.empty:
            raise ValueError(f"Missing VideoMME case {qid} in {parquet_path}")
        rows[str(qid)] = matches.iloc[0]
    return rows


def run_eval_cases(
    *,
    backend: Any,
    rows_by_id: Mapping[str, Any],
    config: EvalConfig,
    duration_fn: Callable[[Path], float] = ffprobe_duration,
) -> dict[str, Any]:
    config.run_root.mkdir(parents=True, exist_ok=True)
    config.workspace_root.mkdir(parents=True, exist_ok=True)
    summary_path = config.run_root / "summary.json"
    results = []
    config_payload = {
        "run_root": str(config.run_root),
        "workspace_root": str(config.workspace_root),
        "cases": list(config.cases),
        "strategies": list(config.strategies),
        "window_sec": config.window_sec,
        "budget": asdict(config.budget),
        "model_path": config.model_path,
        "planner_model_path": config.planner_model_path,
        "planner_api_type": config.planner_api_type,
        "planner_api_base": config.planner_api_base,
        "planner_api_model": config.planner_api_model,
        "planner_api_key_set": bool(config.planner_api_key and config.planner_api_key != "EMPTY"),
        "planner_api_user_key_set": bool(config.planner_api_user_key),
        "planner_api_biz_scene_set": bool(config.planner_api_biz_scene),
        "planner_api_version": config.planner_api_version,
        "planner_api_base_env": config.planner_api_base_env,
        "planner_api_model_env": config.planner_api_model_env,
        "planner_api_key_env": config.planner_api_key_env,
        "planner_api_version_env": config.planner_api_version_env,
        "planner_api_user_key_env": config.planner_api_user_key_env,
        "planner_api_biz_scene_env": config.planner_api_biz_scene_env,
        "planner_api_use_for_tools": config.planner_api_use_for_tools,
        "planner_api_proxy_env_keys": sorted((config.planner_api_proxy_env or {}).keys()),
        "planner_api_proxy_env_set": bool(config.planner_api_proxy_env),
        "planner_api_timeout": config.planner_api_timeout,
        "planner_thinking_token_budget": config.planner_thinking_token_budget,
        "planner_enable_thinking": config.planner_enable_thinking,
        "data_root": str(config.data_root),
        "parquet_path": str(config.parquet_path),
        "video_dir": str(config.video_dir),
        "subtitle_dir": str(config.subtitle_dir),
        "scene_index_cache_dir": str(config.scene_index_cache_dir),
        "scene_index_cache_enabled": config.scene_index_cache_enabled,
        "scene_caption_nframes": config.scene_caption_nframes,
        "scene_index_concurrency": config.scene_index_concurrency,
        "scene_index_frame_fps": config.scene_index_frame_fps,
        "scene_index_max_beats_per_root": config.scene_index_max_beats_per_root,
        "scene_index_max_new_tokens": config.scene_index_max_new_tokens,
        "frame_cache_fps": config.frame_cache_fps,
        "frame_cache_root": str(_frame_cache_root(config)),
        "export_training": config.export_training,
        "ablation_flags": dict(config.ablation_flags or {}),
        "source_config_path": str(config.source_config_path) if config.source_config_path else "",
    }
    (config.run_root / "run_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (config.run_root / "resolved_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = _summary_payload(
        run_id=config.run_root.name,
        case_ids=config.cases,
        config_payload=config_payload,
        results=results,
    )
    print(
        "START videomme_eval "
        + json.dumps({"cases": list(config.cases), "strategies": list(config.strategies)}, sort_keys=True),
        flush=True,
    )
    for qid in config.cases:
        row = rows_by_id[str(qid)]
        video_id = str(row_get(row, "videoID") or row_get(row, "video_id"))
        video_path = str(config.video_dir / f"{video_id}.mp4")
        duration_sec = duration_fn(Path(video_path))
        frame_cache = None
        frame_cache = build_frame_cache_for_video(
            video_path=Path(video_path),
            frame_dir=_frame_cache_dir(config=config, video_id=video_id),
            fps=float(config.frame_cache_fps),
            duration_sec=duration_sec,
        )
        frame_sampler = frame_cache.sample_paths if frame_cache is not None else None
        question = make_question(row)
        gt = str(row_get(row, "answer")).strip().upper()
        case_prefix = f"{qid}_{video_id}"
        case = {
            "question_id": str(qid),
            "video_id": str(row_get(row, "video_id", video_id)),
            "videoID": video_id,
            "task_type": str(row_get(row, "task_type")),
            "duration_sec": round(duration_sec, 1),
            "gt": gt,
            "question": question,
            "options": normalize_options(row_get(row, "options", [])),
            "question_excerpt": compact_text(str(row_get(row, "question")), limit=220),
            "strategies": {},
            "raw_artifacts": {"workspaces": {}},
        }
        if frame_cache is not None:
            case["raw_artifacts"]["frame_cache"] = str(frame_cache.frame_dir)
        print(
            "CASE_START "
            + json.dumps(
                {k: case[k] for k in ["question_id", "videoID", "task_type", "duration_sec", "gt"]},
                sort_keys=True,
            ),
            flush=True,
        )
        for strategy in config.strategies:
            try:
                raw = run_strategy(
                    strategy=strategy,
                    backend=backend,
                    video_path=video_path,
                    video_id=video_id,
                    question=question,
                    duration_sec=duration_sec,
                    run_id=f"{case_prefix}_{strategy}",
                    config=config,
                    frame_sampler=frame_sampler,
                )
                case["strategies"][strategy] = summarize_strategy(raw, gt)
                workspace_path = config.workspace_root / "runs" / f"{case_prefix}_{strategy}"
                case["raw_artifacts"]["workspaces"][strategy] = str(workspace_path)
                if config.export_training:
                    trajectory_path = _export_training_trajectory(
                        workspace_path=workspace_path,
                        run_root=config.run_root,
                        case_id=str(qid),
                        strategy=strategy,
                        question=question,
                        options=case["options"],
                        gt=gt,
                        strategy_summary=case["strategies"][strategy],
                    )
                    if trajectory_path is not None:
                        markdown_path = trajectory_path.with_suffix(".md")
                        case["raw_artifacts"].setdefault("training_trajectories", {})[strategy] = str(trajectory_path)
                        case["raw_artifacts"].setdefault("training_trajectory_markdown", {})[strategy] = str(markdown_path)
                        case["strategies"][strategy]["training_trajectory_path"] = str(trajectory_path)
                        case["strategies"][strategy]["training_trajectory_markdown_path"] = str(markdown_path)
            except Exception as exc:
                case["strategies"][strategy] = {
                    "choice": "",
                    "correct": False,
                    "seconds": None,
                    "status": "error",
                    "rounds": None,
                    "tools": [],
                    "segments": [],
                    "citation_count": 0,
                    "answer_excerpt": "",
                    "error": type(exc).__name__ + ": " + str(exc)[:500],
                }
        results.append(case)
        summary = _summary_payload(
            run_id=config.run_root.name,
            case_ids=config.cases,
            config_payload=config_payload,
            results=results,
        )
        evidence_chains_path = _write_run_evidence_chains(config.run_root, results)
        summary["evidence_chains_path"] = str(evidence_chains_path)
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        compact = {"question_id": qid, "gt": gt, "strategies": case.get("strategies", {})}
        print("CASE_DONE " + json.dumps(compact, ensure_ascii=True, sort_keys=True), flush=True)
    violations = validate_run_summary(RunSummary.from_dict(summary))
    if violations:
        (config.run_root / "summary_violations.json").write_text(
            json.dumps({"violations": violations}, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise SystemExit(2)
    print("DONE summary=" + str(summary_path), flush=True)
    return summary


def _summary_payload(
    *,
    run_id: str,
    case_ids: Sequence[str],
    config_payload: Mapping[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    run_summary = RunSummary.with_defaults(run_id, list(case_ids))
    run_summary.per_case = results
    _populate_run_summary_metrics(run_summary, results)
    run_summary.training_trajectory_exported = _training_trajectory_exported(results)
    workspaces = _workspaces_from_results(results)
    if workspaces:
        compliance, histogram = compute_nframes_metrics(workspaces)
        run_summary.tool_nframes_compliance = compliance
        run_summary.nframes_histogram = histogram
        run_summary.evidence_provenance_completeness = _evidence_provenance_completeness(workspaces)
        _populate_trace_summary_metrics(run_summary, workspaces)
    payload = run_summary.to_dict()
    payload["config"] = dict(config_payload)
    payload["cases"] = results
    return payload


def _export_training_trajectory(
    *,
    workspace_path: Path,
    run_root: Path,
    case_id: str,
    strategy: str,
    question: str,
    options: Sequence[str],
    gt: str,
    strategy_summary: Mapping[str, Any],
) -> Path | None:
    if not workspace_path.exists():
        return None
    selected = str(strategy_summary.get("choice") or "") or None
    trajectory_path = (run_root / "trajectories" / f"{case_id}_{strategy}.json").resolve()
    TrainingTrajectory.from_workspace(
        EvidenceWorkspace(root=workspace_path),
        case_id=case_id,
        question=question,
        options=options,
        ground_truth=gt,
        final_decision=str(strategy_summary.get("status", "")),
        selected_option=selected,
        is_correct=bool(strategy_summary.get("correct")) if selected else None,
        output_path=trajectory_path,
    )
    write_trajectory_markdown(trajectory_path)
    return trajectory_path


def _training_trajectory_exported(results: Sequence[Mapping[str, Any]]) -> bool:
    for case in results:
        raw_artifacts = case.get("raw_artifacts", {})
        if isinstance(raw_artifacts, Mapping) and raw_artifacts.get("training_trajectories"):
            return True
    return False


def _write_run_evidence_chains(run_root: Path, results: Sequence[Mapping[str, Any]]) -> Path:
    path = run_root / "evidence_chains.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in results:
        raw_artifacts = case.get("raw_artifacts", {})
        if not isinstance(raw_artifacts, Mapping):
            continue
        workspace_paths = raw_artifacts.get("workspaces", {})
        if not isinstance(workspace_paths, Mapping):
            continue
        strategies = _case_strategies(case)
        for strategy, workspace_path in workspace_paths.items():
            workspace_root = Path(str(workspace_path))
            if not workspace_root.exists():
                continue
            strategy_summary = strategies.get(str(strategy), {})
            if not isinstance(strategy_summary, Mapping):
                strategy_summary = {}
            chains = EvidenceWorkspace(root=workspace_root).evidence_chain_summaries(max_chains=100)
            rows.append(
                {
                    "case_id": str(case.get("question_id", "")),
                    "strategy": str(strategy),
                    "final_decision": str(strategy_summary.get("status", "")),
                    "selected_option": str(strategy_summary.get("choice", "")),
                    "workspace": workspace_root.as_posix(),
                    "chain_count": len(chains),
                    "chains": [
                        [str(record.get("evidence_id", "")) for record in chain.get("records", [])]
                        for chain in chains
                    ],
                }
            )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    return path


def _populate_run_summary_metrics(summary: RunSummary, results: Sequence[Mapping[str, Any]]) -> None:
    strategy_results = [
        strategy
        for case in results
        for strategy in _case_strategies(case).values()
    ]
    if not strategy_results:
        return
    total = len(strategy_results)
    summary.accuracy = sum(1 for item in strategy_results if bool(item.get("correct"))) / total
    summary.final_rate = sum(1 for item in strategy_results if item.get("status") == "final") / total
    summary.need_more_evidence_rate = (
        sum(1 for item in strategy_results if item.get("status") == "need_more_evidence") / total
    )
    summary.low_confidence_final_rate = (
        sum(1 for item in strategy_results if item.get("status") == "low_confidence_final") / total
    )
    summary.unsupported_final_rate = (
        sum(
            1
            for item in strategy_results
            if item.get("status") == "final" and int(item.get("citation_count", 0) or 0) == 0
        )
        / total
    )


def _populate_trace_summary_metrics(summary: RunSummary, workspaces: Sequence[EvidenceWorkspace]) -> None:
    route_violations = 0
    followup_attempts: list[int] = []
    followup_successes = 0
    context_overflows = 0
    context_turn_token_totals: list[int] = []
    unsupported_citation_finals = 0
    traced_finals = 0
    mutex_conflicts = 0
    timeline_scores: list[float] = []
    degenerate_observations = 0
    total_observations = 0
    normalization_notes = 0
    normalization_rounds = 0
    option_biased_first_queries = 0
    wrong_scope_caption_facts = 0
    caption_fact_downgrades = 0
    caption_fact_observations = 0
    caption_support_finals = 0
    visual_required_caption_finals = 0
    final_cases = 0

    for workspace in workspaces:
        events = _load_trace_events(workspace)
        route_violations += sum(1 for event in events if _event_type(event) == "route_violation")
        attempts, success = _hard_skill_followup_trace_metrics(events)
        followup_attempts.append(attempts)
        if success:
            followup_successes += 1
        overflows, token_totals = _context_budget_trace_metrics(events)
        context_overflows += overflows
        context_turn_token_totals.extend(token_totals)
        unsupported, finals = _unsupported_citation_trace_metrics(workspace, events)
        unsupported_citation_finals += unsupported
        traced_finals += finals
        mutex_conflicts += _mutex_conflict_detection_count(events)
        timeline_scores.extend(_timeline_completeness_scores(events))
        degenerate_observations += _degenerate_observation_count(events)
        total_observations += _observation_count(workspace)
        notes, rounds = _normalization_note_trace_metrics(events)
        normalization_notes += notes
        normalization_rounds += rounds
        caption_metrics = _caption_explore_trace_metrics(workspace)
        option_biased_first_queries += caption_metrics["option_biased_first_query"]
        wrong_scope_caption_facts += caption_metrics["wrong_scope_caption_facts"]
        caption_fact_downgrades += caption_metrics["caption_fact_downgrades"]
        caption_fact_observations += caption_metrics["caption_fact_observations"]
        caption_support_finals += caption_metrics["caption_support_final"]
        visual_required_caption_finals += caption_metrics["visual_required_but_caption_final"]
        final_cases += caption_metrics["final_case"]

    summary.route_violations = route_violations
    summary.context_budget_overflow_count = context_overflows
    if context_turn_token_totals:
        summary.avg_tokens_per_turn = int(sum(context_turn_token_totals) / len(context_turn_token_totals))
    if followup_attempts:
        summary.avg_followups_per_case = sum(followup_attempts) / len(followup_attempts)
        attempted_cases = sum(1 for attempts in followup_attempts if attempts > 0)
        summary.followup_success_rate = (followup_successes / attempted_cases) if attempted_cases else 0.0
    if traced_finals:
        summary.unsupported_citation_rate = unsupported_citation_finals / traced_finals
    summary.mutex_conflict_detection_count = mutex_conflicts
    if timeline_scores:
        summary.timeline_completeness = sum(timeline_scores) / len(timeline_scores)
    if total_observations:
        summary.degenerate_observation_rate = degenerate_observations / total_observations
    if normalization_rounds:
        summary.normalization_notes_per_round = normalization_notes / normalization_rounds
    if workspaces:
        summary.option_biased_first_query_rate = option_biased_first_queries / len(workspaces)
    if caption_fact_observations:
        summary.wrong_scope_caption_fact_rate = wrong_scope_caption_facts / caption_fact_observations
        summary.caption_fact_downgrade_rate = caption_fact_downgrades / caption_fact_observations
    if final_cases:
        summary.caption_support_final_rate = caption_support_finals / final_cases
        summary.visual_required_but_caption_final_rate = visual_required_caption_finals / final_cases


def _caption_explore_trace_metrics(workspace: EvidenceWorkspace) -> dict[str, int]:
    observations = workspace._read_jsonl_dicts("observations.jsonl")
    explore_observations = [
        row for row in observations if str(row.get("tool") or row.get("tool_name") or "") == "explore"
    ]
    first_explore = explore_observations[0] if explore_observations else {}
    first_raw = first_explore.get("raw_output") if isinstance(first_explore.get("raw_output"), Mapping) else {}
    option_biased_first = int(bool((first_raw.get("query_analysis") or {}).get("is_option_biased")) if isinstance(first_raw.get("query_analysis"), Mapping) else False)
    caption_facts = []
    wrong_scope = 0
    downgrades = 0
    for row in explore_observations:
        raw = row.get("raw_output") if isinstance(row.get("raw_output"), Mapping) else {}
        if str(raw.get("mode") or "") not in {"caption_fact", "mixed"} and not raw.get("caption_fact_downgraded"):
            continue
        caption_facts.append(raw)
        condition_match = raw.get("condition_match") if isinstance(raw.get("condition_match"), Mapping) else {}
        if str(condition_match.get("match_level") or "") == "related_but_wrong_scope":
            wrong_scope += 1
        if bool(raw.get("caption_fact_downgraded")):
            downgrades += 1
    final_mem_ids = set()
    for event in workspace._read_jsonl_dicts("trace.jsonl"):
        event_type = str(event.get("type") or event.get("event_type") or "")
        if event_type not in {"answer_accepted", "workspace_answer_accepted", "iterative_final", "low_confidence_final"}:
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        for citation in payload.get("citations") or payload.get("attempted_citations") or ():
            final_mem_ids.add(str(citation))
    caption_support_final = 0
    visual_required_caption_final = 0
    for memory in workspace.memory_entries():
        if memory.entry_id not in final_mem_ids or memory.kind != "caption_support":
            continue
        caption_support_final = 1
        if bool(memory.metadata.get("requires_visual_verify")) or bool(memory.metadata.get("cannot_final_cite")):
            visual_required_caption_final = 1
    return {
        "option_biased_first_query": option_biased_first,
        "wrong_scope_caption_facts": wrong_scope,
        "caption_fact_downgrades": downgrades,
        "caption_fact_observations": len(caption_facts),
        "caption_support_final": caption_support_final,
        "visual_required_but_caption_final": visual_required_caption_final,
        "final_case": 1,
    }


def _evidence_provenance_completeness(workspaces: Sequence[EvidenceWorkspace]) -> float:
    if not workspaces:
        return 0.0
    complete = sum(1 for workspace in workspaces if workspace.evidence_chain_summaries(max_chains=1))
    return complete / len(workspaces)


def _hard_skill_followup_trace_metrics(events: Sequence[Mapping[str, Any]]) -> tuple[int, bool]:
    explicit_attempts = sum(1 for event in events if _event_type(event) == "followup_attempt")
    if explicit_attempts:
        return explicit_attempts, any(
            _event_type(event) == "iterative_final"
            or _event_type(event) == "low_confidence_final"
            or (
                _event_type(event) == "iterative_answer_agent"
                and str(_event_payload(event).get("status", "")) in {"final", "low_confidence_final"}
            )
            for event in events
        )
    in_hard_skill = False
    attempts = 0
    success = False
    for event in events:
        event_type = _event_type(event)
        payload = _event_payload(event)
        if event_type == "hard_skill_runtime":
            in_hard_skill = True
            continue
        if event_type == "tool_use" and in_hard_skill and str(payload.get("tool", "")) in {
            "ground_question",
            "caption_segment",
            "vision_read",
        }:
            attempts += 1
            continue
        if event_type == "iterative_final" and str(payload.get("source", "")) in {
            "hard_skill_runtime",
            "timeline_ordering",
        }:
            success = True
            in_hard_skill = False
            continue
        if event_type == "hard_skill_followup_handoff":
            in_hard_skill = False
    return attempts, success


def _context_budget_trace_metrics(events: Sequence[Mapping[str, Any]]) -> tuple[int, list[int]]:
    overflows = 0
    token_totals: list[int] = []
    for event in events:
        if _event_type(event) != "context_budget_report":
            continue
        payload = _event_payload(event)
        if bool(payload.get("overflow")):
            overflows += 1
        used = payload.get("used_tokens_per_slot", {})
        if not isinstance(used, Mapping):
            continue
        token_totals.append(sum(int(value or 0) for value in used.values()))
    return overflows, token_totals


def _unsupported_citation_trace_metrics(
    workspace: EvidenceWorkspace,
    events: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    observations = {
        str(row.get("observation_id", "")): row
        for row in _load_observations(workspace)
        if row.get("observation_id")
    }
    unsupported_finals = 0
    finals = 0
    for event in events:
        if _event_type(event) != "iterative_final":
            continue
        finals += 1
        payload = _event_payload(event)
        citations = payload.get("citations", [])
        if not isinstance(citations, Sequence) or isinstance(citations, (str, bytes)):
            continue
        if any(_observation_confidence_signal(observations.get(str(citation), {})) == "unsupported" for citation in citations):
            unsupported_finals += 1
    return unsupported_finals, finals


def _mutex_conflict_detection_count(events: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for event in events:
        event_type = _event_type(event)
        payload = _event_payload(event)
        if event_type == "iterative_answer_agent" and str(payload.get("status", "")) != "need_more_evidence":
            continue
        if event_type not in {"iterative_answer_agent", "iterative_final_blocked", "answer_agent_need_more_evidence"}:
            continue
        if "mutex_conflict" in _payload_text(payload).lower():
            count += 1
    return count


def _timeline_completeness_scores(events: Sequence[Mapping[str, Any]]) -> list[float]:
    scores: list[float] = []
    for event in events:
        event_type = _event_type(event)
        payload = _event_payload(event)
        if event_type == "iterative_timeline_temporal_decision":
            explicit = _explicit_completeness_score(payload)
            if explicit is not None:
                scores.append(explicit)
                continue
            matched = payload.get("matched_events", [])
            if isinstance(matched, Sequence) and not isinstance(matched, (str, bytes)) and matched:
                scores.append(1.0)
            continue
        if event_type != "timeline_ordering_missing_entity":
            continue
        explicit = _explicit_completeness_score(payload)
        if explicit is not None:
            scores.append(explicit)
            continue
        targets = _string_list(payload.get("target_facts", []))
        missing = set(_string_list(payload.get("missing_entities", [])))
        if targets:
            satisfied = sum(1 for target in targets if target not in missing)
            scores.append(satisfied / len(targets))
    return scores


def _degenerate_observation_count(events: Sequence[Mapping[str, Any]]) -> int:
    observation_ids = set()
    anonymous_events = 0
    for event in events:
        if _event_type(event) != "tool_output_degenerate":
            continue
        payload = _event_payload(event)
        observation_id = str(payload.get("observation_id", "")).strip()
        if observation_id:
            observation_ids.add(observation_id)
        else:
            anonymous_events += 1
    return len(observation_ids) + anonymous_events


def _normalization_note_trace_metrics(events: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    notes = 0
    rounds = 0
    for event in events:
        if _event_type(event) != "iterative_normalization_empty":
            continue
        payload = _event_payload(event)
        note_payload = payload.get("notes", [])
        if not isinstance(note_payload, Sequence) or isinstance(note_payload, (str, bytes)):
            continue
        rounds += 1
        notes += len(note_payload)
    return notes, rounds


def _observation_count(workspace: EvidenceWorkspace) -> int:
    return len(_load_observations(workspace))


def _load_observations(workspace: EvidenceWorkspace) -> list[dict[str, Any]]:
    path = workspace.root / "observations.jsonl"
    if not path.exists():
        return []
    observations: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                observations.append(payload)
    return observations


def _observation_confidence_signal(observation: Mapping[str, Any]) -> str:
    signal = str(observation.get("confidence_signal", "")).strip().lower()
    if signal:
        return signal
    raw_output = observation.get("raw_output", {})
    if isinstance(raw_output, Mapping):
        return str(raw_output.get("confidence_signal", "")).strip().lower()
    return ""


def _explicit_completeness_score(payload: Mapping[str, Any]) -> float | None:
    if "required_slots" not in payload and "satisfied_slots" not in payload:
        return None
    required = _numeric_slot_count(payload.get("required_slots"))
    if required <= 0:
        return None
    satisfied = _numeric_slot_count(payload.get("satisfied_slots"))
    return max(0.0, min(1.0, satisfied / required))


def _numeric_slot_count(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 0


def _payload_text(payload: Any) -> str:
    if isinstance(payload, Mapping):
        return " ".join(_payload_text(value) for value in payload.values())
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return " ".join(_payload_text(value) for value in payload)
    return str(payload)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if str(item)]


def _load_trace_events(workspace: EvidenceWorkspace) -> list[dict[str, Any]]:
    trace_path = workspace.root / "trace.jsonl"
    if not trace_path.exists():
        return []
    events: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events


def _event_type(event: Mapping[str, Any]) -> str:
    return str(event.get("type") or event.get("event") or "")


def _event_payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = event.get("payload", {})
    return payload if isinstance(payload, Mapping) else {}


def _case_strategies(case: Mapping[str, Any]) -> Mapping[str, Any]:
    strategies = case.get("strategies", {})
    return strategies if isinstance(strategies, Mapping) else {}


def compute_nframes_metrics(
    workspace: EvidenceWorkspace | Sequence[EvidenceWorkspace],
) -> tuple[float, dict[str, dict[int, int]]]:
    workspaces = [workspace] if isinstance(workspace, EvidenceWorkspace) else list(workspace)
    manifests = [manifest for item in workspaces for manifest in item.load_all_manifests()]
    if not manifests:
        return 1.0, {}

    hits = sum(1 for manifest in manifests if manifest.nframes == manifest.target_nframes)
    histogram: dict[str, dict[int, int]] = {}
    for manifest in manifests:
        tool_histogram = histogram.setdefault(manifest.created_by_tool, {})
        tool_histogram[manifest.nframes] = tool_histogram.get(manifest.nframes, 0) + 1
    return hits / len(manifests), _sorted_histogram(histogram)


def _workspaces_from_results(results: Sequence[Mapping[str, Any]]) -> list[EvidenceWorkspace]:
    workspaces = []
    for case in results:
        raw_artifacts = case.get("raw_artifacts", {})
        if not isinstance(raw_artifacts, Mapping):
            continue
        workspace_paths = raw_artifacts.get("workspaces", {})
        if not isinstance(workspace_paths, Mapping):
            continue
        for path in workspace_paths.values():
            workspace_root = Path(str(path))
            if workspace_root.exists():
                workspaces.append(EvidenceWorkspace(root=workspace_root))
    return workspaces


def _sorted_histogram(histogram: Mapping[str, Mapping[int, int]]) -> dict[str, dict[int, int]]:
    return {
        tool: {frames: counts[frames] for frames in sorted(counts)}
        for tool, counts in sorted(histogram.items())
    }


def run_strategy(
    *,
    strategy: str,
    backend: Any,
    video_path: str,
    video_id: str,
    question: str,
    duration_sec: float,
    run_id: str,
    config: EvalConfig,
    frame_sampler: FrameSampler | None = None,
) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    cache = SceneIndexCache(config.scene_index_cache_dir) if config.scene_index_cache_enabled else None
    builder = SceneIndexBuilder(
        backend=backend,
        text_model_id=config.planner_model_path or config.model_path,
        vl_model_id=config.model_path,
        window_sec=config.window_sec,
        caption_nframes=config.scene_caption_nframes,
        root_policy=RootIndexPolicy(
            root_window_sec=float(config.window_sec),
            frame_cache_fps=float(config.scene_index_frame_fps),
            max_beats_per_root=int(config.scene_index_max_beats_per_root),
            max_new_tokens=int(config.scene_index_max_new_tokens),
        ),
        root_concurrency=config.scene_index_concurrency,
        cache=cache,
        clip_root=None if frame_sampler is not None else config.scene_index_cache_dir / "clips",
        frame_sampler=frame_sampler,
    )
    scene_index = builder.build(
        video_id=video_id,
        video_path=video_path,
        duration_sec=duration_sec,
        subtitle_cues=parse_srt_cues(config.subtitle_dir / f"{video_id}.srt"),
    )
    return run_loop(
        backend,
        video_path=video_path,
        question=question,
        duration_sec=duration_sec,
        run_id=run_id,
        scene_index=scene_index,
        workspace_root=config.workspace_root,
        budget=config.budget,
        extract_clips=True,
        frame_sampler=frame_sampler,
        strategy=strategy,
    )


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())
def _frame_cache_root(config: EvalConfig) -> Path:
    return config.frame_cache_root or (config.run_root / "frame_cache")


def _frame_cache_dir(*, config: EvalConfig, video_id: str) -> Path:
    safe_video_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(video_id)).strip("_") or "video"
    fps_label = ("%g" % float(config.frame_cache_fps)).replace(".", "p")
    return _frame_cache_root(config) / f"{safe_video_id}_{fps_label}fps"


def parse_strategies(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_STRATEGIES
    strategies = []
    for value in values:
        strategies.extend(parse_csv(value))
    unknown = [strategy for strategy in strategies if strategy not in STRATEGIES]
    if unknown:
        raise ValueError(f"Unknown strategy: {', '.join(unknown)}")
    return tuple(strategies)


def load_experiment_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except Exception:
        return _parse_minimal_yaml(text)
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config must be a mapping: {path}")
    return dict(payload)


def _parse_minimal_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current_mapping: dict[str, Any] | None = None
    current_indent = 0
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not value:
            mapping: dict[str, Any] = {}
            root[key] = mapping
            current_mapping = mapping
            current_indent = indent
            continue
        target = current_mapping if current_mapping is not None and indent > current_indent else root
        target[key] = _parse_minimal_yaml_scalar(value)
    return root


def _parse_minimal_yaml_scalar(value: str) -> Any:
    value = value.split(" #", 1)[0].strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [_parse_minimal_yaml_scalar(item.strip()) for item in inner.split(",") if item.strip()]
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _config_lookup(config: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if "." in key:
            current: Any = config
            for part in key.split("."):
                if not isinstance(current, Mapping) or part not in current:
                    current = None
                    break
                current = current[part]
            if current is not None:
                return current
        elif key in config:
            return config[key]
    return None


def _arg_or_config(args: argparse.Namespace, config: Mapping[str, Any], attr: str, *keys: str, default: Any = None) -> Any:
    value = getattr(args, attr, None)
    if value is not None:
        return value
    value = _config_lookup(config, *(keys or (attr,)))
    return default if value is None else value


def _as_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_str_mapping(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Expected a mapping for planner API proxy env")
    return {str(key): str(item) for key, item in value.items() if str(key).strip() and str(item).strip()}


def _as_sequence_args(value: Any) -> Sequence[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def _as_cases(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return parse_csv(value)
    if isinstance(value, Sequence):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return parse_csv(str(value))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible VideoMME strategy evaluations.")
    parser.add_argument("--config", type=Path, default=None, help="YAML experiment config; CLI flags override values.")
    parser.add_argument("--strategy", action="append", help="Strategy to run. Repeat or pass comma-separated values.")
    parser.add_argument("--cases", default=None, help="Comma-separated VideoMME question ids.")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--planner-model-path", default=None)
    parser.add_argument("--planner-api-type", default=None)
    parser.add_argument("--planner-api-base", default=None)
    parser.add_argument("--planner-api-model", default=None)
    parser.add_argument("--planner-api-key", default=None)
    parser.add_argument("--planner-api-user-key", default=None)
    parser.add_argument("--planner-api-biz-scene", default=None)
    parser.add_argument("--planner-api-version", default=None)
    parser.add_argument("--planner-api-base-env", default=None)
    parser.add_argument("--planner-api-model-env", default=None)
    parser.add_argument("--planner-api-key-env", default=None)
    parser.add_argument("--planner-api-version-env", default=None)
    parser.add_argument("--planner-api-user-key-env", default=None)
    parser.add_argument("--planner-api-biz-scene-env", default=None)
    parser.add_argument("--planner-api-use-for-tools", dest="planner_api_use_for_tools", action="store_true", default=None)
    parser.add_argument("--planner-api-no-tools", dest="planner_api_use_for_tools", action="store_false")
    parser.add_argument("--planner-api-timeout", type=float, default=None)
    parser.add_argument("--planner-thinking-token-budget", type=int, default=None)
    parser.add_argument("--planner-enable-thinking", dest="planner_enable_thinking", action="store_true", default=None)
    parser.add_argument("--planner-disable-thinking", dest="planner_enable_thinking", action="store_false")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--parquet-path", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--subtitle-dir", type=Path, default=None)
    parser.add_argument("--window-sec", type=float, default=None)
    parser.add_argument("--scene-index-cache-dir", type=Path, default=None)
    parser.add_argument("--no-scene-index-cache", action="store_true", default=None)
    parser.add_argument("--scene-caption-nframes", type=int, default=None)
    parser.add_argument("--scene-index-concurrency", type=int, default=None)
    parser.add_argument("--scene-index-frame-fps", type=float, default=None)
    parser.add_argument("--scene-index-max-beats-per-root", type=int, default=None)
    parser.add_argument("--scene-index-max-new-tokens", type=int, default=None)
    parser.add_argument("--frame-cache-root", type=Path, default=None)
    parser.add_argument("--frame-cache-fps", type=float, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--max-tool-calls-per-round", type=int, default=None)
    parser.add_argument("--default-nframes", type=int, default=None)
    parser.add_argument("--contract-nframes", type=int, default=None)
    parser.add_argument("--high-fps-nframes", type=int, default=None)
    parser.add_argument("--context-budget-tokens", type=int, default=None)
    parser.add_argument(
        "--budget-ratios",
        default=None,
        help="Comma-separated slot ratios, e.g. task:0.1,navigation:0.15,evidence:0.5,feedback:0.25",
    )
    parser.add_argument("--planner-receives-media", action="store_true", default=None)
    parser.add_argument("--no-reserve-final-round", action="store_true", default=None)
    parser.add_argument("--cheap-tool-budget", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--expensive-tool-budget", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--verifier-tool-budget", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument(
        "--hard-skill-runtime",
        action="store_true",
        default=None,
        help="Use deterministic skill runtime for supported routes before falling back to planner loop.",
    )
    parser.add_argument(
        "--use-global-question-rewrite",
        dest="use_global_question_rewrite",
        action="store_true",
        default=None,
        help="Use legacy text-model MCQ-to-open-question rewriting as the canonical planner task.",
    )
    parser.add_argument(
        "--disable-mcq-rewrite",
        dest="use_global_question_rewrite",
        action="store_false",
        help="Legacy compatibility flag; global MCQ rewrite is disabled by default.",
    )
    parser.add_argument("--enable-query-context", dest="enable_query_context", action="store_true", default=None)
    parser.add_argument("--disable-query-context", dest="enable_query_context", action="store_false")
    parser.add_argument("--enable-followup", dest="enable_followup", action="store_true", default=None)
    parser.add_argument("--disable-followup", dest="enable_followup", action="store_false")
    parser.add_argument("--enable-context-budget", dest="enable_context_budget", action="store_true", default=None)
    parser.add_argument("--disable-context-budget", dest="enable_context_budget", action="store_false")
    parser.add_argument("--enable-map-reflux", dest="enable_map_reflux", action="store_true", default=None)
    parser.add_argument("--disable-map-reflux", dest="enable_map_reflux", action="store_false")
    parser.add_argument("--enable-evidence-staging", dest="enable_evidence_staging", action="store_true", default=None)
    parser.add_argument("--disable-evidence-staging", dest="enable_evidence_staging", action="store_false")
    parser.add_argument("--enable-planner-owned-grounding", dest="planner_owned_grounding", action="store_true", default=None)
    parser.add_argument("--disable-planner-owned-grounding", dest="planner_owned_grounding", action="store_false")
    parser.add_argument("--followup-budget", type=int, default=None)
    parser.add_argument(
        "--free-explore",
        action="store_true",
        default=None,
        help="Legacy alias: use the free max-round/tool-call caps and disable the reserved final round.",
    )
    parser.add_argument("--free-max-rounds", type=int, default=None)
    parser.add_argument("--free-max-tool-calls-per-round", type=int, default=None)
    parser.add_argument("--export-training", action="store_true", default=None, help="Export compact TrainingTrajectory JSON per case.")
    parser.add_argument("--allow-any-python", action="store_true", help="Skip the remote Python executable assertion.")
    return parser


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    config_data = load_experiment_config(args.config)
    run_root = _as_path(_arg_or_config(args, config_data, "run_root", default=DEFAULT_RUN_ROOT))
    workspace_root = _as_path(_arg_or_config(args, config_data, "workspace_root", default=run_root / "workspaces"))
    strategies = parse_strategies(_as_sequence_args(_arg_or_config(args, config_data, "strategy", "strategies")))
    cases = _as_cases(_arg_or_config(args, config_data, "cases", default=DEFAULT_CASES))
    budget_ratios = _arg_or_config(args, config_data, "budget_ratios", "budget.ratios")
    context_budget_ratios = parse_budget_ratios(str(budget_ratios)) if budget_ratios else None
    default_nframes_value = _arg_or_config(
        args,
        config_data,
        "default_nframes",
        "nframes",
        "budget.nframes",
        "budget.default_nframes",
        default=SEGMENT_NFRAMES,
    )
    contract_nframes = _arg_or_config(args, config_data, "contract_nframes", "budget.contract_nframes")
    default_nframes = contract_nframes if contract_nframes is not None else default_nframes_value
    context_budget_tokens = int(
        _arg_or_config(args, config_data, "context_budget_tokens", "budget.context_budget_tokens", default=12000)
    )
    if args.enable_context_budget is False or _config_lookup(config_data, "enable_context_budget") is False:
        context_budget_tokens = 10**9
    free_explore = _as_bool(_arg_or_config(args, config_data, "free_explore", default=False))
    free_max_rounds = int(_arg_or_config(args, config_data, "free_max_rounds", "budget.free_max_rounds", default=24))
    free_max_tool_calls = int(
        _arg_or_config(args, config_data, "free_max_tool_calls_per_round", "budget.free_max_tool_calls_per_round", default=4)
    )
    max_rounds = int(
        free_max_rounds
        if free_explore
        else _arg_or_config(args, config_data, "max_rounds", "max_rounds", "budget.max_rounds", default=8)
    )
    max_tool_calls = int(
        free_max_tool_calls
        if free_explore
        else _arg_or_config(
            args,
            config_data,
            "max_tool_calls_per_round",
            "max_tool_calls_per_round",
            "max_tool_calls",
            "budget.max_tool_calls_per_round",
            "budget.max_tool_calls",
            default=2,
        )
    )
    no_reserve_final_round = _as_bool(_arg_or_config(args, config_data, "no_reserve_final_round", default=False))
    reserve_final_round = False if free_explore else not no_reserve_final_round
    planner_owned_grounding = _arg_or_config(
        args,
        config_data,
        "planner_owned_grounding",
        "planner_owned_grounding",
        "budget.planner_owned_grounding",
        default=False,
    )
    use_global_question_rewrite = _as_bool(
        _arg_or_config(args, config_data, "use_global_question_rewrite", "enable_mcq_rewrite", default=False)
    )
    budget = AgentBudget(
        max_rounds=max_rounds,
        max_tool_calls_per_round=max_tool_calls,
        default_nframes=int(default_nframes),
        high_fps_nframes=int(_arg_or_config(args, config_data, "high_fps_nframes", "budget.high_fps_nframes", default=32)),
        context_budget_tokens=context_budget_tokens,
        context_budget_ratios=context_budget_ratios,
        planner_receives_media=_as_bool(_arg_or_config(args, config_data, "planner_receives_media", default=False)),
        reserve_final_round=reserve_final_round,
        max_repeated_programs=max(max_rounds, AgentBudget().max_repeated_programs),
        rewrite_mcq_for_exploration=use_global_question_rewrite,
        hard_skill_runtime=False,
        planner_owned_grounding=_as_bool(planner_owned_grounding),
    )
    if args.enable_followup is False:
        budget = replace(budget, hard_skill_runtime=False)
    elif _as_bool(
        _arg_or_config(args, config_data, "hard_skill_runtime", "hard_skill_runtime", "budget.hard_skill_runtime", default=False)
    ) or args.enable_followup is True:
        budget = replace(budget, hard_skill_runtime=True)
    ablation_flags = {
        "enable_query_context": args.enable_query_context,
        "enable_followup": args.enable_followup,
        "enable_context_budget": args.enable_context_budget,
        "enable_map_reflux": args.enable_map_reflux,
        "enable_evidence_staging": args.enable_evidence_staging,
        "planner_owned_grounding": budget.planner_owned_grounding,
        "enable_mcq_rewrite": use_global_question_rewrite,
        "contract_nframes": contract_nframes,
        "followup_budget": _arg_or_config(args, config_data, "followup_budget", "budget.followup_budget"),
    }
    scene_index_cache_enabled = _arg_or_config(
        args,
        config_data,
        "scene_index_cache_enabled",
        "scene_index_cache_enabled",
        default=None,
    )
    if scene_index_cache_enabled is None:
        scene_index_cache_enabled = not _as_bool(_arg_or_config(args, config_data, "no_scene_index_cache", default=False))
    planner_thinking_budget = _arg_or_config(
        args,
        config_data,
        "planner_thinking_token_budget",
        "planner.thinking_token_budget",
        "planner_api.thinking_token_budget",
        default=None,
    )
    planner_enable_thinking = _arg_or_config(
        args,
        config_data,
        "planner_enable_thinking",
        "planner.enable_thinking",
        "planner_api.enable_thinking",
        default=None,
    )
    planner_api_use_for_tools = _arg_or_config(
        args,
        config_data,
        "planner_api_use_for_tools",
        "planner.api_use_for_tools",
        "planner_api.use_for_tools",
        default=False,
    )
    planner_api_proxy_env = _arg_or_config(
        args,
        config_data,
        "planner_api_proxy_env",
        "planner.proxy_env",
        "planner_api.proxy_env",
        default=None,
    )
    frame_cache_fps = min(
        TOOL_FRAME_CACHE_MAX_FPS,
        max(0.001, float(_arg_or_config(args, config_data, "frame_cache_fps", default=FRAME_CACHE_FPS))),
    )
    scene_index_frame_fps = min(
        frame_cache_fps,
        max(
            0.001,
            float(
                _arg_or_config(
                    args,
                    config_data,
                    "scene_index_frame_fps",
                    "scene_index_frame_fps",
                    "scene_index.fps",
                    "scene.frame_fps",
                    "dense_video_caption.fps",
                    default=ROOT_DVC_FRAME_FPS,
                )
            ),
        ),
    )
    return EvalConfig(
        run_root=run_root,
        workspace_root=workspace_root,
        model_path=str(_arg_or_config(args, config_data, "model_path", default=MODEL_PATH)),
        planner_model_path=str(_arg_or_config(args, config_data, "planner_model_path", default=PLANNER_MODEL_PATH)),
        planner_api_type=str(
            _arg_or_config(args, config_data, "planner_api_type", "planner.api_type", "planner_api.type", default="openai_compatible")
        ),
        planner_api_base=str(
            _arg_or_config(
                args,
                config_data,
                "planner_api_base",
                "planner.api_base",
                "planner_api.base",
                "planner_api.endpoint",
                default="",
            )
        ),
        planner_api_model=str(
            _arg_or_config(
                args,
                config_data,
                "planner_api_model",
                "planner.api_model",
                "planner_api.model",
                "planner_api.deployment",
                default="",
            )
        ),
        planner_api_key=str(
            _arg_or_config(args, config_data, "planner_api_key", "planner.api_key", "planner_api.api_key", default="EMPTY")
        ),
        planner_api_user_key=str(
            _arg_or_config(args, config_data, "planner_api_user_key", "planner.user_key", "planner_api.user_key", default="")
        ),
        planner_api_biz_scene=str(
            _arg_or_config(args, config_data, "planner_api_biz_scene", "planner.biz_scene", "planner_api.biz_scene", default="")
        ),
        planner_api_version=str(
            _arg_or_config(args, config_data, "planner_api_version", "planner.api_version", "planner_api.api_version", default="")
        ),
        planner_api_base_env=str(
            _arg_or_config(
                args,
                config_data,
                "planner_api_base_env",
                "planner.api_base_env",
                "planner_api.base_env",
                "planner_api.endpoint_env",
                default="",
            )
        ),
        planner_api_model_env=str(
            _arg_or_config(
                args,
                config_data,
                "planner_api_model_env",
                "planner.api_model_env",
                "planner_api.model_env",
                "planner_api.deployment_env",
                default="",
            )
        ),
        planner_api_key_env=str(
            _arg_or_config(
                args,
                config_data,
                "planner_api_key_env",
                "planner.api_key_env",
                "planner_api.api_key_env",
                default="",
            )
        ),
        planner_api_version_env=str(
            _arg_or_config(
                args,
                config_data,
                "planner_api_version_env",
                "planner.api_version_env",
                "planner_api.api_version_env",
                default="",
            )
        ),
        planner_api_user_key_env=str(
            _arg_or_config(
                args,
                config_data,
                "planner_api_user_key_env",
                "planner.api_user_key_env",
                "planner_api.user_key_env",
                default="",
            )
        ),
        planner_api_biz_scene_env=str(
            _arg_or_config(
                args,
                config_data,
                "planner_api_biz_scene_env",
                "planner.api_biz_scene_env",
                "planner_api.biz_scene_env",
                default="",
            )
        ),
        planner_api_use_for_tools=_as_bool(planner_api_use_for_tools),
        planner_api_proxy_env=_as_str_mapping(planner_api_proxy_env),
        planner_api_timeout=float(
            _arg_or_config(args, config_data, "planner_api_timeout", "planner.api_timeout", "planner_api.timeout", default=180.0)
        ),
        planner_thinking_token_budget=(
            int(planner_thinking_budget) if planner_thinking_budget is not None else None
        ),
        planner_enable_thinking=(
            _as_bool(planner_enable_thinking) if planner_enable_thinking is not None else None
        ),
        data_root=_as_path(_arg_or_config(args, config_data, "data_root", default=DATA_ROOT)),
        parquet_path=_as_path(_arg_or_config(args, config_data, "parquet_path", default=DEFAULT_PARQUET_PATH)),
        video_dir=_as_path(_arg_or_config(args, config_data, "video_dir", default=DEFAULT_VIDEO_DIR)),
        subtitle_dir=_as_path(_arg_or_config(args, config_data, "subtitle_dir", default=DEFAULT_SUBTITLE_DIR)),
        cases=cases,
        strategies=strategies,
        window_sec=float(_arg_or_config(args, config_data, "window_sec", default=WINDOW_SEC)),
        scene_index_cache_dir=_as_path(
            _arg_or_config(args, config_data, "scene_index_cache_dir", default=DEFAULT_SCENE_INDEX_CACHE_DIR)
        ),
        scene_index_cache_enabled=_as_bool(scene_index_cache_enabled),
        scene_caption_nframes=int(
            _arg_or_config(
                args,
                config_data,
                "scene_caption_nframes",
                "scene_caption_nframes",
                "caption_nframes",
                default=SEGMENT_NFRAMES,
            )
        ),
        scene_index_concurrency=int(
            _arg_or_config(
                args,
                config_data,
                "scene_index_concurrency",
                "scene_index_concurrency",
                "scene.index_concurrency",
                default=1,
            )
        ),
        scene_index_frame_fps=scene_index_frame_fps,
        scene_index_max_beats_per_root=int(
            _arg_or_config(
                args,
                config_data,
                "scene_index_max_beats_per_root",
                "scene_index_max_beats_per_root",
                "scene_index.max_beats_per_root",
                "scene.max_beats_per_root",
                "dense_video_caption.max_beats_per_root",
                default=RootIndexPolicy().max_beats_per_root,
            )
        ),
        scene_index_max_new_tokens=int(
            _arg_or_config(
                args,
                config_data,
                "scene_index_max_new_tokens",
                "scene_index_max_new_tokens",
                "scene_index.max_new_tokens",
                "scene.max_new_tokens",
                "dense_video_caption.max_new_tokens",
                default=RootIndexPolicy().max_new_tokens,
            )
        ),
        frame_cache_fps=frame_cache_fps,
        frame_cache_root=(
            _as_path(_arg_or_config(args, config_data, "frame_cache_root"))
            if _arg_or_config(args, config_data, "frame_cache_root") is not None
            else None
        ),
        budget=budget,
        export_training=_as_bool(_arg_or_config(args, config_data, "export_training", default=False)),
        ablation_flags=ablation_flags,
        source_config_path=args.config,
    )


def build_backend(config: EvalConfig) -> Any:
    planner_api_type = config.planner_api_type.lower().replace("-", "_")
    uses_azure_api = planner_api_type in {"azure", "azure_openai"}
    uses_gemini_gateway = planner_api_type in {"gemini", "gemini_gateway", "ks_gateway", "kigress_gateway"}
    if config.planner_api_base or uses_azure_api or uses_gemini_gateway:
        from visual_coding_agent_harness.backends.openai_chat import OpenAIChatTextBackend
        from visual_coding_agent_harness.backends.routed import RoutedBackend

        text_backend = OpenAIChatTextBackend(
            api_base=config.planner_api_base,
            model=(
                config.planner_api_model
                if uses_azure_api or uses_gemini_gateway
                else config.planner_api_model or config.planner_model_path or config.model_path
            ),
            api_key=config.planner_api_key,
            api_type=config.planner_api_type,
            api_version=config.planner_api_version,
            api_base_env=config.planner_api_base_env,
            model_env=config.planner_api_model_env,
            api_key_env=config.planner_api_key_env,
            api_version_env=config.planner_api_version_env,
            user_key_env=config.planner_api_user_key_env,
            biz_scene_env=config.planner_api_biz_scene_env,
            user_key=config.planner_api_user_key,
            biz_scene=config.planner_api_biz_scene,
            proxy_env=dict(config.planner_api_proxy_env or {}),
            allow_media=config.planner_api_use_for_tools,
            timeout=config.planner_api_timeout,
            thinking_token_budget=config.planner_thinking_token_budget,
            enable_thinking=config.planner_enable_thinking,
        )
        if config.planner_api_use_for_tools:
            return RoutedBackend(text_backend=text_backend, vl_backend=text_backend)

        from visual_coding_agent_harness.backends.qwen_vl import QwenVLBackend

        vl_backend = QwenVLBackend.from_pretrained(config.model_path)
        return RoutedBackend(text_backend=text_backend, vl_backend=vl_backend)

    from visual_coding_agent_harness.backends.qwen_vl import QwenVLBackend

    vl_backend = QwenVLBackend.from_pretrained(config.model_path)
    if not config.planner_model_path:
        return vl_backend

    from visual_coding_agent_harness.backends.qwen_text import QwenTextBackend
    from visual_coding_agent_harness.backends.routed import RoutedBackend

    text_backend = QwenTextBackend.from_pretrained(config.planner_model_path)
    return RoutedBackend(text_backend=text_backend, vl_backend=vl_backend)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    validate_python(allow_any_python=args.allow_any_python)
    config = config_from_args(args)
    rows_by_id = load_rows_by_id(config.parquet_path, config.cases)
    backend = build_backend(config)
    run_eval_cases(backend=backend, rows_by_id=rows_by_id, config=config)


if __name__ == "__main__":
    main()
