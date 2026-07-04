from __future__ import annotations

import argparse
import html
import inspect
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from visual_coding_agent_harness.core.budget import AgentBudget, parse_budget_ratios
from visual_coding_agent_harness.agents.driver import MultiV3Driver
from visual_coding_agent_harness.agents.investigator import Investigator as InvestigatorV3
from visual_coding_agent_harness.agents.reasoner import Reasoner as ReasonerV3
from visual_coding_agent_harness.evals.videomme.dvc_compat import SceneIndex
from visual_coding_agent_harness.evals.videomme.indexing import RootIndexPolicy, SceneIndexBuilder, SubtitleCue
from visual_coding_agent_harness.evals.videomme.indexing import SceneIndexCache
from visual_coding_agent_harness.evals.videomme.outputs import (
    RunSummary,
    export_multi_v3_evidence_chains,
    export_multi_v3_exploration_records,
    export_multi_v3_training_trajectory,
    export_multi_v3_trajectory,
    export_multi_v3_workspace_round_log,
    multi_v3_backend_call_counters,
    multi_v3_tools_and_segments,
    validate as validate_run_summary,
    write_trajectory_markdown,
)
from visual_coding_agent_harness.tools.frame_cache import FrameSampler, build_frame_cache_for_video
from visual_coding_agent_harness.video.build import build_video_workspace
from visual_coding_agent_harness.video.index import Frame
from visual_coding_agent_harness.video.pipeline import sample_shot_frames
from visual_coding_agent_harness.video.overview import build_scene_timeline_overview
from visual_coding_agent_harness.workspace.investigator_ws import InvestigatorWorkspace as InvestigatorWorkspaceV3
from visual_coding_agent_harness.workspace.memo import MemoStore

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
MULTI_V3_STRATEGY = "multi_v3"
DEFAULT_STRATEGIES = (MULTI_V3_STRATEGY,)
STRATEGIES = (MULTI_V3_STRATEGY,)
WINDOW_SEC = 300.0
DEFAULT_NFRAMES = 8
FRAME_CACHE_FPS = 2.0
TOOL_FRAME_CACHE_MAX_FPS = 2.0
ROOT_DVC_FRAME_FPS = 0.5
MAX_EVAL_CASE_CONCURRENCY = 16
MAX_SCENE_INDEX_VIDEO_CONCURRENCY = 16


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
    scene_index_concurrency: int = 1
    scene_index_frame_fps: float = ROOT_DVC_FRAME_FPS
    scene_index_max_new_tokens: int = RootIndexPolicy().max_new_tokens
    frame_cache_fps: float = FRAME_CACHE_FPS
    frame_cache_root: Path | None = None
    scene_index_video_concurrency: int = 1
    eval_case_concurrency: int = 1
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
    strategy: str = MULTI_V3_STRATEGY,
) -> dict[str, Any]:
    start = time.perf_counter()
    workspace_run_root = workspace_root / "runs" / run_id
    workspace_run_root.mkdir(parents=True, exist_ok=True)
    workspace_log_dir = workspace_root.parent / "workspace_logs" / run_id
    if strategy == MULTI_V3_STRATEGY:
        index_frame_sampler = frame_sampler
        verify_frame_sampler = frame_sampler
        if verify_frame_sampler is None and Path(video_path).exists():
            verify_frame_sampler = _default_multi_v3_frame_sampler(
                artifact_dir=workspace_run_root / "artifacts" / "multi_v3_verify_frames"
            )
        video_index = _build_multi_v3_video_index(
            video_path=video_path,
            duration_sec=duration_sec,
            scene_index=scene_index,
            artifact_dir=workspace_run_root / "artifacts" / "multi_v3_index",
            frame_sampler=index_frame_sampler,
        )
        overview = build_scene_timeline_overview(
            video_index,
            output_dir=workspace_run_root / "artifacts" / "multi_v3_overview",
        )
        overview_image_path = getattr(overview, "grid_image_path", getattr(overview, "grid_path", ""))
        investigator_workspace = InvestigatorWorkspaceV3(workspace_run_root / "multi_v3")
        investigator_kwargs = {
            "workspace": investigator_workspace,
            "backend": backend,
            "video_workspace": video_index,
            "memo_store": MemoStore(investigator_workspace.root / "observation_memos.jsonl"),
        }
        if verify_frame_sampler is not None:
            investigator_kwargs["frame_sampler"] = _multi_v3_frame_sampler(video_path=video_path, frame_sampler=verify_frame_sampler)
        reasoner = ReasonerV3(backend=backend)
        investigator = InvestigatorV3(**investigator_kwargs)
        driver = MultiV3Driver(
            reasoner=reasoner,
            investigator=investigator,
            workspace=investigator_workspace,
            max_rounds=budget.max_rounds,
            valid_scene_ids=tuple(chapter.chapter_id for chapter in video_index.chapters),
            video_workspace=video_index,
        )
        result = driver.run(
            question=question,
            options=_extract_option_map(question),
            index_context=video_index.timeline_text(fill_missing_titles=True),
            overview_image_path=overview_image_path,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    seconds = time.perf_counter() - start
    status = _result_status(result)
    answer = str(getattr(result, "answer", ""))
    citations = _result_citations(result)
    confidence = getattr(result, "confidence", "")
    reward_tags = _reward_tags_for_multi_v3(status=status, citations=citations)
    trajectory_path = workspace_run_root / "artifacts" / "trajectories" / "longvideoagent_trajectory.json"
    evidence_chains_path = workspace_run_root / "artifacts" / "evidence_chains" / "evidence_chains.json"
    exploration_records_path = workspace_run_root / "artifacts" / "exploration_records" / "exploration_records.jsonl"
    final_payload = {
        "answer": answer,
        "status": status,
        "citations": list(citations),
        "confidence": confidence,
    }
    trajectory_payload = export_multi_v3_trajectory(
        investigator_workspace,
        question=question,
        video_path=video_path,
        final=final_payload,
        reward_tags=reward_tags,
        output_path=trajectory_path,
    )
    evidence_chains_payload = export_multi_v3_evidence_chains(
        investigator_workspace,
        output_path=evidence_chains_path,
    )
    exploration_records_payload = export_multi_v3_exploration_records(
        investigator_workspace,
        question=question,
        video_path=video_path,
        final=final_payload,
        round_count=_result_round_count(result),
        output_path=exploration_records_path,
    )
    workspace_round_log = export_multi_v3_workspace_round_log(
        investigator_workspace,
        question=question,
        video_path=video_path,
        final=final_payload,
        trajectory_path=trajectory_path,
        evidence_chains_path=evidence_chains_path,
        log_root=workspace_log_dir,
    )
    planner_io_dir = workspace_log_dir
    tools, segments = multi_v3_tools_and_segments(investigator_workspace)
    backend_call_counters = multi_v3_backend_call_counters(investigator_workspace)
    return {
        "answer": answer,
        "strategy": strategy,
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
        "exploration_records_path": str(exploration_records_path),
        "exploration_record_count": int(exploration_records_payload.get("record_count", 0) or 0),
        "workspace_log_dir": str(workspace_log_dir),
        "workspace_round_log_path": workspace_round_log["path"],
        "workspace_round_log_round_count": workspace_round_log["round_count"],
        "planner_io_dir": str(planner_io_dir),
        "planner_prompt_count": len(list(planner_io_dir.glob("*_prompt.txt"))) if planner_io_dir.exists() else 0,
        "backend_call_counters": backend_call_counters,
        **backend_call_counters,
        "reward_tags": reward_tags,
    }


def _multi_v3_frame_sampler(*, video_path: str, frame_sampler: FrameSampler):
    def sample(shot, max_frames: int, *, resolution: str = "high", dense: bool = False) -> tuple[str, ...]:
        return _sample_verify_frames(
            frame_sampler,
            video_path=str(video_path),
            start_sec=float(shot.start_sec),
            end_sec=float(shot.end_sec),
            nframes=int(max_frames),
            resolution=resolution,
            dense=dense,
        )

    return sample


def _default_multi_v3_frame_sampler(*, artifact_dir: Path) -> FrameSampler:
    def sample(
        video_path: str,
        start_sec: float,
        end_sec: float,
        nframes: int,
        *,
        resolution: str = "high",
        dense: bool = False,
    ) -> tuple[str, ...]:
        del dense
        out_dir = artifact_dir / _frame_sample_dir_name(start_sec=start_sec, end_sec=end_sec, nframes=nframes)
        frames = sample_shot_frames(
            video_path,
            float(start_sec),
            float(end_sec),
            n_frames=int(nframes),
            out_dir=out_dir,
            size=None if resolution == "high" else (384, 216),
        )
        return tuple(frame.thumb_path for frame in frames if frame.thumb_path)

    return sample


def _sample_verify_frames(
    frame_sampler: FrameSampler,
    *,
    video_path: str,
    start_sec: float,
    end_sec: float,
    nframes: int,
    resolution: str,
    dense: bool,
) -> tuple[str, ...]:
    try:
        parameters = inspect.signature(frame_sampler).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_keywords = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    if accepts_keywords or "resolution" in parameters or "dense" in parameters:
        return tuple(
            frame_sampler(
                video_path,
                float(start_sec),
                float(end_sec),
                int(nframes),
                resolution=resolution,
                dense=dense,
            )
        )
    return tuple(frame_sampler(video_path, float(start_sec), float(end_sec), int(nframes)))


def _frame_sample_dir_name(*, start_sec: float, end_sec: float, nframes: int) -> str:
    label = f"{float(start_sec):.3f}_{float(end_sec):.3f}_{int(nframes)}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "frames"


def _build_multi_v3_video_index(
    *,
    video_path: str,
    duration_sec: float,
    scene_index: SceneIndex,
    artifact_dir: Path,
    frame_sampler: FrameSampler | None,
):
    ranges = tuple((float(segment.start_sec), float(segment.end_sec)) for segment in scene_index.segments)
    workspace = build_video_workspace(
        video_path,
        duration_sec,
        artifact_dir=artifact_dir,
        asr_cues=_scene_index_asr_cues(scene_index),
        embedding_backend=_ZeroEmbeddingBackend(),
        max_chapters=max(1, len(ranges)),
        max_range_sec=30.0,
        max_beat_sec=30.0,
        index_mode="fast_eval",
        shot_detector=lambda _video_path, _duration: ranges,
        keyframe_sampler=_keyframe_sampler_from_frame_sampler(frame_sampler)
        if frame_sampler is not None
        else _placeholder_keyframe_sampler,
    )
    workspace.save(artifact_dir)
    return workspace


def _keyframe_sampler_from_frame_sampler(frame_sampler: FrameSampler):
    def sample(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
        del out_dir
        paths = tuple(frame_sampler(video_path, float(start_sec), float(end_sec), int(n_frames)))
        return _frames_from_paths(paths, start_sec=float(start_sec), end_sec=float(end_sec))

    return sample


def _frames_from_paths(paths: Sequence[str], *, start_sec: float, end_sec: float) -> tuple[Frame, ...]:
    if not paths:
        return ()
    span = max(0.0, float(end_sec) - float(start_sec))
    frames = []
    for index, path in enumerate(paths, start=1):
        time_sec = float(start_sec) if len(paths) == 1 else float(start_sec) + span * float(index - 1) / float(len(paths) - 1)
        frames.append(Frame(frame_id=f"fr{index:03d}", time_sec=round(time_sec, 3), thumb_path=str(path)))
    return tuple(frames)


def _placeholder_keyframe_sampler(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
    del video_path, end_sec, n_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "frame_001.jpg"
    Image.new("RGB", (256, 144), color=(42, 48, 56)).save(path)
    return (Frame(frame_id="fr001", time_sec=float(start_sec), thumb_path=str(path)),)


class _ZeroEmbeddingBackend:
    embedding_dim = 1

    def encode_images(self, paths: Sequence[str]) -> np.ndarray:
        return np.zeros((len(paths), 1), dtype=np.float32)

    def encode_text(self, queries: Sequence[str]) -> np.ndarray:
        return np.zeros((len(queries), 1), dtype=np.float32)


def _scene_index_asr_cues(scene_index: SceneIndex) -> tuple[dict[str, Any], ...]:
    cues: list[dict[str, Any]] = []
    for segment in scene_index.segments:
        for item in getattr(segment, "asr_sentences", ()) or ():
            if isinstance(item, Mapping) and item.get("text"):
                cues.append(
                    {
                        "start_sec": float(item.get("start_sec", segment.start_sec) or segment.start_sec),
                        "end_sec": float(item.get("end_sec", segment.end_sec) or segment.end_sec),
                        "text": str(item["text"]),
                    }
                )
        summary = str(getattr(segment, "asr_summary", "") or "").strip()
        if summary:
            cues.append({"start_sec": float(segment.start_sec), "end_sec": float(segment.end_sec), "text": summary})
    return tuple(cues)


def _extract_option_map(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for line in str(question or "").splitlines():
        match = re.match(r"^\s*([A-H])[\.\)]\s*(.+?)\s*$", line)
        if match:
            options[match.group(1).upper()] = match.group(2)
    return options


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


def _reward_tags_for_multi_v3(*, status: str, citations: Sequence[str]) -> list[str]:
    tags = [str(status)] if status else []
    tags.append("has_citations" if citations else "missing_citations")
    if citations:
        tags.append("multi_v3_citations")
    return tags


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
        "exploration_records_path",
        "exploration_record_count",
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
        "scene_index_concurrency": config.scene_index_concurrency,
        "scene_index_frame_fps": config.scene_index_frame_fps,
        "scene_index_max_new_tokens": config.scene_index_max_new_tokens,
        "frame_cache_fps": config.frame_cache_fps,
        "frame_cache_root": str(_frame_cache_root(config)),
        "scene_index_video_concurrency": config.scene_index_video_concurrency,
        "eval_case_concurrency": config.eval_case_concurrency,
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
        + json.dumps(
            {
                "cases": list(config.cases),
                "strategies": list(config.strategies),
                "eval_case_concurrency": config.eval_case_concurrency,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    completed: dict[str, dict[str, Any]] = {}

    def record_case(qid: str, case: dict[str, Any]) -> None:
        nonlocal summary
        completed[str(qid)] = case
        results[:] = [completed[str(case_id)] for case_id in config.cases if str(case_id) in completed]
        summary = _summary_payload(
            run_id=config.run_root.name,
            case_ids=config.cases,
            config_payload=config_payload,
            results=results,
        )
        evidence_chains_path = _write_run_evidence_chains(config.run_root, results)
        summary["evidence_chains_path"] = str(evidence_chains_path)
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        compact = {"question_id": qid, "gt": case.get("gt"), "strategies": case.get("strategies", {})}
        print("CASE_DONE " + json.dumps(compact, ensure_ascii=True, sort_keys=True), flush=True)

    if config.eval_case_concurrency <= 1 or len(config.cases) <= 1:
        for qid in config.cases:
            record_case(
                str(qid),
                _run_eval_case(
                    qid=str(qid),
                    backend=backend,
                    row=rows_by_id[str(qid)],
                    config=config,
                    duration_fn=duration_fn,
                ),
            )
    else:
        with ThreadPoolExecutor(max_workers=config.eval_case_concurrency) as executor:
            futures = {
                executor.submit(
                    _run_eval_case,
                    qid=str(qid),
                    backend=backend,
                    row=rows_by_id[str(qid)],
                    config=config,
                    duration_fn=duration_fn,
                ): str(qid)
                for qid in config.cases
            }
            for future in as_completed(futures):
                qid = futures[future]
                record_case(qid, future.result())
    violations = validate_run_summary(RunSummary.from_dict(summary))
    if violations:
        (config.run_root / "summary_violations.json").write_text(
            json.dumps({"violations": violations}, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise SystemExit(2)
    print("DONE summary=" + str(summary_path), flush=True)
    return summary


def _run_eval_case(
    *,
    qid: str,
    backend: Any,
    row: Any,
    config: EvalConfig,
    duration_fn: Callable[[Path], float],
) -> dict[str, Any]:
    video_id = str(row_get(row, "videoID") or row_get(row, "video_id"))
    video_path = str(config.video_dir / f"{video_id}.mp4")
    duration_sec = duration_fn(Path(video_path))
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
    return case


def _build_scene_index(
    *,
    backend: Any,
    video_path: str,
    video_id: str,
    duration_sec: float,
    config: EvalConfig,
    frame_sampler: FrameSampler | None = None,
) -> SceneIndex:
    cache = SceneIndexCache(config.scene_index_cache_dir) if config.scene_index_cache_enabled else None
    vl_model_id = _scene_index_vl_model_id(config)
    builder = SceneIndexBuilder(
        backend=backend,
        text_model_id=config.planner_model_path or config.model_path,
        vl_model_id=vl_model_id,
        window_sec=config.window_sec,
        root_policy=RootIndexPolicy(
            root_window_sec=float(config.window_sec),
            frame_cache_fps=float(config.scene_index_frame_fps),
            max_new_tokens=int(config.scene_index_max_new_tokens),
        ),
        root_concurrency=config.scene_index_concurrency,
        cache=cache,
        clip_root=None if frame_sampler is not None else config.scene_index_cache_dir / "clips",
        frame_sampler=frame_sampler,
    )
    return builder.build(
        video_id=video_id,
        video_path=video_path,
        duration_sec=duration_sec,
        subtitle_cues=parse_srt_cues(config.subtitle_dir / f"{video_id}.srt"),
    )


def _scene_index_vl_model_id(config: EvalConfig) -> str:
    if config.planner_api_use_for_tools and config.planner_api_model:
        return config.planner_api_model
    if config.planner_api_use_for_tools and config.planner_model_path:
        return config.planner_model_path
    return config.model_path


def prewarm_scene_indexes(
    *,
    backend: Any,
    rows_by_id: Mapping[str, Any],
    config: EvalConfig,
    duration_fn: Callable[[Path], float] = ffprobe_duration,
) -> dict[str, Any]:
    config.run_root.mkdir(parents=True, exist_ok=True)
    config.scene_index_cache_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    videos: list[dict[str, Any]] = []
    for qid in config.cases:
        row = rows_by_id[str(qid)]
        video_id = str(row_get(row, "videoID") or row_get(row, "video_id"))
        if video_id in seen:
            continue
        seen.add(video_id)
        videos.append({"question_id": str(qid), "videoID": video_id})

    summary_path = config.run_root / "scene_index_prewarm_summary.json"
    print(
        "START scene_index_prewarm "
        + json.dumps(
            {
                "videos": len(videos),
                "cases": len(config.cases),
                "scene_index_video_concurrency": config.scene_index_video_concurrency,
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    completed: dict[str, dict[str, Any]] = {}

    def write_progress(record: dict[str, Any]) -> None:
        video_id = str(record.get("videoID", ""))
        completed[video_id] = record
        results = [completed[str(item["videoID"])] for item in videos if str(item["videoID"]) in completed]
        summary = {
            "run_id": config.run_root.name,
            "cases": list(config.cases),
            "videos_total": len(videos),
            "videos_done": len(results),
            "videos_ok": sum(1 for item in results if item.get("status") == "done"),
            "videos_error": sum(1 for item in results if item.get("status") == "error"),
            "scene_index_cache_dir": str(config.scene_index_cache_dir),
            "scene_index_video_concurrency": config.scene_index_video_concurrency,
            "videos": results,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        marker = "SCENE_INDEX_DONE" if record.get("status") == "done" else "SCENE_INDEX_ERROR"
        print(marker + " " + json.dumps(record, ensure_ascii=True, sort_keys=True), flush=True)

    if config.scene_index_video_concurrency <= 1 or len(videos) <= 1:
        for item in videos:
            write_progress(
                _prewarm_scene_index_video(
                    backend=backend,
                    item=item,
                    config=config,
                    duration_fn=duration_fn,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=config.scene_index_video_concurrency) as executor:
            futures = {
                executor.submit(
                    _prewarm_scene_index_video,
                    backend=backend,
                    item=item,
                    config=config,
                    duration_fn=duration_fn,
                ): str(item["videoID"])
                for item in videos
            }
            for future in as_completed(futures):
                write_progress(future.result())
    if not completed:
        summary = {
            "run_id": config.run_root.name,
            "cases": list(config.cases),
            "videos_total": 0,
            "videos_done": 0,
            "videos_ok": 0,
            "videos_error": 0,
            "scene_index_cache_dir": str(config.scene_index_cache_dir),
            "scene_index_video_concurrency": config.scene_index_video_concurrency,
            "videos": [],
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    print("DONE scene_index_prewarm_summary=" + str(summary_path), flush=True)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _prewarm_scene_index_video(
    *,
    backend: Any,
    item: Mapping[str, str],
    config: EvalConfig,
    duration_fn: Callable[[Path], float],
) -> dict[str, Any]:
    started = time.time()
    video_id = str(item["videoID"])
    video_path = str(config.video_dir / f"{video_id}.mp4")
    try:
        duration_sec = duration_fn(Path(video_path))
        frame_cache = build_frame_cache_for_video(
            video_path=Path(video_path),
            frame_dir=_frame_cache_dir(config=config, video_id=video_id),
            fps=float(config.frame_cache_fps),
            duration_sec=duration_sec,
        )
        scene_index = _build_scene_index(
            backend=backend,
            video_path=video_path,
            video_id=video_id,
            duration_sec=duration_sec,
            config=config,
            frame_sampler=frame_cache.sample_paths,
        )
        return {
            "question_id": item["question_id"],
            "videoID": video_id,
            "duration_sec": round(duration_sec, 1),
            "segments": len(scene_index.segments),
            "frame_cache": str(frame_cache.frame_dir),
            "seconds": round(time.time() - started, 3),
            "status": "done",
        }
    except Exception as exc:
        return {
            "question_id": item["question_id"],
            "videoID": video_id,
            "duration_sec": None,
            "segments": 0,
            "frame_cache": str(_frame_cache_dir(config=config, video_id=video_id)),
            "seconds": round(time.time() - started, 3),
            "status": "error",
            "error": type(exc).__name__ + ": " + str(exc)[:500],
        }


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
    multi_v3_root = workspace_path / "multi_v3"
    if not multi_v3_root.exists():
        return None
    export_multi_v3_training_trajectory(
        multi_v3_root,
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
            multi_v3_root = workspace_root / "multi_v3"
            if not multi_v3_root.exists():
                continue
            payload = export_multi_v3_evidence_chains(multi_v3_root)
            chains = payload.get("chains", []) if isinstance(payload, Mapping) else []
            chain_rows = [
                [
                    str(record.get("evidence_id", ""))
                    for record in chain.get("records", [])
                    if isinstance(record, Mapping)
                ]
                for chain in chains
                if isinstance(chain, Mapping)
            ]
            rows.append(
                {
                    "case_id": str(case.get("question_id", "")),
                    "strategy": str(strategy),
                    "final_decision": str(strategy_summary.get("status", "")),
                    "selected_option": str(strategy_summary.get("choice", "")),
                    "workspace": workspace_root.as_posix(),
                    "chain_count": len(chains),
                    "chains": chain_rows,
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
    summary.raw_choice_accuracy = summary.accuracy
    summary.grounded_choice_accuracy = (
        sum(
            1
            for item in strategy_results
            if bool(item.get("correct"))
            and item.get("status") == "final"
            and int(item.get("citation_count", 0) or 0) > 0
        )
        / total
    )
    summary.final_rate = sum(1 for item in strategy_results if item.get("status") == "final") / total
    summary.cited_answer_rate = (
        sum(
            1
            for item in strategy_results
            if item.get("status") == "final" and int(item.get("citation_count", 0) or 0) > 0
        )
        / total
    )
    summary.need_more_evidence_rate = (
        sum(1 for item in strategy_results if item.get("status") == "need_more_evidence") / total
    )
    summary.low_confidence_final_rate = (
        sum(1 for item in strategy_results if item.get("status") == "low_confidence_final") / total
    )
    summary.unvalidated_guess_rate = (
        sum(1 for item in strategy_results if item.get("status") == "unvalidated_guess") / total
    )
    summary.unsupported_final_rate = (
        sum(
            1
            for item in strategy_results
            if item.get("status") == "final" and int(item.get("citation_count", 0) or 0) == 0
        )
        / total
    )


def _case_strategies(case: Mapping[str, Any]) -> Mapping[str, Any]:
    strategies = case.get("strategies", {})
    return strategies if isinstance(strategies, Mapping) else {}


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
    scene_index = _build_scene_index(
        backend=backend,
        video_path=video_path,
        video_id=video_id,
        duration_sec=duration_sec,
        config=config,
        frame_sampler=frame_sampler,
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
    parser.add_argument("--scene-index-concurrency", type=int, default=None)
    parser.add_argument("--scene-index-video-concurrency", type=int, default=None)
    parser.add_argument("--eval-case-concurrency", type=int, default=None)
    parser.add_argument("--scene-index-frame-fps", type=float, default=None)
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
    parser.add_argument(
        "--scene-index-only",
        action="store_true",
        help="Only build cached root dense-video captions / scene indexes; do not run workspace QA.",
    )
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
        default=DEFAULT_NFRAMES,
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
    eval_case_concurrency = min(
        MAX_EVAL_CASE_CONCURRENCY,
        max(
            1,
            int(
                _arg_or_config(
                    args,
                    config_data,
                    "eval_case_concurrency",
                    "eval.case_concurrency",
                    "eval.concurrency",
                    "case_concurrency",
                    default=1,
                )
            ),
        ),
    )
    scene_index_video_concurrency = min(
        MAX_SCENE_INDEX_VIDEO_CONCURRENCY,
        max(
            1,
            int(
                _arg_or_config(
                    args,
                    config_data,
                    "scene_index_video_concurrency",
                    "scene_index.video_concurrency",
                    "dense_video_caption.video_concurrency",
                    default=1,
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
        scene_index_video_concurrency=scene_index_video_concurrency,
        eval_case_concurrency=eval_case_concurrency,
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
    if args.scene_index_only:
        prewarm_scene_indexes(backend=backend, rows_by_id=rows_by_id, config=config)
    else:
        run_eval_cases(backend=backend, rows_by_id=rows_by_id, config=config)


if __name__ == "__main__":
    main()
