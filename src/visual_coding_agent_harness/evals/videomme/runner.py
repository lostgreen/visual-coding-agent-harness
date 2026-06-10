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

from visual_coding_agent_harness.agents.iterative_agent import AgentBudget
from visual_coding_agent_harness.agents.context_budget import parse_budget_ratios
from visual_coding_agent_harness.backends.base import BackendRequest
from visual_coding_agent_harness.evals.videomme.scene_index_builder import SceneIndexBuilder, SubtitleCue
from visual_coding_agent_harness.evals.videomme.scene_index_cache import SceneIndexCache
from visual_coding_agent_harness.iterative_smoke import run_iterative_smoke
from visual_coding_agent_harness.tools.frame_cache import FrameSampler, build_frame_cache_for_video
from visual_coding_agent_harness.video_index import SceneIndex, VideoSegment, fixed_window_scene_index
from visual_coding_agent_harness.workspace import EvidenceWorkspace

from .summary_schema import RunSummary, validate as validate_run_summary
from .training_trajectory import TrainingTrajectory
from .trajectory_markdown import write_trajectory_markdown


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
DEFAULT_STRATEGIES = ("direct_full_video", "agent_v2")
STRATEGIES = ("direct_full_video", "empty_index_loop", "subtitle_index_loop", "agent_v2")
WINDOW_SEC = 300.0
DIRECT_NFRAMES = 64
SEGMENT_NFRAMES = 8
MAX_PIXELS = 151200
FRAME_CACHE_FPS = 2.0


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
    window_sec: float = WINDOW_SEC
    scene_index_mode: str = "dual-source"
    scene_index_cache_dir: Path = DEFAULT_SCENE_INDEX_CACHE_DIR
    scene_index_cache_enabled: bool = True
    scene_caption_nframes: int = SEGMENT_NFRAMES
    frame_cache_fps: float = FRAME_CACHE_FPS
    frame_cache_root: Path | None = None
    budget: AgentBudget = AgentBudget()
    export_training: bool = False
    ablation_flags: Mapping[str, Any] | None = None


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
        "VideoMME multiple-choice question. Answer with exactly one option letter (A/B/C/D) first, "
        "then one short evidence-based reason.\n"
        "Do not use outside knowledge unless it is directly supported by the video evidence.\n"
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


def parse_srt(path: Path) -> list[tuple[float, str]]:
    return [(cue.start_sec, cue.text) for cue in parse_srt_cues(path)]


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


def subtitle_scene_index(
    *,
    video_path: str,
    video_id: str,
    duration_sec: float,
    subtitle_dir: Path,
    window_sec: float = WINDOW_SEC,
) -> SceneIndex:
    base = fixed_window_scene_index(
        video_path=video_path,
        duration_sec=duration_sec,
        window_sec=window_sec,
        source="fixed_window_subtitle",
    )
    buckets = [[] for _ in base.segments]
    for start_sec, text in parse_srt(subtitle_dir / f"{video_id}.srt"):
        idx = min(int(start_sec // window_sec), len(buckets) - 1)
        if idx >= 0:
            buckets[idx].append(text)
    enriched = []
    for segment, texts in zip(base.segments, buckets):
        excerpt = compact_text(" ".join(texts), limit=720)
        caption = f"ASR/subtitle excerpt: {excerpt}" if excerpt else ""
        enriched.append(
            VideoSegment(
                segment_id=segment.segment_id,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                low_fps_caption=caption,
                source="fixed_window_subtitle",
            )
        )
    return SceneIndex(video_path=video_path, duration_sec=duration_sec, segments=enriched)


def direct_answer(backend: Any, *, video_path: str, question: str, duration_sec: float) -> dict[str, Any]:
    start = time.perf_counter()
    response = backend.generate(
        BackendRequest(
            task="videomme_direct_qa",
            prompt=(
                "Answer the multiple-choice question directly from the sampled full-video context. "
                "Start with exactly one option letter. Mention uncertainty if the sampled context is insufficient.\n"
                f"Video duration: {duration_sec:.1f} seconds.\n{question}"
            ),
            media_path=video_path,
            media_type="video",
            max_new_tokens=256,
            metadata={"nframes": DIRECT_NFRAMES, "max_pixels": MAX_PIXELS},
        )
    )
    seconds = time.perf_counter() - start
    return {"answer": response.text.strip(), "choice": extract_choice(response.text), "seconds": round(seconds, 3), "status": "ok"}


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
) -> dict[str, Any]:
    start = time.perf_counter()
    result = run_iterative_smoke(
        base_dir=workspace_root,
        backend=backend,
        media_path=video_path,
        question=question,
        duration_sec=duration_sec,
        window_sec=WINDOW_SEC,
        run_id=run_id,
        scene_index=scene_index,
        budget=budget,
        extract_clips=extract_clips,
        frame_sampler=frame_sampler,
    )
    seconds = time.perf_counter() - start
    workspace = EvidenceWorkspace(root=workspace_root / "runs" / run_id)
    reward_tags = _reward_tags_for_result(workspace=workspace, status=result.status, citations=result.citations)
    trajectory_payload = workspace.export_longvideoagent_trajectory(
        question=question,
        video_path=video_path,
        final={
            "answer": result.answer,
            "status": result.status,
            "citations": list(result.citations),
            "confidence": result.confidence,
        },
        verifier_result={"status": result.status},
        reward_tags=reward_tags,
    )
    evidence_chains_payload = workspace.export_evidence_chains()
    trajectory_path = workspace.root / "artifacts" / "trajectories" / "longvideoagent_trajectory.json"
    evidence_chains_path = workspace.root / "artifacts" / "evidence_chains" / "evidence_chains.json"
    planner_io_dir = workspace.root / "artifacts" / "planner_io"
    tools = []
    segments = []
    for round_item in result.rounds:
        for step in round_item.program:
            tools.append(str(step.get("tool", "")))
            args = step.get("args", {}) if isinstance(step.get("args", {}), Mapping) else {}
            if args.get("segment_id"):
                segments.append(str(args["segment_id"]))
    return {
        "answer": result.answer,
        "choice": extract_choice(result.answer),
        "status": result.status,
        "confidence": result.confidence,
        "citations": list(result.citations),
        "rounds": len(result.rounds),
        "tools": tools,
        "segments": segments,
        "seconds": round(seconds, 3),
        "trajectory_path": str(trajectory_path),
        "trajectory_action_count": len(trajectory_payload.get("actions", [])),
        "evidence_chains_path": str(evidence_chains_path),
        "evidence_chain_count": int(evidence_chains_payload.get("chain_count", 0) or 0),
        "planner_io_dir": str(planner_io_dir),
        "planner_prompt_count": len(list(planner_io_dir.glob("*_prompt.txt"))) if planner_io_dir.exists() else 0,
        "reward_tags": reward_tags,
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
        "planner_io_dir",
        "planner_prompt_count",
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
        "cases": list(config.cases),
        "strategies": list(config.strategies),
        "window_sec": config.window_sec,
        "budget": asdict(config.budget),
        "model_path": config.model_path,
        "planner_model_path": config.planner_model_path,
        "scene_index_mode": config.scene_index_mode,
        "scene_index_cache_dir": str(config.scene_index_cache_dir),
        "scene_index_cache_enabled": config.scene_index_cache_enabled,
        "scene_caption_nframes": config.scene_caption_nframes,
        "frame_cache_fps": config.frame_cache_fps,
        "frame_cache_root": str(_frame_cache_root(config)),
        "export_training": config.export_training,
        "ablation_flags": dict(config.ablation_flags or {}),
    }
    (config.run_root / "run_config.json").write_text(
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
        if _uses_frame_cache(config.strategies):
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
                if strategy != "direct_full_video":
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
    if strategy == "direct_full_video":
        return direct_answer(backend, video_path=video_path, question=question, duration_sec=duration_sec)

    if strategy == "empty_index_loop":
        scene_index = fixed_window_scene_index(
            video_path=video_path,
            duration_sec=duration_sec,
            window_sec=config.window_sec,
            source="fixed_window_empty",
        )
    elif config.scene_index_mode == "dual-source":
        cache = SceneIndexCache(config.scene_index_cache_dir) if config.scene_index_cache_enabled else None
        builder = SceneIndexBuilder(
            backend=backend,
            text_model_id=config.planner_model_path or config.model_path,
            vl_model_id=config.model_path,
            window_sec=config.window_sec,
            caption_nframes=config.scene_caption_nframes,
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
    else:
        scene_index = subtitle_scene_index(
            video_path=video_path,
            video_id=video_id,
            duration_sec=duration_sec,
            subtitle_dir=config.subtitle_dir,
            window_sec=config.window_sec,
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
    )


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _uses_frame_cache(strategies: Sequence[str]) -> bool:
    return any(strategy != "direct_full_video" for strategy in strategies)


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible VideoMME strategy evaluations.")
    parser.add_argument("--strategy", action="append", help="Strategy to run. Repeat or pass comma-separated values.")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="Comma-separated VideoMME question ids.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--planner-model-path", default=PLANNER_MODEL_PATH)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--subtitle-dir", type=Path, default=DEFAULT_SUBTITLE_DIR)
    parser.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    parser.add_argument("--scene-index-mode", choices=("subtitle", "dual-source"), default="dual-source")
    parser.add_argument("--scene-index-cache-dir", type=Path, default=DEFAULT_SCENE_INDEX_CACHE_DIR)
    parser.add_argument("--no-scene-index-cache", action="store_true")
    parser.add_argument("--scene-caption-nframes", type=int, default=SEGMENT_NFRAMES)
    parser.add_argument("--frame-cache-root", type=Path, default=None)
    parser.add_argument("--frame-cache-fps", type=float, default=FRAME_CACHE_FPS)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--max-tool-calls-per-round", type=int, default=2)
    parser.add_argument("--default-nframes", type=int, default=SEGMENT_NFRAMES)
    parser.add_argument("--contract-nframes", type=int, default=None)
    parser.add_argument("--high-fps-nframes", type=int, default=32)
    parser.add_argument("--context-budget-tokens", type=int, default=12000)
    parser.add_argument(
        "--budget-ratios",
        default=None,
        help="Comma-separated slot ratios, e.g. task:0.1,navigation:0.15,evidence:0.5,feedback:0.25",
    )
    parser.add_argument("--planner-receives-media", action="store_true")
    parser.add_argument("--no-reserve-final-round", action="store_true")
    parser.add_argument("--cheap-tool-budget", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--expensive-tool-budget", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--verifier-tool-budget", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument(
        "--hard-skill-runtime",
        action="store_true",
        help="Use deterministic skill runtime for supported routes before falling back to planner loop.",
    )
    parser.add_argument(
        "--disable-global-gist-route",
        action="store_true",
        help="Skip the automatic gist_global shortcut so debugging runs capture planner-loop IO.",
    )
    parser.add_argument(
        "--use-global-question-rewrite",
        dest="use_global_question_rewrite",
        action="store_true",
        default=False,
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
        help="Legacy alias: use the free max-round/tool-call caps and disable the reserved final round.",
    )
    parser.add_argument("--free-max-rounds", type=int, default=24)
    parser.add_argument("--free-max-tool-calls-per-round", type=int, default=4)
    parser.add_argument("--export-training", action="store_true", help="Export compact TrainingTrajectory JSON per case.")
    parser.add_argument("--allow-any-python", action="store_true", help="Skip the remote Python executable assertion.")
    return parser


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    workspace_root = args.workspace_root or (args.run_root / "workspaces")
    strategies = parse_strategies(args.strategy)
    context_budget_ratios = parse_budget_ratios(args.budget_ratios) if args.budget_ratios else None
    default_nframes = args.contract_nframes if args.contract_nframes is not None else args.default_nframes
    context_budget_tokens = args.context_budget_tokens
    if args.enable_context_budget is False:
        context_budget_tokens = 10**9
    max_rounds = int(args.free_max_rounds if args.free_explore else args.max_rounds)
    max_tool_calls = int(args.free_max_tool_calls_per_round if args.free_explore else args.max_tool_calls_per_round)
    reserve_final_round = False if args.free_explore else not args.no_reserve_final_round
    budget = AgentBudget(
        max_rounds=max_rounds,
        max_tool_calls_per_round=max_tool_calls,
        default_nframes=default_nframes,
        high_fps_nframes=args.high_fps_nframes,
        context_budget_tokens=context_budget_tokens,
        context_budget_ratios=context_budget_ratios,
        planner_receives_media=args.planner_receives_media,
        reserve_final_round=reserve_final_round,
        max_repeated_programs=max(max_rounds, AgentBudget().max_repeated_programs),
        disable_global_gist_route=args.disable_global_gist_route,
        rewrite_mcq_for_exploration=bool(args.use_global_question_rewrite),
        hard_skill_runtime="agent_v2" in strategies,
        planner_owned_grounding=(
            bool(args.planner_owned_grounding)
            if args.planner_owned_grounding is not None
            else "agent_v2" in strategies
        ),
    )
    if args.enable_followup is False:
        budget = replace(budget, hard_skill_runtime=False)
    elif args.hard_skill_runtime or args.enable_followup is True:
        budget = replace(budget, hard_skill_runtime=True)
    ablation_flags = {
        "enable_query_context": args.enable_query_context,
        "enable_followup": args.enable_followup,
        "enable_context_budget": args.enable_context_budget,
        "enable_map_reflux": args.enable_map_reflux,
        "enable_evidence_staging": args.enable_evidence_staging,
        "planner_owned_grounding": budget.planner_owned_grounding,
        "enable_mcq_rewrite": bool(args.use_global_question_rewrite),
        "contract_nframes": args.contract_nframes,
        "followup_budget": args.followup_budget,
    }
    return EvalConfig(
        run_root=args.run_root,
        workspace_root=workspace_root,
        model_path=args.model_path,
        planner_model_path=args.planner_model_path,
        data_root=args.data_root,
        parquet_path=args.parquet_path,
        video_dir=args.video_dir,
        subtitle_dir=args.subtitle_dir,
        cases=parse_csv(args.cases),
        strategies=strategies,
        window_sec=args.window_sec,
        scene_index_mode=args.scene_index_mode,
        scene_index_cache_dir=args.scene_index_cache_dir,
        scene_index_cache_enabled=not args.no_scene_index_cache,
        scene_caption_nframes=args.scene_caption_nframes,
        frame_cache_fps=args.frame_cache_fps,
        frame_cache_root=args.frame_cache_root,
        budget=budget,
        export_training=args.export_training,
        ablation_flags=ablation_flags,
    )


def build_backend(config: EvalConfig) -> Any:
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
