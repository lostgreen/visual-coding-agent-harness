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
from visual_coding_agent_harness.backends.base import BackendRequest
from visual_coding_agent_harness.iterative_smoke import run_iterative_smoke
from visual_coding_agent_harness.video_index import SceneIndex, VideoSegment, fixed_window_scene_index
from visual_coding_agent_harness.workspace import EvidenceWorkspace

try:
    from runs.summary_schema import RunSummary, validate as validate_run_summary
except ModuleNotFoundError:
    from summary_schema import RunSummary, validate as validate_run_summary


REMOTE_PYTHON = "/home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python"
MODEL_PATH = "/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct"
DATA_ROOT = Path(
    "/ytech_m2v5_hdd/workspace/kling_mm/Datasets/VLMEvalKit_Dataset_Cache/HFCache/"
    "datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b"
)
DEFAULT_PARQUET_PATH = DATA_ROOT / "videomme/test-00000-of-00001.parquet"
DEFAULT_VIDEO_DIR = DATA_ROOT / "video"
DEFAULT_SUBTITLE_DIR = DATA_ROOT / "subtitle"
DEFAULT_RUN_ROOT = Path("runs/videomme_agent_eval")
DEFAULT_CASES = ("605-1", "611-2", "612-1")
DEFAULT_STRATEGIES = ("direct_full_video", "agent_v2")
STRATEGIES = ("direct_full_video", "empty_index_loop", "subtitle_index_loop", "agent_v2")
WINDOW_SEC = 300.0
DIRECT_NFRAMES = 64
SEGMENT_NFRAMES = 8
MAX_PIXELS = 151200


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
    window_sec: float = WINDOW_SEC
    budget: AgentBudget = AgentBudget()


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
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    cues: list[tuple[float, str]] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        time_index = next((idx for idx, line in enumerate(lines) if "-->" in line), None)
        if time_index is None:
            continue
        start = lines[time_index].split("-->")[0].strip()
        body = " ".join(lines[time_index + 1 :])
        cleaned = clean_subtitle_text(body)
        if cleaned:
            cues.append((parse_time(start), cleaned))
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
    }
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
                )
                case["strategies"][strategy] = summarize_strategy(raw, gt)
                if strategy != "direct_full_video":
                    case["raw_artifacts"]["workspaces"][strategy] = str(
                        config.workspace_root / "runs" / f"{case_prefix}_{strategy}"
                    )
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

    for workspace in workspaces:
        events = _load_trace_events(workspace)
        route_violations += sum(1 for event in events if _event_type(event) == "route_violation")
        attempts, success = _hard_skill_followup_trace_metrics(events)
        followup_attempts.append(attempts)
        if success:
            followup_successes += 1

    summary.route_violations = route_violations
    if followup_attempts:
        summary.avg_followups_per_case = sum(followup_attempts) / len(followup_attempts)
        attempted_cases = sum(1 for attempts in followup_attempts if attempts > 0)
        summary.followup_success_rate = (followup_successes / attempted_cases) if attempted_cases else 0.0


def _evidence_provenance_completeness(workspaces: Sequence[EvidenceWorkspace]) -> float:
    if not workspaces:
        return 0.0
    complete = sum(1 for workspace in workspaces if workspace.evidence_chain_summaries(max_chains=1))
    return complete / len(workspaces)


def _hard_skill_followup_trace_metrics(events: Sequence[Mapping[str, Any]]) -> tuple[int, bool]:
    in_hard_skill = False
    attempts = 0
    success = False
    for event in events:
        event_type = _event_type(event)
        payload = _event_payload(event)
        if event_type == "hard_skill_runtime":
            in_hard_skill = True
            continue
        if event_type == "tool_use" and in_hard_skill and str(payload.get("tool", "")) == "ground_question":
            attempts += 1
            continue
        if event_type == "iterative_final" and str(payload.get("source", "")) == "hard_skill_runtime":
            success = True
            in_hard_skill = False
            continue
        if event_type == "hard_skill_followup_handoff":
            in_hard_skill = False
    return attempts, success


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
    )


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


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
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--subtitle-dir", type=Path, default=DEFAULT_SUBTITLE_DIR)
    parser.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--max-tool-calls-per-round", type=int, default=2)
    parser.add_argument("--default-nframes", type=int, default=SEGMENT_NFRAMES)
    parser.add_argument("--high-fps-nframes", type=int, default=32)
    parser.add_argument("--planner-receives-media", action="store_true")
    parser.add_argument("--no-reserve-final-round", action="store_true")
    parser.add_argument("--cheap-tool-budget", type=int, default=16)
    parser.add_argument("--expensive-tool-budget", type=int, default=6)
    parser.add_argument("--verifier-tool-budget", type=int, default=2)
    parser.add_argument(
        "--hard-skill-runtime",
        action="store_true",
        help="Use deterministic skill runtime for supported routes before falling back to planner loop.",
    )
    parser.add_argument(
        "--free-explore",
        action="store_true",
        help="Disable per-class and reserved-final policy budgets; keep only emergency safety caps.",
    )
    parser.add_argument("--free-max-rounds", type=int, default=24)
    parser.add_argument("--free-max-tool-calls-per-round", type=int, default=4)
    parser.add_argument("--allow-any-python", action="store_true", help="Skip the remote Python executable assertion.")
    return parser


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    workspace_root = args.workspace_root or (args.run_root / "workspaces")
    budget = (
        AgentBudget.free_explore(
            max_rounds=args.free_max_rounds,
            max_tool_calls_per_round=args.free_max_tool_calls_per_round,
        )
        if args.free_explore
        else AgentBudget(
            max_rounds=args.max_rounds,
            max_tool_calls_per_round=args.max_tool_calls_per_round,
            default_nframes=args.default_nframes,
            high_fps_nframes=args.high_fps_nframes,
            planner_receives_media=args.planner_receives_media,
            reserve_final_round=not args.no_reserve_final_round,
            cheap_tool_budget=args.cheap_tool_budget,
            expensive_tool_budget=args.expensive_tool_budget,
            verifier_tool_budget=args.verifier_tool_budget,
        )
    )
    if args.hard_skill_runtime:
        budget = replace(budget, hard_skill_runtime=True)
    return EvalConfig(
        run_root=args.run_root,
        workspace_root=workspace_root,
        model_path=args.model_path,
        data_root=args.data_root,
        parquet_path=args.parquet_path,
        video_dir=args.video_dir,
        subtitle_dir=args.subtitle_dir,
        cases=parse_csv(args.cases),
        strategies=parse_strategies(args.strategy),
        window_sec=args.window_sec,
        budget=budget,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    validate_python(allow_any_python=args.allow_any_python)
    config = config_from_args(args)
    rows_by_id = load_rows_by_id(config.parquet_path, config.cases)
    from visual_coding_agent_harness.backends.qwen_vl import QwenVLBackend

    backend = QwenVLBackend.from_pretrained(config.model_path)
    run_eval_cases(backend=backend, rows_by_id=rows_by_id, config=config)


if __name__ == "__main__":
    main()
