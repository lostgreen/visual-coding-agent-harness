#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

from vcah.interactive_agents import VisionInvestigator, WorkspaceReasoner
from vcah.model_client import OpenAICompatibleClient
from vcah.multiround import RUN_ARTIFACT_NAMES, VirtualVideoMultiRoundDriver
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
from vcah.video import probe_duration
from vcah.virtual_index import build_virtual_beat_index
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
    load_srt_as_virtual_cues,
    materialize_lowfps_frame_cache,
)
from vcah.videomme_v2 import score_videomme_v2_answer


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


def main() -> None:
    args = _parse_args()
    seeds = tuple(dict.fromkeys(int(seed) for seed in (args.seeds or (args.seed,))))
    if not seeds:
        raise ValueError("Provide at least one seed")
    reasoner_api, investigator_api = load_role_clients(
        shared_config=args.config,
        reasoner_config=args.reasoner_config,
        investigator_config=args.investigator_config,
    )
    case_group = _load_case_group(Path(args.case_group)) if args.case_group else None
    case_ids = tuple(case_group["case_ids"]) if case_group else tuple(args.case_ids or DEFAULT_CASE_IDS)
    if case_group and args.construction == "single_segment":
        args.construction = str(case_group["construction"])
    if args.mode == "long":
        if not case_group:
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
    dataset_root = Path(args.dataset_root)

    def run_one(case_id: str, seed: int) -> Mapping[str, Any]:
        reasoner_api.set_requested_seed(seed)
        investigator_api.set_requested_seed(seed)
        workspace_root = run.root / "workspaces" / case_id
        if len(seeds) > 1:
            workspace_root /= f"seed-{seed}"
        workspace = build_workspace(
            dataset_root,
            workspace_root,
            case_id=case_id,
            seed=seed,
            min_duration_sec=float(args.min_duration_sec),
            max_duration_sec=None if args.max_duration_sec is None else float(args.max_duration_sec),
            segment_sec=float(args.segment_sec),
            construction=str(args.construction),
            chunk_sec=float(args.chunk_sec),
        )
        ensure_index(
            workspace,
            low_fps=float(args.low_fps),
            beat_sec=float(args.beat_sec),
            rebuild=False,
        )
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
            "selected_option": result.selected_option,
            "citations": list(result.citations),
            "correct": score_videomme_v2_answer(result.answer, workspace.case.gold),
            "reference_valid": result.reference_valid,
            "reference_reason": result.reference_reason,
            "rounds": result.rounds,
            "investigation_count": result.investigation_count,
            "workspace": str(workspace.root_dir),
            "trace": str(workspace.root_dir / "interactions.jsonl"),
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
        "multi_seed_report": aggregate_seed_results([dict(item.get("replay", {}) or {}) for item in summaries]),
    }
    summary_path = write_immutable_summary(run, payload)
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "summary": str(summary_path),
                "case_count": len(summaries),
                "correct": payload["correct"],
                "seeds": list(seeds),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _replay_run_config(
    args: argparse.Namespace,
    *,
    selected: Sequence[str],
    seeds: Sequence[int],
    reasoner_api: OpenAICompatibleClient,
    investigator_api: OpenAICompatibleClient,
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
            key: file_checksum(Path(value)) for key, value in source_paths.items() if value
        },
    }


def build_workspace(
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
) -> VirtualVideoWorkspace:
    rows = _load_rows(dataset_root)
    by_qid = {str(row["question_id"]): row for row in rows}
    target = by_qid[str(case_id)]
    rng = random.Random(seed + sum(ord(char) for char in str(case_id)))
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
        segments = _build_segments(
            dataset_root,
            rows,
            target,
            rng=rng,
            min_duration_sec=min_duration_sec,
            segment_sec=segment_sec,
        )
    target_segments = tuple(segment for segment in segments if segment.role == "target")
    case = VirtualVideoCase(
        case_id=str(case_id),
        question=str(target["question"]),
        options=_options_mapping(target["options"]),
        gold=str(target["answer"]),
        target_segment_id=target_segments[0].segment_id,
        target_virtual_interval=(target_segments[0].virtual_start_sec, target_segments[-1].virtual_end_sec),
        metadata={
            "source_video_id": str(target["videoID"]),
            "min_duration_sec": min_duration_sec,
            "max_duration_sec": max_duration_sec,
            "seed": seed,
            "construction": construction,
            "target_segment_ids": [segment.segment_id for segment in target_segments],
            "target_virtual_intervals": [
                [segment.virtual_start_sec, segment.virtual_end_sec] for segment in target_segments
            ],
        },
    )
    workspace = VirtualVideoWorkspace.create(
        root_dir,
        manifest=VirtualVideoManifest(workspace_id=str(case_id), segments=tuple(segments)),
        case=case,
    )
    cues = []
    for segment in segments:
        cues.extend(
            load_srt_as_virtual_cues(
                dataset_root / "subtitle" / f"{segment.source_video_id}.srt",
                segment,
            )
        )
    workspace.write_asr_virtual_cues(tuple(cues))
    return workspace


def ensure_index(
    workspace: VirtualVideoWorkspace,
    *,
    low_fps: float,
    beat_sec: float,
    rebuild: bool,
) -> None:
    if (workspace.root_dir / "beat_index.json").exists() and workspace.frame_manifest.exists() and not rebuild:
        return
    frames = materialize_lowfps_frame_cache(workspace, fps=low_fps)
    build_virtual_beat_index(workspace, frames, beat_sec=beat_sec)


def run_case(
    workspace: VirtualVideoWorkspace,
    *,
    reasoner_api: OpenAICompatibleClient | None = None,
    investigator_api: OpenAICompatibleClient | None = None,
    max_rounds: int,
    max_investigations: int,
) -> Any:
    if reasoner_api is None or investigator_api is None:
        raise ValueError("run_case requires both reasoner_api and investigator_api")
    artifact_names = ("interactions.jsonl", *RUN_ARTIFACT_NAMES)
    existing = tuple(name for name in artifact_names if (workspace.root_dir / name).exists())
    if existing:
        raise FileExistsError(f"workspace already contains run artifacts: {', '.join(existing)}")
    trace_path = workspace.root_dir / "interactions.jsonl"
    trace_path.touch(exist_ok=False)
    driver = VirtualVideoMultiRoundDriver(
        reasoner=WorkspaceReasoner(reasoner_api, trace_path=trace_path),
        investigator=VisionInvestigator(workspace, api=investigator_api, trace_path=trace_path),
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
) -> tuple[OpenAICompatibleClient, OpenAICompatibleClient]:
    reasoner_value = reasoner_config or shared_config
    investigator_value = investigator_config or shared_config
    if not reasoner_value or not investigator_value:
        raise ValueError("Provide --config or both --reasoner-config and --investigator-config")
    return (
        OpenAICompatibleClient.from_yaml(Path(reasoner_value), section="reasoner_api"),
        OpenAICompatibleClient.from_yaml(Path(investigator_value), section="investigator_api"),
    )


def _write_model_roles(
    workspace: VirtualVideoWorkspace,
    *,
    reasoner_api: OpenAICompatibleClient,
    investigator_api: OpenAICompatibleClient,
) -> None:
    path = workspace.root_dir / "run_summary.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["models"] = {"reasoner": reasoner_api.model, "investigator": investigator_api.model}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


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
    specs = [
        {
            "role": "target",
            "video_id": target_video,
            "start": 0.0,
            "end": min(_duration(dataset_root, target_video), segment_sec),
        }
    ]
    pool = _long_video_pool(dataset_root, rows, exclude={target_video}, min_duration_sec=600.0)
    rng.shuffle(pool)
    total = float(specs[0]["end"])
    for video_id, duration in pool:
        length = min(float(segment_sec), duration)
        start = 0.0 if duration <= length else rng.uniform(0.0, duration - length)
        specs.append({"role": "distractor", "video_id": video_id, "start": start, "end": start + length})
        total += length
        if total >= min_duration_sec:
            break
    rng.shuffle(specs)
    if total < min_duration_sec:
        raise RuntimeError(f"Only built {total:.1f}s, below requested {min_duration_sec:.1f}s")
    return _segments_from_specs(dataset_root, specs)


def _build_source_only_segments(
    dataset_root: Path,
    target: Mapping[str, Any],
    *,
    chunk_sec: float,
) -> tuple[VirtualVideoSegment, ...]:
    video_id = str(target["videoID"])
    duration = _duration(dataset_root, video_id)
    chunk = max(1.0, float(chunk_sec))
    specs = []
    start = 0.0
    while start < duration:
        end = min(duration, start + chunk)
        specs.append({"role": "target", "video_id": video_id, "start": start, "end": end})
        start = end
    return _segments_from_specs(dataset_root, specs)


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
    specs = list(_video_chunks("target", target_video, _duration(dataset_root, target_video), chunk_width=chunk_width))
    pool = _long_video_pool(dataset_root, rows, exclude={target_video}, min_duration_sec=chunk_width)
    rng.shuffle(pool)
    total = sum(float(spec["end"]) - float(spec["start"]) for spec in specs)
    for video_id, duration in pool:
        chunks = list(_video_chunks("distractor", video_id, duration, chunk_width=chunk_width))
        rng.shuffle(chunks)
        for spec in chunks:
            if max_duration_sec is not None and total >= max_duration_sec:
                break
            specs.append(spec)
            total += float(spec["end"]) - float(spec["start"])
            if total >= min_duration_sec:
                break
        if total >= min_duration_sec:
            break
    if total < min_duration_sec:
        raise RuntimeError(f"Only built {total:.1f}s, below requested {min_duration_sec:.1f}s")
    return _segments_from_specs(dataset_root, _interleave_specs(specs, rng=rng))


def _segments_from_specs(
    dataset_root: Path,
    specs: Sequence[Mapping[str, Any]],
) -> tuple[VirtualVideoSegment, ...]:
    segments = []
    cursor = 0.0
    for index, spec in enumerate(specs, start=1):
        duration = float(spec["end"]) - float(spec["start"])
        segments.append(
            VirtualVideoSegment(
                segment_id=f"seg_{index:04d}",
                source_video_id=str(spec["video_id"]),
                source_path=str(dataset_root / "video" / f"{spec['video_id']}.mp4"),
                source_start_sec=round(float(spec["start"]), 3),
                source_end_sec=round(float(spec["end"]), 3),
                virtual_start_sec=round(cursor, 3),
                virtual_end_sec=round(cursor + duration, 3),
                role=str(spec["role"]),
            )
        )
        cursor += duration
    return tuple(segments)


def _video_chunks(
    role: str,
    video_id: str,
    duration_sec: float,
    *,
    chunk_width: float,
) -> tuple[dict[str, Any], ...]:
    chunks = []
    start = 0.0
    while start < duration_sec:
        end = min(duration_sec, start + chunk_width)
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
        video_id = str(row["videoID"])
        if video_id in seen or str(row.get("duration")) != "long":
            continue
        if (dataset_root / "video" / f"{video_id}.mp4").exists():
            duration = _duration(dataset_root, video_id)
            if duration >= min_duration_sec:
                pool.append((video_id, duration))
                seen.add(video_id)
    return pool


def _interleave_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    rng: random.Random,
) -> list[Mapping[str, Any]]:
    items = list(specs)
    rng.shuffle(items)
    for _ in range(4):
        changed = False
        for index in range(1, len(items)):
            if items[index]["video_id"] != items[index - 1]["video_id"]:
                continue
            swap = next(
                (position for position in range(index + 1, len(items)) if items[position]["video_id"] != items[index]["video_id"]),
                None,
            )
            if swap is not None:
                items[index], items[swap] = items[swap], items[index]
                changed = True
        if not changed:
            break
    return items


def _duration(dataset_root: Path, video_id: str) -> float:
    return probe_duration(str(dataset_root / "video" / f"{video_id}.mp4"))


def _options_mapping(value: Any) -> dict[str, str]:
    labels = "ABCDEFGH"
    values = list(value) if not isinstance(value, str) else [part.strip() for part in value.split("|")]
    result = {}
    for index, item in enumerate(values):
        text = str(item).strip()
        label = labels[index]
        if len(text) > 2 and text[0].upper() in labels and text[1] == ".":
            label, text = text[0].upper(), text[2:].strip()
        result[label] = text
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the workspace-based Reasoner/Investigator Video-MME evaluation")
    parser.add_argument(
        "--dataset-root",
        default=(
            "/ytech_m2v5_hdd/workspace/kling_mm/Datasets/VLMEvalKit_Dataset_Cache/HFCache/"
            "datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b"
        ),
    )
    parser.add_argument(
        "--out-root",
        default="/m2v_intern/xuboshen/zgw/VideoAgent/virtual_videomme_interactive/runs",
        help="Parent directory for create-exclusive replay runs",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--config", help="Shared API config used for both roles")
    parser.add_argument("--reasoner-config", help="Text Reasoner API config")
    parser.add_argument("--investigator-config", help="Multimodal Investigator API config")
    cases = parser.add_mutually_exclusive_group()
    cases.add_argument("--case-ids", nargs="*")
    cases.add_argument("--case-group")
    parser.add_argument("--mode", choices=("smoke", "all", "long"), default="smoke")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--seeds", type=int, nargs="+")
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
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
