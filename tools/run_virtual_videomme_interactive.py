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

from vcah.investigator import InvestigationEvidence, InvestigationReport, VirtualVideoInvestigator
from vcah.multiround import InvestigationTask, ReasonerDecision, VirtualVideoMultiRoundDriver
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


DEFAULT_CASE_IDS = ("477-2", "548-1", "371-1", "311-1", "314-3", "315-1")


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    api = OpenAICompatibleVisionClient.from_yaml(Path(args.config))
    case_ids = tuple(args.case_ids or DEFAULT_CASE_IDS)
    selected = case_ids[:1] if args.mode == "smoke" else case_ids
    summaries = []
    for case_id in selected:
        workspace = build_or_load_workspace(
            dataset_root,
            out_root / "workspaces" / case_id,
            case_id=case_id,
            seed=int(args.seed),
            min_duration_sec=float(args.min_duration_sec),
            segment_sec=float(args.segment_sec),
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
    segment_sec: float,
    rebuild: bool,
) -> VirtualVideoWorkspace:
    if root_dir.exists() and (root_dir / "case.json").exists() and not rebuild:
        return VirtualVideoWorkspace.load(root_dir)
    rows = _load_rows(dataset_root)
    by_qid = {str(row["question_id"]): row for row in rows}
    target = by_qid[str(case_id)]
    rng = random.Random(seed + sum(ord(ch) for ch in str(case_id)))
    segments = _build_segments(dataset_root, rows, target, rng=rng, min_duration_sec=min_duration_sec, segment_sec=segment_sec)
    manifest = VirtualVideoManifest(workspace_id=str(case_id), segments=tuple(segments))
    target_segment = next(segment for segment in segments if segment.role == "target")
    case = VirtualVideoCase(
        case_id=str(case_id),
        question=str(target["question"]),
        options=_options_mapping(target["options"]),
        gold=str(target["answer"]),
        target_segment_id=target_segment.segment_id,
        target_virtual_interval=(target_segment.virtual_start_sec, target_segment.virtual_end_sec),
        metadata={"source_video_id": str(target["videoID"]), "min_duration_sec": min_duration_sec, "seed": seed},
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
            prompt = _answer_prompt(kwargs, evidence_digest)
            raw = self.api.chat(prompt)
            parsed = _parse_json(raw)
            self._trace("reasoner_answer", prompt, raw, parsed)
            return ReasonerDecision(
                action="answer",
                answer=str(parsed.get("answer") or ""),
                citations=tuple(parsed.get("citations") or (evidence_digest[-1]["evidence_id"],)),
            )
        image_paths = [str(row.get("overview_thumbnail_grid_path")) for row in overview.get("segment_overviews", ())]
        prompt = _investigate_prompt(kwargs)
        raw = self.api.chat(prompt, image_paths=image_paths)
        parsed = _parse_json(raw)
        self._trace("reasoner_investigate", prompt, raw, parsed)
        tasks = parsed.get("tasks") or []
        return ReasonerDecision(action="investigate", tasks=tuple(tasks[:4]))

    def _trace(self, kind: str, prompt: str, raw: str, parsed: Mapping[str, Any]) -> None:
        _append_jsonl(self.trace_path, {"type": kind, "prompt": prompt, "raw": raw, "parsed": dict(parsed), "time": time.time()})


class GeminiInvestigator(VirtualVideoInvestigator):
    def __init__(self, workspace: VirtualVideoWorkspace, *, api: OpenAICompatibleVisionClient, trace_path: Path) -> None:
        super().__init__(workspace)
        self.api = api
        self.trace_path = trace_path

    def _investigate_task(self, task: Any) -> InvestigationReport:
        segment_packet = self.open_segment(str(getattr(task, "segment_id", "") or self.workspace.manifest.segments[0].segment_id))
        window = _select_window_with_model(self.api, task, segment_packet, self.trace_path)
        low = self.inspect_window(window[0], window[1], fps=0.5, max_frames=64, query_id=f"{getattr(task, 'query_id')}_preview")
        high = self.inspect_window(window[0], window[1], fps=2.0, max_frames=64, query_id=str(getattr(task, "query_id")))
        prompt = _evidence_prompt(self.workspace, task, segment_packet, high)
        frame_paths = [str(row["path"]) for row in high["frames"][:16]]
        raw = self.api.chat(prompt, image_paths=frame_paths, max_tokens=700)
        parsed = _parse_json(raw)
        _append_jsonl(self.trace_path, {"type": "investigator_evidence", "query_id": getattr(task, "query_id"), "window": list(window), "raw": raw, "parsed": parsed, "time": time.time()})
        evidence = InvestigationEvidence(
            evidence_id=f"ev_{getattr(task, 'query_id')}_001",
            summary=str(parsed.get("summary") or raw)[:1200],
            modality="visual",
            sampling=dict(high["sampling"]),
            virtual_time_range=(float(window[0]), float(window[1])),
            source_lineage=tuple(dict(item) for item in high["source_lineage"]),
            supporting_frames=tuple(frame_paths),
            confidence=float(parsed.get("confidence", 0.6) or 0.6),
        )
        return InvestigationReport(
            query_id=str(getattr(task, "query_id")),
            status="satisfied",
            evidence=(evidence,),
            cost={"tool_trace": ("open_segment", "inspect_window:0.5", "inspect_window:2.0"), "frames": len(frame_paths), "vlm_calls": 2},
        )


def _select_window_with_model(api: OpenAICompatibleVisionClient, task: Any, segment_packet: Mapping[str, Any], trace_path: Path) -> tuple[float, float]:
    prompt = (
        "You are an Investigator. Choose the most relevant virtual-time window inside this segment.\n"
        "Return JSON only: {\"start_sec\": float, \"end_sec\": float, \"reason\": string}.\n"
        f"Task: {getattr(task, 'goal', '')}\nExpected evidence: {getattr(task, 'expected_evidence', '')}\n"
        f"Segment packet (text only): {json.dumps(_compact_segment_packet(segment_packet), ensure_ascii=False)}"
    )
    image_paths = []
    for beat in segment_packet.get("beats", ())[:12]:
        image_paths.extend(str(path) for path in beat.get("thumbnail_grid_paths", ())[:1])
    raw = api.chat(prompt, image_paths=image_paths, max_tokens=400)
    parsed = _parse_json(raw)
    _append_jsonl(trace_path, {"type": "investigator_select_window", "query_id": getattr(task, "query_id"), "raw": raw, "parsed": parsed, "time": time.time()})
    seg_start, seg_end = segment_packet["virtual_time_range"]
    start = max(float(seg_start), float(parsed.get("start_sec", seg_start) or seg_start))
    end = min(float(seg_end), float(parsed.get("end_sec", start + 60.0) or start + 60.0))
    if end <= start:
        end = min(float(seg_end), start + 60.0)
    return round(start, 3), round(end, 3)


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
        f"Question: {kwargs['question']}\nOptions: {json.dumps(kwargs['options'], ensure_ascii=False)}\n"
        f"Workspace overview: {json.dumps(kwargs['workspace_overview'], ensure_ascii=False)[:6000]}"
    )


def _answer_prompt(kwargs: Mapping[str, Any], evidence_digest: Sequence[Mapping[str, Any]]) -> str:
    return (
        "Answer the multiple-choice question using only cited evidence. Return JSON only: "
        "{\"answer\":\"A. ...\", \"citations\":[\"ev_...\"]}.\n"
        f"Question: {kwargs['question']}\nOptions: {json.dumps(kwargs['options'], ensure_ascii=False)}\n"
        f"Evidence: {json.dumps(list(evidence_digest), ensure_ascii=False)}"
    )


def _evidence_prompt(workspace: VirtualVideoWorkspace, task: Any, segment_packet: Mapping[str, Any], window: Mapping[str, Any]) -> str:
    return (
        "You are the Investigator. Inspect these sampled frames and ASR. Return JSON only: "
        "{\"summary\":\"visual evidence summary\", \"confidence\":0.0-1.0}.\n"
        f"Question: {workspace.case.question}\nOptions: {json.dumps(dict(workspace.case.options), ensure_ascii=False)}\n"
        f"Task: {getattr(task, 'goal', '')}\nSegment: {json.dumps(_compact_segment_packet(segment_packet), ensure_ascii=False)[:3000]}\n"
        f"Window metadata: {json.dumps({k: window[k] for k in ['virtual_time_range','sampling','asr_cues','source_lineage']}, ensure_ascii=False)[:5000]}"
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
    parser.add_argument("--mode", choices=("smoke", "all"), default="smoke")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--min-duration-sec", type=float, default=18000.0)
    parser.add_argument("--segment-sec", type=float, default=600.0)
    parser.add_argument("--low-fps", type=float, default=0.1)
    parser.add_argument("--beat-sec", type=float, default=60.0)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-investigations", type=int, default=20)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
