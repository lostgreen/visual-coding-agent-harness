from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import requests
import yaml
from PIL import Image, ImageDraw

from vcah.xlebench import LifeLogColdIndex, LifeLogInvestigator, LifeLogRetriever, load_xlebench_manifest, write_investigation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate X-LeBench candidate selection or investigation with an OpenAI-compatible VLM.")
    parser.add_argument("--cases-dir", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=("selector", "investigator"), default="selector")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--inspect-top-n", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-window-sec", type=float, default=30.0)
    parser.add_argument("--max-windows-per-round", type=int, default=4)
    parser.add_argument("--max-total-investigator-calls", type=int, default=10)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--frames-per-candidate", type=int, default=1)
    parser.add_argument("--max-image-edge", type=int, default=512)
    args = parser.parse_args()

    os.environ.setdefault("VCAH_ALLOW_PLACEHOLDER_VISUAL", "1")
    cases_dir = Path(args.cases_dir)
    index_dir = Path(args.index_dir)
    out_dir = Path(args.out_dir)
    trace_dir = out_dir / "traces"
    sheet_dir = out_dir / "contact_sheets"
    trace_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(Path(args.config))
    manifest = load_xlebench_manifest(cases_dir)
    index = LifeLogColdIndex.load(index_dir)
    api = _OpenAICompatibleVLM(config, max_image_edge=args.max_image_edge, frames_per_window=args.frames_per_candidate)
    if args.mode == "investigator":
        _run_investigator_mode(manifest, index, api, args, out_dir, started=time.time())
        return
    retriever = LifeLogRetriever(index)

    records: list[dict[str, Any]] = []
    started = time.time()
    for case in manifest.cases:
        result = retriever.retrieve(case.question, scope=case.scope, top_k=args.top_k)
        candidates = result.candidates[: args.top_k]
        candidate_payloads = [_candidate_payload(item, offset + 1, args.frames_per_candidate) for offset, item in enumerate(candidates)]
        image_paths = [frame["path"] for item in candidate_payloads for frame in item["frames"]]
        prompt = _build_prompt(case.case_id, case.question, candidate_payloads)
        trace: dict[str, Any] = {
            "case_id": case.case_id,
            "question": case.question,
            "gt_intervals": [
                {
                    "video_uid": interval.video_uid,
                    "source_start_sec": interval.source_start_sec,
                    "source_end_sec": interval.source_end_sec,
                    "virtual_start_sec": interval.virtual_start_sec,
                    "virtual_end_sec": interval.virtual_end_sec,
                }
                for interval in case.gt_intervals
            ],
            "retrieval_debug": {
                "top_k": args.top_k,
                "candidate_count": len(candidates),
                "per_level_hits": {key: list(value) for key, value in result.per_level_hits.items()},
                "per_channel_hits": {key: list(value) for key, value in result.per_channel_hits.items()},
            },
            "candidates": candidate_payloads,
            "prompt_text": prompt,
            "model": api.model,
            "config_path": str(Path(args.config)),
        }
        try:
            response_text = api.chat(prompt, image_paths=image_paths, max_image_edge=args.max_image_edge)
            parsed = _parse_json_response(response_text)
            selected = parsed.get("selected_candidate") if isinstance(parsed, dict) else None
            selected_candidate = _candidate_by_id(candidate_payloads, selected)
            correct = bool(
                selected_candidate
                and any(_overlap_ratio(selected_candidate, interval) > 0.0 for interval in case.gt_intervals)
            )
            trace.update(
                {
                    "raw_response_text": response_text,
                    "parsed_response": parsed,
                    "selected_candidate": selected,
                    "correct": correct,
                    "max_gt_overlap_ratio": max(
                        (_overlap_ratio(selected_candidate, interval) for interval in case.gt_intervals),
                        default=0.0,
                    )
                    if selected_candidate
                    else 0.0,
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 - traces need compact failure fingerprints.
            trace.update(
                {
                    "raw_response_text": "",
                    "parsed_response": None,
                    "selected_candidate": None,
                    "correct": False,
                    "max_gt_overlap_ratio": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        sheet_path = sheet_dir / f"{_safe_id(case.case_id)}.jpg"
        _write_contact_sheet(image_paths, sheet_path)
        trace["contact_sheet"] = str(sheet_path)
        (trace_dir / f"{_safe_id(case.case_id)}.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        records.append(trace)

    attempted = sum(1 for item in records if not item.get("error"))
    correct = sum(1 for item in records if item.get("correct"))
    metrics = {
        "case_count": len(records),
        "attempted": attempted,
        "failed": len(records) - attempted,
        "correct": correct,
        "accuracy": correct / max(1, len(records)),
        "accuracy_definition": "selected_candidate source interval overlaps any GT interval by >0 seconds",
        "model": api.model,
        "api_base": api.base,
        "top_k": args.top_k,
        "frames_per_candidate": args.frames_per_candidate,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "cases": [
            {
                "case_id": item["case_id"],
                "selected_candidate": item.get("selected_candidate"),
                "correct": item.get("correct"),
                "max_gt_overlap_ratio": item.get("max_gt_overlap_ratio"),
                "error": item.get("error"),
            }
            for item in records
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


class _OpenAICompatibleVLM:
    def __init__(self, config: dict[str, Any], *, max_image_edge: int = 512, frames_per_window: int = 1) -> None:
        planner = config.get("planner_api") or config
        self.base = str(planner["base"]).rstrip("/")
        self.model = str(planner["model"])
        self.api_key = str(planner["api_key"])
        self.timeout = float(planner.get("timeout", 300))
        self.max_image_edge = int(max_image_edge)
        self.frames_per_window = max(1, int(frames_per_window))
        self.interactions: list[dict[str, Any]] = []
        for key, value in (planner.get("proxy_env") or {}).items():
            os.environ[str(key)] = str(value)

    def chat(self, prompt: str, *, image_paths: list[str], max_image_edge: int) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(path), max_image_edge)}})
        response = requests.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": 600,
            },
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            snippet = response.text[:500].replace(self.api_key, "<redacted>")
            raise RuntimeError(f"HTTP {response.status_code}: {snippet}")
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"])

    def select_xle_investigation(self, question: str, *, candidates: tuple[Any, ...], candidate_digest: str, max_windows: int, **kwargs: Any) -> dict[str, Any]:
        del candidates, kwargs
        prompt = _build_reasoner_selection_prompt(question, candidate_digest, max_windows=max_windows)
        raw_response = self.chat(prompt, image_paths=[], max_image_edge=self.max_image_edge)
        parsed = _parse_json_response(raw_response)
        if not isinstance(parsed, dict):
            parsed = {"inspect_windows": []}
        self.interactions.append(
            {
                "type": "reasoner_select_investigation",
                "question": question,
                "prompt_text": prompt,
                "raw_response_text": raw_response,
                "parsed_response": parsed,
            }
        )
        parsed.setdefault("investigator_request", "Inspect the selected windows for evidence relevant to the question.")
        return parsed

    def investigate_xle_window(self, question: str, *, candidate: Any, subwindow: Any, frame_refs: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
        selected_frames = tuple(frame_refs)[: self.frames_per_window]
        image_paths = [frame.path for frame in selected_frames]
        prompt = _build_investigator_prompt(question, candidate, subwindow, selected_frames)
        if not image_paths:
            return {"claim": "", "answer": "", "evidence": "", "confidence": 0.0}
        raw_response = self.chat(prompt, image_paths=image_paths, max_image_edge=self.max_image_edge)
        parsed = _parse_json_response(raw_response)
        if not isinstance(parsed, dict):
            parsed = {"evidence": str(parsed)}
        self.interactions.append(
            {
                "question": question,
                "candidate_id": subwindow.candidate_id,
                "window": {
                    "video_uid": subwindow.video_uid,
                    "source_start_sec": subwindow.source_start_sec,
                    "source_end_sec": subwindow.source_end_sec,
                    "virtual_start_sec": subwindow.virtual_start_sec,
                    "virtual_end_sec": subwindow.virtual_end_sec,
                },
                "frame_paths": image_paths,
                "prompt_text": prompt,
                "raw_response_text": raw_response,
                "parsed_response": parsed,
            }
        )
        return {
            "claim": str(parsed.get("claim") or parsed.get("evidence") or "").strip(),
            "answer": str(parsed.get("answer") or "").strip(),
            "evidence": str(parsed.get("evidence") or "").strip(),
            "confidence": float(parsed.get("confidence") or 0.0),
        }


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _run_investigator_mode(manifest: Any, index: LifeLogColdIndex, api: _OpenAICompatibleVLM, args: argparse.Namespace, out_dir: Path, *, started: float) -> None:
    trace_dir = out_dir / "traces"
    interaction_dir = out_dir / "model_interactions"
    sheet_dir = out_dir / "contact_sheets"
    trace_dir.mkdir(parents=True, exist_ok=True)
    interaction_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    investigator = LifeLogInvestigator(
        index,
        model=api,  # type: ignore[arg-type]
        max_steps=args.max_steps,
        inspect_top_n=args.inspect_top_n,
        retrieve_top_k=args.top_k,
        max_window_sec=args.max_window_sec,
        max_windows_per_round=args.max_windows_per_round,
        max_total_investigator_calls=args.max_total_investigator_calls,
        min_confidence=args.min_confidence,
    )
    records = []
    for case in manifest.cases:
        interaction_start = len(api.interactions)
        try:
            result = investigator.answer(case)
            selected = result.selected_interval
            correct = bool(selected and any(_window_overlap_ratio(selected, interval) > 0.0 for interval in case.gt_intervals))
            max_overlap = max((_window_overlap_ratio(selected, interval) for interval in case.gt_intervals), default=0.0) if selected else 0.0
            error = None
            write_investigation_report(result, trace_dir / f"{_safe_id(case.case_id)}.json")
            image_paths = [frame.path for item in result.evidence for frame in item.frame_refs]
            _write_contact_sheet(image_paths, sheet_dir / f"{_safe_id(case.case_id)}.jpg")
            summary = {
                "case_id": case.case_id,
                "answer": result.answer,
                "selected_interval": _window_payload(result.selected_interval),
                "verified_claim": result.verified_claim.claim_id if result.verified_claim else None,
                "correct": correct,
                "max_gt_overlap_ratio": max_overlap,
                "error": error,
            }
        except Exception as exc:  # noqa: BLE001 - traces need compact failure fingerprints.
            summary = {
                "case_id": case.case_id,
                "answer": "",
                "selected_interval": None,
                "verified_claim": None,
                "correct": False,
                "max_gt_overlap_ratio": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        interactions = api.interactions[interaction_start:]
        (interaction_dir / f"{_safe_id(case.case_id)}.json").write_text(
            json.dumps(interactions, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        records.append(summary)
    attempted = sum(1 for item in records if not item.get("error"))
    correct = sum(1 for item in records if item.get("correct"))
    metrics = {
        "mode": "investigator",
        "case_count": len(records),
        "attempted": attempted,
        "failed": len(records) - attempted,
        "correct": correct,
        "accuracy": correct / max(1, len(records)),
        "accuracy_definition": "selected investigator interval overlaps any GT interval by >0 seconds",
        "model": api.model,
        "api_base": api.base,
        "top_k": args.top_k,
        "inspect_top_n": args.inspect_top_n,
        "max_steps": args.max_steps,
        "max_window_sec": args.max_window_sec,
        "max_windows_per_round": args.max_windows_per_round,
        "max_total_investigator_calls": args.max_total_investigator_calls,
        "frames_per_window": args.frames_per_candidate,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "cases": records,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


def _candidate_payload(candidate: Any, candidate_id: int, frames_per_candidate: int) -> dict[str, Any]:
    frames = []
    for frame in candidate.frame_refs[: max(0, frames_per_candidate)]:
        frames.append(
            {
                "path": frame.path,
                "video_uid": frame.video_uid,
                "source_time_sec": frame.source_time_sec,
                "virtual_time_sec": frame.virtual_time_sec,
            }
        )
    return {
        "candidate_id": candidate_id,
        "video_uid": candidate.video_uid,
        "beat_id": candidate.beat_id,
        "source_start_sec": candidate.source_start_sec,
        "source_end_sec": candidate.source_end_sec,
        "virtual_start_sec": candidate.virtual_start_sec,
        "virtual_end_sec": candidate.virtual_end_sec,
        "score": candidate.score,
        "modalities": list(candidate.modalities),
        "frames": frames,
    }


def _build_prompt(case_id: str, question: str, candidates: list[dict[str, Any]]) -> str:
    compact = [
        {
            "candidate_id": item["candidate_id"],
            "video_uid": item["video_uid"],
            "source_sec": [round(item["source_start_sec"], 2), round(item["source_end_sec"], 2)],
            "virtual_sec": [round(item["virtual_start_sec"], 2), round(item["virtual_end_sec"], 2)],
            "modalities": item["modalities"],
            "frames": [
                {
                    "source_time_sec": round(frame["source_time_sec"], 2),
                    "virtual_time_sec": round(frame["virtual_time_sec"], 2),
                }
                for frame in item["frames"]
            ],
        }
        for item in candidates
    ]
    return (
        "You are evaluating a long-video retrieval result for X-LeBench.\n"
        "Choose the single candidate window whose image evidence best answers the question. "
        "The attached images are ordered exactly as the candidate/frame list below: all frames for candidate 1, "
        "then all frames for candidate 2, and so on.\n"
        "If none of the candidates contain useful evidence, set selected_candidate to null.\n\n"
        f"case_id: {case_id}\n"
        f"question: {question}\n"
        f"candidates_json: {json.dumps(compact, ensure_ascii=False)}\n\n"
        "Return only valid JSON with this schema: "
        '{"selected_candidate": number|null, "answer": string, "evidence": string, "confidence": number}'
    )


def _build_investigator_prompt(question: str, candidate: Any, subwindow: Any, frame_refs: tuple[Any, ...]) -> str:
    compact = {
        "candidate_id": subwindow.candidate_id,
        "video_uid": subwindow.video_uid,
        "candidate_source_sec": [round(candidate.source_start_sec, 2), round(candidate.source_end_sec, 2)],
        "subwindow_source_sec": [round(subwindow.source_start_sec, 2), round(subwindow.source_end_sec, 2)],
        "subwindow_virtual_sec": [round(subwindow.virtual_start_sec, 2), round(subwindow.virtual_end_sec, 2)],
        "frames": [
            {
                "source_time_sec": round(frame.source_time_sec, 2),
                "virtual_time_sec": round(frame.virtual_time_sec, 2),
            }
            for frame in frame_refs
        ],
    }
    return (
        "You are the investigator in an X-LeBench long-video QA loop.\n"
        "Inspect only this subwindow and its attached frames. Form one evidence-backed claim if the frames answer "
        "or materially constrain the question. If the frames are not useful, return an empty claim and low confidence.\n\n"
        f"question: {question}\n"
        f"subwindow_json: {json.dumps(compact, ensure_ascii=False)}\n\n"
        "Return only valid JSON with this schema: "
        '{"claim": string, "answer": string, "evidence": string, "confidence": number}'
    )


def _build_reasoner_selection_prompt(question: str, candidate_digest: str, *, max_windows: int) -> str:
    return (
        "You are the reasoner in an X-LeBench long-video QA loop.\n"
        "You will receive cold retrieval candidates as metadata. Do not inspect every candidate. "
        "Choose only the few source-time windows that are most worth visual/text inspection for answering the question.\n"
        "Prefer narrow windows near promising frame times. Use source seconds from the candidate digest. "
        "If a candidate is long, select the specific subwindow to inspect rather than the full candidate.\n\n"
        f"question: {question}\n"
        f"max_windows: {int(max_windows)}\n"
        f"candidate_digest:\n{candidate_digest}\n\n"
        "Return only valid JSON with this schema: "
        '{"answer_hypothesis": string|null, "investigator_request": string, '
        '"inspect_windows": [{"candidate_id": number, "start_sec": number, "end_sec": number}], '
        '"stop_reason": string|null}'
    )


def _image_data_url(path: Path, max_edge: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_edge, max_edge))
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=80, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _parse_json_response(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _candidate_by_id(candidates: list[dict[str, Any]], selected: Any) -> dict[str, Any] | None:
    if selected is None:
        return None
    try:
        selected_id = int(selected)
    except (TypeError, ValueError):
        return None
    for candidate in candidates:
        if int(candidate["candidate_id"]) == selected_id:
            return candidate
    return None


def _overlap_ratio(candidate: dict[str, Any], interval: Any) -> float:
    if candidate["video_uid"] != interval.video_uid:
        return 0.0
    intersection = max(
        0.0,
        min(float(candidate["source_end_sec"]), interval.source_end_sec)
        - max(float(candidate["source_start_sec"]), interval.source_start_sec),
    )
    return intersection / max(1e-9, interval.source_end_sec - interval.source_start_sec)


def _window_overlap_ratio(window: Any, interval: Any) -> float:
    if window is None or window.video_uid != interval.video_uid:
        return 0.0
    intersection = max(
        0.0,
        min(float(window.source_end_sec), interval.source_end_sec)
        - max(float(window.source_start_sec), interval.source_start_sec),
    )
    return intersection / max(1e-9, interval.source_end_sec - interval.source_start_sec)


def _window_payload(window: Any) -> dict[str, Any] | None:
    if window is None:
        return None
    return {
        "video_uid": window.video_uid,
        "source_start_sec": window.source_start_sec,
        "source_end_sec": window.source_end_sec,
        "virtual_start_sec": window.virtual_start_sec,
        "virtual_end_sec": window.virtual_end_sec,
    }


def _write_contact_sheet(image_paths: list[str], path: Path) -> None:
    if not image_paths:
        return
    thumbs = []
    for index, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((160, 100))
            canvas = Image.new("RGB", (160, 122), "white")
            canvas.paste(thumb, ((160 - thumb.width) // 2, 0))
            ImageDraw.Draw(canvas).text((4, 104), str(index), fill=(0, 0, 0))
            thumbs.append(canvas)
    cols = min(5, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 160, rows * 122), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % cols) * 160, (index // cols) * 122))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="JPEG", quality=85)


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "case"


if __name__ == "__main__":
    main()
