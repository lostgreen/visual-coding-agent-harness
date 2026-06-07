#!/usr/bin/env python
"""Audit VideoMME MCQ exploration rewrites with compact risk tags."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from visual_coding_agent_harness.agents.open_questions import rewrite_exploration_question_with_model
from visual_coding_agent_harness.agents.question_policy import classify_question_route
from visual_coding_agent_harness.evals.videomme.runner import (
    DEFAULT_PARQUET_PATH,
    MODEL_PATH,
    PLANNER_MODEL_PATH,
    compact_text,
    make_question,
    normalize_options,
    row_get,
)


DETECTOR_PATTERNS = (
    r"\btarget\s+(?:items?|entities?|list|temporal)\b",
    r"\bunordered\s+list\b",
    r"\bfor each target\b",
    r"\bmentions?\s+of\b",
    r"\b(?:determine|decide|verify|confirm)\s+(?:whether|if)\b",
    r"\bpresent\s+or\s+absent\b",
    r"\bdoes\s+(?:not\s+)?present\b",
    r"\blook\s+for\b",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--planner-model-path", default=PLANNER_MODEL_PATH)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=6112)
    parser.add_argument("--include-cases", default="611-2")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=12)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_sample(args.parquet_path, sample_size=args.sample_size, seed=args.seed, include_cases=args.include_cases)
    backend = _build_rewrite_backend(model_path=args.model_path, planner_model_path=args.planner_model_path)

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for _, row in df.iterrows():
        question = make_question(row)
        route = classify_question_route(question)
        rewrite = rewrite_exploration_question_with_model(backend, question=question, route_hint=route)
        record = _record(row=row, question=question, route=route, rewrite=rewrite)
        records.append(record)
        print(
            "REWRITE_AUDIT_CASE "
            + json.dumps(
                {
                    "question_id": record["question_id"],
                    "route": record["route"],
                    "used_model": record["used_model"],
                    "risk_tags": record["risk_tags"],
                    "target_count": record["target_count"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = _summary(records=records, seconds=time.perf_counter() - started, args=args)
    (args.out_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=True, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    samples = [record for record in records if record["risk_tags"]][: args.max_samples]
    (args.out_dir / "risk_samples.json").write_text(
        json.dumps(samples, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("REWRITE_AUDIT_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _build_rewrite_backend(*, model_path: str, planner_model_path: str) -> Any:
    if planner_model_path:
        from visual_coding_agent_harness.backends.qwen_text import QwenTextBackend

        return QwenTextBackend.from_pretrained(planner_model_path)

    from visual_coding_agent_harness.backends.qwen_vl import QwenVLBackend

    return QwenVLBackend.from_pretrained(model_path)


def _load_sample(parquet_path: Path, *, sample_size: int, seed: int, include_cases: str) -> Any:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    include = [item.strip() for item in include_cases.split(",") if item.strip()]
    qid_series = df["question_id"].astype(str)
    picked_indexes: list[int] = []
    for qid in include:
        matches = list(df.index[qid_series.eq(qid)])
        picked_indexes.extend(matches[:1])
    remaining = [idx for idx in list(df.index) if idx not in set(picked_indexes)]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    target_count = max(0, sample_size - len(picked_indexes))
    picked_indexes.extend(remaining[:target_count])
    return df.loc[picked_indexes].reset_index(drop=True)


def _record(*, row: Any, question: str, route: str, rewrite: Any) -> dict[str, Any]:
    exploration_question = str(rewrite.exploration_question or "")
    target_entities = [str(item) for item in rewrite.target_entities]
    options = normalize_options(row_get(row, "options", []))
    tags = _risk_tags(
        question=question,
        route=route,
        exploration_question=exploration_question,
        target_entities=target_entities,
        options=options,
    )
    return {
        "question_id": str(row_get(row, "question_id")),
        "video_id": str(row_get(row, "video_id", row_get(row, "videoID"))),
        "task_type": str(row_get(row, "task_type")),
        "route": route,
        "used_model": bool(rewrite.used_model),
        "fallback_reason": str(rewrite.fallback_reason),
        "target_count": len(target_entities),
        "target_entities": target_entities,
        "risk_tags": tags,
        "question_excerpt": compact_text(str(row_get(row, "question")), limit=180),
        "exploration_question": compact_text(exploration_question, limit=420),
        "raw_rewrite_excerpt": compact_text(str(rewrite.raw_text), limit=420),
    }


def _risk_tags(
    *,
    question: str,
    route: str,
    exploration_question: str,
    target_entities: Sequence[str],
    options: Sequence[str],
) -> list[str]:
    tags: list[str] = []
    lowered = exploration_question.lower()
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in DETECTOR_PATTERNS):
        tags.append("target_detector_prompt")
    if route == "temporal_order" and _target_mentions(exploration_question, target_entities) >= 2:
        tags.append("temporal_targets_in_tool_question")
    if _option_leak(exploration_question, options):
        tags.append("option_surface_leak")
    if re.search(r"\b(?:not\s+present|no\s+target|does\s+not\s+show|does\s+not\s+present)\b", lowered):
        tags.append("negative_absence_prompt")
    if route == "temporal_order" and not target_entities:
        tags.append("missing_temporal_targets")
    if not exploration_question.strip():
        tags.append("empty_rewrite")
    return tags


def _target_mentions(text: str, targets: Sequence[str]) -> int:
    normalized_text = _normalize(text)
    return sum(1 for target in targets if _normalize(target) and _normalize(target) in normalized_text)


def _option_leak(text: str, options: Sequence[str]) -> bool:
    if re.search(r"(?m)^\s*[A-H][\).:-]\s+\S+", text):
        return True
    if re.search(r"\b(?:option|choice|answer)\s*[A-H]\b", text, flags=re.IGNORECASE):
        return True
    normalized_text = _normalize(text)
    return any(_normalized_option_leaks(option, normalized_text=normalized_text) for option in options)


def _normalized_option_leaks(option: str, *, normalized_text: str) -> bool:
    normalized_option = _normalize(re.sub(r"^\s*[A-H][\).:-]\s+", "", str(option or ""), flags=re.IGNORECASE))
    if not normalized_option:
        return False
    if normalized_option in normalized_text:
        return True
    option_terms = [term for term in normalized_option.split() if len(term) >= 4]
    return len(option_terms) >= 2 and all(re.search(rf"\b{re.escape(term)}\b", normalized_text) for term in option_terms)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _summary(*, records: Sequence[Mapping[str, Any]], seconds: float, args: argparse.Namespace) -> dict[str, Any]:
    risk_counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    for record in records:
        route_counts[str(record["route"])] = route_counts.get(str(record["route"]), 0) + 1
        for tag in record["risk_tags"]:
            risk_counts[str(tag)] = risk_counts.get(str(tag), 0) + 1
    return {
        "sample_size": len(records),
        "seconds": round(seconds, 3),
        "parquet_path": str(args.parquet_path),
        "model_path": str(args.model_path),
        "planner_model_path": str(args.planner_model_path),
        "seed": args.seed,
        "route_counts": route_counts,
        "risk_counts": risk_counts,
        "used_model_count": sum(1 for record in records if record["used_model"]),
        "fallback_count": sum(1 for record in records if not record["used_model"]),
        "artifact_paths": {
            "records": str(args.out_dir / "records.jsonl"),
            "summary": str(args.out_dir / "summary.json"),
            "risk_samples": str(args.out_dir / "risk_samples.json"),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
