#!/usr/bin/env python3
"""Compare full-question and anchor-only Caption retrieval on frozen labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.diagnose_mmlifelong_occurrence_candidate_coverage import (
    DeterministicCaptionReplay,
    _manifest_case_ids,
)
from tools.run_mmlifelong_gemini_ocr import _caption_packets
from vcah.occurrence_anchor_recall import build_anchor_recall_report
from vcah.virtual_video import VirtualVideoWorkspace


VARIANTS = (
    "frozen_seed_best_packet",
    "full_question_hybrid",
    "anchor_only_hybrid",
    "anchor_event_lexical",
    "anchor_event_hybrid",
)


def load_anchor_recall_cases(
    run_root: Path,
    *,
    relation_spec: Mapping[str, Any],
    query_spec: Mapping[str, Any],
    replay: DeterministicCaptionReplay,
    case_ids: Sequence[str],
    top_k: int,
) -> tuple[dict[str, Any], ...]:
    relation_cases = dict(relation_spec.get("cases", {}) or {})
    query_cases = dict(query_spec.get("cases", {}) or {})
    frozen = bool(query_spec.get("queries_frozen_before_anchor_outcomes"))
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        relation = relation_cases.get(case_id)
        queries = query_cases.get(case_id)
        if not isinstance(relation, Mapping) or not isinstance(queries, Mapping):
            raise ValueError(f"missing anchor recall specification: {case_id}")
        run_dir = Path(run_root) / "cases" / case_id
        workspace = VirtualVideoWorkspace.load(run_dir)
        packets = tuple(_caption_packets(run_dir))
        config_digests = {
            str(packet.get("config_digest", "") or "") for packet in packets
        }
        if len(config_digests) != 1 or not all(config_digests):
            raise ValueError(f"expected one frozen Caption config: {case_id}")
        anchor_intervals = tuple(relation.get("anchor_intervals", ()) or ())
        probe = {
            **dict(packets[0]),
            "time_range": None,
            "segment_ids": [],
            "source_video_ids": [],
            "expand_neighbors": 0,
            "index_mode": "hybrid",
            "query_strategy": "joint",
        }
        full_question = (workspace.case.question,)
        anchor_query = (str(queries.get("anchor_query", "") or ""),)
        event_queries = tuple(queries.get("anchor_event_queries", ()) or ())
        if not anchor_query[0] or not event_queries:
            raise ValueError(f"empty anchor query specification: {case_id}")
        ranks = {
            "frozen_seed_best_packet": _best_frozen_rank(packets, anchor_intervals),
            "full_question_hybrid": _search_rank(
                replay,
                workspace,
                probe,
                full_question,
                anchor_intervals,
                top_k=top_k,
            ),
            "anchor_only_hybrid": _search_rank(
                replay,
                workspace,
                probe,
                anchor_query,
                anchor_intervals,
                top_k=top_k,
            ),
            "anchor_event_lexical": _search_rank(
                replay,
                workspace,
                {**probe, "index_mode": "lexical"},
                event_queries,
                anchor_intervals,
                top_k=top_k,
            ),
            "anchor_event_hybrid": _search_rank(
                replay,
                workspace,
                probe,
                event_queries,
                anchor_intervals,
                top_k=top_k,
            ),
        }
        rows.append(
            {
                "case_id": case_id,
                "anchor_description": str(relation.get("anchor_description", "") or ""),
                "anchor_intervals": [list(value) for value in anchor_intervals],
                "queries_frozen_before_anchor_outcomes": frozen,
                "ranks": ranks,
            }
        )
    return tuple(rows)


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MM-Lifelong WP16-4A Anchor Recall",
        "",
        "## 结论",
        "",
        f"`{report['decision']}`",
        "",
        "本轮只测 Question -> anchor occurrence，不运行 evidence search 或 QA。",
        "",
        "## Anchor Recall",
        "",
        "| Variant | R@1 | R@3 | R@5 | R@10 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in report["variants"].items():
        values = []
        for k in (1, 3, 5, 10):
            row = summary["recall"][f"at_{k}"]
            values.append(f"{row['count']}/{row['case_count']}")
        lines.append(
            f"| `{name}` | {' | '.join(values)} | "
            f"{summary['mean_reciprocal_rank']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 逐题排名",
            "",
            "| Case | Anchor | Frozen | Full Q | Anchor only | Event lexical | Event hybrid |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case in report["per_case"]:
        ranks = case["ranks"]
        suffix = str(case["case_id"]).rsplit("-", 1)[-1]

        def rank(name: str) -> str:
            return str(ranks[name]) if ranks[name] is not None else "-"

        lines.append(
            f"| `{suffix}` | {case['anchor_description']} | "
            f"{rank('frozen_seed_best_packet')} | "
            f"{rank('full_question_hybrid')} | "
            f"{rank('anchor_only_hybrid')} | "
            f"{rank('anchor_event_lexical')} | "
            f"{rank('anchor_event_hybrid')} |"
        )
    lines.extend(
        [
            "",
            "## 有效性边界",
            "",
            f"- Structural gate: {report['structural_gate_passed']}",
            "- Gold anchor interval 只进入 evaluator，不进入检索。",
            "- Anchor-only query 只保留题目中的 anchor；不包含 target 或答案值。",
            "- Event queries 是人工 Caption-grounded 上界，不是自动 runtime 结果。",
            "- 当前没有 question-independent global OCR index，因此不伪造 OCR Anchor Recall。",
            "- 使用冻结 MiniLM-L6-v2 CPU embedding；generative/VLM/judge calls 均为 0。",
            "- Endpoint 不是 gate；10-case relation subset underpowered。",
        ]
    )
    return "\n".join(lines) + "\n"


def _best_frozen_rank(
    packets: Sequence[Mapping[str, Any]],
    anchor_intervals: Sequence[Sequence[float]],
) -> int | None:
    ranks = [
        rank
        for packet in packets
        if (
            rank := _best_hit_rank(
                tuple(packet.get("hits", ()) or ()), anchor_intervals
            )
        )
        is not None
    ]
    return min(ranks) if ranks else None


def _search_rank(
    replay: DeterministicCaptionReplay,
    workspace: VirtualVideoWorkspace,
    packet: Mapping[str, Any],
    queries: Sequence[str],
    anchor_intervals: Sequence[Sequence[float]],
    *,
    top_k: int,
) -> int | None:
    result = replay.search(
        workspace,
        packet,
        queries,
        top_k=max(1, int(top_k)),
        expand_neighbors_override=0,
    )
    return _best_hit_rank(tuple(result.get("hits", ()) or ()), anchor_intervals)


def _best_hit_rank(
    hits: Sequence[Any],
    anchor_intervals: Sequence[Sequence[float]],
) -> int | None:
    ranks: list[int] = []
    for fallback_rank, value in enumerate(hits, start=1):
        if not isinstance(value, Mapping):
            continue
        raw_range = tuple(value.get("range", ()) or ())
        if len(raw_range) != 2:
            raw_range = (
                value.get("virtual_start_sec"),
                value.get("virtual_end_sec"),
            )
        if len(raw_range) != 2 or any(item is None for item in raw_range):
            continue
        interval = tuple(sorted((float(raw_range[0]), float(raw_range[1]))))
        if any(_overlap(interval, anchor) > 0.0 for anchor in anchor_intervals):
            ranks.append(max(1, int(value.get("rank", fallback_rank) or fallback_rank)))
    return min(ranks) if ranks else None


def _overlap(left: Sequence[float], right: Sequence[float]) -> float:
    left_start, left_end = sorted((float(left[0]), float(left[1])))
    right_start, right_end = sorted((float(right[0]), float(right[1])))
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--relation-spec", required=True)
    parser.add_argument("--expected-relation-spec-sha256", required=True)
    parser.add_argument("--anchor-query-spec", required=True)
    parser.add_argument("--expected-anchor-query-spec-sha256", required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.case_manifest)
    relation_spec_path = Path(args.relation_spec)
    query_spec_path = Path(args.anchor_query_spec)
    _require_sha(manifest_path, args.expected_manifest_sha256, "manifest")
    _require_sha(
        relation_spec_path,
        args.expected_relation_spec_sha256,
        "relation spec",
    )
    _require_sha(
        query_spec_path,
        args.expected_anchor_query_spec_sha256,
        "anchor query spec",
    )
    manifest_ids = set(_manifest_case_ids(_read_json_value(manifest_path)))
    case_ids = tuple(dict.fromkeys(str(value) for value in args.case_ids))
    if len(case_ids) != int(args.expected_cases) or not set(case_ids) <= manifest_ids:
        raise ValueError("selected cases do not match frozen manifest subset")
    replay = DeterministicCaptionReplay(
        model_id=str(args.embedding_model),
        revision=str(args.embedding_revision),
        device=str(args.embedding_device),
        batch_size=max(1, int(args.embedding_batch_size)),
    )
    cases = load_anchor_recall_cases(
        Path(args.run_root),
        relation_spec=_read_json(relation_spec_path),
        query_spec=_read_json(query_spec_path),
        replay=replay,
        case_ids=case_ids,
        top_k=max(1, int(args.top_k)),
    )
    report = build_anchor_recall_report(
        cases,
        expected_cases=int(args.expected_cases),
        variant_order=VARIANTS,
    )
    report["provenance"] = {
        "label": str(args.provenance),
        "source_run_root": str(Path(args.run_root)),
        "case_manifest": str(manifest_path),
        "case_manifest_sha256": _sha256(manifest_path),
        "relation_spec": str(relation_spec_path),
        "relation_spec_sha256": _sha256(relation_spec_path),
        "anchor_query_spec": str(query_spec_path),
        "anchor_query_spec_sha256": _sha256(query_spec_path),
        "embedding": dict(replay.adapter.manifest),
        "top_k": max(1, int(args.top_k)),
        "generative_model_calls": 0,
        "vlm_calls": 0,
        "judge_calls": 0,
    }
    _write_json(Path(args.out_json), report)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), encoding="utf-8")
    event_at5 = report["variants"]["anchor_event_hybrid"]["recall"]["at_5"]
    print(
        "ANCHOR_RECALL_DONE "
        f"decision={report['decision']} "
        f"structural={report['structural_gate_passed']} "
        f"event_at5={event_at5['count']}/{event_at5['case_count']}",
        flush=True,
    )


def _require_sha(path: Path, expected: str, label: str) -> None:
    if _sha256(path) != str(expected):
        raise ValueError(f"{label} SHA256 mismatch")


def _read_json(path: Path) -> dict[str, Any]:
    value = _read_json_value(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _read_json_value(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
