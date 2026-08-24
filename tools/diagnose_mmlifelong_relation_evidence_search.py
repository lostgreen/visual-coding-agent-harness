#!/usr/bin/env python3
"""Evaluate oracle-anchor relation-guided evidence search without model calls."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.diagnose_mmlifelong_occurrence_candidate_coverage import (
    _manifest_case_ids,
)
from tools.run_mmlifelong_gemini_ocr import _caption_packets
from vcah.caption_lexical_index import CaptionLexicalIndex
from vcah.occurrence_relation_evidence import (
    build_relation_evidence_report,
)
from vcah.relation_evidence_search import (
    build_oracle_fixed_window,
    relation_guided_evidence_search,
)
from vcah.virtual_video import VirtualVideoWorkspace


def load_relation_evidence_cases(
    run_root: Path,
    *,
    evaluation_record_root: Path,
    ocr_root: Path,
    spec: Mapping[str, Any],
    case_ids: Sequence[str],
    fixed_distance: int,
) -> tuple[dict[str, Any], ...]:
    specifications = dict(spec.get("cases", {}) or {})
    labels_frozen = bool(spec.get("labels_frozen_before_primary_outcomes"))
    cases: list[dict[str, Any]] = []
    for case_id in case_ids:
        case_spec = specifications.get(case_id)
        if not isinstance(case_spec, Mapping):
            raise ValueError(f"missing relation search specification: {case_id}")
        run_dir = Path(run_root) / "cases" / case_id
        workspace = VirtualVideoWorkspace.load(run_dir)
        if str(case_spec.get("question", "") or "") != workspace.case.question:
            raise ValueError(f"question mismatch for {case_id}")
        packets = tuple(_caption_packets(run_dir))
        config_digests = {
            str(packet.get("config_digest", "") or "") for packet in packets
        }
        if len(config_digests) != 1 or not all(config_digests):
            raise ValueError(f"expected one frozen Caption config: {case_id}")
        config_digest = next(iter(config_digests))
        lexical = CaptionLexicalIndex.from_asset_root(
            workspace.asset_root,
            config_digest=config_digest,
        )
        ocr_result = _read_json(
            Path(ocr_root) / "cases" / case_id / "ocr.ui_aware_v1.json"
        )
        if str(ocr_result.get("status", "") or "") != "success":
            raise ValueError(f"selected OCR result did not succeed: {case_id}")
        ocr_rows = tuple(ocr_result.get("ocr_rows", ()) or ())
        source_map = {
            segment.segment_id: segment.source_video_id
            for segment in workspace.manifest.segments
        }
        direction = str(case_spec.get("relation", "") or "")
        channels = tuple(case_spec.get("target_evidence_type", ()) or ())
        anchor_intervals = tuple(case_spec.get("anchor_intervals", ()) or ())
        fixed = build_oracle_fixed_window(
            lexical.passages,
            anchor_intervals=anchor_intervals,
            direction=direction,
            evidence_channels=channels,
            ocr_rows=ocr_rows,
            distance=int(fixed_distance),
            index_digest=lexical.index_digest,
            config_digest=config_digest,
            source_video_id_by_segment=source_map,
            max_gap_sec=float(case_spec.get("max_gap_sec", 600.0)),
        )
        bounded = relation_guided_evidence_search(
            lexical.passages,
            anchor_intervals=anchor_intervals,
            direction=direction,
            target_event_term_groups=tuple(
                case_spec.get("target_event_term_groups", ()) or ()
            ),
            evidence_channels=channels,
            target_text_sources=tuple(
                case_spec.get("target_text_sources", ("caption",)) or ("caption",)
            ),
            ocr_rows=ocr_rows,
            max_passages=int(case_spec.get("max_passages", 80)),
            max_elapsed_sec=float(case_spec.get("max_elapsed_sec", 2400.0)),
            max_gap_sec=float(case_spec.get("max_gap_sec", 600.0)),
            index_digest=lexical.index_digest,
            config_digest=config_digest,
            source_video_id_by_segment=source_map,
        )
        evaluation_record = _read_json(
            Path(evaluation_record_root) / case_id / "evaluation_case.json"
        )
        cases.append(
            {
                "case_id": case_id,
                "question": workspace.case.question,
                "anchor_description": str(
                    case_spec.get("anchor_description", "") or ""
                ),
                "anchor_intervals": [list(value) for value in anchor_intervals],
                "relation": direction,
                "target_event_description": str(
                    case_spec.get("target_event_description", "") or ""
                ),
                "target_event_term_groups": [
                    list(value)
                    for value in tuple(
                        case_spec.get("target_event_term_groups", ()) or ()
                    )
                ],
                "target_evidence_type": list(channels),
                "evidence_intervals": [
                    list(value)
                    for value in tuple(
                        case_spec.get("target_evidence_intervals", ()) or ()
                    )
                ],
                "official_clue_intervals": list(
                    evaluation_record.get("clue_intervals", ()) or ()
                ),
                "labels_frozen_before_primary_outcomes": labels_frozen,
                "variants": {
                    f"fixed_d{int(fixed_distance)}": _compact_result(fixed),
                    "bounded_search": _compact_result(bounded),
                },
            }
        )
    return tuple(cases)


def render_markdown(report: Mapping[str, Any]) -> str:
    fixed_name = next(name for name in report["variants"] if name.startswith("fixed_d"))
    fixed = report["variants"][fixed_name]
    bounded = report["variants"]["bounded_search"]

    def ratio(row: Mapping[str, Any]) -> str:
        rate = row.get("rate")
        suffix = "NA" if rate is None else f"{100.0 * float(rate):.2f}%"
        return f"{row['count']}/{row['case_count']} ({suffix})"

    lines = [
        "# MM-Lifelong WP16-4 Oracle Anchor + Bounded Search",
        "",
        "## 结论",
        "",
        f"`{report['decision']}`",
        "",
        (
            "本轮直接提供人工核对过的正确 anchor，只测试：给定正确事件后，"
            "能否沿 before/after 关系找到目标事件并及时停止。"
        ),
        "",
        "## 主要结果",
        "",
        f"| 指标 | {fixed_name} | Bounded search |",
        "|---|---:|---:|",
        (
            f"| Evidence recall | {ratio(fixed['evidence_recall'])} | "
            f"{ratio(bounded['evidence_recall'])} |"
        ),
        (
            "| Channel-bound evidence recall | "
            f"{ratio(fixed['bound_evidence_recall'])} | "
            f"{ratio(bounded['bound_evidence_recall'])} |"
        ),
        (
            f"| Stop success | {ratio(fixed['stop_success_rate'])} | "
            f"{ratio(bounded['stop_success_rate'])} |"
        ),
        (
            f"| Wrong stop | {ratio(fixed['wrong_stop_rate'])} | "
            f"{ratio(bounded['wrong_stop_rate'])} |"
        ),
        (
            "| Mean passages visited | "
            f"{fixed['passages_visited']['mean']:.2f} | "
            f"{bounded['passages_visited']['mean']:.2f} |"
        ),
        (
            "| P95 passages visited | "
            f"{fixed['passages_visited']['p95']} | "
            f"{bounded['passages_visited']['p95']} |"
        ),
        "",
        "## 逐题",
        "",
        (
            f"| Case | Anchor -> target | {fixed_name} evidence/cost | "
            "Bounded stop/evidence/cost |"
        ),
        "|---|---|---:|---:|",
    ]
    for case in report["per_case"]:
        fixed_case = case["metrics"][fixed_name]
        bounded_case = case["metrics"]["bounded_search"]
        suffix = str(case["case_id"]).rsplit("-", 1)[-1]
        lines.append(
            f"| `{suffix}` | {case['anchor_description']} "
            f"{case['relation']} {case['target_event_description']} | "
            f"{int(fixed_case['evidence'])}/{fixed_case['visited_passage_count']} | "
            f"{int(bounded_case['stop_success'])}/"
            f"{int(bounded_case['evidence'])}/"
            f"{bounded_case['visited_passage_count']} |"
        )
    lines.extend(
        [
            "",
            "## 有效性边界",
            "",
            f"- Structural gate: {report['structural_gate_passed']}",
            "- `0115` 已排除；它是 multi-occurrence state comparison。",
            "- Anchor 和 target interval 是人工标注，只进入 evaluator。",
            "- Target 词不包含官方答案值，但来自题目与 Caption timeline 的探索性核对。",
            "- OCR 仍由官方区间 oracle-localize，因此这是机制上界，不是正式 runtime 结果。",
            "- 本轮 generative/VLM/judge calls 均为 0；没有运行 QA。",
            "- Endpoint 数值不是 gate；10-case relation subset underpowered。",
        ]
    )
    return "\n".join(lines) + "\n"


def _compact_result(result: Mapping[str, Any]) -> dict[str, Any]:
    hits = tuple(result.get("hits", ()) or ())
    compact_hits = [_compact_hit(hit) for hit in hits]
    stop = result.get("stop_hit")
    return {
        "status": str(result.get("status", "") or ""),
        "anchor_hit": compact_hits[0] if compact_hits else None,
        "stop_hit": _compact_hit(stop) if stop is not None else None,
        "hits": compact_hits,
        "visited_passage_count": int(result.get("visited_passage_count", 0) or 0),
        "stop_success": bool(result.get("stop_success")),
        "stop_reason": str(result.get("stop_reason", "") or ""),
        "matched_target_terms": list(result.get("matched_target_terms", ()) or ()),
    }


def _compact_hit(hit: Any) -> dict[str, Any]:
    row = asdict(hit) if not isinstance(hit, Mapping) else dict(hit)
    metadata = dict(row.get("metadata", {}) or {})
    return {
        "passage_id": str(row.get("passage_id", "") or ""),
        "time_range": [
            float(row.get("virtual_start_sec", 0.0) or 0.0),
            float(row.get("virtual_end_sec", 0.0) or 0.0),
        ],
        "evidence_channels_observed": list(
            metadata.get("evidence_channels_observed", ()) or ()
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--evaluation-record-root", required=True)
    parser.add_argument("--ocr-root", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--relation-spec", required=True)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--fixed-distance", type=int, default=20)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.case_manifest)
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != str(args.expected_manifest_sha256):
        raise ValueError("case manifest SHA256 mismatch")
    relation_spec_path = Path(args.relation_spec)
    relation_spec_sha = _sha256(relation_spec_path)
    if relation_spec_sha != str(args.expected_spec_sha256):
        raise ValueError("relation specification SHA256 mismatch")
    manifest_ids = set(_manifest_case_ids(_read_json_value(manifest_path)))
    case_ids = tuple(dict.fromkeys(str(value) for value in args.case_ids))
    if len(case_ids) != int(args.expected_cases) or not set(case_ids) <= manifest_ids:
        raise ValueError("selected cases do not match frozen manifest subset")
    spec = _read_json(relation_spec_path)
    cases = load_relation_evidence_cases(
        Path(args.run_root),
        evaluation_record_root=Path(args.evaluation_record_root),
        ocr_root=Path(args.ocr_root),
        spec=spec,
        case_ids=case_ids,
        fixed_distance=max(1, int(args.fixed_distance)),
    )
    fixed_name = f"fixed_d{max(1, int(args.fixed_distance))}"
    report = build_relation_evidence_report(
        cases,
        expected_cases=int(args.expected_cases),
        variant_order=(fixed_name, "bounded_search"),
    )
    report["provenance"] = {
        "label": str(args.provenance),
        "source_run_root": str(Path(args.run_root)),
        "ocr_root": str(Path(args.ocr_root)),
        "case_manifest": str(manifest_path),
        "case_manifest_sha256": manifest_sha,
        "relation_spec": str(relation_spec_path),
        "relation_spec_sha256": relation_spec_sha,
        "case_ids": list(case_ids),
        "fixed_distance": max(1, int(args.fixed_distance)),
        "generative_model_calls": 0,
        "vlm_calls": 0,
        "judge_calls": 0,
        "ocr_localization": "official_clue_intervals_oracle_diagnostic",
    }
    _write_json(Path(args.out_json), report)
    output_md = Path(args.out_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(report), encoding="utf-8")
    bounded = report["variants"]["bounded_search"]["evidence_recall"]
    print(
        "RELATION_EVIDENCE_DONE "
        f"decision={report['decision']} "
        f"structural={report['structural_gate_passed']} "
        f"evidence={bounded['count']}/{bounded['case_count']}",
        flush=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = _read_json_value(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _read_json_value(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
