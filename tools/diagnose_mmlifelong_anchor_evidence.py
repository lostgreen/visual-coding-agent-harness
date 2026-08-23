#!/usr/bin/env python3
"""Evaluate deterministic anchor-conditioned evidence expansion with no model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.diagnose_mmlifelong_occurrence_candidate_coverage import (
    _manifest_case_ids,
    _optional_interval,
)
from tools.run_mmlifelong_gemini_ocr import _caption_packets
from vcah.anchor_evidence import expand_anchor_conditioned_evidence
from vcah.caption_evidence_bundle import build_caption_evidence_bundle_set
from vcah.caption_lexical_index import CaptionLexicalIndex
from vcah.caption_schema import CaptionHitV1
from vcah.occurrence_anchor_evidence import build_anchor_evidence_report
from vcah.occurrence_ocr import ocr_query_overlap
from vcah.virtual_video import VirtualVideoWorkspace


def load_anchor_evidence_cases(
    run_root: Path,
    *,
    evaluation_record_root: Path,
    ocr_root: Path,
    case_ids: Sequence[str],
    distances: Sequence[int],
    seed_top_k: int,
    max_gap_sec: float,
    anchor_spec: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    specifications = dict((anchor_spec or {}).get("cases", {}) or {})
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        run_dir = Path(run_root) / "cases" / case_id
        workspace = VirtualVideoWorkspace.load(run_dir)
        record = _read_json(
            Path(evaluation_record_root) / case_id / "evaluation_case.json"
        )
        clues = tuple(
            [float(interval[0]), float(interval[1])]
            for interval in tuple(record.get("clue_intervals", ()) or ())
            if _optional_interval(interval) is not None
        )
        spec = specifications.get(case_id, {})
        if not isinstance(spec, Mapping):
            raise ValueError(f"{case_id}: anchor specification is not an object")
        anchors = tuple(
            [float(interval[0]), float(interval[1])]
            for interval in tuple(spec.get("anchor_intervals", ()) or ())
            if _optional_interval(interval) is not None
        )
        evidence = tuple(
            [float(interval[0]), float(interval[1])]
            for interval in tuple(spec.get("evidence_intervals", clues) or ())
            if _optional_interval(interval) is not None
        )
        ocr_result = _read_json(
            Path(ocr_root) / "cases" / case_id / "ocr.ui_aware_v1.json"
        )
        if str(ocr_result.get("status", "") or "") != "success":
            raise ValueError(f"{case_id}: selected OCR result did not succeed")
        ocr_rows = tuple(ocr_result.get("ocr_rows", ()) or ())
        packets: list[dict[str, Any]] = []
        request: dict[str, Any] | None = None
        for packet in _caption_packets(run_dir):
            config_digest = str(packet.get("config_digest", "") or "")
            lexical = CaptionLexicalIndex.from_asset_root(
                workspace.asset_root,
                config_digest=config_digest,
            )
            seed_hits = tuple(
                _caption_hit(hit)
                for hit in tuple(packet.get("hits", ()) or ())[: max(1, int(seed_top_k))]
                if isinstance(hit, Mapping)
            )
            baseline_bundle = build_caption_evidence_bundle_set(seed_hits)
            variant_rows: dict[str, dict[str, Any]] = {
                "baseline": _variant_row(seed_hits, seed_hits, baseline_bundle)
            }
            source_map = {
                segment.segment_id: segment.source_video_id
                for segment in workspace.manifest.segments
            }
            for distance in distances:
                result = expand_anchor_conditioned_evidence(
                    lexical.passages,
                    seed_hits,
                    question=workspace.case.question,
                    ocr_rows=ocr_rows,
                    distance=int(distance),
                    index_digest=str(packet.get("index_digest", "") or ""),
                    config_digest=config_digest,
                    source_video_id_by_segment=source_map,
                    max_gap_sec=float(max_gap_sec),
                )
                request = dict(result["request"])
                variant_rows[f"distance_{int(distance)}"] = _variant_row(
                    result["seed_hits"],
                    result["hits"],
                    result["evidence_bundle_set"],
                )
            queries = tuple(str(value) for value in packet.get("queries", ()) or ())
            scoped_rows = tuple(
                row
                for row in ocr_rows
                if _ocr_row_in_scope(row, packet)
            )
            global_overlap = ocr_query_overlap(ocr_rows, queries)
            scoped_overlap = ocr_query_overlap(scoped_rows, queries)
            packets.append(
                {
                    "variants": variant_rows,
                    "scope_diagnostic": {
                        "global_overlap": global_overlap,
                        "scoped_overlap": scoped_overlap,
                        "scope_blocked_semantic_evidence": (
                            int(global_overlap["matched_token_count"]) > 0
                            and int(scoped_overlap["matched_token_count"]) == 0
                        ),
                    },
                }
            )
        if request is None:
            raise ValueError(f"{case_id}: no frozen Caption packet")
        rows.append(
            {
                "case_id": case_id,
                "request": request,
                "anchor_intervals": list(anchors),
                "evidence_intervals": list(evidence),
                "packets": packets,
            }
        )
    return tuple(rows)


def render_markdown(report: Mapping[str, Any]) -> str:
    selected = str(report.get("selected_variant", "") or "baseline")
    baseline = report["variants"]["baseline"]
    treatment = report["variants"][selected]

    def recall(row: Mapping[str, Any], key: str) -> str:
        value = row[key]
        rate = value.get("rate")
        suffix = "NA" if rate is None else f"{100.0 * float(rate):.2f}%"
        return f"{value['count']}/{value['case_count']} ({suffix})"

    lines = [
        "# MM-Lifelong WP16-3 Anchor-Conditioned Evidence Expansion",
        "",
        "## 结论",
        "",
        f"`{report['decision']}`",
        "",
        (
            "这是零新增模型调用的机制诊断：先保留冻结 Caption seed，再按问题中的 "
            "before/after 关系单向扩展，并把已有 OCR 行绑定到扩展 passage。"
        ),
        "",
        "## 分层召回",
        "",
        "| 指标 @5 | Baseline | " + selected + " |",
        "|---|---:|---:|",
        (
            "| Anchor seed recall（仅独立标注题） | "
            f"{recall(baseline, 'anchor_seed_at_5')} | "
            f"{recall(treatment, 'anchor_seed_at_5')} |"
        ),
        (
            f"| Evidence recall | {recall(baseline, 'evidence_at_5')} | "
            f"{recall(treatment, 'evidence_at_5')} |"
        ),
        (
            "| Channel-matched evidence recall | "
            f"{recall(baseline, 'channel_evidence_at_5')} | "
            f"{recall(treatment, 'channel_evidence_at_5')} |"
        ),
        (
            "| Bound evidence recall（仅独立标注题） | "
            f"{recall(baseline, 'bound_evidence_at_5')} | "
            f"{recall(treatment, 'bound_evidence_at_5')} |"
        ),
        "",
        "## 逐题",
        "",
        (
            "| Case | 关系 | 目标通道 | Anchor 标注 | Baseline evidence | "
            "Treatment evidence/channel/bound |"
        ),
        "|---|---|---|---:|---:|---:|",
    ]
    for row in report["per_case"]:
        request = row["request"]
        base = row["metrics_at_5"]["baseline"]
        treated = row["metrics_at_5"][selected]
        case_suffix = str(row["case_id"]).rsplit("-", 1)[-1]
        lines.append(
            f"| `{case_suffix}` | {request.get('relation') or '-'} | "
            f"{', '.join(request.get('evidence_channels', ()) or ())} | "
            f"{row['anchor_labeled']} | {base['evidence']} | "
            f"{treated['evidence']}/{treated['channel']}/{treated['bound']} |"
        )
    scope = report["scope_diagnostics"]
    lines.extend(
        [
            "",
            "## Admission 诊断",
            "",
            (
                "- Scope 阻断语义证据的 cases: "
                f"{', '.join(scope['scope_blocked_case_ids']) or 'none'}"
            ),
            (
                "- 纯数字弱命中的 cases: "
                f"{', '.join(scope['weak_numeric_only_case_ids']) or 'none'}"
            ),
            "",
            "## 有效性边界",
            "",
            f"- Structural gate: {report['structural_gate_passed']}",
            f"- Eligible cases: {report['eligible_case_count']}/{report['case_count']}",
            f"- Independently anchor-labeled cases: {report['anchor_labeled_case_count']}",
            "- 官方区间只进入 evaluator，不进入扩展策略。",
            (
                "- OCR 帧由官方区间 oracle-localize，因此结果只是机制上界，"
                "不是正式 retrieval 提升。"
            ),
            (
                "- 没有独立 anchor 标注的题不报告 Anchor Recall 或 "
                "Bound Evidence Recall。"
            ),
            "- Endpoint 数值不是 structural gate；frozen39/精选子集均 underpowered。",
        ]
    )
    return "\n".join(lines) + "\n"


def _variant_row(
    seed_hits: Sequence[CaptionHitV1],
    hits: Sequence[CaptionHitV1],
    bundle_set: Mapping[str, Any],
) -> dict[str, Any]:
    context_hits = tuple(
        hit
        for hit in hits
        if hit.metadata.get("anchor_evidence_contract")
    )
    return {
        "seed_hit_ids": [hit.passage_id for hit in seed_hits],
        "context_hit_count": len(context_hits),
        "bundle_set": _compact_bundle_set(bundle_set),
    }


def _compact_bundle_set(bundle_set: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "bundles": [
            {
                "rank": int(bundle.get("rank", 0) or 0),
                "seed_passage_ids": list(bundle.get("seed_passage_ids", ()) or ()),
                "member_passages": [
                    {
                        "passage_id": str(member.get("passage_id", "") or ""),
                        "time_range": list(member.get("time_range", ()) or ()),
                        "role": str(member.get("role", "") or ""),
                        "context_links": [
                            dict(link)
                            for link in tuple(member.get("context_links", ()) or ())
                            if isinstance(link, Mapping)
                        ],
                        "evidence_channels_observed": list(
                            member.get("evidence_channels_observed", ()) or ()
                        ),
                    }
                    for member in tuple(bundle.get("member_passages", ()) or ())
                    if isinstance(member, Mapping)
                ],
            }
            for bundle in tuple(bundle_set.get("bundles", ()) or ())
            if isinstance(bundle, Mapping)
        ]
    }


def _caption_hit(value: Mapping[str, Any]) -> CaptionHitV1:
    fields = {
        "passage_id",
        "caption_id",
        "rank",
        "lexical_score",
        "dense_score",
        "fused_score",
        "virtual_start_sec",
        "virtual_end_sec",
        "wall_clock_begin",
        "wall_clock_end",
        "text",
        "interval_precision",
        "source_pointer",
        "metadata",
    }
    return CaptionHitV1(**{key: value[key] for key in fields})


def _ocr_row_in_scope(row: Mapping[str, Any], packet: Mapping[str, Any]) -> bool:
    requested_segments = {
        str(value) for value in tuple(packet.get("segment_ids", ()) or ()) if str(value)
    }
    row_segments = {
        str(value) for value in tuple(row.get("segment_ids", ()) or ()) if str(value)
    }
    if requested_segments and row_segments and requested_segments.isdisjoint(row_segments):
        return False
    time_range = _optional_interval(packet.get("time_range"))
    if time_range is None:
        return True
    times = tuple(
        float(value)
        for value in tuple(row.get("virtual_times_sec", ()) or ())
        if isinstance(value, (int, float))
    )
    return bool(times) and any(time_range[0] <= value <= time_range[1] for value in times)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--evaluation-record-root", required=True)
    parser.add_argument("--ocr-root", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--anchor-spec")
    parser.add_argument("--seed-top-k", type=int, default=5)
    parser.add_argument("--distances", type=int, nargs="+", default=(1, 8, 20))
    parser.add_argument("--max-gap-sec", type=float, default=600.0)
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
    manifest_ids = set(_manifest_case_ids(_read_json_value(manifest_path)))
    case_ids = tuple(dict.fromkeys(str(value) for value in args.case_ids))
    if len(case_ids) != int(args.expected_cases) or not set(case_ids) <= manifest_ids:
        raise ValueError("selected cases do not match expected frozen manifest subset")
    distances = tuple(sorted({max(1, int(value)) for value in args.distances}))
    anchor_spec = (
        _read_json(Path(args.anchor_spec)) if str(args.anchor_spec or "") else None
    )
    cases = load_anchor_evidence_cases(
        Path(args.run_root),
        evaluation_record_root=Path(args.evaluation_record_root),
        ocr_root=Path(args.ocr_root),
        case_ids=case_ids,
        distances=distances,
        seed_top_k=max(1, int(args.seed_top_k)),
        max_gap_sec=max(0.0, float(args.max_gap_sec)),
        anchor_spec=anchor_spec,
    )
    variants = tuple(f"distance_{distance}" for distance in distances)
    report = build_anchor_evidence_report(
        cases,
        expected_cases=int(args.expected_cases),
        variant_order=variants,
    )
    report["provenance"] = {
        "label": str(args.provenance),
        "source_run_root": str(Path(args.run_root)),
        "ocr_root": str(Path(args.ocr_root)),
        "case_manifest": str(manifest_path),
        "case_manifest_sha256": manifest_sha,
        "case_ids": list(case_ids),
        "seed_top_k": int(args.seed_top_k),
        "distances": list(distances),
        "max_gap_sec": float(args.max_gap_sec),
        "generative_model_calls": 0,
        "vlm_calls": 0,
        "judge_calls": 0,
        "ocr_localization": "official_clue_intervals_oracle_diagnostic",
    }
    _write_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render_markdown(report), encoding="utf-8")
    print(
        "ANCHOR_EVIDENCE_DONE "
        f"decision={report['decision']} "
        f"structural={report['structural_gate_passed']} "
        f"selected={report.get('selected_variant') or 'none'}",
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
