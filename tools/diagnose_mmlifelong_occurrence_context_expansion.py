#!/usr/bin/env python3
"""Evaluate post-retrieval caption context expansion with zero model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.diagnose_mmlifelong_occurrence_candidate_coverage import (
    DeterministicCaptionReplay,
    _caption_packet,
    _manifest_case_ids,
    _optional_interval,
    _read_json,
    _read_jsonl,
    load_cases,
)
from vcah.caption_context import CAPTION_CONTEXT_CONTRACT
from vcah.caption_evidence_bundle import build_caption_evidence_bundle_set
from vcah.caption_lexical_index import normalize_caption_query
from vcah.occurrence_candidate_coverage import build_candidate_coverage_report
from vcah.occurrence_context_expansion import (
    build_occurrence_context_expansion_report,
)
from vcah.virtual_video import VirtualVideoWorkspace


def load_context_cases(
    run_root: Path,
    *,
    evaluation_record_root: Path,
    replay: DeterministicCaptionReplay,
    expected_index_digest: str,
    seed_top_k: int,
    distances: Sequence[int],
    context_max_gap_sec: float,
) -> tuple[dict[str, Any], ...]:
    cases: list[dict[str, Any]] = []
    variants = tuple(f"neighbors_{int(distance)}" for distance in distances)
    for prediction_path in sorted(Path(run_root).glob("cases/*/prediction.json")):
        run_dir = prediction_path.parent
        workspace = VirtualVideoWorkspace.load(run_dir)
        case_id = workspace.case.case_id
        record = _read_json(
            Path(evaluation_record_root) / case_id / "evaluation_case.json"
        )
        clues = tuple(
            [float(interval[0]), float(interval[1])]
            for interval in tuple(record.get("clue_intervals", ()) or ())
            if _optional_interval(interval) is not None
        )
        packets: list[dict[str, Any]] = []
        observations = _read_jsonl(run_dir / "observation_log.jsonl")
        for observation in observations:
            config = observation.get("sampling_config", {})
            if not isinstance(config, Mapping) or not isinstance(
                config.get("occurrence_set"), Mapping
            ):
                continue
            frozen_packet = _caption_packet(run_dir, observation)
            queries = tuple(
                str(query)
                for query in tuple(frozen_packet.get("queries", ()) or ())
                if normalize_caption_query(str(query))
            )
            packet_config = {
                "config_digest": str(frozen_packet.get("config_digest", "") or ""),
                "index_mode": str(frozen_packet.get("index_mode", "") or ""),
                "query_strategy": str(
                    frozen_packet.get("query_strategy", "joint") or "joint"
                ),
                "time_range": frozen_packet.get("time_range"),
                "segment_ids": list(frozen_packet.get("segment_ids", ()) or ()),
                "source_video_ids": list(
                    frozen_packet.get("source_video_ids", ()) or ()
                ),
                "expand_neighbors": 0,
            }
            baseline_packet = replay.search(
                workspace,
                packet_config,
                queries,
                top_k=seed_top_k,
                expand_neighbors_override=0,
            )
            _require_index_digest(
                baseline_packet,
                expected=expected_index_digest,
                case_id=case_id,
            )
            baseline_hits = tuple(baseline_packet.get("hits", ()) or ())
            variant_rows: dict[str, dict[str, Any]] = {
                "baseline": _variant_row(
                    baseline_packet,
                    build_caption_evidence_bundle_set(baseline_hits),
                )
            }
            for distance, variant in zip(distances, variants):
                expanded_packet = replay.search(
                    workspace,
                    packet_config,
                    queries,
                    top_k=seed_top_k,
                    expand_neighbors_override=0,
                    context_neighbors=int(distance),
                    context_max_gap_sec=context_max_gap_sec,
                )
                _require_index_digest(
                    expanded_packet,
                    expected=expected_index_digest,
                    case_id=case_id,
                )
                bundle_set = expanded_packet.get("evidence_bundle_set")
                if not isinstance(bundle_set, Mapping):
                    raise ValueError(f"{case_id}: context bundle set missing")
                variant_rows[variant] = _variant_row(
                    expanded_packet,
                    bundle_set,
                )
            packets.append(
                {
                    "attempt_id": str(observation.get("attempt_id", "") or ""),
                    "variants": variant_rows,
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "clues": list(clues),
                "packets": packets,
            }
        )
    return tuple(cases)


def render_markdown(report: Mapping[str, Any]) -> str:
    cohort = report["cohort"]
    contract = report["evaluation_contract"]
    lines = [
        "# MM-Lifelong WP16-1 Query-Conditioned Context Expansion",
        "",
        "## Decision",
        "",
        f"`{report['decision']}`",
        "",
        (
            "This is a zero-generative-call comparison of fixed retrieval seeds "
            "against post-retrieval same-source temporal expansion."
        ),
        "",
        "## Cohort",
        "",
        f"- Cases: {cohort['case_count']}",
        f"- Baseline present at bundle@5: {cohort['baseline_candidate_present_at_5']}",
        f"- Baseline absent at bundle@5: {cohort['baseline_candidate_absent_at_5']}",
        f"- Selected variant: {report.get('selected_variant') or 'none'}",
        "- Gold coverage is computed from exact member-passage intervals, never the bundle span.",
        "",
        "## Coverage",
        "",
        "| Variant | R@1 | R@3 | R@5 | Recovered | Regressed | Mean context hits | Cross-caption |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _coverage_row("baseline", report["baseline"]),
    ]
    for name, row in report["variants"].items():
        lines.append(_coverage_row(name, row))
    lines.extend(
        [
            "",
            "## Frozen Decision Rule",
            "",
            (
                f"Proceed only if the smallest variant reaches at least "
                f"{contract['target_recall_count']} cases at bundle@5 or recovers "
                f"{contract['target_recovery_count']} baseline-absent cases, with zero regressions."
            ),
            "",
            "## Validity",
            "",
            f"- Structural gate: {report['structural_gate_passed']}",
            f"- Structural error count: {len(report['structural_errors'])}",
            "- Retrieval seed IDs must match exactly across all variants.",
            "- Every context passage carries a proven same-source link to a seed passage.",
            "- A bundle may list multiple source IDs when a boundary passage itself spans segments.",
            "- Bundles preserve member-event boundaries and assert only temporal relatedness.",
            "- frozen39 remains exploratory and underpowered.",
        ]
    )
    return "\n".join(lines) + "\n"


def _variant_row(
    packet: Mapping[str, Any],
    bundle_set: Mapping[str, Any],
) -> dict[str, Any]:
    seed_hits = tuple(packet.get("seed_hits", packet.get("hits", ())) or ())
    hits = tuple(packet.get("hits", ()) or ())
    context_hits = tuple(
        hit
        for hit in hits
        if isinstance(hit, Mapping)
        and isinstance(hit.get("metadata"), Mapping)
        and hit["metadata"].get("context_expansion_contract")
        == CAPTION_CONTEXT_CONTRACT
    )
    return {
        "seed_hit_ids": [
            str(hit.get("passage_id", "") or "")
            for hit in seed_hits
            if isinstance(hit, Mapping)
        ],
        "seed_hit_count": len(seed_hits),
        "context_hit_count": len(context_hits),
        "cross_caption_context_count": sum(
            bool(hit["metadata"].get("cross_caption", False))
            for hit in context_hits
        ),
        "bundle_set": _compact_bundle_set(bundle_set),
    }


def _compact_bundle_set(bundle_set: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": str(bundle_set.get("schema_version", "") or ""),
        "context_contract": str(bundle_set.get("context_contract", "") or ""),
        "bundles": [
            {
                "bundle_id": str(bundle.get("bundle_id", "") or ""),
                "rank": int(bundle.get("rank", 0) or 0),
                "seed_passage_ids": list(bundle.get("seed_passage_ids", ()) or ()),
                "context_passage_ids": list(
                    bundle.get("context_passage_ids", ()) or ()
                ),
                "source_video_ids": list(
                    bundle.get("source_video_ids", ()) or ()
                ),
                "event_boundaries_preserved": bool(
                    bundle.get("event_boundaries_preserved", False)
                ),
                "member_passages": [
                    {
                        "passage_id": str(member.get("passage_id", "") or ""),
                        "time_range": list(member.get("time_range", ()) or ()),
                        "role": str(member.get("role", "") or ""),
                        "cross_caption": bool(
                            member.get("cross_caption", False)
                        ),
                        "context_links": [
                            dict(link)
                            for link in tuple(
                                member.get("context_links", ()) or ()
                            )
                            if isinstance(link, Mapping)
                        ],
                    }
                    for member in tuple(bundle.get("member_passages", ()) or ())
                    if isinstance(member, Mapping)
                ],
            }
            for bundle in tuple(bundle_set.get("bundles", ()) or ())
            if isinstance(bundle, Mapping)
        ],
    }


def _require_index_digest(
    packet: Mapping[str, Any],
    *,
    expected: str,
    case_id: str,
) -> None:
    if str(packet.get("index_digest", "") or "") != str(expected):
        raise ValueError(f"{case_id}: caption index digest mismatch")


def _coverage_row(name: str, row: Mapping[str, Any]) -> str:
    def cell(key: str) -> str:
        value = row[key]
        rate = value.get("rate")
        return f"{value['count']} ({100.0 * float(rate):.2f}%)" if rate is not None else "NA"

    cost = row.get("context_cost", {})
    return (
        f"| {name} | {cell('at_1')} | {cell('at_3')} | {cell('at_5')} | "
        f"{row.get('recovered_from_baseline_absent_count', 0)} | "
        f"{row.get('regressed_case_count', 0)} | "
        f"{float(cost.get('mean_context_hit_count', 0.0)):.3f} | "
        f"{int(cost.get('total_cross_caption_context_count', 0) or 0)} |"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--evaluation-record-root", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--expected-candidate-present", type=int, required=True)
    parser.add_argument("--expected-candidate-absent", type=int, required=True)
    parser.add_argument("--caption-config-digest", required=True)
    parser.add_argument("--caption-index-digest", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--seed-top-k", type=int, default=5)
    parser.add_argument("--neighbor-distances", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--context-max-gap-sec", type=float, default=180.0)
    parser.add_argument("--target-recall-count", type=int, default=22)
    parser.add_argument("--target-recovery-count", type=int, default=8)
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
    manifest_ids = set(
        _manifest_case_ids(json.loads(manifest_path.read_text(encoding="utf-8")))
    )
    if len(manifest_ids) != int(args.expected_cases):
        raise ValueError("case manifest count mismatch")
    distances = tuple(sorted({max(1, int(value)) for value in args.neighbor_distances}))
    replay = DeterministicCaptionReplay(
        model_id=str(args.embedding_model),
        revision=str(args.embedding_revision),
        device=str(args.embedding_device),
        batch_size=int(args.embedding_batch_size),
    )
    frozen_cases = load_cases(
        Path(args.run_root),
        evaluation_record_root=Path(args.evaluation_record_root),
        replay=replay,
        expected_caption_digest=str(args.caption_config_digest),
        expected_index_digest=str(args.caption_index_digest),
        replay_top_k=20,
    )
    frozen_report = build_candidate_coverage_report(
        frozen_cases,
        expected_cases=int(args.expected_cases),
        expected_candidate_present=int(args.expected_candidate_present),
        expected_candidate_absent=int(args.expected_candidate_absent),
    )
    if not frozen_report["structural_gate_passed"]:
        raise ValueError("frozen retrieval replay structural gate failed")
    cases = load_context_cases(
        Path(args.run_root),
        evaluation_record_root=Path(args.evaluation_record_root),
        replay=replay,
        expected_index_digest=str(args.caption_index_digest),
        seed_top_k=max(1, int(args.seed_top_k)),
        distances=distances,
        context_max_gap_sec=max(0.0, float(args.context_max_gap_sec)),
    )
    if {case["case_id"] for case in cases} != manifest_ids:
        raise ValueError("run cases do not match the frozen manifest")
    variants = tuple(f"neighbors_{distance}" for distance in distances)
    report = build_occurrence_context_expansion_report(
        cases,
        expected_cases=int(args.expected_cases),
        variant_order=variants,
        target_recall_count=int(args.target_recall_count),
        target_recovery_count=int(args.target_recovery_count),
    )
    report["provenance"] = {
        "label": str(args.provenance),
        "source_run_root": str(Path(args.run_root)),
        "case_manifest": str(manifest_path),
        "case_manifest_sha256": manifest_sha,
        "caption_config_digest": str(args.caption_config_digest),
        "caption_index_digest": str(args.caption_index_digest),
        "embedding_model": replay.adapter.model_id,
        "embedding_revision": replay.adapter.model_version,
        "seed_top_k": int(args.seed_top_k),
        "neighbor_distances": list(distances),
        "context_max_gap_sec": float(args.context_max_gap_sec),
        "generative_model_calls": 0,
        "vlm_calls": 0,
        "judge_calls": 0,
    }
    _write_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render_markdown(report), encoding="utf-8")
    print(
        "CONTEXT_EXPANSION_DONE "
        f"decision={report['decision']} "
        f"structural={report['structural_gate_passed']} "
        f"selected={report.get('selected_variant') or 'none'}",
        flush=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
