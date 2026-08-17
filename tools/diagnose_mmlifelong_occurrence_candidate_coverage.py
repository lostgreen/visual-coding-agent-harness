#!/usr/bin/env python3
"""Run the zero-generative-call WP16-0 occurrence candidate coverage audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from vcah.caption_lexical_index import normalize_caption_query
from vcah.embedding_adapter import SentenceTransformerEmbeddingAdapter
from vcah.investigator import VirtualVideoInvestigator
from vcah.occurrence_candidate_coverage import (
    FAILURE_CATEGORIES,
    RECALL_KS,
    build_candidate_coverage_report,
)
from vcah.virtual_video import VirtualVideoWorkspace


class DeterministicCaptionReplay:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: str,
        batch_size: int,
    ) -> None:
        self.adapter = SentenceTransformerEmbeddingAdapter(
            model_id,
            revision=revision,
            device=device,
            normalize=True,
            batch_size=batch_size,
        )
        self._investigators: dict[tuple[str, str, str], VirtualVideoInvestigator] = {}

    def search(
        self,
        workspace: VirtualVideoWorkspace,
        packet: Mapping[str, Any],
        queries: Sequence[str],
        *,
        top_k: int,
    ) -> Mapping[str, Any]:
        mode = str(packet.get("index_mode", "") or "").strip().casefold()
        strategy = str(packet.get("query_strategy", "") or "joint").strip().casefold()
        config_digest = str(packet.get("config_digest", "") or "").strip()
        key = (str(workspace.asset_root.resolve()), config_digest, strategy)
        investigator = self._investigators.get(key)
        if investigator is None:
            investigator = VirtualVideoInvestigator(
                workspace,
                caption_embedding_adapter=self.adapter,
                caption_config_digest=config_digest,
                caption_query_strategy=strategy,
            )
            self._investigators[key] = investigator
        time_range = _optional_interval(packet.get("time_range"))
        return investigator.search_caption(
            queries,
            time_range=time_range,
            segment_ids=tuple(packet.get("segment_ids", ()) or ()),
            source_video_ids=tuple(packet.get("source_video_ids", ()) or ()),
            top_k=top_k,
            expand_neighbors=int(packet.get("expand_neighbors", 0) or 0),
            index_mode=mode,
        )


def load_cases(
    run_root: Path,
    *,
    evaluation_record_root: Path,
    replay: DeterministicCaptionReplay,
    expected_caption_digest: str,
    expected_index_digest: str,
    replay_top_k: int,
) -> tuple[dict[str, Any], ...]:
    cases: list[dict[str, Any]] = []
    for prediction_path in sorted(Path(run_root).glob("cases/*/prediction.json")):
        run_dir = prediction_path.parent
        workspace = VirtualVideoWorkspace.load(run_dir)
        run_config = _read_json(run_dir / "run_config.json")
        case_id = workspace.case.case_id
        if str(run_config.get("caption_config_digest", "")) != expected_caption_digest:
            raise ValueError(f"{case_id}: caption config digest mismatch")
        embedding = run_config.get("embedding", {})
        if not isinstance(embedding, Mapping):
            raise ValueError(f"{case_id}: missing embedding identity")
        if str(embedding.get("model_id", "")) != replay.adapter.model_id:
            raise ValueError(f"{case_id}: embedding model mismatch")
        if str(embedding.get("model_version", "")) != replay.adapter.model_version:
            raise ValueError(f"{case_id}: embedding revision mismatch")

        state = _read_json(run_dir / "occurrence_resolution_state.json")
        observations = _read_jsonl(run_dir / "observation_log.jsonl")
        record = _read_json(
            Path(evaluation_record_root) / case_id / "evaluation_case.json"
        )
        clues = tuple(
            [float(interval[0]), float(interval[1])]
            for interval in tuple(record.get("clue_intervals", ()) or ())
            if _optional_interval(interval) is not None
        )
        packets: list[dict[str, Any]] = []
        normalized_queries: set[str] = set()
        query_contexts: dict[str, tuple[str, Mapping[str, Any]]] = {}
        for observation in observations:
            config = observation.get("sampling_config", {})
            if not isinstance(config, Mapping):
                continue
            if not isinstance(config.get("occurrence_set"), Mapping):
                continue
            frozen_packet = _caption_packet(run_dir, observation)
            occurrence_set = frozen_packet.get("occurrence_set")
            if not isinstance(occurrence_set, Mapping):
                continue
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
                "expand_neighbors": int(frozen_packet.get("expand_neighbors", 0) or 0),
            }
            observed_top_k = int(frozen_packet.get("top_k", 0) or 0)
            replay_recorded_depth = replay.search(
                workspace,
                packet_config,
                queries,
                top_k=observed_top_k,
            )
            replay_top20 = replay.search(
                workspace,
                packet_config,
                queries,
                top_k=replay_top_k,
            )
            observed_hit_ids = _hit_ids(tuple(frozen_packet.get("hits", ()) or ()))
            replay_hit_ids = _hit_ids(
                tuple(replay_recorded_depth.get("hits", ()) or ())
            )
            observed_occurrence_ids = _occurrence_ids(occurrence_set)
            replay_occurrence_ids = _occurrence_ids(
                replay_recorded_depth.get("occurrence_set", {})
            )
            replay_index_digest = str(
                replay_recorded_depth.get("index_digest", "") or ""
            )
            recorded_index_digest = str(frozen_packet.get("index_digest", "") or "")
            packets.append(
                {
                    "attempt_id": str(observation.get("attempt_id", "") or ""),
                    "observed_top_k": observed_top_k,
                    "recorded_depth_replay_match": (
                        observed_hit_ids == replay_hit_ids
                        and observed_occurrence_ids == replay_occurrence_ids
                    ),
                    "index_digest_match": (
                        recorded_index_digest == replay_index_digest
                        and replay_index_digest == expected_index_digest
                    ),
                    "observed_hits": [
                        _compact_hit(hit)
                        for hit in tuple(frozen_packet.get("hits", ()) or ())
                        if isinstance(hit, Mapping)
                    ],
                    "observed_candidates": [
                        _compact_candidate(candidate)
                        for candidate in tuple(
                            occurrence_set.get("candidates", ()) or ()
                        )
                        if isinstance(candidate, Mapping)
                    ],
                    "replay_hits": [
                        _compact_hit(hit)
                        for hit in tuple(replay_top20.get("hits", ()) or ())
                        if isinstance(hit, Mapping)
                    ],
                    "replay_candidates": [
                        _compact_candidate(candidate)
                        for candidate in tuple(
                            replay_top20.get("occurrence_set", {}).get("candidates", ())
                            or ()
                        )
                        if isinstance(candidate, Mapping)
                    ],
                }
            )
            for query in queries:
                normalized = normalize_caption_query(query)
                normalized_queries.add(normalized)
                context_digest = _digest(
                    {
                        "query": normalized,
                        "time_range": packet_config["time_range"],
                        "segment_ids": packet_config["segment_ids"],
                        "source_video_ids": packet_config["source_video_ids"],
                        "index_mode": packet_config["index_mode"],
                    }
                )
                query_contexts.setdefault(context_digest, (query, packet_config))

        top1_candidates: list[dict[str, Any]] = []
        for context_digest, (query, packet_config) in sorted(query_contexts.items()):
            top1 = replay.search(
                workspace,
                packet_config,
                (query,),
                top_k=1,
            )
            candidates = tuple(
                top1.get("occurrence_set", {}).get("candidates", ()) or ()
            )
            if candidates and isinstance(candidates[0], Mapping):
                candidate = _compact_candidate(candidates[0])
                candidate["query_context_digest"] = context_digest
                top1_candidates.append(candidate)

        final_candidates, retired_set_ids = _final_state_candidates(state)
        cases.append(
            {
                "case_id": case_id,
                "clues": list(clues),
                "packets": packets,
                "final_candidates": final_candidates,
                "retired_set_ids": retired_set_ids,
                "normalized_query_count": len(normalized_queries),
                "query_context_count": len(query_contexts),
                "query_top1_candidates": top1_candidates,
                "replay_available": bool(packets),
            }
        )
    return tuple(cases)


def render_markdown(report: Mapping[str, Any]) -> str:
    cohort = report["cohort"]
    recall = report["recall"]
    partition = report["candidate_absent_failure_partition"]["categories"]
    crowding = report["duplicate_crowding"]
    query = report["query_coverage"]
    branch = report["branch_evidence"]
    lines = [
        "# MM-Lifelong WP16-0 Candidate Coverage Audit",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "This audit makes no generative-model, VLM, or judge calls. It first reproduces each frozen packet at its recorded depth, then runs the same deterministic caption retriever at depth 20.",
        "",
        f"Structural gate passed: **{report['structural_gate_passed']}**",
        f"Cohort: **{cohort['case_count']}** cases; candidate present **{cohort['candidate_present_count']}**; candidate absent **{cohort['candidate_absent_count']}**.",
        "",
        "## Candidate Recall",
        "",
        "| Scope | R@1 | R@3 | R@5 | R@10 | R@20 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Final scoped", "final_scoped"),
        ("Observed trajectory", "observed_trajectory"),
        ("Counterfactual depth-20", "counterfactual_top20"),
    ):
        row = recall[key]
        values = [
            _fmt_recall(row.get(f"at_{k}", {})) if f"at_{k}" in row else "NA"
            for k in RECALL_KS
        ]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Observed packets have heterogeneous depths. Each observed R@K uses only cases with at least one recorded packet at depth K or deeper; the depth-20 row uses all 39 cases.",
            "",
            "## Candidate-Absent Partition",
            "",
            "| Failure category | Count | Rate |",
            "|---|---:|---:|",
        ]
    )
    for category in FAILURE_CATEGORIES:
        row = partition[category]
        lines.append(f"| {category} | {row['count']} | {_fmt(row['rate'])} |")
    lines.extend(
        [
            "",
            "## Duplicate Crowding",
            "",
            "| Retrieval view | Hit slots | Occurrence clusters | Slots lost | Loss rate | Affected cases |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    observed = crowding["observed_first5"]
    lines.append(
        f"| Observed top-5 | {observed['hit_slots']} | {observed['occurrence_clusters']} | "
        f"{observed['slots_consumed_by_same_occurrence']} | {_fmt(observed['slot_loss_rate'])} | "
        f"{observed['affected_case_count']} |"
    )
    for k in RECALL_KS:
        row = crowding["counterfactual_top20"][f"at_{k}"]
        lines.append(
            f"| Replay top-{k} | {row['hit_slots']} | {row['occurrence_clusters']} | "
            f"{row['slots_consumed_by_same_occurrence']} | {_fmt(row['slot_loss_rate'])} | "
            f"{row['affected_case_count']} |"
        )
    lines.extend(
        [
            "",
            "## Query Coverage",
            "",
            f"Candidate-absent cases: **{query['candidate_absent_case_count']}**.",
            f"Mean normalized queries: **{_fmt(query['mean_normalized_query_count'])}**; mean top-1 episodes: **{_fmt(query['mean_top1_episode_count'])}**.",
            f"Retrieved-episode collapse: **{query['episode_collapse_count']}** cases ({_fmt(query['episode_collapse_rate'])}).",
            "Semantic-template collapse is not inferred from hidden query text.",
            "",
            "## Frozen Branch Rule",
            "",
            f"A branch requires **{branch['dominance_case_count_required']}** of the candidate-absent cases (60%).",
            f"Coverage-preservation evidence: **{branch['coverage_preservation_case_count']}** ({_fmt(branch['coverage_preservation_rate'])}).",
            f"Query/representation evidence: **{branch['query_or_representation_case_count']}** ({_fmt(branch['query_or_representation_rate'])}).",
            "",
            "Recall@K and crowding are diagnostics only; K was not tuned on these outcomes. R5 and all behavioral policies remain unchanged.",
            "",
            "frozen39 is underpowered and repeatedly used for mechanism development. Day-test140 and Week remain sealed.",
            "",
        ]
    )
    return "\n".join(lines)


def _final_state_candidates(
    state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    active_set_id = str(state.get("active_set_id", "") or "")
    retired_ids = [
        str(value) for value in tuple(state.get("retired_set_ids", ()) or ())
    ]
    for raw_set in tuple(state.get("sets", ()) or ()):
        if not isinstance(raw_set, Mapping):
            continue
        if str(raw_set.get("set_id", "") or "") != active_set_id:
            continue
        candidates = [
            _compact_candidate(candidate)
            for candidate in tuple(raw_set.get("candidates", ()) or ())
            if isinstance(candidate, Mapping)
        ]
        return candidates, retired_ids
    return [], retired_ids


def _caption_packet(run_dir: Path, observation: Mapping[str, Any]) -> dict[str, Any]:
    raw_output = observation.get("raw_output")
    if isinstance(raw_output, Mapping):
        payload = raw_output
    elif isinstance(raw_output, str) and raw_output.strip():
        value = json.loads(raw_output)
        payload = value if isinstance(value, Mapping) else {}
    else:
        payload = {}
    pointer = str(payload.get("raw_output_pointer", "") or "").strip()
    if not pointer:
        raise ValueError(f"{run_dir.name}: caption observation missing packet pointer")
    packet_path = Path(pointer)
    if not packet_path.is_absolute():
        packet_path = Path(run_dir) / packet_path
    packet_path = packet_path.resolve()
    caption_root = (Path(run_dir) / "caption_search").resolve()
    if packet_path.parent != caption_root:
        raise ValueError(f"{run_dir.name}: caption packet escaped case directory")
    return _read_json(packet_path)


def _compact_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "occurrence_id": str(candidate.get("occurrence_id", "") or ""),
        "rank": int(candidate.get("rank", 0) or 0),
        "time_range": list(candidate.get("time_range", ()) or ()),
        "source_video_ids": [
            str(value) for value in tuple(candidate.get("source_video_ids", ()) or ())
        ],
        "segment_ids": [
            str(value) for value in tuple(candidate.get("segment_ids", ()) or ())
        ],
        "passage_ids": [
            str(value) for value in tuple(candidate.get("passage_ids", ()) or ())
        ],
    }


def _compact_hit(hit: Mapping[str, Any]) -> dict[str, Any]:
    metadata = hit.get("metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return {
        "passage_id": str(hit.get("passage_id", "") or ""),
        "caption_id": str(hit.get("caption_id", "") or ""),
        "rank": int(hit.get("rank", 0) or 0),
        "fused_score": float(hit.get("fused_score", 0.0) or 0.0),
        "virtual_start_sec": float(hit.get("virtual_start_sec", 0.0) or 0.0),
        "virtual_end_sec": float(hit.get("virtual_end_sec", 0.0) or 0.0),
        "metadata": {
            "source_video_ids": [
                str(value)
                for value in tuple(metadata.get("source_video_ids", ()) or ())
            ],
            "source_segments": [
                str(value) for value in tuple(metadata.get("source_segments", ()) or ())
            ],
        },
    }


def _hit_ids(hits: Sequence[Any]) -> tuple[str, ...]:
    return tuple(
        str(hit.get("passage_id", "") or "") for hit in hits if isinstance(hit, Mapping)
    )


def _occurrence_ids(occurrence_set: Any) -> tuple[str, ...]:
    if not isinstance(occurrence_set, Mapping):
        return ()
    return tuple(
        str(candidate.get("occurrence_id", "") or "")
        for candidate in tuple(occurrence_set.get("candidates", ()) or ())
        if isinstance(candidate, Mapping)
    )


def _manifest_case_ids(payload: Any) -> tuple[str, ...]:
    if isinstance(payload, Mapping):
        raw = payload.get("case_ids", payload.get("cases", ()))
    else:
        raw = payload
    case_ids: list[str] = []
    for row in tuple(raw or ()):
        if isinstance(row, Mapping):
            value = row.get("case_id", row.get("id", ""))
        else:
            value = row
        if str(value or ""):
            case_ids.append(str(value))
    return tuple(case_ids)


def _optional_interval(value: Any) -> tuple[float, float] | None:
    try:
        if value is None or len(value) != 2:
            return None
        start, end = sorted((float(value[0]), float(value[1])))
    except (TypeError, ValueError):
        return None
    return (start, end) if end > start else None


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    return "NA" if not isinstance(value, (int, float)) else f"{float(value):.4f}"


def _fmt_recall(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "NA"
    rate = value.get("rate")
    count = value.get("count")
    eligible = value.get("eligible_case_count")
    if not isinstance(rate, (int, float)):
        return "NA"
    return f"{float(rate):.4f} ({count}/{eligible})"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(value)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and isinstance((value := json.loads(line)), Mapping)
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    parser.add_argument("--replay-top-k", type=int, default=20)
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
    if int(args.replay_top_k) != 20:
        raise ValueError("WP16-0 replay depth is frozen at 20")

    replay = DeterministicCaptionReplay(
        model_id=str(args.embedding_model),
        revision=str(args.embedding_revision),
        device=str(args.embedding_device),
        batch_size=int(args.embedding_batch_size),
    )
    cases = load_cases(
        Path(args.run_root),
        evaluation_record_root=Path(args.evaluation_record_root),
        replay=replay,
        expected_caption_digest=str(args.caption_config_digest),
        expected_index_digest=str(args.caption_index_digest),
        replay_top_k=int(args.replay_top_k),
    )
    run_ids = {case["case_id"] for case in cases}
    if run_ids != manifest_ids:
        raise ValueError("run cases do not match the frozen manifest")
    report = build_candidate_coverage_report(
        cases,
        expected_cases=int(args.expected_cases),
        expected_candidate_present=int(args.expected_candidate_present),
        expected_candidate_absent=int(args.expected_candidate_absent),
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
        "generative_model_calls": 0,
        "judge_calls": 0,
    }
    _write_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render_markdown(report), encoding="utf-8")
    partition = report["candidate_absent_failure_partition"]["categories"]
    partition_counts = ",".join(
        f"{key}:{partition[key]['count']}" for key in FAILURE_CATEGORIES
    )
    print(
        "CANDIDATE_COVERAGE_DONE "
        f"decision={report['decision']} "
        f"structural={report['structural_gate_passed']} "
        f"present={report['cohort']['candidate_present_count']} "
        f"absent={report['cohort']['candidate_absent_count']} "
        f"partition={partition_counts}",
        flush=True,
    )


if __name__ == "__main__":
    main()
