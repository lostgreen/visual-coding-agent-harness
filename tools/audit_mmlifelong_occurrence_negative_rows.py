#!/usr/bin/env python3
"""Prepare and summarize a blinded audit of negative-sidecar rows."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Mapping, Sequence

from vcah.occurrence_negative_sidecar import (
    load_negative_sidecar_snapshot,
    stable_digest,
)


VALID_VERDICTS = frozenset({"true_contradiction", "false_contradiction", "unclear"})
BLIND_JUDGMENT_PROTOCOL = {
    "name": "citation_grounded_contradiction_precision",
    "unit": "one emitted contradiction row",
    "input": [
        "question and options",
        "one stated constraint",
        "one candidate's visible caption passages",
        "the cited passage subset",
    ],
    "verdicts": {
        "true_contradiction": (
            "The cited visible text directly provides evidence that the candidate "
            "does not satisfy the stated constraint."
        ),
        "false_contradiction": (
            "The cited text merely lacks support, is irrelevant, is ambiguous, "
            "or supports the candidate, without directly contradicting it."
        ),
        "unclear": "The cited text is insufficient for a reliable judgment.",
    },
    "non_negotiable_rule": "Absence of support is not contradiction.",
    "scope_note": (
        "This audit measures whether the cited caption evidence grounds the "
        "contradiction claim; it does not establish video-world truth."
    ),
}


def prepare_audit(
    repeats: Mapping[str, Path],
    *,
    positive_run_root: Path,
    replay_fixture_root: Path,
    frozen_rows: Sequence[Mapping[str, Any]] = (),
    reliability_fraction: float = 0.20,
    seed: int = 20260817,
) -> tuple[dict[str, Any], dict[str, Any]]:
    blind_items: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    case_ids = tuple(
        sorted(
            {
                path.parent.name
                for root in repeats.values()
                for path in Path(root).glob("cases/*/sidecar_result.json")
            }
        )
    )
    snapshots = {
        case_id: load_negative_sidecar_snapshot(
            Path(positive_run_root) / "cases" / case_id,
            replay_fixture_path=(
                Path(replay_fixture_root) / "cases" / f"{case_id}.json"
            ),
        )
        for case_id in case_ids
    }
    winner_cases = _winner_case_metadata(frozen_rows, case_ids=case_ids)
    for repeat_label, root in repeats.items():
        for path in sorted(Path(root).glob("cases/*/sidecar_result.json")):
            result = _read_json(path)
            case_id = str(result.get("case_id", path.parent.name) or path.parent.name)
            snapshot = snapshots[case_id]
            constraints = {
                str(row["constraint_id"]): row for row in snapshot.constraints
            }
            candidates = {
                str(row.get("occurrence_id", "") or ""): row
                for row in snapshot.candidates
            }
            for row_index, row in enumerate(
                tuple(result.get("contradiction_rows", ()) or ())
            ):
                if not isinstance(row, Mapping):
                    continue
                constraint_id = str(row.get("constraint_id", "") or "")
                occurrence_id = str(row.get("occurrence_id", "") or "")
                if constraint_id not in constraints or occurrence_id not in candidates:
                    raise ValueError(
                        f"{case_id}: audited row is outside frozen snapshot"
                    )
                cited_ids = tuple(
                    dict.fromkeys(
                        str(value)
                        for value in tuple(row.get("evidence_passage_ids", ()) or ())
                        if str(value)
                    )
                )
                semantic_claim_digest = stable_digest(
                    {
                        "case_id": case_id,
                        "constraint_id": constraint_id,
                        "occurrence_id": occurrence_id,
                        "cited_passage_ids": sorted(cited_ids),
                    }
                )
                winner_case = winner_cases.get(case_id, {})
                occurrence_id_digest = stable_digest(occurrence_id)
                targets_selected_winner = bool(
                    occurrence_id_digest
                    == str(winner_case.get("selected_occurrence_id_digest", "") or "")
                )
                item_id = stable_digest(
                    {
                        "repeat": repeat_label,
                        "case_id": case_id,
                        "row_index": row_index,
                        "constraint_id": constraint_id,
                        "occurrence_id": occurrence_id,
                        "cited_ids": cited_ids,
                    }
                )[:24]
                candidate = candidates[occurrence_id]
                blind_items.append(
                    {
                        "audit_item_id": item_id,
                        "question": snapshot.question,
                        "options": snapshot.options,
                        "constraint": {
                            "constraint_type": constraints[constraint_id][
                                "constraint_type"
                            ],
                            "description": constraints[constraint_id]["description"],
                        },
                        "candidate_label": "candidate-"
                        + stable_digest(occurrence_id)[:10],
                        "candidate_passages": [
                            {
                                "passage_id": str(passage.get("passage_id", "") or ""),
                                "time_range": list(passage.get("time_range", ()) or ()),
                                "caption_excerpt": str(
                                    passage.get("caption_excerpt", "") or ""
                                ),
                                "cited": str(passage.get("passage_id", "") or "")
                                in cited_ids,
                            }
                            for passage in tuple(
                                candidate.get("representative_passages", ()) or ()
                            )
                            if isinstance(passage, Mapping)
                        ],
                        "audit_question": (
                            "Do the cited visible passages directly contradict this "
                            "candidate on the stated constraint? Absence of support "
                            "does not count as contradiction."
                        ),
                        "allowed_verdicts": sorted(VALID_VERDICTS),
                    }
                )
                key_rows.append(
                    {
                        "audit_item_id": item_id,
                        "repeat_label": repeat_label,
                        "case_id": case_id,
                        "constraint_id": constraint_id,
                        "constraint_type": constraints[constraint_id][
                            "constraint_type"
                        ],
                        "occurrence_id_digest": occurrence_id_digest,
                        "semantic_claim_digest": semantic_claim_digest,
                        "targets_selected_winner": targets_selected_winner,
                        "cited_passage_count": len(cited_ids),
                    }
                )
    blind_items.sort(key=lambda row: row["audit_item_id"])
    key_rows.sort(key=lambda row: row["audit_item_id"])
    fraction = max(0.0, min(1.0, float(reliability_fraction)))
    reliability_count = min(
        len(blind_items), int(math.ceil(len(blind_items) * fraction))
    )
    rng = random.Random(seed)
    reliability_sample_item_ids = sorted(
        rng.sample(
            [str(row["audit_item_id"]) for row in blind_items],
            reliability_count,
        )
    )
    protocol_digest = stable_digest(BLIND_JUDGMENT_PROTOCOL)
    return (
        {
            "schema_version": "MMLifelongOccurrenceNegativeRowBlindAuditV2",
            "item_count": len(blind_items),
            "judgment_protocol": BLIND_JUDGMENT_PROTOCOL,
            "judgment_protocol_digest": protocol_digest,
            "items": blind_items,
            "blinding_checks": {
                "selection_outcomes_absent": True,
                "reference_labels_absent": True,
            },
        },
        {
            "schema_version": "MMLifelongOccurrenceNegativeRowAuditKeyV2",
            "item_count": len(key_rows),
            "rows": key_rows,
            "winner_cases": [winner_cases[case_id] for case_id in sorted(winner_cases)],
            "judgment_protocol_digest": protocol_digest,
            "reliability_fraction": fraction,
            "reliability_sample_count": reliability_count,
            "reliability_sample_seed": int(seed),
            "reliability_sample_item_ids": reliability_sample_item_ids,
            "blind_items_digest": stable_digest(blind_items),
        },
    )


def analyze_judgments(
    key: Mapping[str, Any],
    judgments: Mapping[str, Any],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    key_rows = {
        str(row.get("audit_item_id", "") or ""): row
        for row in tuple(key.get("rows", ()) or ())
        if isinstance(row, Mapping)
    }
    judgment_rows = _rows_by_item_id(judgments.get("judgments", ()))
    unknown_ids = sorted(set(judgment_rows) - set(key_rows))
    if unknown_ids:
        raise ValueError("judgments contain unknown audit item IDs")
    normalized_by_id, invalid_ids = _normalize_judgments(key_rows, judgment_rows)
    normalized = list(normalized_by_id.values())
    primary_precision = _precision_report(
        normalized, samples=bootstrap_samples, seed=seed
    )
    unique_claim_rows, discordant_claim_count = _unique_semantic_claim_rows(normalized)
    unique_claim_precision = _precision_report(
        unique_claim_rows, samples=bootstrap_samples, seed=seed + 37
    )
    by_type = {
        constraint_type: _precision_report(
            [row for row in normalized if row["constraint_type"] == constraint_type],
            samples=bootstrap_samples,
            seed=seed + 101 + index,
        )
        for index, constraint_type in enumerate(
            sorted({str(row["constraint_type"]) for row in normalized})
        )
    }
    unique_by_type = {
        constraint_type: _precision_report(
            [
                row
                for row in unique_claim_rows
                if row["constraint_type"] == constraint_type
            ],
            samples=bootstrap_samples,
            seed=seed + 503 + index,
        )
        for index, constraint_type in enumerate(
            sorted({str(row["constraint_type"]) for row in unique_claim_rows})
        )
    }
    reliability = _reliability_report(
        key,
        normalized_by_id,
        judgments.get("reliability_judgments", ()),
    )
    winner_discrimination = _winner_discrimination_report(
        key_rows,
        normalized_by_id,
        key.get("winner_cases", ()),
        bootstrap_samples=bootstrap_samples,
        seed=seed + 1009,
    )
    missing_ids = sorted(set(key_rows) - set(judgment_rows))
    expected_protocol_digest = str(key.get("judgment_protocol_digest", "") or "")
    protocol_digest_match = bool(
        not expected_protocol_digest
        or str(judgments.get("judgment_protocol_digest", "") or "")
        == expected_protocol_digest
    )
    primary_complete = bool(
        len(normalized) == len(key_rows) and not invalid_ids and not missing_ids
    )
    complete = bool(
        primary_complete
        and protocol_digest_match
        and reliability["complete"]
        and winner_discrimination["complete"]
    )
    return {
        "schema_version": "MMLifelongOccurrenceNegativeRowAuditAnalysisV2",
        "metric_name": "citation_grounded_contradiction_precision",
        "scope_note": BLIND_JUDGMENT_PROTOCOL["scope_note"],
        "expected_item_count": len(key_rows),
        "judgment_count": len(judgment_rows),
        "valid_judgment_count": len(normalized),
        "invalid_judgment_item_ids": invalid_ids,
        "missing_judgment_count": len(missing_ids),
        "protocol_digest_match": protocol_digest_match,
        "emitted_row_precision": primary_precision,
        "unique_semantic_claim_count": len(unique_claim_rows),
        "duplicate_emitted_row_count": len(normalized) - len(unique_claim_rows),
        "discordant_duplicate_claim_count": discordant_claim_count,
        "duplicate_claim_disagreement_policy": (
            "Any non-unanimous duplicate-claim labels resolve to unclear."
        ),
        "unique_semantic_claim_precision": unique_claim_precision,
        # Compatibility aliases for the V1 report consumers.
        "unclear_count": primary_precision["unclear_count"],
        "true_contradiction_count": primary_precision["true_count"],
        "false_contradiction_count": primary_precision["false_count"],
        "row_precision": primary_precision["precision"],
        "row_precision_wilson95": primary_precision["wilson95"],
        "case_cluster_bootstrap": primary_precision["case_cluster_bootstrap"],
        "by_constraint_type": by_type,
        "unique_by_constraint_type": unique_by_type,
        "reliability": reliability,
        "reliability_disagreement_policy": (
            "Primary labels define endpoints; repeat labels measure reliability "
            "and never replace primary labels."
        ),
        "winner_discrimination": winner_discrimination,
        "primary_complete": primary_complete,
        "complete": complete,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    row_precision = report["emitted_row_precision"]
    claim_precision = report["unique_semantic_claim_precision"]
    reliability = report["reliability"]
    lines = [
        "# WP12 Citation-Grounded Contradiction Audit",
        "",
        f"Complete: **{report['complete']}**",
        "",
        str(report["scope_note"]),
        "",
        (
            f"Judgments: {report['valid_judgment_count']}/"
            f"{report['expected_item_count']}; unclear: {row_precision['unclear_count']}."
        ),
        (
            "Emitted-row precision (Wilson 95%): "
            f"{_fmt(row_precision['precision'])} "
            f"{_fmt_ci(row_precision['wilson95'])}."
        ),
        (
            "Emitted-row case-cluster bootstrap 95% CI: "
            f"{_fmt_ci(row_precision['case_cluster_bootstrap']['ci95'])}."
        ),
        (
            "Unique semantic-claim precision (Wilson/bootstrap 95%): "
            f"{_fmt(claim_precision['precision'])} "
            f"{_fmt_ci(claim_precision['wilson95'])} / "
            f"{_fmt_ci(claim_precision['case_cluster_bootstrap']['ci95'])}."
        ),
        (
            "Unique claims / duplicate emitted rows / discordant duplicate claims: "
            f"{report['unique_semantic_claim_count']} / "
            f"{report['duplicate_emitted_row_count']} / "
            f"{report['discordant_duplicate_claim_count']}."
        ),
        "",
        "## Reliability",
        "",
        (
            f"Independent rejudgments: {reliability['paired_count']}/"
            f"{reliability['expected_count']}; exact agreement "
            f"{_fmt(reliability['agreement_rate'])} "
            f"{_fmt_ci(reliability['agreement_wilson95'])}; "
            f"Cohen's kappa {_fmt(reliability['cohen_kappa'])}."
        ),
        str(report["reliability_disagreement_policy"]),
        "",
        "## Precision by constraint type",
        "",
        "| Constraint type | Emitted true/false/unclear | Emitted precision (Wilson/bootstrap 95%) | Unique precision (Wilson/bootstrap 95%) |",
        "|---|---:|---:|---:|",
    ]
    for constraint_type, row in report["by_constraint_type"].items():
        unique = report["unique_by_constraint_type"].get(constraint_type, {})
        lines.append(
            f"| {constraint_type} | {row['true_count']}/{row['false_count']}/"
            f"{row['unclear_count']} | {_fmt(row['precision'])} "
            f"{_fmt_ci(row['wilson95'])} / "
            f"{_fmt_ci(row['case_cluster_bootstrap']['ci95'])} | "
            f"{_fmt(unique.get('precision'))} "
            f"{_fmt_ci(unique.get('wilson95', ()))} / "
            f"{_fmt_ci(unique.get('case_cluster_bootstrap', {}).get('ci95', ()))} |"
        )
    discrimination = report["winner_discrimination"]
    if not discrimination.get("available"):
        lines.extend(
            [
                "",
                "## Post-Unblind Winner Discrimination",
                "",
                "Winner metadata unavailable; no discrimination result was computed.",
            ]
        )
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "",
            "## Post-Unblind Winner Discrimination",
            "",
            (
                "Conclusion: **"
                f"{discrimination['conclusion']}"
                "**. Raw and validated endpoint values were not validity gates."
            ),
            "",
            "| Repeat | Signal | False winner | Candidate-present winner | Strict-correct winner | False-candidate gap (95%) | False-strict gap (95%) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for repeat_label, repeat in discrimination["per_repeat"].items():
        for signal in ("raw", "validated"):
            row = repeat[signal]
            lines.append(
                f"| {repeat_label} | {signal} | "
                f"{row['false_hit_count']}/{row['false_count']} | "
                f"{row['candidate_present_hit_count']}/"
                f"{row['candidate_present_count']} | "
                f"{row['strict_correct_hit_count']}/{row['strict_correct_count']} | "
                f"{_fmt(row['false_candidate_gap'])} "
                f"{_fmt_ci(row['false_candidate_gap_bootstrap']['ci95'])} | "
                f"{_fmt(row['false_strict_gap'])} "
                f"{_fmt_ci(row['false_strict_gap_bootstrap']['ci95'])} |"
            )
    stability = discrimination["validated_repeat_stability"]
    lines.extend(
        [
            "",
            (
                "Validated winner-flag repeat agreement: "
                f"{stability['agreement_count']}/{stability['case_count']} "
                f"({_fmt(stability['agreement_rate'])}), "
                f"Cohen's kappa {_fmt(stability['cohen_kappa'])}."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _precision_report(
    rows: Sequence[Mapping[str, Any]], *, samples: int, seed: int
) -> dict[str, Any]:
    adjudicated = [row for row in rows if row["verdict"] != "unclear"]
    true_count = sum(row["verdict"] == "true_contradiction" for row in adjudicated)
    false_count = len(adjudicated) - true_count
    return {
        "true_count": true_count,
        "false_count": false_count,
        "unclear_count": len(rows) - len(adjudicated),
        "precision": true_count / len(adjudicated) if adjudicated else None,
        "wilson95": _wilson_interval(true_count, len(adjudicated)),
        "case_cluster_bootstrap": _case_cluster_bootstrap(
            rows, samples=samples, seed=seed
        ),
    }


def _rows_by_item_id(values: Any) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("audit_item_id", "") or ""): row
        for row in tuple(values or ())
        if isinstance(row, Mapping) and str(row.get("audit_item_id", "") or "")
    }


def _normalize_judgments(
    key_rows: Mapping[str, Mapping[str, Any]],
    judgment_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    normalized: dict[str, dict[str, Any]] = {}
    invalid_ids: list[str] = []
    for item_id, row in judgment_rows.items():
        verdict = str(row.get("verdict", "") or "").strip().casefold()
        if verdict not in VALID_VERDICTS:
            invalid_ids.append(item_id)
            continue
        normalized[item_id] = {**dict(key_rows[item_id]), "verdict": verdict}
    return normalized, sorted(invalid_ids)


def _unique_semantic_claim_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        digest = str(
            row.get("semantic_claim_digest", row.get("audit_item_id", "")) or ""
        )
        grouped.setdefault(digest, []).append(row)
    unique_rows: list[dict[str, Any]] = []
    discordant = 0
    for digest, group in sorted(grouped.items()):
        verdicts = {str(row["verdict"]) for row in group}
        if len(verdicts) == 1:
            verdict = next(iter(verdicts))
        else:
            verdict = "unclear"
            discordant += 1
        unique_rows.append(
            {
                **dict(group[0]),
                "semantic_claim_digest": digest,
                "verdict": verdict,
                "emitted_row_count": len(group),
            }
        )
    return unique_rows, discordant


def _reliability_report(
    key: Mapping[str, Any],
    primary: Mapping[str, Mapping[str, Any]],
    reliability_values: Any,
) -> dict[str, Any]:
    expected_ids = tuple(
        str(value)
        for value in tuple(key.get("reliability_sample_item_ids", ()) or ())
        if str(value)
    )
    reliability_rows = _rows_by_item_id(reliability_values)
    unknown_ids = sorted(set(reliability_rows) - set(expected_ids))
    if unknown_ids:
        raise ValueError("reliability judgments contain unsampled audit item IDs")
    invalid_ids: list[str] = []
    normalized: dict[str, str] = {}
    for item_id, row in reliability_rows.items():
        verdict = str(row.get("verdict", "") or "").strip().casefold()
        if verdict not in VALID_VERDICTS:
            invalid_ids.append(item_id)
            continue
        normalized[item_id] = verdict
    paired_ids = tuple(
        item_id
        for item_id in expected_ids
        if item_id in normalized and item_id in primary
    )
    pairs = [
        (str(primary[item_id]["verdict"]), normalized[item_id])
        for item_id in paired_ids
    ]
    agreements = sum(left == right for left, right in pairs)
    confusion = {
        left: {
            right: sum(pair == (left, right) for pair in pairs)
            for right in sorted(VALID_VERDICTS)
        }
        for left in sorted(VALID_VERDICTS)
    }
    missing_ids = sorted(set(expected_ids) - set(reliability_rows))
    return {
        "expected_count": len(expected_ids),
        "judgment_count": len(reliability_rows),
        "paired_count": len(pairs),
        "agreement_count": agreements,
        "agreement_rate": agreements / len(pairs) if pairs else None,
        "agreement_wilson95": _wilson_interval(agreements, len(pairs)),
        "cohen_kappa": _cohen_kappa(pairs),
        "confusion_matrix": confusion,
        "invalid_judgment_item_ids": sorted(invalid_ids),
        "missing_judgment_count": len(missing_ids),
        "complete": bool(
            len(pairs) == len(expected_ids) and not invalid_ids and not missing_ids
        ),
    }


def _cohen_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(left == right for left, right in pairs) / total
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(
        (left_counts[label] / total) * (right_counts[label] / total)
        for label in VALID_VERDICTS
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def _winner_discrimination_report(
    key_rows: Mapping[str, Mapping[str, Any]],
    judgments: Mapping[str, Mapping[str, Any]],
    winner_case_values: Any,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    winner_cases = {
        str(row.get("case_id", "") or ""): row
        for row in tuple(winner_case_values or ())
        if isinstance(row, Mapping) and str(row.get("case_id", "") or "")
    }
    if not winner_cases:
        return {
            "available": False,
            "complete": False,
            "conclusion": "WINNER_METADATA_UNAVAILABLE",
            "per_repeat": {},
        }
    false_ids = tuple(
        case_id
        for case_id, row in winner_cases.items()
        if row.get("winner_class") == "false_winner"
    )
    candidate_ids = tuple(
        case_id
        for case_id, row in winner_cases.items()
        if row.get("winner_class") == "candidate_present_winner"
    )
    strict_ids = tuple(
        case_id
        for case_id in candidate_ids
        if winner_cases[case_id].get("strict_correct") is True
    )
    repeat_labels = tuple(
        sorted(
            {
                str(row.get("repeat_label", "") or "")
                for row in key_rows.values()
                if str(row.get("repeat_label", "") or "")
            }
        )
    )
    per_repeat: dict[str, Any] = {}
    raw_hits_by_repeat: dict[str, set[str]] = {}
    validated_hits_by_repeat: dict[str, set[str]] = {}
    for repeat_index, repeat_label in enumerate(repeat_labels):
        repeat_rows = [
            row
            for row in key_rows.values()
            if row.get("repeat_label") == repeat_label
            and row.get("targets_selected_winner") is True
        ]
        raw_hits = {str(row["case_id"]) for row in repeat_rows}
        validated_hits = {
            str(row["case_id"])
            for row in repeat_rows
            if str(row.get("audit_item_id", "") or "") in judgments
            and judgments[str(row["audit_item_id"])]["verdict"] == "true_contradiction"
        }
        raw_hits_by_repeat[repeat_label] = raw_hits
        validated_hits_by_repeat[repeat_label] = validated_hits
        per_repeat[repeat_label] = {
            "raw": _winner_signal_metrics(
                raw_hits,
                false_ids=false_ids,
                candidate_ids=candidate_ids,
                strict_ids=strict_ids,
                samples=bootstrap_samples,
                seed=seed + repeat_index * 211,
            ),
            "validated": _winner_signal_metrics(
                validated_hits,
                false_ids=false_ids,
                candidate_ids=candidate_ids,
                strict_ids=strict_ids,
                samples=bootstrap_samples,
                seed=seed + repeat_index * 211 + 97,
            ),
        }
    primary_complete = len(judgments) == len(key_rows)
    candidate_lower_bounds = [
        row["validated"]["false_candidate_gap_bootstrap"]["ci95"][0]
        for row in per_repeat.values()
    ]
    strict_lower_bounds = [
        row["validated"]["false_strict_gap_bootstrap"]["ci95"][0]
        for row in per_repeat.values()
    ]
    established = bool(
        primary_complete
        and candidate_lower_bounds
        and strict_lower_bounds
        and all(
            value is not None and value > 0
            for value in (*candidate_lower_bounds, *strict_lower_bounds)
        )
    )
    scored_ids = tuple((*false_ids, *candidate_ids))
    return {
        "available": True,
        "complete": primary_complete,
        "false_winner_count": len(false_ids),
        "candidate_present_winner_count": len(candidate_ids),
        "strict_correct_winner_count": len(strict_ids),
        "per_repeat": per_repeat,
        "raw_repeat_stability": _binary_repeat_stability(
            raw_hits_by_repeat, case_ids=scored_ids
        ),
        "validated_repeat_stability": _binary_repeat_stability(
            validated_hits_by_repeat, case_ids=scored_ids
        ),
        "validated_discrimination_established": established,
        "conclusion": (
            "VALIDATED_WINNER_DISCRIMINATION_ESTABLISHED"
            if established
            else "VALIDATED_WINNER_DISCRIMINATION_NOT_ESTABLISHED"
        ),
    }


def _winner_signal_metrics(
    hit_ids: set[str],
    *,
    false_ids: Sequence[str],
    candidate_ids: Sequence[str],
    strict_ids: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    false_hits = sum(case_id in hit_ids for case_id in false_ids)
    candidate_hits = sum(case_id in hit_ids for case_id in candidate_ids)
    strict_hits = sum(case_id in hit_ids for case_id in strict_ids)
    false_rate = false_hits / len(false_ids) if false_ids else None
    candidate_rate = candidate_hits / len(candidate_ids) if candidate_ids else None
    strict_rate = strict_hits / len(strict_ids) if strict_ids else None
    return {
        "false_count": len(false_ids),
        "false_hit_count": false_hits,
        "false_rate": false_rate,
        "false_rate_wilson95": _wilson_interval(false_hits, len(false_ids)),
        "candidate_present_count": len(candidate_ids),
        "candidate_present_hit_count": candidate_hits,
        "candidate_present_rate": candidate_rate,
        "candidate_present_rate_wilson95": _wilson_interval(
            candidate_hits, len(candidate_ids)
        ),
        "strict_correct_count": len(strict_ids),
        "strict_correct_hit_count": strict_hits,
        "strict_correct_rate": strict_rate,
        "strict_correct_rate_wilson95": _wilson_interval(strict_hits, len(strict_ids)),
        "false_candidate_gap": (
            false_rate - candidate_rate
            if false_rate is not None and candidate_rate is not None
            else None
        ),
        "false_candidate_gap_bootstrap": _winner_gap_bootstrap(
            hit_ids,
            false_ids=false_ids,
            candidate_ids=candidate_ids,
            samples=samples,
            seed=seed,
        ),
        "false_strict_gap": (
            false_rate - strict_rate
            if false_rate is not None and strict_rate is not None
            else None
        ),
        "false_strict_gap_bootstrap": _winner_gap_bootstrap(
            hit_ids,
            false_ids=false_ids,
            candidate_ids=strict_ids,
            samples=samples,
            seed=seed + 43,
        ),
    }


def _binary_repeat_stability(
    hits_by_repeat: Mapping[str, set[str]], *, case_ids: Sequence[str]
) -> dict[str, Any]:
    labels = tuple(sorted(hits_by_repeat))
    if len(labels) != 2:
        return {
            "available": False,
            "case_count": len(case_ids),
            "agreement_count": 0,
            "agreement_rate": None,
            "cohen_kappa": None,
        }
    pairs = [
        (
            "hit" if case_id in hits_by_repeat[labels[0]] else "clear",
            "hit" if case_id in hits_by_repeat[labels[1]] else "clear",
        )
        for case_id in case_ids
    ]
    agreements = sum(left == right for left, right in pairs)
    return {
        "available": True,
        "case_count": len(case_ids),
        "agreement_count": agreements,
        "agreement_rate": agreements / len(pairs) if pairs else None,
        "agreement_wilson95": _wilson_interval(agreements, len(pairs)),
        "cohen_kappa": _cohen_kappa(pairs),
    }


def _winner_gap_bootstrap(
    hit_ids: set[str],
    *,
    false_ids: Sequence[str],
    candidate_ids: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not false_ids or not candidate_ids or samples <= 0:
        return {"samples": samples, "valid_samples": 0, "ci95": [None, None]}
    rng = random.Random(seed)
    gaps: list[float] = []
    for _ in range(samples):
        sampled_false = [rng.choice(false_ids) for _ in false_ids]
        sampled_candidate = [rng.choice(candidate_ids) for _ in candidate_ids]
        gaps.append(
            mean(case_id in hit_ids for case_id in sampled_false)
            - mean(case_id in hit_ids for case_id in sampled_candidate)
        )
    gaps.sort()
    return {
        "samples": samples,
        "valid_samples": len(gaps),
        "ci95": [_quantile(gaps, 0.025), _quantile(gaps, 0.975)],
    }


def _case_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, samples: int, seed: int
) -> dict[str, Any]:
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    case_ids = sorted(by_case)
    if not case_ids or samples <= 0:
        return {"samples": samples, "valid_samples": 0, "ci95": [None, None]}
    rng = random.Random(seed)
    precisions: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(case_ids) for _ in case_ids]
        sampled_rows = [row for case_id in sampled for row in by_case[case_id]]
        adjudicated = [row for row in sampled_rows if row["verdict"] != "unclear"]
        if not adjudicated:
            continue
        precisions.append(
            mean(row["verdict"] == "true_contradiction" for row in adjudicated)
        )
    precisions.sort()
    return {
        "samples": samples,
        "valid_samples": len(precisions),
        "ci95": [
            _quantile(precisions, 0.025),
            _quantile(precisions, 0.975),
        ],
    }


def _wilson_interval(successes: int, total: int) -> list[float | None]:
    if total <= 0:
        return [None, None]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - spread), min(1.0, center + spread)]


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1 - weight) + values[upper] * weight)


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _fmt_ci(values: Sequence[Any]) -> str:
    items = tuple(values or ())
    if len(items) != 2 or any(value is None for value in items):
        return "[NA, NA]"
    return f"[{_fmt(items[0])}, {_fmt(items[1])}]"


def _winner_case_metadata(
    frozen_rows: Sequence[Mapping[str, Any]], *, case_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    frozen = {
        str(row.get("case_id", "") or ""): row
        for row in frozen_rows
        if str(row.get("arm", "") or "") == "a4"
        and str(row.get("case_id", "") or "") in set(case_ids)
    }
    metadata: dict[str, dict[str, Any]] = {}
    for case_id in case_ids:
        row = frozen.get(case_id)
        if row is None:
            continue
        selected = tuple(
            str(value)
            for value in tuple(row.get("selected_occurrence_ids", ()) or ())
            if str(value)
        )
        selected_id = selected[0] if len(selected) == 1 else ""
        candidate_present = row.get("candidate_recall_resolved_set") is True
        false_winner = row.get("candidate_recall_resolved_set") is False
        selected_resolution = bool(
            row.get("final_resolution") == "selected" and selected_id
        )
        winner_class = "not_scored"
        if selected_resolution and false_winner:
            winner_class = "false_winner"
        elif selected_resolution and candidate_present:
            winner_class = "candidate_present_winner"
        metadata[case_id] = {
            "case_id": case_id,
            "winner_class": winner_class,
            "strict_correct": bool(
                winner_class == "candidate_present_winner"
                and row.get("osa_strict") is True
            ),
            "selected_occurrence_id_digest": (
                stable_digest(selected_id) if selected_id else ""
            ),
        }
    return metadata


def _load_frozen_rows(
    run_root: Path, *, evaluation_record_root: Path
) -> tuple[dict[str, Any], ...]:
    module_path = Path(__file__).with_name("analyze_mmlifelong_occurrence_agent.py")
    spec = importlib.util.spec_from_file_location(
        "negative_row_frozen_analysis", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load occurrence analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(
        module.collect_rows(
            (Path(run_root),),
            evaluation_record_root=Path(evaluation_record_root),
        )
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _parse_repeat(value: str) -> tuple[str, Path]:
    label, separator, path = str(value).partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("repeat must be LABEL=PATH")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repeat", action="append", type=_parse_repeat, required=True)
    prepare.add_argument("--positive-run-root", required=True)
    prepare.add_argument("--replay-fixture-root", required=True)
    prepare.add_argument("--evaluation-record-root")
    prepare.add_argument("--reliability-fraction", type=float, default=0.20)
    prepare.add_argument("--seed", type=int, default=20260817)
    prepare.add_argument("--output-items-json", required=True)
    prepare.add_argument("--output-key-json", required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--key-json", required=True)
    analyze.add_argument("--judgments-json", required=True)
    analyze.add_argument("--bootstrap-samples", type=int, default=10000)
    analyze.add_argument("--seed", type=int, default=20260817)
    analyze.add_argument("--output-json", required=True)
    analyze.add_argument("--output-md", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        frozen_rows = (
            _load_frozen_rows(
                Path(args.positive_run_root),
                evaluation_record_root=Path(args.evaluation_record_root),
            )
            if args.evaluation_record_root
            else ()
        )
        blind, key = prepare_audit(
            dict(args.repeat),
            positive_run_root=Path(args.positive_run_root),
            replay_fixture_root=Path(args.replay_fixture_root),
            frozen_rows=frozen_rows,
            reliability_fraction=args.reliability_fraction,
            seed=args.seed,
        )
        _write_json(Path(args.output_items_json), blind)
        _write_json(Path(args.output_key_json), key)
        print(f"NEGATIVE_ROW_AUDIT_PREPARED items={blind['item_count']}", flush=True)
        return
    report = analyze_judgments(
        _read_json(Path(args.key_json)),
        _read_json(Path(args.judgments_json)),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    _write_json(Path(args.output_json), report)
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print(
        f"NEGATIVE_ROW_AUDIT_ANALYZED complete={report['complete']} "
        f"precision={_fmt(report['row_precision'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
