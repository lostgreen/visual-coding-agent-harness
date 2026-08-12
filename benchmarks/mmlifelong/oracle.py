"""Answer-free runtime interventions for the MM-Lifelong oracle ladder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from vcah.caption_lexical_index import render_caption_hits
from vcah.caption_occurrence import build_caption_occurrence_set
from vcah.caption_schema import CaptionHitV1, CaptionPassageV1, stable_digest
from vcah.caption_semantic_index import load_caption_passages
from vcah.multiround import InvestigationTask
from vcah.virtual_video import VirtualVideoWorkspace, virtual_to_source_windows


ORACLE_ARMS = (
    "o0",
    "c0",
    "o1",
    "o1.5",
    "o1.75",
    "o1.75-forced",
    "o2",
    "o2-center",
)
_CANDIDATE_INCLUSION_ARMS = frozenset(
    {"o1", "o1.5", "o1.75", "o1.75-forced"}
)
_POINT_ANCHOR_ARMS = frozenset({"o1.75", "o1.75-forced", "o2-center"})
_GUIDANCE_ARMS = frozenset({"o1.5", *_POINT_ANCHOR_ARMS})
_EXACT_LOCATOR_ARMS = frozenset({"o2", "o2-center"})
_FULL_RECALL_ARMS = frozenset(
    {*_CANDIDATE_INCLUSION_ARMS, *_EXACT_LOCATOR_ARMS}
)
_FORBIDDEN_MANIFEST_KEYS = frozenset(
    {
        "answer",
        "gold",
        "gold_answer",
        "reference_answer",
        "target_gt",
    }
)


@dataclass(frozen=True)
class OracleIntervention:
    case_id: str
    normalized_clue_intervals: tuple[tuple[float, float], ...]
    experiment_seed: int
    caption_config_digest: str
    source_manifest_digest: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OracleIntervention":
        forbidden = _forbidden_keys(value)
        if forbidden:
            raise ValueError(
                "oracle intervention contains answer-bearing keys: "
                + ", ".join(sorted(forbidden))
            )
        intervals = tuple(
            _interval(item)
            for item in tuple(value.get("normalized_clue_intervals", ()) or ())
        )
        if not intervals:
            raise ValueError("oracle intervention requires at least one clue interval")
        return cls(
            case_id=str(value["case_id"]),
            normalized_clue_intervals=intervals,
            experiment_seed=int(value.get("experiment_seed", 0) or 0),
            caption_config_digest=str(value["caption_config_digest"]),
            source_manifest_digest=str(value.get("source_manifest_digest", "") or ""),
        )

    @property
    def digest(self) -> str:
        return stable_digest(
            {
                "case_id": self.case_id,
                "normalized_clue_intervals": [
                    list(item) for item in self.normalized_clue_intervals
                ],
                "experiment_seed": self.experiment_seed,
                "caption_config_digest": self.caption_config_digest,
                "source_manifest_digest": self.source_manifest_digest,
            }
        )


def load_oracle_intervention(path: Path) -> OracleIntervention:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"oracle intervention must be a JSON object: {path}")
    return OracleIntervention.from_mapping(payload)


def bootstrap_tasks(
    *,
    arm: str,
    question: str,
    index_mode: str,
    top_k: int = 12,
) -> tuple[InvestigationTask, ...]:
    normalized_arm = _arm(arm)
    if normalized_arm == "o0":
        return ()
    return (
        InvestigationTask(
            query_id="bootstrap_caption_candidates",
            goal=str(question),
            inspection_mode="search_caption",
            caption_queries=(str(question),),
            top_k=top_k,
            index_mode=index_mode,
            expected_evidence="locator candidates",
        ),
    )


class CaptionPacketIntervention:
    def __init__(
        self,
        *,
        arm: str,
        intervention: OracleIntervention,
        workspace: VirtualVideoWorkspace,
        audit_path: Path,
    ) -> None:
        self.arm = _arm(arm)
        if self.arm == "o0":
            raise ValueError("O0 does not use a Caption packet intervention")
        if intervention.case_id != workspace.case.case_id:
            raise ValueError("oracle intervention case_id does not match workspace")
        if intervention.source_manifest_digest:
            actual_manifest_digest = _file_sha256(
                workspace.asset_root / "virtual_timeline.json"
            )
            if actual_manifest_digest != intervention.source_manifest_digest:
                raise ValueError("oracle intervention source manifest digest mismatch")
        self.intervention = intervention
        self.workspace = workspace
        self.audit_path = Path(audit_path)
        self.applied = False
        self._audit: dict[str, Any] = {
            "schema_version": "MMLifelongOracleInterventionAuditV1",
            "arm": self.arm,
            "case_id": workspace.case.case_id,
            "applied": False,
            "intervention_digest": intervention.digest,
        }
        if self.arm in _FULL_RECALL_ARMS:
            passages, digest, _ = load_caption_passages(
                workspace.asset_root,
                config_digest=intervention.caption_config_digest,
            )
            if digest != intervention.caption_config_digest:
                raise ValueError("loaded Caption digest does not match intervention")
            self.passages = passages
        else:
            self.passages = ()

    @property
    def audit(self) -> Mapping[str, Any]:
        return dict(self._audit)

    def __call__(self, packet: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.applied:
            return packet
        self.applied = True
        if str(packet.get("config_digest", "")) != self.intervention.caption_config_digest:
            raise ValueError("Caption packet digest does not match oracle intervention")
        natural_hits = tuple(_hit(item) for item in tuple(packet.get("hits", ()) or ()))
        if not natural_hits:
            raise ValueError("bootstrap Caption search returned no candidates")
        natural_recall = _clue_recall(
            natural_hits,
            self.intervention.normalized_clue_intervals,
        )
        injected_count = 0
        if self.arm == "c0":
            final_hits = list(natural_hits)
        elif self.arm in _CANDIDATE_INCLUSION_ARMS:
            final_hits, injected_count = self._candidate_inclusion(natural_hits)
        elif self.arm in _EXACT_LOCATOR_ARMS:
            final_hits = self._exact_locators(natural_hits)
        else:
            raise ValueError(f"unsupported oracle arm: {self.arm}")
        shuffle_seed = _shuffle_seed(
            self.intervention.experiment_seed,
            self.intervention.case_id,
        )
        random.Random(shuffle_seed).shuffle(final_hits)
        ranked_hits = tuple(
            replace(hit, rank=index)
            for index, hit in enumerate(final_hits, start=1)
        )
        final_recall = _clue_recall(
            ranked_hits,
            self.intervention.normalized_clue_intervals,
        )
        if self.arm in _FULL_RECALL_ARMS and final_recall < 1.0:
            raise ValueError(
                f"{self.arm} failed to include every clue interval: {final_recall}"
            )
        oracle_guidance = (
            self._oracle_guidance(ranked_hits)
            if self.arm in _GUIDANCE_ARMS
            else None
        )
        transformed = {
            **dict(packet),
            "hits": [asdict(hit) for hit in ranked_hits],
            "occurrence_set": build_caption_occurrence_set(ranked_hits),
            "rendered": render_caption_hits(ranked_hits),
            **({"oracle_guidance": oracle_guidance} if oracle_guidance else {}),
        }
        selected_candidates = tuple(
            oracle_guidance.get("selected_candidates", ())
            if oracle_guidance
            else ()
        )
        point_anchors = tuple(
            oracle_guidance.get("point_anchors", ())
            if oracle_guidance
            else ()
        )
        self._audit = {
            "schema_version": "MMLifelongOracleInterventionAuditV2",
            "arm": self.arm,
            "case_id": self.intervention.case_id,
            "applied": True,
            "intervention_digest": self.intervention.digest,
            "caption_config_digest": self.intervention.caption_config_digest,
            "natural_candidate_count": len(natural_hits),
            "final_candidate_count": len(ranked_hits),
            "natural_clue_recall": natural_recall,
            "final_clue_recall": final_recall,
            "injected_candidate_count": injected_count,
            "exact_locator_count": (
                len(self.intervention.normalized_clue_intervals)
                if self.arm in _EXACT_LOCATOR_ARMS
                else 0
            ),
            "guidance_type": (
                str(oracle_guidance.get("guidance_type", ""))
                if oracle_guidance
                else ""
            ),
            "exact_boundaries_visible": self.arm in _EXACT_LOCATOR_ARMS,
            "anchor_execution_policy": (
                "force_if_requested"
                if self.arm == "o1.75-forced"
                else "agent_controlled"
                if self.arm in _POINT_ANCHOR_ARMS
                else ""
            ),
            "selected_candidate_count": len(selected_candidates),
            "selected_candidate_ranks": [
                int(candidate["candidate_rank"])
                for candidate in selected_candidates
            ],
            "selected_candidate_passage_ids": [
                str(candidate["passage_id"])
                for candidate in selected_candidates
            ],
            "selected_candidate_intervals": [
                list(candidate["inspection_range"])
                for candidate in selected_candidates
            ],
            "selected_candidate_clue_recall": (
                _clue_recall(
                    tuple(
                        ranked_hits[int(candidate["candidate_rank"]) - 1]
                        for candidate in selected_candidates
                    ),
                    self.intervention.normalized_clue_intervals,
                )
                if selected_candidates
                else None
            ),
            "anchor_count": len(
                tuple(oracle_guidance.get("anchor_timestamps_sec", ()))
                if oracle_guidance
                else ()
            ),
            "anchor_timestamps_sec": list(
                tuple(oracle_guidance.get("anchor_timestamps_sec", ()))
                if oracle_guidance
                else ()
            ),
            "point_anchor_candidate_ranks": [
                int(anchor["selected_candidate_rank"])
                for anchor in point_anchors
            ],
            "point_anchor_candidate_passage_ids": [
                str(anchor["selected_candidate_passage_id"])
                for anchor in point_anchors
            ],
            "shuffle_seed_digest": hashlib.sha256(
                str(shuffle_seed).encode("utf-8")
            ).hexdigest(),
            "candidate_passage_ids": [hit.passage_id for hit in ranked_hits],
            "candidate_intervals": [
                [hit.virtual_start_sec, hit.virtual_end_sec]
                for hit in ranked_hits
            ],
        }
        _write_json(self.audit_path, self._audit)
        return transformed

    def _oracle_guidance(
        self,
        ranked_hits: Sequence[CaptionHitV1],
    ) -> dict[str, Any]:
        selected_by_clue = self._selected_candidates_by_clue(ranked_hits)
        selected_by_rank = {hit.rank: hit for hit in selected_by_clue}
        selected = tuple(selected_by_rank[rank] for rank in sorted(selected_by_rank))
        exact_locators = self.arm == "o2-center"
        guidance: dict[str, Any] = {
            "schema_version": "MMLifelongOracleGuidanceV1",
            "arm": "o1.75" if self.arm == "o1.75-forced" else self.arm,
            "guidance_type": "selected_coarse_candidates",
            "scope": "answer_free_locator_only",
            "selected_candidate_guarantee": (
                "exact_annotated_occurrence"
                if exact_locators
                else "overlaps_annotated_occurrence"
            ),
            "boundary_visibility": "exact" if exact_locators else "hidden",
            "selected_candidates": [
                {
                    "candidate_rank": hit.rank,
                    "passage_id": hit.passage_id,
                    "inspection_range": [
                        hit.virtual_start_sec,
                        hit.virtual_end_sec,
                    ],
                    "interval_precision": hit.interval_precision,
                }
                for hit in selected
            ],
            "agent_controls": ["window_width", "zoom", "stopping"],
        }
        if self.arm in _POINT_ANCHOR_ARMS:
            point_anchors = [
                {
                    "anchor_timestamp_sec": round((clue[0] + clue[1]) / 2.0, 3),
                    "selected_candidate_rank": hit.rank,
                    "selected_candidate_passage_id": hit.passage_id,
                }
                for clue, hit in zip(
                    self.intervention.normalized_clue_intervals,
                    selected_by_clue,
                )
            ]
            guidance.update(
                {
                    "guidance_type": (
                        "exact_locators_with_point_anchors"
                        if exact_locators
                        else "selected_coarse_candidates_with_point_anchors"
                    ),
                    "anchor_guarantee": "annotated_occurrence_center",
                    "anchor_timestamps_sec": [
                        anchor["anchor_timestamp_sec"] for anchor in point_anchors
                    ],
                    "point_anchors": point_anchors,
                }
            )
        return guidance

    def _selected_candidates_by_clue(
        self,
        ranked_hits: Sequence[CaptionHitV1],
    ) -> tuple[CaptionHitV1, ...]:
        selected: list[CaptionHitV1] = []
        for clue in self.intervention.normalized_clue_intervals:
            overlapping = [
                hit for hit in ranked_hits if _overlap(_hit_range(hit), clue)
            ]
            if not overlapping:
                raise ValueError(f"no final Caption candidate overlaps clue interval {clue}")
            clue_midpoint = sum(clue) / 2.0
            chosen = min(
                overlapping,
                key=lambda hit: (
                    -_intersection(_hit_range(hit), clue),
                    abs(sum(_hit_range(hit)) / 2.0 - clue_midpoint),
                    hit.rank,
                    hit.passage_id,
                ),
            )
            selected.append(chosen)
        return tuple(selected)

    def _candidate_inclusion(
        self,
        natural_hits: Sequence[CaptionHitV1],
    ) -> tuple[list[CaptionHitV1], int]:
        selected = list(natural_hits)
        injected = 0
        clues = self.intervention.normalized_clue_intervals
        for clue in clues:
            if any(_overlap(_hit_range(hit), clue) for hit in selected):
                continue
            passage = _best_overlapping_passage(self.passages, clue)
            protected = {
                index
                for index, hit in enumerate(selected)
                if any(_overlap(_hit_range(hit), other) for other in clues)
            }
            replacement_index = next(
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if index not in protected
                    and selected[index].passage_id != passage.passage_id
                ),
                None,
            )
            if replacement_index is None:
                raise ValueError("candidate pool has no replaceable distractor")
            template = selected[replacement_index]
            selected[replacement_index] = self._passage_hit(
                passage,
                template=template,
            )
            injected += 1
        return selected, injected

    def _exact_locators(
        self,
        natural_hits: Sequence[CaptionHitV1],
    ) -> list[CaptionHitV1]:
        hits: list[CaptionHitV1] = []
        for index, clue in enumerate(
            self.intervention.normalized_clue_intervals,
            start=1,
        ):
            passage = _best_overlapping_passage(self.passages, clue)
            template = natural_hits[min(index - 1, len(natural_hits) - 1)]
            hits.append(
                self._passage_hit(
                    passage,
                    template=template,
                    exact_interval=clue,
                    locator_index=index,
                )
            )
        return hits

    def _passage_hit(
        self,
        passage: CaptionPassageV1,
        *,
        template: CaptionHitV1,
        exact_interval: tuple[float, float] | None = None,
        locator_index: int = 0,
    ) -> CaptionHitV1:
        start_sec, end_sec = exact_interval or (
            passage.virtual_start_sec,
            passage.virtual_end_sec,
        )
        windows = virtual_to_source_windows(
            self.workspace.manifest,
            start_sec,
            max(end_sec, start_sec + 0.001),
        )
        metadata = {
            **dict(passage.metadata),
            "source_segments": list(
                dict.fromkeys(window.segment_id for window in windows)
            ),
            "source_video_ids": list(
                dict.fromkeys(window.source_video_id for window in windows)
            ),
        }
        passage_id = passage.passage_id
        source_pointer = (
            f"caption://{self.intervention.caption_config_digest}/{passage.passage_id}"
        )
        interval_precision = "passage"
        if exact_interval is not None:
            material = (
                f"{self.intervention.case_id}:{locator_index}:"
                f"{start_sec:.3f}:{end_sec:.3f}"
            )
            suffix = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
            passage_id = f"candidate_{suffix}"
            source_pointer = f"{source_pointer}#interval={suffix}"
            interval_precision = "exact_interval"
        return CaptionHitV1(
            passage_id=passage_id,
            caption_id=passage.caption_id,
            rank=template.rank,
            lexical_score=template.lexical_score,
            dense_score=template.dense_score,
            fused_score=template.fused_score,
            virtual_start_sec=start_sec,
            virtual_end_sec=end_sec,
            wall_clock_begin=None,
            wall_clock_end=None,
            text=passage.text,
            interval_precision=interval_precision,
            source_pointer=source_pointer,
            metadata=metadata,
        )


def _best_overlapping_passage(
    passages: Sequence[CaptionPassageV1],
    clue: tuple[float, float],
) -> CaptionPassageV1:
    candidates = [
        passage
        for passage in passages
        if _overlap(
            (passage.virtual_start_sec, passage.virtual_end_sec),
            clue,
        )
    ]
    if not candidates:
        raise ValueError(f"no frozen Caption passage overlaps clue interval {clue}")
    clue_midpoint = sum(clue) / 2.0
    return min(
        candidates,
        key=lambda passage: (
            -_intersection(
                (passage.virtual_start_sec, passage.virtual_end_sec),
                clue,
            ),
            abs(
                (
                    passage.anchor_virtual_sec
                    if passage.anchor_virtual_sec is not None
                    else (passage.virtual_start_sec + passage.virtual_end_sec) / 2.0
                )
                - clue_midpoint
            ),
            passage.passage_id,
        ),
    )


def _clue_recall(
    hits: Sequence[CaptionHitV1],
    clues: Sequence[tuple[float, float]],
) -> float:
    recalled = sum(
        any(_overlap(_hit_range(hit), clue) for hit in hits)
        for clue in clues
    )
    return recalled / len(clues) if clues else 0.0


def _hit(value: CaptionHitV1 | Mapping[str, Any]) -> CaptionHitV1:
    return value if isinstance(value, CaptionHitV1) else CaptionHitV1(**dict(value))


def _hit_range(hit: CaptionHitV1) -> tuple[float, float]:
    return hit.virtual_start_sec, hit.virtual_end_sec


def _interval(value: Sequence[float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"expected [start, end], got {value!r}")
    start_sec, end_sec = float(value[0]), float(value[1])
    if not math.isfinite(start_sec) or not math.isfinite(end_sec) or end_sec <= start_sec:
        raise ValueError(f"invalid clue interval: {value!r}")
    return round(start_sec, 3), round(end_sec, 3)


def _overlap(
    left: tuple[float, float],
    right: tuple[float, float],
) -> bool:
    return min(left[1], right[1]) > max(left[0], right[0])


def _intersection(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _arm(value: str) -> str:
    arm = str(value or "o0").strip().casefold()
    if arm not in ORACLE_ARMS:
        raise ValueError(f"unsupported MM-Lifelong oracle arm: {arm}")
    return arm


def _shuffle_seed(experiment_seed: int, case_id: str) -> int:
    material = f"{int(experiment_seed)}:{case_id}:candidate-pool-v1"
    return int.from_bytes(
        hashlib.sha256(material.encode("utf-8")).digest()[:8],
        "big",
    )


def _forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().casefold()
            if key in _FORBIDDEN_MANIFEST_KEYS:
                found.add(str(raw_key))
            found.update(_forbidden_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_forbidden_keys(item))
    return found


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
