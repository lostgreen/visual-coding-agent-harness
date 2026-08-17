from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from vcah.caption_schema import stable_digest
from vcah.occurrence_sufficiency import (
    DEFAULT_SUFFICIENCY_CANDIDATE_LIMIT,
    OccurrenceEvidenceReport,
    OccurrenceSufficiencyDecision,
    SUFFICIENCY_OPERATION,
    aggregate_occurrence_evidence,
    validate_evidence_operation,
)


OCCURRENCE_METHOD_ARMS = (
    "none",
    "a0",
    "a1",
    "a1-flat",
    "a2",
    "a2-clean",
    "a3",
    "a4",
)
FORBIDDEN_AGENT_VISIBLE_KEYS = frozenset(
    {
        "clue_intervals",
        "evaluation_case",
        "gold",
        "gold_answer",
        "gold_clue_intervals",
        "normalized_clue_intervals",
        "oracle_guidance",
        "oracle_intervention",
        "reference_answer",
    }
)
DEFAULT_CARD_CANDIDATE_LIMIT = 8
DEFAULT_CARD_EXCERPT_LIMIT = 3
DEFAULT_CARD_EXCERPT_CHARS = 240
DEFAULT_CARD_QUERY_LIMIT = 4
DEFAULT_CARD_QUERY_CHARS = 160
OCCURRENCE_RESOLUTION_OPS = frozenset({"keep", "eliminate", "select", "reopen"})
SCOPED_OCCURRENCE_RESOLUTION_OPS = frozenset(
    {
        "keep",
        "eliminate",
        "select",
        "reopen",
        "defer",
        "no_match",
        SUFFICIENCY_OPERATION,
    }
)


@dataclass
class OccurrenceResolutionStateV1:
    """Runtime-owned state; the Reasoner remains responsible for every semantic choice."""

    states: dict[str, str] = field(default_factory=dict)
    current_visible_ids: tuple[str, ...] = ()
    selected_occurrence_id: str = ""
    selection_required: bool = False
    revision: int = 0

    def sync_visible(self, occurrence_ids: Sequence[str]) -> bool:
        newly_visible = tuple(
            dict.fromkeys(
                str(value or "").strip()
                for value in occurrence_ids
                if str(value or "").strip()
            )
        )
        # An occurrence handle remains agent-visible after first exposure. Retrieval
        # packets can change between rounds, but invalidating an earlier handle makes
        # a semantically valid later selection fail for purely mechanical reasons.
        visible = tuple(dict.fromkeys((*self.current_visible_ids, *newly_visible)))
        changed = visible != self.current_visible_ids
        self.current_visible_ids = visible
        if len(visible) > 1:
            self.selection_required = True
        for occurrence_id in visible:
            self.states.setdefault(occurrence_id, "active")
        if self.selected_occurrence_id not in set(visible):
            if self.states.get(self.selected_occurrence_id) == "selected":
                self.states[self.selected_occurrence_id] = "active"
            self.selected_occurrence_id = ""
        if changed:
            self.revision += 1
        return changed

    def validate_ops(
        self, operations: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        errors: list[dict[str, Any]] = []
        visible = set(self.current_visible_ids)
        shadow_states = dict(self.states)
        shadow_selected = self.selected_occurrence_id
        for index, operation in enumerate(operations):
            if not isinstance(operation, Mapping):
                errors.append(
                    {
                        "code": "occurrence_op_must_be_object",
                        "occurrence_op_index": index,
                    }
                )
                continue
            op = str(operation.get("op", operation.get("type", "")) or "").casefold()
            occurrence_id = str(operation.get("occurrence_id", "") or "").strip()
            if op not in OCCURRENCE_RESOLUTION_OPS:
                errors.append(
                    {
                        "code": "unsupported_occurrence_op",
                        "occurrence_op_index": index,
                        "op": op,
                    }
                )
                continue
            if not occurrence_id or occurrence_id not in visible:
                errors.append(
                    {
                        "code": "occurrence_id_not_currently_visible",
                        "occurrence_op_index": index,
                        "occurrence_id": occurrence_id,
                    }
                )
                continue
            if op == "select" and shadow_states.get(occurrence_id) == "eliminated":
                errors.append(
                    {
                        "code": "eliminated_occurrence_requires_reopen",
                        "occurrence_op_index": index,
                        "occurrence_id": occurrence_id,
                    }
                )
                continue
            if op == "keep" and shadow_states.get(occurrence_id) == "eliminated":
                errors.append(
                    {
                        "code": "eliminated_occurrence_requires_reopen",
                        "occurrence_op_index": index,
                        "occurrence_id": occurrence_id,
                    }
                )
                continue
            shadow_selected = _apply_occurrence_op(
                shadow_states,
                selected_occurrence_id=shadow_selected,
                op=op,
                occurrence_id=occurrence_id,
            )
        return errors

    def apply_ops(
        self, operations: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        normalized = tuple(dict(item) for item in operations if isinstance(item, Mapping))
        errors = self.validate_ops(normalized)
        if errors:
            return {"accepted": False, "errors": errors, "applied": []}
        applied: list[dict[str, str]] = []
        for operation in normalized:
            op = str(operation.get("op", operation.get("type", "")) or "").casefold()
            occurrence_id = str(operation.get("occurrence_id", "") or "").strip()
            self.selected_occurrence_id = _apply_occurrence_op(
                self.states,
                selected_occurrence_id=self.selected_occurrence_id,
                op=op,
                occurrence_id=occurrence_id,
            )
            applied.append({"op": op, "occurrence_id": occurrence_id})
        if applied:
            self.revision += 1
        return {"accepted": True, "errors": [], "applied": applied}

    @property
    def viable_occurrence_ids(self) -> tuple[str, ...]:
        return tuple(
            occurrence_id
            for occurrence_id in self.current_visible_ids
            if self.states.get(occurrence_id, "active") != "eliminated"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "OccurrenceResolutionStateV1",
            "revision": self.revision,
            "current_visible_occurrence_ids": list(self.current_visible_ids),
            "candidates": [
                {
                    "occurrence_id": occurrence_id,
                    "status": self.states.get(occurrence_id, "active"),
                }
                for occurrence_id in self.current_visible_ids
            ],
            "viable_occurrence_ids": list(self.viable_occurrence_ids),
            "selected_occurrence_id": self.selected_occurrence_id or None,
            "selection_required": self.selection_required,
        }

    def save(self, path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)


@dataclass
class OccurrenceSetStateV2:
    """One semantically scoped candidate set produced by one locator attempt."""

    set_id: str
    semantic_target: tuple[str, ...] = ()
    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    states: dict[str, str] = field(default_factory=dict)
    selected_occurrence_ids: tuple[str, ...] = ()
    resolution: str = "unresolved"
    lifecycle: str = "active"
    evidence_report: OccurrenceEvidenceReport | None = None
    sufficiency: OccurrenceSufficiencyDecision | None = None
    revision: int = 0

    @property
    def viable_occurrence_ids(self) -> tuple[str, ...]:
        return tuple(
            occurrence_id
            for occurrence_id in self.candidates
            if self.states.get(occurrence_id, "active") != "eliminated"
        )

    def sync(
        self,
        *,
        semantic_target: Sequence[str],
        candidates: Sequence[Mapping[str, Any]],
    ) -> bool:
        target = tuple(
            dict.fromkeys(
                str(value or "").strip()
                for value in semantic_target
                if str(value or "").strip()
            )
        )
        normalized: dict[str, dict[str, Any]] = {}
        for raw_candidate in candidates:
            if not isinstance(raw_candidate, Mapping):
                continue
            occurrence_id = str(raw_candidate.get("occurrence_id", "") or "").strip()
            if occurrence_id:
                normalized[occurrence_id] = dict(raw_candidate)
        changed = target != self.semantic_target or normalized != self.candidates
        if changed:
            self.evidence_report = None
            self.sufficiency = None
        self.semantic_target = target
        self.candidates = normalized
        for occurrence_id in normalized:
            self.states.setdefault(occurrence_id, "active")
        stale = set(self.states) - set(normalized)
        for occurrence_id in stale:
            self.states.pop(occurrence_id, None)
        selected = tuple(
            occurrence_id
            for occurrence_id in self.selected_occurrence_ids
            if occurrence_id in normalized
        )
        if selected != self.selected_occurrence_ids:
            self.selected_occurrence_ids = selected
            if not selected and self.resolution == "selected":
                self.resolution = "unresolved"
            changed = True
        if changed:
            self.revision += 1
        return changed

    def to_dict(self) -> dict[str, Any]:
        return {
            "set_id": self.set_id,
            "locator_attempt_id": self.set_id,
            "semantic_target": list(self.semantic_target),
            "revision": self.revision,
            "resolution": self.resolution,
            "lifecycle": self.lifecycle,
            "candidates": [
                {
                    "occurrence_id": occurrence_id,
                    "status": self.states.get(occurrence_id, "active"),
                    "time_range": list(
                        self.candidates[occurrence_id].get("time_range", ()) or ()
                    ),
                }
                for occurrence_id in self.candidates
            ],
            "viable_occurrence_ids": list(self.viable_occurrence_ids),
            "selected_occurrence_ids": list(self.selected_occurrence_ids),
            "evidence_report": (
                self.evidence_report.to_dict()
                if self.evidence_report is not None
                else None
            ),
            "sufficiency": (
                self.sufficiency.to_dict() if self.sufficiency is not None else None
            ),
        }


@dataclass
class OccurrenceResolutionStateV2:
    """Scoped, abstainable occurrence arbitration owned mechanically by Runtime."""

    sets: dict[str, OccurrenceSetStateV2] = field(default_factory=dict)
    active_set_id: str = ""
    sufficiency_enabled: bool = False
    revision: int = 0

    @property
    def activated(self) -> bool:
        return bool(self.sets)

    @property
    def active_set(self) -> OccurrenceSetStateV2 | None:
        active = self.sets.get(self.active_set_id)
        return active if active is not None and active.lifecycle == "active" else None

    @property
    def retired_set_ids(self) -> tuple[str, ...]:
        return tuple(
            set_id
            for set_id, occurrence_set in self.sets.items()
            if occurrence_set.lifecycle == "retired"
        )

    @property
    def candidate_count(self) -> int:
        active = self.active_set
        return len(active.candidates) if active is not None else 0

    @property
    def resolution_required(self) -> bool:
        return self.candidate_count >= 1

    @property
    def arbitration_required(self) -> bool:
        return self.candidate_count >= 2

    @property
    def selection_required(self) -> bool:
        active = self.active_set
        return bool(active is not None and active.resolution == "unresolved")

    @property
    def search_required(self) -> bool:
        active = self.active_set
        return bool(active is not None and active.resolution == "deferred")

    @property
    def resolution_committed(self) -> bool:
        active = self.active_set
        return bool(
            active is not None and active.resolution in {"selected", "no_match"}
        )

    @property
    def selected_occurrence_ids(self) -> tuple[str, ...]:
        active = self.active_set
        return active.selected_occurrence_ids if active is not None else ()

    @property
    def sufficiency_required(self) -> bool:
        active = self.active_set
        return bool(
            self.sufficiency_enabled
            and active is not None
            and active.resolution in {"unresolved", "deferred"}
            and active.sufficiency is None
        )

    @property
    def sufficiency_scope_occurrence_ids(self) -> tuple[str, ...]:
        active = self.active_set
        if not self.sufficiency_enabled or active is None:
            return ()
        return active.viable_occurrence_ids[:DEFAULT_SUFFICIENCY_CANDIDATE_LIMIT]

    @property
    def sufficiency_out_of_scope_occurrence_ids(self) -> tuple[str, ...]:
        active = self.active_set
        if not self.sufficiency_enabled or active is None:
            return ()
        return active.viable_occurrence_ids[DEFAULT_SUFFICIENCY_CANDIDATE_LIMIT:]

    def evidence_scope_candidates(self) -> tuple[dict[str, Any], ...]:
        active = self.active_set
        if not self.sufficiency_enabled or active is None:
            return ()
        rows: list[dict[str, Any]] = []
        for occurrence_id in self.sufficiency_scope_occurrence_ids:
            candidate = active.candidates[occurrence_id]
            raw_card = candidate.get("candidate_card")
            card = raw_card if isinstance(raw_card, Mapping) else {}
            passages = [
                {
                    "passage_id": str(passage.get("passage_id", "") or ""),
                    "time_range": list(passage.get("time_range", ()) or ()),
                    "caption_excerpt": str(
                        passage.get("caption_excerpt", "") or ""
                    ),
                    "query_matches": list(
                        passage.get("query_matches", ()) or ()
                    ),
                    "evidence_role": "locator_only",
                    "answer_support": False,
                }
                for passage in tuple(
                    card.get("representative_passages", ()) or ()
                )
                if isinstance(passage, Mapping)
            ]
            rows.append(
                {
                    "occurrence_id": occurrence_id,
                    "time_range": list(candidate.get("time_range", ()) or ()),
                    "passage_ids": list(candidate.get("passage_ids", ()) or ()),
                    "representative_passages": passages,
                    "evidence_role": "locator_only",
                    "answer_support": False,
                }
            )
        return tuple(rows)

    def sync_sets(self, occurrence_sets: Sequence[Mapping[str, Any]]) -> bool:
        # A selected/no-match result is the scoped decision endpoint. Later
        # retrieval packets cannot silently replace it; search-more remains
        # available while the active set is unresolved or deferred.
        if self.resolution_committed:
            return False
        changed = False
        for raw_set in occurrence_sets:
            if not isinstance(raw_set, Mapping):
                continue
            candidates = tuple(
                candidate
                for candidate in tuple(raw_set.get("candidates", ()) or ())
                if isinstance(candidate, Mapping)
                and str(candidate.get("occurrence_id", "") or "").strip()
            )
            if len(candidates) < 1:
                continue
            set_id = str(
                raw_set.get("attempt_id", raw_set.get("set_id", "")) or ""
            ).strip()
            if not set_id:
                set_id = "set_" + stable_digest(
                    sorted(
                        str(candidate.get("occurrence_id", "") or "")
                        for candidate in candidates
                    )
                )[:20]
            semantic_target = tuple(raw_set.get("semantic_target", ()) or ())
            occurrence_set = self.sets.get(set_id)
            is_new = occurrence_set is None
            if occurrence_set is None:
                occurrence_set = OccurrenceSetStateV2(set_id=set_id)
                self.sets[set_id] = occurrence_set
            if occurrence_set.sync(
                semantic_target=semantic_target,
                candidates=candidates,
            ):
                changed = True
            if is_new:
                previous = self.active_set
                if previous is not None:
                    previous.lifecycle = "retired"
                    if previous.resolution == "unresolved":
                        previous.resolution = "deferred"
                    previous.revision += 1
                occurrence_set.lifecycle = "active"
                self.active_set_id = set_id
                changed = True
        if changed:
            self.revision += 1
        return changed

    def validate_ops(
        self, operations: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        operation_names = tuple(
            str(operation.get("op", operation.get("type", "")) or "")
            .strip()
            .casefold()
            for operation in operations
            if isinstance(operation, Mapping)
        )
        if SUFFICIENCY_OPERATION in operation_names and len(operation_names) != 1:
            return [
                {
                    "code": "occurrence_evidence_transaction_must_be_isolated",
                    "occurrence_op_index": operation_names.index(
                        SUFFICIENCY_OPERATION
                    ),
                }
            ]
        shadow = self._clone()
        allow_resolution_transaction = not shadow.resolution_committed
        errors: list[dict[str, Any]] = []
        for index, operation in enumerate(operations):
            error = shadow._apply_one(
                operation,
                index=index,
                allow_resolution_transaction=allow_resolution_transaction,
            )
            if error is not None:
                if error.get("code") == "_occurrence_evidence_validation_errors":
                    errors.extend(
                        dict(row)
                        for row in tuple(error.get("errors", ()) or ())
                        if isinstance(row, Mapping)
                    )
                else:
                    errors.append(error)
        return errors

    def apply_ops(
        self, operations: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        normalized = tuple(
            dict(item) for item in operations if isinstance(item, Mapping)
        )
        errors = self.validate_ops(normalized)
        if errors:
            return {"accepted": False, "errors": errors, "applied": []}
        allow_resolution_transaction = not self.resolution_committed
        applied: list[dict[str, Any]] = []
        runtime_occurrence_ops: list[dict[str, Any]] = []
        evidence_report: dict[str, Any] | None = None
        gate_decision: dict[str, Any] | None = None
        for index, operation in enumerate(normalized):
            error = self._apply_one(
                operation,
                index=index,
                allow_resolution_transaction=allow_resolution_transaction,
            )
            assert error is None
            operation_name = str(
                operation.get("op", operation.get("type", "")) or ""
            ).strip().casefold()
            if operation_name == SUFFICIENCY_OPERATION:
                active = self.active_set
                assert (
                    active is not None
                    and active.evidence_report is not None
                    and active.sufficiency is not None
                )
                evidence_report = {
                    **active.evidence_report.to_dict(),
                    "evidence_report_digest": active.evidence_report.digest,
                }
                gate_decision = active.sufficiency.to_dict()
                applied.append(active.evidence_report.to_operation())
                runtime_op = {
                    "op": active.sufficiency.resolution_op,
                    "set_id": active.set_id,
                    "source": "runtime_sufficiency_gate",
                }
                if active.sufficiency.winner_occurrence_id:
                    runtime_op["occurrence_id"] = (
                        active.sufficiency.winner_occurrence_id
                    )
                runtime_occurrence_ops.append(runtime_op)
            else:
                applied.append(_normalized_scoped_occurrence_op(operation))
        if applied:
            self.revision += 1
        return {
            "accepted": True,
            "errors": [],
            "applied": applied,
            "runtime_occurrence_ops": runtime_occurrence_ops,
            "evidence_report": evidence_report,
            "gate_decision": gate_decision,
        }

    def active_locators(self) -> tuple[dict[str, Any], ...]:
        occurrence_set = self.active_set
        if occurrence_set is None:
            return ()
        return tuple(
            {
                "set_id": occurrence_set.set_id,
                "locator_attempt_id": occurrence_set.set_id,
                "occurrence_id": occurrence_id,
                "time_range": list(
                    occurrence_set.candidates[occurrence_id].get("time_range", ())
                    or ()
                ),
                "status": "selected_for_active_set",
            }
            for occurrence_id in occurrence_set.selected_occurrence_ids
            if occurrence_id in occurrence_set.candidates
        )

    def retired_locators(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "set_id": occurrence_set.set_id,
                "locator_attempt_id": occurrence_set.set_id,
                "occurrence_id": occurrence_id,
                "time_range": list(
                    occurrence_set.candidates[occurrence_id].get("time_range", ())
                    or ()
                ),
                "status": "retired_history",
            }
            for occurrence_set in self.sets.values()
            if occurrence_set.lifecycle == "retired"
            for occurrence_id in occurrence_set.selected_occurrence_ids
            if occurrence_id in occurrence_set.candidates
        )

    def to_dict(self) -> dict[str, Any]:
        active = self.active_set
        return {
            "schema_version": "OccurrenceResolutionStateV2",
            "revision": self.revision,
            "active_set_id": self.active_set_id or None,
            "selection_required": self.selection_required,
            "search_required": self.search_required,
            "active_resolution": active.resolution if active is not None else None,
            "candidate_count": self.candidate_count,
            "resolution_required": self.resolution_required,
            "arbitration_required": self.arbitration_required,
            "selected_occurrence_ids": list(self.selected_occurrence_ids),
            "sufficiency_enabled": self.sufficiency_enabled,
            "sufficiency_required": self.sufficiency_required,
            "sufficiency_candidate_limit": (
                DEFAULT_SUFFICIENCY_CANDIDATE_LIMIT
                if self.sufficiency_enabled
                else None
            ),
            "sufficiency_scope_occurrence_ids": list(
                self.sufficiency_scope_occurrence_ids
            ),
            "sufficiency_out_of_scope_occurrence_ids": list(
                self.sufficiency_out_of_scope_occurrence_ids
            ),
            "active_evidence_report": (
                active.evidence_report.to_dict()
                if active is not None and active.evidence_report is not None
                else None
            ),
            "active_sufficiency": (
                active.sufficiency.to_dict()
                if active is not None and active.sufficiency is not None
                else None
            ),
            "retired_set_ids": list(self.retired_set_ids),
            "active_locators": list(self.active_locators()),
            "retired_locators": list(self.retired_locators()),
            "sets": [occurrence_set.to_dict() for occurrence_set in self.sets.values()],
        }

    def to_reasoner_dict(self) -> dict[str, Any]:
        """Expose lifecycle state without leaking evidence or gate internals."""
        if self.sufficiency_required:
            active = self.active_set
            assert active is not None
            return {
                "schema_version": "OccurrenceResolutionStateV2",
                "revision": self.revision,
                "active_set_id": self.active_set_id,
                "selection_required": True,
                "search_required": False,
                "active_resolution": active.resolution,
                "candidate_count": len(active.candidates),
                "resolution_required": True,
                "arbitration_required": len(active.candidates) >= 2,
                "selected_occurrence_ids": [],
                "sufficiency_enabled": True,
                "sufficiency_required": True,
                "sufficiency_candidate_limit": (
                    DEFAULT_SUFFICIENCY_CANDIDATE_LIMIT
                ),
                "evidence_scope": {
                    "set_id": active.set_id,
                    "semantic_target": list(active.semantic_target),
                    "candidates": list(self.evidence_scope_candidates()),
                },
                "additional_candidate_count": len(
                    self.sufficiency_out_of_scope_occurrence_ids
                ),
            }
        payload = self.to_dict()
        payload.pop("active_evidence_report", None)
        payload.pop("active_sufficiency", None)
        for occurrence_set in payload["sets"]:
            occurrence_set.pop("evidence_report", None)
            occurrence_set.pop("sufficiency", None)
        return payload

    def save(self, path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    def _clone(self) -> OccurrenceResolutionStateV2:
        return OccurrenceResolutionStateV2(
            sets={
                set_id: OccurrenceSetStateV2(
                    set_id=value.set_id,
                    semantic_target=value.semantic_target,
                    candidates={
                        occurrence_id: dict(candidate)
                        for occurrence_id, candidate in value.candidates.items()
                    },
                    states=dict(value.states),
                    selected_occurrence_ids=value.selected_occurrence_ids,
                    resolution=value.resolution,
                    lifecycle=value.lifecycle,
                    evidence_report=value.evidence_report,
                    sufficiency=value.sufficiency,
                    revision=value.revision,
                )
                for set_id, value in self.sets.items()
            },
            active_set_id=self.active_set_id,
            sufficiency_enabled=self.sufficiency_enabled,
            revision=self.revision,
        )

    def _apply_one(
        self,
        operation: Mapping[str, Any],
        *,
        index: int,
        allow_resolution_transaction: bool = False,
    ) -> dict[str, Any] | None:
        if not isinstance(operation, Mapping):
            return {
                "code": "occurrence_op_must_be_object",
                "occurrence_op_index": index,
            }
        normalized = _normalized_scoped_occurrence_op(operation)
        op = str(normalized["op"])
        set_id = str(normalized["set_id"])
        occurrence_id = str(normalized.get("occurrence_id", ""))
        if op not in SCOPED_OCCURRENCE_RESOLUTION_OPS:
            return {
                "code": "unsupported_occurrence_op",
                "occurrence_op_index": index,
                "op": op,
            }
        occurrence_set = self.sets.get(set_id)
        if occurrence_set is None:
            return {
                "code": "occurrence_set_not_visible",
                "occurrence_op_index": index,
                "set_id": set_id,
            }
        if set_id != self.active_set_id or occurrence_set.lifecycle != "active":
            return {
                "code": "occurrence_set_not_active",
                "occurrence_op_index": index,
                "set_id": set_id,
                "active_set_id": self.active_set_id,
            }
        if (
            occurrence_set.resolution in {"selected", "no_match"}
            and not allow_resolution_transaction
        ):
            return {
                "code": "occurrence_resolution_already_committed",
                "occurrence_op_index": index,
                "set_id": set_id,
                "resolution": occurrence_set.resolution,
            }
        if op == SUFFICIENCY_OPERATION:
            if not self.sufficiency_enabled:
                return {
                    "code": "occurrence_sufficiency_not_enabled",
                    "occurrence_op_index": index,
                    "set_id": set_id,
                }
            if occurrence_id:
                return {
                    "code": "occurrence_sufficiency_forbids_occurrence_id",
                    "occurrence_op_index": index,
                    "set_id": set_id,
                }
            if (
                occurrence_set.evidence_report is not None
                or occurrence_set.sufficiency is not None
            ):
                return {
                    "code": "occurrence_sufficiency_already_assessed",
                    "occurrence_op_index": index,
                    "set_id": set_id,
                    "current_sufficiency_verdict": (
                        occurrence_set.sufficiency.verdict
                    ),
                    "sufficient_occurrence_ids": list(
                        occurrence_set.sufficiency.sufficient_occurrence_ids
                    ),
                }
            report, errors = validate_evidence_operation(
                operation,
                set_id=set_id,
                candidates=occurrence_set.candidates,
                viable_occurrence_ids=self.sufficiency_scope_occurrence_ids,
                out_of_scope_occurrence_ids=(
                    self.sufficiency_out_of_scope_occurrence_ids
                ),
                operation_index=index,
            )
            if errors:
                return {
                    "code": "_occurrence_evidence_validation_errors",
                    "errors": [dict(error) for error in errors],
                }
            assert report is not None
            decision = aggregate_occurrence_evidence(report)
            occurrence_set.evidence_report = report
            occurrence_set.sufficiency = decision
            for candidate_id, status in tuple(occurrence_set.states.items()):
                if status == "selected":
                    occurrence_set.states[candidate_id] = "active"
            occurrence_set.selected_occurrence_ids = ()
            if decision.winner_occurrence_id:
                occurrence_set.states[decision.winner_occurrence_id] = "selected"
                occurrence_set.selected_occurrence_ids = (
                    decision.winner_occurrence_id,
                )
                occurrence_set.resolution = "selected"
            else:
                occurrence_set.resolution = "no_match"
            occurrence_set.revision += 1
            return None
        if op in {"defer", "no_match"}:
            if occurrence_id:
                return {
                    "code": "set_resolution_op_forbids_occurrence_id",
                    "occurrence_op_index": index,
                    "set_id": set_id,
                    "op": op,
                }
            if self.sufficiency_enabled and (
                occurrence_set.sufficiency is None
                or occurrence_set.sufficiency.verdict != "insufficient"
            ):
                return {
                    "code": "occurrence_sufficiency_requires_insufficient",
                    "occurrence_op_index": index,
                    "set_id": set_id,
                    "op": op,
                    "current_sufficiency_verdict": (
                        occurrence_set.sufficiency.verdict
                        if occurrence_set.sufficiency is not None
                        else None
                    ),
                    "sufficient_occurrence_ids": list(
                        occurrence_set.sufficiency.sufficient_occurrence_ids
                        if occurrence_set.sufficiency is not None
                        else ()
                    ),
                }
            occurrence_set.selected_occurrence_ids = ()
            for candidate_id, status in tuple(occurrence_set.states.items()):
                if status == "selected":
                    occurrence_set.states[candidate_id] = "active"
            occurrence_set.resolution = "deferred" if op == "defer" else "no_match"
            occurrence_set.revision += 1
            return None
        if not occurrence_id or occurrence_id not in occurrence_set.candidates:
            return {
                "code": "occurrence_id_not_in_set",
                "occurrence_op_index": index,
                "set_id": set_id,
                "occurrence_id": occurrence_id,
            }
        current = occurrence_set.states.get(occurrence_id, "active")
        if op in {"keep", "select"} and current == "eliminated":
            return {
                "code": "eliminated_occurrence_requires_reopen",
                "occurrence_op_index": index,
                "set_id": set_id,
                "occurrence_id": occurrence_id,
            }
        selected = list(occurrence_set.selected_occurrence_ids)
        if op == "select":
            if self.sufficiency_enabled and occurrence_set.sufficiency is None:
                return {
                    "code": "occurrence_sufficiency_assessment_required",
                    "occurrence_op_index": index,
                    "set_id": set_id,
                    "occurrence_id": occurrence_id,
                }
            if self.sufficiency_enabled and (
                occurrence_set.sufficiency is not None
                and occurrence_set.sufficiency.verdict != "sufficient"
            ):
                return {
                    "code": "occurrence_sufficiency_forbids_selection",
                    "occurrence_op_index": index,
                    "set_id": set_id,
                    "occurrence_id": occurrence_id,
                    "current_sufficiency_verdict": (
                        occurrence_set.sufficiency.verdict
                    ),
                    "sufficient_occurrence_ids": list(
                        occurrence_set.sufficiency.sufficient_occurrence_ids
                    ),
                }
            if self.sufficiency_enabled and (
                occurrence_set.sufficiency is not None
                and occurrence_id
                not in occurrence_set.sufficiency.sufficient_occurrence_ids
            ):
                return {
                    "code": "occurrence_sufficiency_candidate_not_supported",
                    "occurrence_op_index": index,
                    "set_id": set_id,
                    "occurrence_id": occurrence_id,
                    "current_sufficiency_verdict": (
                        occurrence_set.sufficiency.verdict
                    ),
                    "sufficient_occurrence_ids": list(
                        occurrence_set.sufficiency.sufficient_occurrence_ids
                    ),
                }
            if occurrence_set.resolution == "no_match":
                return {
                    "code": "no_match_set_requires_reopen",
                    "occurrence_op_index": index,
                    "set_id": set_id,
                }
            occurrence_set.states[occurrence_id] = "selected"
            if occurrence_id not in selected:
                selected.append(occurrence_id)
            occurrence_set.selected_occurrence_ids = tuple(selected)
            occurrence_set.resolution = "selected"
        elif op == "eliminate":
            occurrence_set.evidence_report = None
            occurrence_set.sufficiency = None
            occurrence_set.states[occurrence_id] = "eliminated"
            occurrence_set.selected_occurrence_ids = tuple(
                value for value in selected if value != occurrence_id
            )
            if not occurrence_set.selected_occurrence_ids:
                occurrence_set.resolution = "unresolved"
        elif op == "reopen":
            occurrence_set.evidence_report = None
            occurrence_set.sufficiency = None
            occurrence_set.states[occurrence_id] = "active"
            occurrence_set.selected_occurrence_ids = tuple(
                value for value in selected if value != occurrence_id
            )
            occurrence_set.resolution = "unresolved"
        else:
            if op == "keep":
                occurrence_set.evidence_report = None
                occurrence_set.sufficiency = None
            if current != "selected":
                occurrence_set.states[occurrence_id] = "active"
            if occurrence_set.resolution in {"deferred", "no_match"}:
                occurrence_set.resolution = "unresolved"
        occurrence_set.revision += 1
        return None


def _normalized_scoped_occurrence_op(
    operation: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = {
        "op": str(operation.get("op", operation.get("type", "")) or "")
        .strip()
        .casefold(),
        "set_id": str(
            operation.get("set_id", operation.get("locator_attempt_id", "")) or ""
        ).strip(),
    }
    occurrence_id = str(operation.get("occurrence_id", "") or "").strip()
    if occurrence_id:
        normalized["occurrence_id"] = occurrence_id
    return normalized


def _apply_occurrence_op(
    states: dict[str, str],
    *,
    selected_occurrence_id: str,
    op: str,
    occurrence_id: str,
) -> str:
    selected = selected_occurrence_id
    if op == "select":
        if selected and selected != occurrence_id and states.get(selected) == "selected":
            states[selected] = "active"
        states[occurrence_id] = "selected"
        return occurrence_id
    if op == "eliminate":
        states[occurrence_id] = "eliminated"
        return "" if selected == occurrence_id else selected
    if op == "reopen":
        states[occurrence_id] = "active"
        return "" if selected == occurrence_id else selected
    if states.get(occurrence_id) != "selected":
        states[occurrence_id] = "active"
    return selected


def validate_occurrence_method_configuration(
    *,
    method_arm: str,
    oracle_arm: str,
    oracle_intervention: str | Path | None,
) -> str:
    arm = str(method_arm or "none").strip().casefold()
    if arm not in OCCURRENCE_METHOD_ARMS:
        raise ValueError(f"unsupported occurrence method arm: {arm}")
    if arm != "none" and (
        str(oracle_arm or "o0").strip().casefold() != "o0"
        or bool(str(oracle_intervention or "").strip())
    ):
        raise ValueError(
            f"occurrence method arm {arm} requires O0 and no oracle intervention"
        )
    return arm


def assert_no_oracle_packet(packet: Mapping[str, Any], *, surface: str) -> None:
    forbidden_paths = tuple(sorted(_forbidden_key_paths(packet)))
    if forbidden_paths:
        raise ValueError(
            f"no-oracle runtime gate failed for {surface}: "
            + ", ".join(forbidden_paths)
        )


class OccurrencePacketTransform:
    """Audit or reshape occurrence evidence without changing retrieval."""

    def __init__(
        self,
        *,
        arm: str,
        audit_path: Path,
        case_id: str = "",
        caption_config_digest: str = "",
        replay_fixture_path: Path | None = None,
        replay_record_path: Path | None = None,
        replay_prime: bool = False,
        candidate_limit: int = DEFAULT_CARD_CANDIDATE_LIMIT,
        excerpt_limit: int = DEFAULT_CARD_EXCERPT_LIMIT,
        excerpt_chars: int = DEFAULT_CARD_EXCERPT_CHARS,
        query_limit: int = DEFAULT_CARD_QUERY_LIMIT,
        query_chars: int = DEFAULT_CARD_QUERY_CHARS,
    ) -> None:
        normalized = validate_occurrence_method_configuration(
            method_arm=arm,
            oracle_arm="o0",
            oracle_intervention=None,
        )
        if normalized == "none":
            raise ValueError("OccurrencePacketTransform requires a method arm")
        self.arm = normalized
        self.audit_path = Path(audit_path)
        self.case_id = str(case_id or "")
        self.caption_config_digest = str(caption_config_digest or "")
        if replay_fixture_path is not None and replay_record_path is not None:
            raise ValueError(
                "occurrence replay fixture and record paths are mutually exclusive"
            )
        self.replay_fixture_path = (
            Path(replay_fixture_path) if replay_fixture_path is not None else None
        )
        self.replay_record_path = (
            Path(replay_record_path) if replay_record_path is not None else None
        )
        self.replay_prime = bool(replay_prime)
        if self.replay_prime and self.replay_fixture_path is None:
            raise ValueError("occurrence replay prime requires a replay fixture")
        self._replay_packets: tuple[dict[str, Any], ...] = ()
        self._replay_fixture_digest = ""
        self._replay_request_identity_digests: list[str] = []
        self._replay_consumed_identity_digests: list[str] = []
        self._replay_post_fixture_reuse_count = 0
        self._recorded_packets: list[dict[str, Any]] = []
        if self.replay_fixture_path is not None:
            fixture = _load_occurrence_replay_fixture(
                self.replay_fixture_path,
                case_id=self.case_id,
                caption_config_digest=self.caption_config_digest,
            )
            self._replay_packets = tuple(
                dict(row["packet"])
                for row in tuple(fixture.get("packets", ()) or ())
                if isinstance(row, Mapping) and isinstance(row.get("packet"), Mapping)
            )
            if not self._replay_packets:
                raise ValueError("occurrence replay fixture contains no packets")
            self._replay_fixture_digest = stable_digest(fixture)
        self.candidate_limit = max(1, int(candidate_limit))
        self.excerpt_limit = max(1, int(excerpt_limit))
        self.excerpt_chars = max(1, int(excerpt_chars))
        self.query_limit = max(1, int(query_limit))
        self.query_chars = max(1, int(query_chars))
        self._surface_checks: list[str] = []
        self._call_count = 0
        self._retrieval_parity_passed = True
        self._card_counts: list[int] = []
        self._representations: list[str] = []
        self._visible_excerpt_digests: list[str] = []
        self._retrieval_identity_digests: list[str] = []
        self._text_budget_parity_checks: list[dict[str, Any]] = []
        self._write_audit()

    @property
    def audit(self) -> Mapping[str, Any]:
        return self._audit_payload()

    @property
    def replay_prime_task_spec(self) -> Mapping[str, Any]:
        if not self.replay_prime or not self._replay_packets:
            raise ValueError("occurrence replay prime is not configured")
        packet = self._replay_packets[0]
        raw_range = packet.get("time_range")
        time_range = (
            (float(raw_range[0]), float(raw_range[1]))
            if isinstance(raw_range, Sequence)
            and not isinstance(raw_range, (str, bytes))
            and len(raw_range) == 2
            else None
        )
        return {
            "queries": tuple(
                dict.fromkeys(
                    str(value).strip()
                    for value in tuple(packet.get("queries", ()) or ())
                    if str(value).strip()
                )
            )[:5],
            "time_range": time_range,
            "segment_ids": tuple(
                str(value).strip()
                for value in tuple(packet.get("segment_ids", ()) or ())
                if str(value).strip()
            ),
            "source_video_ids": tuple(
                str(value).strip()
                for value in tuple(packet.get("source_video_ids", ()) or ())
                if str(value).strip()
            ),
            "top_k": max(1, int(packet.get("top_k", 12) or 12)),
            "expand_neighbors": max(
                0, int(packet.get("expand_neighbors", 0) or 0)
            ),
            "index_mode": str(packet.get("index_mode", "hybrid") or "hybrid"),
        }

    def validate_surface(self, packet: Mapping[str, Any], *, surface: str) -> None:
        assert_no_oracle_packet(packet, surface=surface)
        self._surface_checks.append(str(surface))
        self._write_audit()

    def __call__(self, packet: Mapping[str, Any]) -> Mapping[str, Any]:
        assert_no_oracle_packet(packet, surface="caption_packet_before_transform")
        natural_packet = dict(packet)
        if self.replay_record_path is not None:
            self._record_packet(natural_packet)
        if self._replay_packets:
            packet = self._consume_replay_packet(natural_packet)
        retrieval_before = _retrieval_identity(packet)
        representation = "identity"
        visible_excerpt_digest = stable_digest([])
        if self.arm == "a0":
            transformed: Mapping[str, Any] = packet
            card_count = 0
        else:
            transformed = dict(packet)
            occurrence_set = dict(transformed.get("occurrence_set") or {})
            cards = build_occurrence_candidate_cards(
                transformed,
                candidate_limit=self.candidate_limit,
                excerpt_limit=self.excerpt_limit,
                excerpt_chars=self.excerpt_chars,
                query_limit=self.query_limit,
                query_chars=self.query_chars,
            )
            flat_passages = flatten_occurrence_candidate_cards(cards)
            flat_queries = flatten_occurrence_candidate_queries(cards)
            grouped_text_digest = occurrence_visible_text_digest(cards=cards)
            flat_text_digest = occurrence_visible_text_digest(
                flat_passages=flat_passages,
                flat_queries=flat_queries,
            )
            grouped_text_chars = occurrence_visible_text_chars(cards=cards)
            flat_text_chars = occurrence_visible_text_chars(
                flat_passages=flat_passages,
                flat_queries=flat_queries,
            )
            text_budget_parity = {
                "grouped_text_digest": grouped_text_digest,
                "flat_text_digest": flat_text_digest,
                "grouped_text_chars": grouped_text_chars,
                "flat_text_chars": flat_text_chars,
                "passed": (
                    grouped_text_digest == flat_text_digest
                    and grouped_text_chars == flat_text_chars
                ),
            }
            if not text_budget_parity["passed"]:
                raise ValueError(
                    "grouped and flat occurrence text budgets differ"
                )
            occurrence_set["method_arm"] = (
                "scoped" if self.arm in {"a2-clean", "a3", "a4"} else self.arm
            )
            if self.arm == "a1-flat":
                representation = "flat"
                occurrence_set["flat_candidate_passages"] = flat_passages
                occurrence_set["flat_candidate_queries"] = flat_queries
            else:
                representation = "grouped"
                occurrence_set["candidate_cards"] = cards
            occurrence_set["candidate_card_budget"] = self._card_budget()
            transformed["occurrence_set"] = occurrence_set
            card_count = len(cards)
            visible_excerpt_digest = candidate_card_excerpt_digest(cards)
            self._text_budget_parity_checks.append(text_budget_parity)
        retrieval_after = _retrieval_identity(transformed)
        parity_passed = retrieval_before == retrieval_after
        self._retrieval_parity_passed &= parity_passed
        if not parity_passed:
            raise ValueError("occurrence method transform changed retrieval identity")
        assert_no_oracle_packet(
            transformed, surface="caption_packet_after_transform"
        )
        self._call_count += 1
        self._card_counts.append(card_count)
        self._representations.append(representation)
        self._visible_excerpt_digests.append(visible_excerpt_digest)
        self._retrieval_identity_digests.append(retrieval_before)
        self._write_audit()
        return transformed

    def _record_packet(self, packet: Mapping[str, Any]) -> None:
        packet_copy = _json_copy(packet)
        assert_no_oracle_packet(packet_copy, surface="occurrence_replay_record")
        self._recorded_packets.append(
            {
                "ordinal": len(self._recorded_packets) + 1,
                "retrieval_identity_digest": _retrieval_identity(packet_copy),
                "packet": packet_copy,
            }
        )
        fixture = {
            "schema_version": "MMLifelongOccurrenceReplayV1",
            "case_id": self.case_id,
            "caption_config_digest": self.caption_config_digest,
            "source_method_arm": self.arm,
            "packets": list(self._recorded_packets),
        }
        assert self.replay_record_path is not None
        _write_json_atomic(self.replay_record_path, fixture)
        self._replay_fixture_digest = stable_digest(fixture)

    def _consume_replay_packet(
        self, natural_packet: Mapping[str, Any]
    ) -> dict[str, Any]:
        index = len(self._replay_consumed_identity_digests)
        if index >= len(self._replay_packets):
            if not self.replay_prime:
                raise ValueError(
                    "occurrence replay fixture exhausted before runtime Caption searches completed"
                )
            self._replay_request_identity_digests.append(
                _retrieval_identity(natural_packet)
            )
            self._replay_post_fixture_reuse_count += 1
            return _json_copy(self._replay_packets[-1])
        replay_packet = _json_copy(self._replay_packets[index])
        assert_no_oracle_packet(replay_packet, surface="occurrence_replay_packet")
        required = {
            "config_digest",
            "hits",
            "index_digest",
            "occurrence_set",
            "query_fingerprint",
            "rendered",
        }
        missing = sorted(required - set(replay_packet))
        if missing:
            raise ValueError(
                "occurrence replay packet missing required fields: "
                + ", ".join(missing)
            )
        if str(replay_packet.get("config_digest", "") or "") != self.caption_config_digest:
            raise ValueError("occurrence replay packet Caption digest mismatch")
        self._replay_request_identity_digests.append(
            _retrieval_identity(natural_packet)
        )
        self._replay_consumed_identity_digests.append(
            _retrieval_identity(replay_packet)
        )
        return replay_packet

    def _card_budget(self) -> dict[str, int]:
        return {
            "candidate_limit": self.candidate_limit,
            "excerpt_limit_per_candidate": self.excerpt_limit,
            "excerpt_char_limit": self.excerpt_chars,
            "query_limit_per_candidate": self.query_limit,
            "query_char_limit": self.query_chars,
        }

    def _audit_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "MMLifelongNoOracleRuntimeAuditV3",
            "method_arm": self.arm,
            "no_oracle_runtime_gate_passed": True,
            "forbidden_agent_visible_keys": sorted(FORBIDDEN_AGENT_VISIBLE_KEYS),
            "forbidden_key_paths_seen": [],
            "surface_checks": list(self._surface_checks),
            "caption_packet_call_count": self._call_count,
            "retrieval_parity_passed": self._retrieval_parity_passed,
            "candidate_card_counts": list(self._card_counts),
            "candidate_card_budget": self._card_budget(),
            "representations": list(self._representations),
            "visible_excerpt_digests": list(self._visible_excerpt_digests),
            "retrieval_identity_digests": list(
                self._retrieval_identity_digests
            ),
            "text_budget_parity_checks": list(
                self._text_budget_parity_checks
            ),
            "text_budget_parity_passed": all(
                bool(row.get("passed"))
                for row in self._text_budget_parity_checks
            ),
            "occurrence_replay": {
                "mode": (
                    "replay"
                    if self._replay_packets
                    else "record"
                    if self.replay_record_path is not None
                    else "live"
                ),
                "fixture_digest": self._replay_fixture_digest or None,
                "expected_packet_count": len(self._replay_packets),
                "consumed_packet_count": len(
                    self._replay_consumed_identity_digests
                ),
                "recorded_packet_count": len(self._recorded_packets),
                "prime_requested": self.replay_prime,
                "prime_consumed": (
                    self.replay_prime
                    and bool(self._replay_consumed_identity_digests)
                ),
                "post_fixture_reuse_count": self._replay_post_fixture_reuse_count,
                "consumption_complete": (
                    len(self._replay_consumed_identity_digests)
                    == len(self._replay_packets)
                    if self._replay_packets
                    else None
                ),
                "consumed_prefix_valid": (
                    bool(self._replay_consumed_identity_digests)
                    and self._replay_consumed_identity_digests
                    == [
                        _retrieval_identity(packet)
                        for packet in self._replay_packets[
                            : len(self._replay_consumed_identity_digests)
                        ]
                    ]
                    if self._replay_packets
                    else None
                ),
                "request_identity_digests": list(
                    self._replay_request_identity_digests
                ),
                "consumed_identity_digests": list(
                    self._replay_consumed_identity_digests
                ),
            },
        }

    def _write_audit(self) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.audit_path.with_name(f".{self.audit_path.name}.tmp")
        temporary.write_text(
            json.dumps(self._audit_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.audit_path)


def _load_occurrence_replay_fixture(
    path: Path,
    *,
    case_id: str,
    caption_config_digest: str,
) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("occurrence replay fixture must be a JSON object")
    fixture = dict(value)
    if fixture.get("schema_version") != "MMLifelongOccurrenceReplayV1":
        raise ValueError("unsupported occurrence replay fixture schema")
    if case_id and str(fixture.get("case_id", "") or "") != case_id:
        raise ValueError("occurrence replay fixture case_id mismatch")
    if (
        caption_config_digest
        and str(fixture.get("caption_config_digest", "") or "")
        != caption_config_digest
    ):
        raise ValueError("occurrence replay fixture Caption digest mismatch")
    assert_no_oracle_packet(fixture, surface="occurrence_replay_fixture")
    return fixture


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(dict(value), ensure_ascii=False))
    if not isinstance(copied, Mapping):
        raise TypeError("occurrence replay packet must serialize to an object")
    return dict(copied)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def build_occurrence_candidate_cards(
    packet: Mapping[str, Any],
    *,
    candidate_limit: int = DEFAULT_CARD_CANDIDATE_LIMIT,
    excerpt_limit: int = DEFAULT_CARD_EXCERPT_LIMIT,
    excerpt_chars: int = DEFAULT_CARD_EXCERPT_CHARS,
    query_limit: int = DEFAULT_CARD_QUERY_LIMIT,
    query_chars: int = DEFAULT_CARD_QUERY_CHARS,
) -> list[dict[str, Any]]:
    occurrence_set = packet.get("occurrence_set")
    if not isinstance(occurrence_set, Mapping):
        return []
    hits_by_passage: dict[str, dict[str, Any]] = {}
    for retrieval_rank, raw_hit in enumerate(
        tuple(packet.get("hits", ()) or ()), start=1
    ):
        if not isinstance(raw_hit, Mapping):
            continue
        hit = dict(raw_hit)
        passage_id = str(hit.get("passage_id", "") or "")
        if not passage_id:
            continue
        hit["_retrieval_rank"] = retrieval_rank
        hits_by_passage.setdefault(passage_id, hit)

    cards: list[dict[str, Any]] = []
    raw_candidates = tuple(occurrence_set.get("candidates", ()) or ())
    for raw_candidate in raw_candidates[: max(1, int(candidate_limit))]:
        if not isinstance(raw_candidate, Mapping):
            continue
        candidate = dict(raw_candidate)
        passage_ids = tuple(str(value) for value in candidate.get("passage_ids", ()))
        candidate_hits = sorted(
            (hits_by_passage[value] for value in passage_ids if value in hits_by_passage),
            key=lambda hit: int(hit.get("_retrieval_rank", 0) or 0),
        )
        representative_passages = [
            _representative_passage(
                hit,
                excerpt_chars=max(1, int(excerpt_chars)),
                query_limit=max(1, int(query_limit)),
                query_chars=max(1, int(query_chars)),
            )
            for hit in candidate_hits[: max(1, int(excerpt_limit))]
        ]
        cards.append(
            {
                "occurrence_id": str(candidate.get("occurrence_id", "") or ""),
                "rank": int(candidate.get("rank", len(cards) + 1) or len(cards) + 1),
                "time_range": list(candidate.get("time_range", ()) or ()),
                "source_video_ids": list(candidate.get("source_video_ids", ()) or ()),
                "matched_queries": _candidate_queries(
                    candidate,
                    candidate_hits,
                    limit=max(1, int(query_limit)),
                    char_limit=max(1, int(query_chars)),
                ),
                "representative_passages": representative_passages,
                "evidence_role": "locator_only",
                "answer_support": False,
            }
        )
    return cards


def candidate_cards_by_occurrence(
    occurrence_set: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(card.get("occurrence_id", "") or ""): dict(card)
        for card in tuple(occurrence_set.get("candidate_cards", ()) or ())
        if isinstance(card, Mapping) and str(card.get("occurrence_id", "") or "")
    }


def flatten_occurrence_candidate_cards(
    cards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose A1's excerpt budget without mechanically binding text to an occurrence."""
    flattened: list[dict[str, Any]] = []
    for card in cards:
        for passage in tuple(card.get("representative_passages", ()) or ()):
            if not isinstance(passage, Mapping):
                continue
            flattened.append(
                {
                    "passage_id": str(passage.get("passage_id", "") or ""),
                    "time_range": list(passage.get("time_range", ()) or ()),
                    "caption_excerpt": str(
                        passage.get("caption_excerpt", "") or ""
                    ),
                    "query_matches": list(passage.get("query_matches", ()) or ()),
                    "evidence_role": "locator_only",
                    "answer_support": False,
                }
            )
    return flattened


def flatten_occurrence_candidate_queries(
    cards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for card in cards:
        for query in tuple(card.get("matched_queries", ()) or ()):
            text = str(query or "").strip()
            if not text:
                continue
            flattened.append(
                {
                    "query": text,
                    "evidence_role": "locator_only",
                    "answer_support": False,
                }
            )
    return flattened


def candidate_card_excerpt_digest(cards: Sequence[Mapping[str, Any]]) -> str:
    excerpts = [
            {
                "caption_excerpt": str(
                    passage.get("caption_excerpt", "") or ""
                ),
                "query_matches": list(passage.get("query_matches", ()) or ()),
            }
            for card in cards
            for passage in tuple(card.get("representative_passages", ()) or ())
            if isinstance(passage, Mapping)
        ]
    return occurrence_excerpt_digest(excerpts)


def occurrence_excerpt_digest(excerpts: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {
            "caption_excerpt": str(item.get("caption_excerpt", "") or ""),
            "query_matches": list(item.get("query_matches", ()) or ()),
        }
        for item in excerpts
        if isinstance(item, Mapping)
    ]
    return stable_digest(sorted(normalized, key=stable_digest))


def occurrence_visible_text_digest(
    *,
    cards: Sequence[Mapping[str, Any]] = (),
    flat_passages: Sequence[Mapping[str, Any]] = (),
    flat_queries: Sequence[Mapping[str, Any]] = (),
) -> str:
    inventory = _occurrence_visible_text_inventory(
        cards=cards,
        flat_passages=flat_passages,
        flat_queries=flat_queries,
    )
    return stable_digest(sorted(inventory, key=stable_digest))


def _occurrence_visible_text_inventory(
    *,
    cards: Sequence[Mapping[str, Any]] = (),
    flat_passages: Sequence[Mapping[str, Any]] = (),
    flat_queries: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, str]]:
    inventory: list[dict[str, Any]] = []
    for card in cards:
        for query in tuple(card.get("matched_queries", ()) or ()):
            inventory.append({"kind": "candidate_query", "text": str(query or "")})
        for passage in tuple(card.get("representative_passages", ()) or ()):
            if not isinstance(passage, Mapping):
                continue
            inventory.append(
                {
                    "kind": "caption_excerpt",
                    "text": str(passage.get("caption_excerpt", "") or ""),
                }
            )
            inventory.extend(
                {"kind": "passage_query", "text": str(query or "")}
                for query in tuple(passage.get("query_matches", ()) or ())
            )
    for passage in flat_passages:
        if not isinstance(passage, Mapping):
            continue
        inventory.append(
            {
                "kind": "caption_excerpt",
                "text": str(passage.get("caption_excerpt", "") or ""),
            }
        )
        inventory.extend(
            {"kind": "passage_query", "text": str(query or "")}
            for query in tuple(passage.get("query_matches", ()) or ())
        )
    inventory.extend(
        {
            "kind": "candidate_query",
            "text": str(query.get("query", "") or ""),
        }
        for query in flat_queries
        if isinstance(query, Mapping)
    )
    return inventory


def occurrence_visible_text_chars(
    *,
    cards: Sequence[Mapping[str, Any]] = (),
    flat_passages: Sequence[Mapping[str, Any]] = (),
    flat_queries: Sequence[Mapping[str, Any]] = (),
) -> int:
    inventory = _occurrence_visible_text_inventory(
        cards=cards,
        flat_passages=flat_passages,
        flat_queries=flat_queries,
    )
    return sum(len(str(row["text"])) for row in inventory)


def _representative_passage(
    hit: Mapping[str, Any], *, excerpt_chars: int, query_limit: int, query_chars: int
) -> dict[str, Any]:
    metadata = hit.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    return {
        "passage_id": str(hit.get("passage_id", "") or ""),
        "time_range": [
            float(hit.get("virtual_start_sec", 0.0) or 0.0),
            float(hit.get("virtual_end_sec", 0.0) or 0.0),
        ],
        "caption_excerpt": str(hit.get("text", "") or "")[:excerpt_chars],
        "query_matches": _query_strings(
            metadata.get("query_matches") or metadata.get("matched_queries", ()),
            limit=query_limit,
            char_limit=query_chars,
        ),
    }


def _candidate_queries(
    candidate: Mapping[str, Any],
    hits: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    char_limit: int,
) -> list[str]:
    values: list[Any] = list(candidate.get("query_matches", ()) or ())
    for hit in hits:
        metadata = hit.get("metadata")
        if isinstance(metadata, Mapping):
            values.extend(
                tuple(
                    metadata.get("query_matches")
                    or metadata.get("matched_queries", ())
                    or ()
                )
            )
    return _query_strings(values, limit=limit, char_limit=char_limit)


def _query_strings(value: Any, *, limit: int, char_limit: int) -> list[str]:
    raw_values = (value,) if isinstance(value, (str, Mapping)) else tuple(value or ())
    queries: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        query = (
            str(raw.get("query", "") or "").strip()
            if isinstance(raw, Mapping)
            else str(raw or "").strip()
        )
        if query and query not in seen:
            seen.add(query)
            queries.append(query[:char_limit])
            if len(queries) >= limit:
                break
    return queries


def occurrence_replay_identity(packet: Mapping[str, Any]) -> str:
    occurrence_set = packet.get("occurrence_set")
    candidates = (
        occurrence_set.get("candidates")
        if isinstance(occurrence_set, Mapping)
        else None
    )
    return stable_digest(
        {
            "queries": packet.get("queries"),
            "time_range": packet.get("time_range"),
            "segment_ids": packet.get("segment_ids"),
            "source_video_ids": packet.get("source_video_ids"),
            "top_k": packet.get("top_k"),
            "index_digest": packet.get("index_digest"),
            "query_fingerprint": packet.get("query_fingerprint"),
            "hits": packet.get("hits"),
            "occurrence_candidates": candidates,
            "rendered": packet.get("rendered"),
        }
    )


def _retrieval_identity(packet: Mapping[str, Any]) -> str:
    return occurrence_replay_identity(packet)


def _forbidden_key_paths(value: Any, *, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}"
            if key.casefold() in FORBIDDEN_AGENT_VISIBLE_KEYS:
                paths.append(path)
            paths.extend(_forbidden_key_paths(child, prefix=path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            paths.extend(_forbidden_key_paths(child, prefix=f"{prefix}[{index}]"))
    return paths
