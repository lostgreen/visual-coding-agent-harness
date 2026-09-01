"""Validated slot-memory transactions for WP17 construction experiments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


WP17_SLOT_TRANSACTION_CONTRACT = "WP17-slot-memory-transaction-v2"
WP17_SLOT_STATE_CONTRACT = "WP17-slot-memory-state-v2"
WP17_SLOT_CAPSULE_CONTRACT = "WP17-slot-memory-capsule-v3"
WP17_CAPSULE_PROVENANCE_CONTRACT = "WP17-slot-capsule-provenance-summary-v2"
WP17_SLOT_REPAIR_CONTRACT = "WP17-slot-memory-repair-v1"
WP17_BUDGET_TOKENIZER = "VCAH-unicode-budget-tokenizer-v1"
WP17_SLOT_LIFECYCLE_POLICY_V9 = "WP17-slot-lifecycle-v9"
WP17_SLOT_LIFECYCLE_POLICY_V10 = "WP17-slot-lifecycle-reliability-v10"
WP17_CLOSED_SWEEP_AFTER_UNTOUCHED_TRANSACTIONS = 1
WP17_SLOT_NAMES = (
    "location",
    "active_encounter",
    "active_participants",
    "equipped_or_held_item",
    "recent_state_change",
    "occurrence_counter",
    "current_activity",
)
WP17_SLOT_OPERATIONS = (
    "write",
    "update",
    "retain",
    "close",
    "archive",
    "evict",
)
WP17_OBSERVATION_KINDS = (
    "entity",
    "event",
    "state",
    "relation",
    "visible_text",
    "activity",
    "location",
)
WP17_MAX_OBSERVATIONS = 16
WP17_TARGET_OBSERVATION_EVIDENCE_IDS = 6
WP17_MAX_OBSERVATION_PARTICIPANTS = 12
WP17_MAX_STRUCTURED_EVENT_ITEMS = 12
WP17_MAX_OUTPUT_JSON_CHARS = 10_000
_WORKING_STATUSES = {"active", "closed"}
_TOKEN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", re.UNICODE)


class SlotTransactionError(ValueError):
    """Raised when a model-authored slot transaction violates the contract."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "slot_validation_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details = deepcopy(dict(details or {}))

    def repair_contract(self) -> dict[str, Any]:
        return {
            "contract": WP17_SLOT_REPAIR_CONTRACT,
            "error_code": self.code,
            "message": str(self),
            "details": deepcopy(self.details),
        }


_FORBIDDEN_MODEL_KEYS = {
    "question",
    "options",
    "gold",
    "gold_answer",
    "official_intervals",
    "case_id",
    "case_ids",
    "source_path",
}


def budget_tokens(text: str) -> tuple[str, ...]:
    """Return deterministic protocol tokens used for both C1 and C2 budgets."""
    return tuple(_TOKEN_RE.findall(str(text)))


def budget_token_count(text: str) -> int:
    return len(budget_tokens(text))


def tail_budget_text(text: str, *, max_tokens: int) -> str:
    """Keep a token-bounded suffix without rewriting its bytes or punctuation."""
    limit = max(0, int(max_tokens))
    source = str(text)
    matches = tuple(_TOKEN_RE.finditer(source))
    if not matches or limit == 0:
        return ""
    if len(matches) <= limit:
        return source
    return source[matches[-limit].start() :]


def validate_observations(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_evidence_ids: Sequence[str],
    evidence_id_map: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    if len(rows) > WP17_MAX_OBSERVATIONS:
        raise SlotTransactionError(
            f"observations exceed the frozen limit of {WP17_MAX_OBSERVATIONS}"
        )
    allowed = {str(value) for value in allowed_evidence_ids}
    canonical = {
        str(key): str(value)
        for key, value in dict(evidence_id_map or {}).items()
    }
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        row = dict(raw)
        observation_id = str(row.get("observation_id", "") or "").strip()
        if not observation_id or observation_id in seen:
            raise SlotTransactionError("observation IDs must be nonempty and unique")
        seen.add(observation_id)
        kind = str(row.get("kind", "") or "").strip().casefold()
        if kind not in WP17_OBSERVATION_KINDS:
            raise SlotTransactionError(f"unsupported observation kind: {kind}")
        fact = str(row.get("fact", "") or "").strip()
        if not fact:
            raise SlotTransactionError("every observation requires a fact")
        evidence_ids = tuple(
            dict.fromkeys(str(value).strip() for value in row.get("evidence_ids", ()) if str(value).strip())
        )
        if not evidence_ids or not set(evidence_ids).issubset(allowed):
            raise SlotTransactionError("observation evidence must resolve inside the current packet")
        canonical_evidence_ids = tuple(canonical.get(value, value) for value in evidence_ids)
        participants = tuple(
            dict.fromkeys(str(value).strip() for value in row.get("participants", ()) if str(value).strip())
        )
        if len(participants) > WP17_MAX_OBSERVATION_PARTICIPANTS:
            raise SlotTransactionError(
                "observation participants exceed the frozen per-observation limit"
            )
        normalized.append(
            {
                "observation_id": observation_id,
                "kind": kind,
                "fact": fact,
                "evidence_ids": list(canonical_evidence_ids),
                "participants": list(participants),
            }
        )
    return tuple(normalized)


def validate_structured_event_record(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    _reject_forbidden_model_keys(row)
    required_lists = ("entities", "events", "state_changes", "relations", "occurrence_refs")
    for key in required_lists:
        if key not in row:
            row[key] = []
        if not isinstance(row[key], list):
            singleton = row[key]
            if singleton is None or singleton == "":
                row[key] = []
            elif isinstance(singleton, (Mapping, str, int, float, bool)):
                row[key] = [deepcopy(singleton)]
            else:
                raise SlotTransactionError(
                    f"structured_event_record.{key} must be a list or singleton"
                )
        if len(row[key]) > WP17_MAX_STRUCTURED_EVENT_ITEMS:
            raise SlotTransactionError(
                f"structured_event_record.{key} exceeds the frozen item limit"
            )
    summary = str(row.get("summary", "") or "").strip()
    if not summary:
        raise SlotTransactionError("structured_event_record.summary is required")
    return {key: deepcopy(row[key]) for key in required_lists} | {"summary": summary}


@dataclass
class SlotMemoryState:
    """Arm-local working slots plus an append-only long-term lifecycle ledger."""

    arm: str
    token_budget: int = 600
    lifecycle_policy: str = WP17_SLOT_LIFECYCLE_POLICY_V9
    closed_sweep_after_untouched_transactions: int = (
        WP17_CLOSED_SWEEP_AFTER_UNTOUCHED_TRANSACTIONS
    )
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    ledger: list[dict[str, Any]] = field(default_factory=list)
    transaction_index: int = 0

    def __post_init__(self) -> None:
        self.arm = str(self.arm)
        self.token_budget = int(self.token_budget)
        if self.token_budget <= 0:
            raise ValueError("slot token budget must be positive")
        self.lifecycle_policy = str(self.lifecycle_policy)
        if self.lifecycle_policy not in {
            WP17_SLOT_LIFECYCLE_POLICY_V9,
            WP17_SLOT_LIFECYCLE_POLICY_V10,
        }:
            raise ValueError(f"unknown slot lifecycle policy: {self.lifecycle_policy}")
        self.closed_sweep_after_untouched_transactions = int(
            self.closed_sweep_after_untouched_transactions
        )
        if self.closed_sweep_after_untouched_transactions <= 0:
            raise ValueError("closed slot sweep horizon must be positive")
        unknown = set(self.records) - set(WP17_SLOT_NAMES)
        if unknown:
            raise ValueError(f"unknown slot records: {sorted(unknown)}")

    def apply(
        self,
        payload: Mapping[str, Any],
        *,
        segment_id: str,
        allowed_evidence_ids: Sequence[str],
        evidence_id_map: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        if payload.get("contract") != WP17_SLOT_TRANSACTION_CONTRACT:
            raise SlotTransactionError("slot transaction contract mismatch")
        observations = validate_observations(
            tuple(payload.get("observations", ()) or ()),
            allowed_evidence_ids=allowed_evidence_ids,
            evidence_id_map=evidence_id_map,
        )
        ser = validate_structured_event_record(
            dict(payload.get("structured_event_record", {}) or {})
        )
        operation_rows = tuple(payload.get("slot_operations", ()) or ())
        if not all(isinstance(row, Mapping) for row in operation_rows):
            raise SlotTransactionError("slot_operations must contain objects")
        observations_by_id = {row["observation_id"]: row for row in observations}

        working_before = {
            name for name, record in self.records.items() if record.get("status") in _WORKING_STATUSES
        }
        normalized_operations = self._normalize_operations(
            operation_rows,
            observations_by_id=observations_by_id,
        )
        touched = {row["slot"] for row in normalized_operations}

        candidate = deepcopy(self.records)
        transaction_events: list[dict[str, Any]] = []
        for operation in normalized_operations:
            event = self._apply_one(
                candidate,
                operation,
                segment_id=str(segment_id),
                observations_by_id=observations_by_id,
            )
            transaction_events.append(event)
        for name in sorted(working_before - touched):
            record = candidate[name]
            transaction_events.append(
                {
                    "event": "slot_lifecycle",
                    "segment_id": str(segment_id),
                    "slot": name,
                    "operation": "implicit_retain",
                    "from_status": str(record["status"]),
                    "to_status": str(record["status"]),
                    "from_version": int(record["version"]),
                    "to_version": int(record["version"]),
                    "observation_ids": [],
                    "provenance": list(record["provenance"]),
                }
            )
        self._validate_cross_slot_invariants(candidate)

        prior_records = self.records
        prior_ledger = self.ledger
        prior_index = self.transaction_index
        self.records = candidate
        self.transaction_index += 1
        self.ledger = list(self.ledger) + transaction_events
        try:
            lifecycle_policy_events = self._enforce_lifecycle_policy(
                segment_id=str(segment_id),
                touched=touched,
            )
            self.ledger.extend(lifecycle_policy_events)
            budget_events = self._enforce_budget(segment_id=str(segment_id))
        except Exception:
            self.records = prior_records
            self.ledger = prior_ledger
            self.transaction_index = prior_index
            raise
        self.ledger.extend(budget_events)
        automatic_events = lifecycle_policy_events + budget_events
        capsule = self.capsule()
        result = {
            "contract": WP17_SLOT_TRANSACTION_CONTRACT,
            "segment_id": str(segment_id),
            "transaction_index": self.transaction_index,
            "observations": [dict(row) for row in observations],
            "slot_operations": [dict(row) for row in normalized_operations],
            "structured_event_record": ser,
            "lifecycle_events": transaction_events + automatic_events,
            "capsule": capsule,
            "state_digest": self.digest(),
            "long_term_ledger_count": len(self.ledger),
        }
        return result

    def capsule(self) -> dict[str, Any]:
        slots = [
            {
                "slot": name,
                "version": int(record["version"]),
                "status": str(record["status"]),
                "value": deepcopy(record["value"]),
                "provenance_count": len(record["provenance"]),
                "provenance_digest": hashlib.sha256(
                    _canonical_json(list(record["provenance"])).encode("utf-8")
                ).hexdigest(),
                "last_verified_segment_id": str(record["last_verified_segment_id"]),
            }
            for name, record in sorted(self.records.items())
            if record.get("status") in _WORKING_STATUSES
        ]
        versions = {
            name: int(record["version"])
            for name, record in sorted(self.records.items())
        }
        working_names = {row["slot"] for row in slots}
        visible_slots = [
            {
                "slot": row["slot"],
                "version": row["version"],
                "status": row["status"],
                "value": deepcopy(row["value"]),
            }
            for row in slots
        ]
        inactive_versions = {
            name: version for name, version in versions.items() if name not in working_names
        }
        visible_payload = {
            "slots": visible_slots,
            "inactive_versions": inactive_versions,
        }
        context = "" if not slots and not versions else _canonical_json(visible_payload)
        semantic_context = "" if not slots else _canonical_json(
            {
                "slots": [
                    {"slot": row["slot"], "value": deepcopy(row["value"])}
                    for row in slots
                ]
            }
        )
        payload = {
            "contract": WP17_SLOT_CAPSULE_CONTRACT,
            "provenance_projection_contract": WP17_CAPSULE_PROVENANCE_CONTRACT,
            "arm": self.arm,
            "slots": slots,
            "versions": versions,
            "context": context,
        }
        if self.lifecycle_policy == WP17_SLOT_LIFECYCLE_POLICY_V10:
            payload["lifecycle_policy"] = self.lifecycle_policy
        payload["tokenizer"] = WP17_BUDGET_TOKENIZER
        payload["token_count"] = budget_token_count(context)
        payload["semantic_token_count"] = budget_token_count(semantic_context)
        payload["overhead_token_count"] = max(
            0, payload["token_count"] - payload["semantic_token_count"]
        )
        payload["overhead_share"] = (
            payload["overhead_token_count"] / payload["token_count"]
            if payload["token_count"]
            else 0.0
        )
        payload["token_budget"] = self.token_budget
        payload["within_budget"] = payload["token_count"] <= self.token_budget
        return payload

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "contract": WP17_SLOT_STATE_CONTRACT,
            "arm": self.arm,
            "token_budget": self.token_budget,
            "transaction_index": self.transaction_index,
            "records": deepcopy(self.records),
            "ledger": deepcopy(self.ledger),
            "digest": self.digest(),
        }
        if self.lifecycle_policy == WP17_SLOT_LIFECYCLE_POLICY_V10:
            payload["lifecycle_policy"] = self.lifecycle_policy
            payload["closed_sweep_after_untouched_transactions"] = (
                self.closed_sweep_after_untouched_transactions
            )
        return payload

    @classmethod
    def from_snapshot(cls, payload: Mapping[str, Any]) -> "SlotMemoryState":
        if payload.get("contract") != WP17_SLOT_STATE_CONTRACT:
            raise ValueError("slot state contract mismatch")
        state = cls(
            arm=str(payload["arm"]),
            token_budget=int(payload["token_budget"]),
            lifecycle_policy=str(
                payload.get("lifecycle_policy", WP17_SLOT_LIFECYCLE_POLICY_V9)
            ),
            closed_sweep_after_untouched_transactions=int(
                payload.get(
                    "closed_sweep_after_untouched_transactions",
                    WP17_CLOSED_SWEEP_AFTER_UNTOUCHED_TRANSACTIONS,
                )
            ),
            records=deepcopy(dict(payload.get("records", {}) or {})),
            ledger=deepcopy(list(payload.get("ledger", ()) or ())),
            transaction_index=int(payload.get("transaction_index", 0) or 0),
        )
        if str(payload.get("digest", "") or "") and state.digest() != payload["digest"]:
            raise ValueError("slot state snapshot digest mismatch")
        return state

    def digest(self) -> str:
        payload = {
            "arm": self.arm,
            "token_budget": self.token_budget,
            "transaction_index": self.transaction_index,
            "records": self.records,
            "ledger": self.ledger,
        }
        if self.lifecycle_policy == WP17_SLOT_LIFECYCLE_POLICY_V10:
            payload["lifecycle_policy"] = self.lifecycle_policy
            payload["closed_sweep_after_untouched_transactions"] = (
                self.closed_sweep_after_untouched_transactions
            )
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def _normalize_operations(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        observations_by_id: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        normalized = []
        operation_counts: dict[str, int] = {}
        for raw in rows:
            row = dict(raw)
            operation = str(row.get("operation", "") or "").strip().casefold()
            slot = str(row.get("slot", "") or "").strip().casefold()
            if operation not in WP17_SLOT_OPERATIONS:
                raise SlotTransactionError(f"unsupported slot operation: {operation}")
            if slot not in WP17_SLOT_NAMES:
                raise SlotTransactionError(
                    "slot operation references an unknown slot",
                    code="unknown_slot",
                    details={"slot": slot, "allowed_slots": list(WP17_SLOT_NAMES)},
                )
            operation_counts[slot] = operation_counts.get(slot, 0) + 1
            if operation_counts[slot] > 3:
                raise SlotTransactionError(
                    "each known slot can receive at most three sequential operations",
                    code="too_many_slot_operations",
                    details={"slot": slot, "maximum": 3},
                )
            observation_ids = tuple(
                dict.fromkeys(
                    str(value).strip()
                    for value in row.get("observation_ids", ())
                    if str(value).strip()
                )
            )
            unknown_observation_ids = sorted(
                set(observation_ids) - set(observations_by_id)
            )
            if unknown_observation_ids:
                raise SlotTransactionError(
                    "slot operation references unknown observation IDs "
                    f"{unknown_observation_ids}; valid observation IDs are "
                    f"{sorted(observations_by_id)}",
                    code="unknown_observation_ids",
                    details={
                        "slot": slot,
                        "operation": operation,
                        "unknown_observation_ids": unknown_observation_ids,
                        "valid_observation_ids": sorted(observations_by_id),
                    },
                )
            if operation in {"write", "update", "close"} and not observation_ids:
                raise SlotTransactionError(f"{operation} requires current observations")
            if operation in {"write", "update"} and "value" not in row:
                raise SlotTransactionError(f"{operation} requires a value")
            if operation in {"retain", "close", "archive", "evict"} and "value" in row:
                raise SlotTransactionError(f"{operation} cannot rewrite slot value")
            if "value" in row:
                _reject_forbidden_model_keys(row["value"])
            normalized.append(
                {
                    "operation": operation,
                    "slot": slot,
                    "expected_version": int(row.get("expected_version", 0) or 0),
                    "observation_ids": list(observation_ids),
                    **({"value": deepcopy(row["value"])} if "value" in row else {}),
                }
            )
        return tuple(normalized)

    def _apply_one(
        self,
        records: dict[str, dict[str, Any]],
        operation: Mapping[str, Any],
        *,
        segment_id: str,
        observations_by_id: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        name = str(operation["slot"])
        action = str(operation["operation"])
        prior = deepcopy(records.get(name))
        prior_version = int(prior.get("version", 0) if prior else 0)
        if int(operation["expected_version"]) != prior_version:
            raise SlotTransactionError(
                f"slot version mismatch for {name}",
                code="slot_version_mismatch",
                details={
                    "slot": name,
                    "operation": action,
                    "expected_version": int(operation["expected_version"]),
                    "actual_version": prior_version,
                    "current_status": str(
                        prior.get("status", "absent") if prior else "absent"
                    ),
                    "repair_operations": self._repair_operations(
                        operation,
                        prior_status=str(
                            prior.get("status", "absent") if prior else "absent"
                        ),
                        prior_version=prior_version,
                    ),
                },
            )
        prior_status = str(prior.get("status", "absent") if prior else "absent")
        observation_ids = tuple(operation["observation_ids"])
        current_evidence = tuple(
            dict.fromkeys(
                evidence_id
                for observation_id in observation_ids
                for evidence_id in observations_by_id[observation_id]["evidence_ids"]
            )
        )

        if action == "write":
            if prior_status in _WORKING_STATUSES:
                required = (
                    ["close", "archive", "write"]
                    if prior_status == "active"
                    else ["archive", "write"]
                )
                raise SlotTransactionError(
                    f"cannot write working slot {name}",
                    code="write_on_working_slot",
                    details={
                        "slot": name,
                        "current_status": prior_status,
                        "current_version": prior_version,
                        "required_operation_sequence": required,
                        "repair_operations": self._repair_operations(
                            operation,
                            prior_status=prior_status,
                            prior_version=prior_version,
                        ),
                    },
                )
            record = {
                "slot": name,
                "version": prior_version + 1,
                "status": "active",
                "value": deepcopy(operation["value"]),
                "provenance": list(current_evidence),
                "last_verified_segment_id": segment_id,
            }
        elif action == "update":
            if prior_status != "active":
                raise SlotTransactionError(
                    f"cannot update non-active slot {name}",
                    code="update_on_non_active_slot",
                    details={
                        "slot": name,
                        "current_status": prior_status,
                        "current_version": prior_version,
                        "repair_operations": self._repair_operations(
                            operation,
                            prior_status=prior_status,
                            prior_version=prior_version,
                        ),
                    },
                )
            value_changed = operation["value"] != prior["value"]
            record = {
                **prior,
                "version": prior_version + 1,
                "value": deepcopy(operation["value"]),
                "provenance": (
                    list(current_evidence)
                    if value_changed
                    else list(
                        dict.fromkeys(tuple(prior["provenance"]) + current_evidence)
                    )
                ),
                "last_verified_segment_id": segment_id,
            }
        elif action == "retain":
            if prior_status not in _WORKING_STATUSES:
                raise SlotTransactionError(
                    f"cannot retain non-working slot {name}",
                    code="retain_on_non_working_slot",
                    details={
                        "slot": name,
                        "current_status": prior_status,
                        "current_version": prior_version,
                        "repair_operations": self._repair_operations(
                            operation,
                            prior_status=prior_status,
                            prior_version=prior_version,
                        ),
                    },
                )
            record = (
                {
                    **prior,
                    "version": prior_version + 1,
                    "provenance": list(
                        dict.fromkeys(tuple(prior["provenance"]) + current_evidence)
                    ),
                    "last_verified_segment_id": segment_id,
                }
                if observation_ids
                else prior
            )
        elif action == "close":
            if (
                self.lifecycle_policy == WP17_SLOT_LIFECYCLE_POLICY_V10
                and prior_status in {"closed", "archived", "evicted"}
            ):
                record = prior
                action = "redundant_close"
            else:
                if prior_status != "active":
                    raise SlotTransactionError(
                        f"cannot close non-active slot {name}",
                        code="close_on_non_active_slot",
                        details={
                            "slot": name,
                            "current_status": prior_status,
                            "current_version": prior_version,
                            "repair_operations": self._repair_operations(
                                operation,
                                prior_status=prior_status,
                                prior_version=prior_version,
                            ),
                        },
                    )
                record = {
                    **prior,
                    "version": prior_version + 1,
                    "status": "closed",
                    "provenance": list(
                        dict.fromkeys(tuple(prior["provenance"]) + current_evidence)
                    ),
                    "last_verified_segment_id": segment_id,
                    "closed_at_transaction_index": self.transaction_index + 1,
                }
        elif action == "archive":
            if (
                self.lifecycle_policy == WP17_SLOT_LIFECYCLE_POLICY_V10
                and prior_status in {"archived", "evicted"}
                and not observation_ids
            ):
                record = prior
                action = "redundant_archive"
            else:
                if prior_status != "closed" or observation_ids:
                    raise SlotTransactionError(
                        f"archive requires a closed unchanged slot: {name}",
                        code="archive_on_non_closed_slot",
                        details={
                            "slot": name,
                            "current_status": prior_status,
                            "current_version": prior_version,
                            "repair_operations": self._repair_operations(
                                operation,
                                prior_status=prior_status,
                                prior_version=prior_version,
                            ),
                        },
                    )
                record = {**prior, "version": prior_version + 1, "status": "archived"}
        elif action == "evict":
            if (
                self.lifecycle_policy == WP17_SLOT_LIFECYCLE_POLICY_V10
                and prior_status == "evicted"
                and not observation_ids
            ):
                record = prior
                action = "redundant_evict"
            else:
                if prior_status not in {"closed", "archived"} or observation_ids:
                    raise SlotTransactionError(
                        f"evict requires a closed/archived unchanged slot: {name}",
                        code="evict_on_non_terminal_slot",
                        details={
                            "slot": name,
                            "current_status": prior_status,
                            "current_version": prior_version,
                            "repair_operations": self._repair_operations(
                                operation,
                                prior_status=prior_status,
                                prior_version=prior_version,
                            ),
                        },
                    )
                record = {**prior, "version": prior_version + 1, "status": "evicted"}
        else:  # pragma: no cover - normalized above
            raise AssertionError(action)
        records[name] = record
        return {
            "event": "slot_lifecycle",
            "segment_id": segment_id,
            "slot": name,
            "operation": action,
            "from_status": prior_status,
            "to_status": str(record["status"]),
            "from_version": prior_version,
            "to_version": int(record["version"]),
            "observation_ids": list(observation_ids),
            "provenance": list(record["provenance"]),
        }

    def _repair_operations(
        self,
        operation: Mapping[str, Any],
        *,
        prior_status: str,
        prior_version: int,
    ) -> list[dict[str, Any]]:
        action = str(operation["operation"])
        slot = str(operation["slot"])
        observation_ids = list(operation.get("observation_ids", ()) or ())

        def row(
            value: str,
            version: int,
            *,
            include_observations: bool = False,
            include_value: bool = False,
        ) -> dict[str, Any]:
            result: dict[str, Any] = {
                "operation": value,
                "slot": slot,
                "expected_version": int(version),
                "observation_ids": observation_ids if include_observations else [],
            }
            if include_value:
                result["value"] = deepcopy(operation["value"])
            return result

        if action == "write":
            if prior_status == "active":
                return [
                    row("close", prior_version, include_observations=True),
                    row("archive", prior_version + 1),
                    row(
                        "write",
                        prior_version + 2,
                        include_observations=True,
                        include_value=True,
                    ),
                ]
            if prior_status == "closed":
                return [
                    row("archive", prior_version),
                    row(
                        "write",
                        prior_version + 1,
                        include_observations=True,
                        include_value=True,
                    ),
                ]
            return [
                row(
                    "write",
                    prior_version,
                    include_observations=True,
                    include_value=True,
                )
            ]
        if action == "update":
            if prior_status == "active":
                return [
                    row(
                        "update",
                        prior_version,
                        include_observations=True,
                        include_value=True,
                    )
                ]
            if prior_status == "closed":
                return [
                    row("archive", prior_version),
                    row(
                        "write",
                        prior_version + 1,
                        include_observations=True,
                        include_value=True,
                    ),
                ]
            return [
                row(
                    "write",
                    prior_version,
                    include_observations=True,
                    include_value=True,
                )
            ]
        if action == "retain":
            return (
                [row("retain", prior_version, include_observations=bool(observation_ids))]
                if prior_status in _WORKING_STATUSES
                else []
            )
        if action == "close":
            if prior_status == "active" or (
                self.lifecycle_policy == WP17_SLOT_LIFECYCLE_POLICY_V10
                and prior_status in {"closed", "archived", "evicted"}
            ):
                return [row("close", prior_version, include_observations=True)]
            return []
        if action == "archive":
            if prior_status == "closed" or (
                self.lifecycle_policy == WP17_SLOT_LIFECYCLE_POLICY_V10
                and prior_status in {"archived", "evicted"}
            ):
                return [row("archive", prior_version)]
            return []
        if action == "evict":
            if prior_status in {"closed", "archived"} or (
                self.lifecycle_policy == WP17_SLOT_LIFECYCLE_POLICY_V10
                and prior_status == "evicted"
            ):
                return [row("evict", prior_version)]
            return []
        return []

    def _validate_cross_slot_invariants(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        participants = records.get("active_participants")
        if not participants or participants.get("status") != "active":
            return
        encounter = records.get("active_encounter")
        if not encounter or encounter.get("status") != "active":
            raise SlotTransactionError(
                "active_participants requires an active encounter",
                code="participants_without_active_encounter",
                details={
                    "participant_slot": "active_participants",
                    "encounter_slot": "active_encounter",
                    "encounter_status": (
                        str(encounter.get("status")) if encounter else "absent"
                    ),
                    "repair": "close_or_archive_active_participants_before_closing_encounter",
                },
            )
        participant_value = participants.get("value")
        encounter_value = encounter.get("value")
        if not isinstance(participant_value, Mapping) or not isinstance(encounter_value, Mapping):
            raise SlotTransactionError(
                "encounter/participant values must be objects",
                code="encounter_participant_value_shape",
                details={"repair": "write_both_slots_with_the_required_object_shapes"},
            )
        event_ref = str(participant_value.get("event_ref", "") or "")
        event_id = str(encounter_value.get("event_id", "") or "")
        people = participant_value.get("participants")
        if not event_ref or event_ref != event_id or not isinstance(people, list) or not people:
            raise SlotTransactionError(
                "active_participants must bind participants to active_encounter.event_id",
                code="participant_encounter_binding_mismatch",
                details={
                    "repair": "update_or_close_active_participants_in_the_same_transaction"
                },
            )

    def _enforce_budget(self, *, segment_id: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while True:
            capsule = self.capsule()
            if capsule["within_budget"]:
                break
            eligible = [
                (name, record)
                for name, record in self.records.items()
                if record.get("status") == "closed"
            ]
            if not eligible:
                active_count = sum(
                    record.get("status") == "active" for record in self.records.values()
                )
                raise SlotTransactionError(
                    "active slot capsule uses "
                    f"{capsule['token_count']} tokens across {active_count} active slots and "
                    f"exceeds the frozen {self.token_budget}-token budget; keep segment-local "
                    "details in structured_event_record, write fewer working slots, and shorten "
                    "slot values",
                    code="active_capsule_budget_exceeded",
                    details={
                        "token_count": int(capsule["token_count"]),
                        "token_budget": self.token_budget,
                        "active_slot_count": active_count,
                    },
                )
            name, prior = min(
                eligible,
                key=lambda item: (
                    str(item[1].get("last_verified_segment_id", "")),
                    int(item[1].get("version", 0)),
                    item[0],
                ),
            )
            record = {**prior, "version": int(prior["version"]) + 1, "status": "evicted"}
            self.records[name] = record
            events.append(
                {
                    "event": "runtime_budget_evict",
                    "segment_id": segment_id,
                    "slot": name,
                    "operation": "evict",
                    "from_status": "closed",
                    "to_status": "evicted",
                    "from_version": int(prior["version"]),
                    "to_version": int(record["version"]),
                    "provenance": list(record["provenance"]),
                    "long_term_memory_preserved": True,
                }
            )
        return events

    def _enforce_lifecycle_policy(
        self,
        *,
        segment_id: str,
        touched: set[str],
    ) -> list[dict[str, Any]]:
        if self.lifecycle_policy != WP17_SLOT_LIFECYCLE_POLICY_V10:
            return []
        events = []
        for name, prior in sorted(self.records.items()):
            if prior.get("status") != "closed" or name in touched:
                continue
            closed_at = int(
                prior.get("closed_at_transaction_index", self.transaction_index - 1)
            )
            if (
                self.transaction_index - closed_at
                < self.closed_sweep_after_untouched_transactions
            ):
                continue
            record = {
                **prior,
                "version": int(prior["version"]) + 2,
                "status": "evicted",
            }
            self.records[name] = record
            events.append(
                {
                    "event": "runtime_lifecycle_sweep",
                    "segment_id": segment_id,
                    "slot": name,
                    "operation": "runtime_lifecycle_sweep",
                    "from_status": "closed",
                    "to_status": "evicted",
                    "from_version": int(prior["version"]),
                    "to_version": int(record["version"]),
                    "observation_ids": [],
                    "provenance": list(record["provenance"]),
                    "archive_then_evict": True,
                    "long_term_memory_preserved": True,
                }
            )
        return events


def parse_transaction_response(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    candidates = [text]
    if "```" in text:
        candidates.extend(
            part.strip().removeprefix("json").strip()
            for index, part in enumerate(text.split("```"))
            if index % 2 == 1
        )
    left = text.find("{")
    right = text.rfind("}")
    if 0 <= left < right:
        candidates.append(text[left : right + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return dict(payload)
    return None


def validate_construction_base(
    payload: Mapping[str, Any],
    *,
    allowed_evidence_ids: Sequence[str],
    evidence_id_map: Mapping[str, str] | None = None,
    enforce_output_size: bool = True,
) -> dict[str, Any]:
    """Validate observations and SER without accepting any slot transition."""
    _reject_forbidden_model_keys(payload)
    if enforce_output_size and len(_canonical_json(payload)) > WP17_MAX_OUTPUT_JSON_CHARS:
        raise SlotTransactionError(
            f"construction output exceeds {WP17_MAX_OUTPUT_JSON_CHARS} JSON characters",
            code="construction_output_too_large",
            details={"maximum_json_chars": WP17_MAX_OUTPUT_JSON_CHARS},
        )
    if payload.get("contract") != WP17_SLOT_TRANSACTION_CONTRACT:
        raise SlotTransactionError(
            "construction output contract mismatch",
            code="transaction_contract_mismatch",
            details={"expected_contract": WP17_SLOT_TRANSACTION_CONTRACT},
        )
    operations = tuple(payload.get("slot_operations", ()) or ())
    if not all(isinstance(row, Mapping) for row in operations):
        raise SlotTransactionError(
            "slot_operations must contain objects",
            code="slot_operations_not_objects",
        )
    observations = validate_observations(
        tuple(payload.get("observations", ()) or ()),
        allowed_evidence_ids=allowed_evidence_ids,
        evidence_id_map=evidence_id_map,
    )
    ser = validate_structured_event_record(
        dict(payload.get("structured_event_record", {}) or {})
    )
    return {
        "contract": WP17_SLOT_TRANSACTION_CONTRACT,
        "observations": [dict(row) for row in observations],
        "slot_operations": [deepcopy(dict(row)) for row in operations],
        "structured_event_record": ser,
    }


def validate_construction_output(
    payload: Mapping[str, Any],
    *,
    arm: str,
    segment_id: str,
    allowed_evidence_ids: Sequence[str],
    state: SlotMemoryState | None = None,
    evidence_id_map: Mapping[str, str] | None = None,
    enforce_output_size: bool = True,
) -> dict[str, Any]:
    """Validate one arm output and apply state only for the E1C2 treatment."""
    normalized_arm = str(arm).strip().casefold()
    base = validate_construction_base(
        payload,
        allowed_evidence_ids=allowed_evidence_ids,
        evidence_id_map=evidence_id_map,
        enforce_output_size=enforce_output_size,
    )
    if normalized_arm == "e1c2":
        if state is None:
            raise SlotTransactionError("E1C2 requires an arm-local slot state")
        return state.apply(
            payload,
            segment_id=str(segment_id),
            allowed_evidence_ids=allowed_evidence_ids,
            evidence_id_map=evidence_id_map,
        )
    operations = tuple(base["slot_operations"])
    if operations:
        raise SlotTransactionError("non-slot arms must return empty slot_operations")
    return {
        "contract": WP17_SLOT_TRANSACTION_CONTRACT,
        "segment_id": str(segment_id),
        "observations": base["observations"],
        "slot_operations": [],
        "structured_event_record": base["structured_event_record"],
    }


def _reject_forbidden_model_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().casefold()
            if key in _FORBIDDEN_MODEL_KEYS:
                raise SlotTransactionError(f"forbidden construction field: {key}")
            _reject_forbidden_model_keys(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _reject_forbidden_model_keys(child)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
