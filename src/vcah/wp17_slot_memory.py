"""Validated slot-memory transactions for WP17 construction experiments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


WP17_SLOT_TRANSACTION_CONTRACT = "WP17-slot-memory-transaction-v1"
WP17_SLOT_STATE_CONTRACT = "WP17-slot-memory-state-v1"
WP17_SLOT_CAPSULE_CONTRACT = "WP17-slot-memory-capsule-v1"
WP17_BUDGET_TOKENIZER = "VCAH-unicode-budget-tokenizer-v1"
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
_WORKING_STATUSES = {"active", "closed"}
_TOKEN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", re.UNICODE)


class SlotTransactionError(ValueError):
    """Raised when a model-authored slot transaction violates the contract."""


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
    """Keep the most recent protocol tokens without exceeding ``max_tokens``."""
    limit = max(0, int(max_tokens))
    tokens = budget_tokens(text)
    if not tokens or limit == 0:
        return ""
    return " ".join(tokens[-limit:])


def validate_observations(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_evidence_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    allowed = {str(value) for value in allowed_evidence_ids}
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
        participants = tuple(
            dict.fromkeys(str(value).strip() for value in row.get("participants", ()) if str(value).strip())
        )
        normalized.append(
            {
                "observation_id": observation_id,
                "kind": kind,
                "fact": fact,
                "evidence_ids": list(evidence_ids),
                "participants": list(participants),
            }
        )
    return tuple(normalized)


def validate_structured_event_record(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    _reject_forbidden_model_keys(row)
    required_lists = ("entities", "events", "state_changes", "relations", "occurrence_refs")
    for key in required_lists:
        if not isinstance(row.get(key), list):
            raise SlotTransactionError(f"structured_event_record.{key} must be a list")
    summary = str(row.get("summary", "") or "").strip()
    if not summary:
        raise SlotTransactionError("structured_event_record.summary is required")
    return {key: deepcopy(row[key]) for key in required_lists} | {"summary": summary}


@dataclass
class SlotMemoryState:
    """Arm-local working slots plus an append-only long-term lifecycle ledger."""

    arm: str
    token_budget: int = 600
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    ledger: list[dict[str, Any]] = field(default_factory=list)
    transaction_index: int = 0

    def __post_init__(self) -> None:
        self.arm = str(self.arm)
        self.token_budget = int(self.token_budget)
        if self.token_budget <= 0:
            raise ValueError("slot token budget must be positive")
        unknown = set(self.records) - set(WP17_SLOT_NAMES)
        if unknown:
            raise ValueError(f"unknown slot records: {sorted(unknown)}")

    def apply(
        self,
        payload: Mapping[str, Any],
        *,
        segment_id: str,
        allowed_evidence_ids: Sequence[str],
    ) -> dict[str, Any]:
        if payload.get("contract") != WP17_SLOT_TRANSACTION_CONTRACT:
            raise SlotTransactionError("slot transaction contract mismatch")
        observations = validate_observations(
            tuple(payload.get("observations", ()) or ()),
            allowed_evidence_ids=allowed_evidence_ids,
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
        missing_lifecycle = working_before - touched
        if missing_lifecycle:
            raise SlotTransactionError(
                "working slots require an explicit lifecycle operation: "
                + ",".join(sorted(missing_lifecycle))
            )

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
        self._validate_cross_slot_invariants(candidate)

        prior_records = self.records
        prior_ledger = self.ledger
        prior_index = self.transaction_index
        self.records = candidate
        self.transaction_index += 1
        self.ledger = list(self.ledger) + transaction_events
        try:
            automatic_events = self._enforce_budget(segment_id=str(segment_id))
        except Exception:
            self.records = prior_records
            self.ledger = prior_ledger
            self.transaction_index = prior_index
            raise
        self.ledger.extend(automatic_events)
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
                "provenance": list(record["provenance"]),
                "last_verified_segment_id": str(record["last_verified_segment_id"]),
            }
            for name, record in sorted(self.records.items())
            if record.get("status") in _WORKING_STATUSES
        ]
        versions = {
            name: int(record["version"])
            for name, record in sorted(self.records.items())
        }
        context = "" if not slots and not versions else _canonical_json(
            {"slots": slots, "versions": versions}
        )
        payload = {
            "contract": WP17_SLOT_CAPSULE_CONTRACT,
            "arm": self.arm,
            "slots": slots,
            "versions": versions,
            "context": context,
        }
        payload["tokenizer"] = WP17_BUDGET_TOKENIZER
        payload["token_count"] = budget_token_count(context)
        payload["token_budget"] = self.token_budget
        payload["within_budget"] = payload["token_count"] <= self.token_budget
        return payload

    def snapshot(self) -> dict[str, Any]:
        return {
            "contract": WP17_SLOT_STATE_CONTRACT,
            "arm": self.arm,
            "token_budget": self.token_budget,
            "transaction_index": self.transaction_index,
            "records": deepcopy(self.records),
            "ledger": deepcopy(self.ledger),
            "digest": self.digest(),
        }

    @classmethod
    def from_snapshot(cls, payload: Mapping[str, Any]) -> "SlotMemoryState":
        if payload.get("contract") != WP17_SLOT_STATE_CONTRACT:
            raise ValueError("slot state contract mismatch")
        state = cls(
            arm=str(payload["arm"]),
            token_budget=int(payload["token_budget"]),
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
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def _normalize_operations(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        observations_by_id: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        normalized = []
        touched: set[str] = set()
        for raw in rows:
            row = dict(raw)
            operation = str(row.get("operation", "") or "").strip().casefold()
            slot = str(row.get("slot", "") or "").strip().casefold()
            if operation not in WP17_SLOT_OPERATIONS:
                raise SlotTransactionError(f"unsupported slot operation: {operation}")
            if slot not in WP17_SLOT_NAMES or slot in touched:
                raise SlotTransactionError("each known slot can be operated on at most once")
            touched.add(slot)
            observation_ids = tuple(
                dict.fromkeys(
                    str(value).strip()
                    for value in row.get("observation_ids", ())
                    if str(value).strip()
                )
            )
            if not set(observation_ids).issubset(observations_by_id):
                raise SlotTransactionError("slot operation references an unknown observation")
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
            raise SlotTransactionError(f"slot version mismatch for {name}")
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
                raise SlotTransactionError(f"cannot write active working slot {name}")
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
                raise SlotTransactionError(f"cannot update non-active slot {name}")
            record = {
                **prior,
                "version": prior_version + 1,
                "value": deepcopy(operation["value"]),
                "provenance": list(dict.fromkeys(tuple(prior["provenance"]) + current_evidence)),
                "last_verified_segment_id": segment_id,
            }
        elif action == "retain":
            if prior_status not in _WORKING_STATUSES:
                raise SlotTransactionError(f"cannot retain non-working slot {name}")
            if observation_ids:
                raise SlotTransactionError("retain cannot attach new observations")
            record = prior
        elif action == "close":
            if prior_status != "active":
                raise SlotTransactionError(f"cannot close non-active slot {name}")
            record = {
                **prior,
                "version": prior_version + 1,
                "status": "closed",
                "provenance": list(dict.fromkeys(tuple(prior["provenance"]) + current_evidence)),
                "last_verified_segment_id": segment_id,
            }
        elif action == "archive":
            if prior_status != "closed" or observation_ids:
                raise SlotTransactionError(f"archive requires a closed unchanged slot: {name}")
            record = {**prior, "version": prior_version + 1, "status": "archived"}
        elif action == "evict":
            if prior_status not in {"closed", "archived"} or observation_ids:
                raise SlotTransactionError(f"evict requires a closed/archived unchanged slot: {name}")
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

    def _validate_cross_slot_invariants(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        participants = records.get("active_participants")
        if not participants or participants.get("status") != "active":
            return
        encounter = records.get("active_encounter")
        if not encounter or encounter.get("status") != "active":
            raise SlotTransactionError("active_participants requires an active encounter")
        participant_value = participants.get("value")
        encounter_value = encounter.get("value")
        if not isinstance(participant_value, Mapping) or not isinstance(encounter_value, Mapping):
            raise SlotTransactionError("encounter/participant values must be objects")
        event_ref = str(participant_value.get("event_ref", "") or "")
        event_id = str(encounter_value.get("event_id", "") or "")
        people = participant_value.get("participants")
        if not event_ref or event_ref != event_id or not isinstance(people, list) or not people:
            raise SlotTransactionError("active_participants must bind participants to active_encounter.event_id")

    def _enforce_budget(self, *, segment_id: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while not self.capsule()["within_budget"]:
            eligible = [
                (name, record)
                for name, record in self.records.items()
                if record.get("status") == "closed"
            ]
            if not eligible:
                raise SlotTransactionError("active slot capsule exceeds the frozen token budget")
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


def validate_construction_output(
    payload: Mapping[str, Any],
    *,
    arm: str,
    segment_id: str,
    allowed_evidence_ids: Sequence[str],
    state: SlotMemoryState | None = None,
) -> dict[str, Any]:
    """Validate one arm output and apply state only for the E1C2 treatment."""
    normalized_arm = str(arm).strip().casefold()
    _reject_forbidden_model_keys(payload)
    if payload.get("contract") != WP17_SLOT_TRANSACTION_CONTRACT:
        raise SlotTransactionError("construction output contract mismatch")
    if normalized_arm == "e1c2":
        if state is None:
            raise SlotTransactionError("E1C2 requires an arm-local slot state")
        return state.apply(
            payload,
            segment_id=str(segment_id),
            allowed_evidence_ids=allowed_evidence_ids,
        )
    operations = tuple(payload.get("slot_operations", ()) or ())
    if operations:
        raise SlotTransactionError("non-slot arms must return empty slot_operations")
    observations = validate_observations(
        tuple(payload.get("observations", ()) or ()),
        allowed_evidence_ids=allowed_evidence_ids,
    )
    ser = validate_structured_event_record(
        dict(payload.get("structured_event_record", {}) or {})
    )
    return {
        "contract": WP17_SLOT_TRANSACTION_CONTRACT,
        "segment_id": str(segment_id),
        "observations": [dict(row) for row in observations],
        "slot_operations": [],
        "structured_event_record": ser,
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
