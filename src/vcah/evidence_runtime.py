from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Sequence

from vcah.evidence_state import (
    DEPENDENCY_TYPES,
    EVIDENCE_KINDS,
    EVIDENCE_ROLES,
    EvidenceObligation,
    EvidenceObligationState,
)
from vcah.temporal_scope import TEMPORAL_RELATIONS, TEMPORAL_SELECTIONS, TemporalScope
from vcah.workspace import ObservationLog, WorkingDocument


@dataclass(frozen=True)
class EvidenceRequirementSpec:
    name: str
    goal: str
    kind: str = "generic"
    role: str = "answer_bearing"
    depends_on: tuple[str, ...] = ()
    dependency_type: str = "locator"
    temporal_relation: str = ""
    temporal_selection: str = "unspecified"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        index: int,
    ) -> "EvidenceRequirementSpec":
        name = str(value.get("name", value.get("id", f"requirement_{index}")) or "").strip()
        goal = str(value.get("goal", value.get("observable_goal", "")) or "").strip()
        kind = str(value.get("kind", value.get("evidence_kind", "generic")) or "generic").casefold()
        role = str(value.get("role", "answer_bearing") or "answer_bearing").casefold()
        dependency_type = str(value.get("dependency_type", "locator") or "locator").casefold()
        if kind not in EVIDENCE_KINDS:
            kind = "generic"
        if role not in EVIDENCE_ROLES:
            role = "answer_bearing"
        if dependency_type not in DEPENDENCY_TYPES:
            dependency_type = "locator"
        temporal_relation = str(
            value.get("temporal_relation", value.get("relation", "")) or ""
        ).strip().casefold()
        temporal_relation = {
            "during": "within",
            "while": "within",
            "immediately_after": "after",
            "immediately after": "after",
            "prior": "before",
        }.get(temporal_relation, temporal_relation)
        if temporal_relation not in TEMPORAL_RELATIONS:
            temporal_relation = ""
        temporal_selection = str(
            value.get("temporal_selection", value.get("selection", "unspecified"))
            or "unspecified"
        ).strip().casefold()
        if temporal_selection not in TEMPORAL_SELECTIONS:
            temporal_selection = "unspecified"
        raw_dependencies = value.get("depends_on", ()) or ()
        if isinstance(raw_dependencies, str):
            raw_dependencies = (raw_dependencies,)
        depends_on = tuple(
            dict.fromkeys(
                str(item.get("requirement", item.get("name", "")) if isinstance(item, Mapping) else item).strip()
                for item in raw_dependencies
                if str(item.get("requirement", item.get("name", "")) if isinstance(item, Mapping) else item).strip()
            )
        )
        return cls(
            name=name or f"requirement_{index}",
            goal=goal,
            kind=kind,
            role=role,
            depends_on=depends_on,
            dependency_type=dependency_type,
            temporal_relation=temporal_relation,
            temporal_selection=temporal_selection,
        )


@dataclass(frozen=True)
class EvidencePlan:
    requirements: tuple[EvidenceRequirementSpec, ...]
    source: str = "reasoner"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        question: str,
    ) -> "EvidencePlan":
        raw = value.get("requirements", ()) if isinstance(value, Mapping) else ()
        specs = tuple(
            EvidenceRequirementSpec.from_mapping(item, index=index)
            for index, item in enumerate(raw, start=1)
            if isinstance(item, Mapping)
        )[:6]
        specs = tuple(spec for spec in specs if spec.goal)
        if not specs:
            return cls.fallback(question)
        if not any(spec.role in {"answer_bearing", "disambiguation"} for spec in specs):
            specs = (
                *specs[:5],
                EvidenceRequirementSpec(
                    name="answer",
                    goal=str(question or "Answer the question from direct observation.").strip(),
                    role="answer_bearing",
                ),
            )
        return cls(requirements=_unique_requirement_names(specs), source="reasoner")

    @classmethod
    def fallback(cls, question: str) -> "EvidencePlan":
        return cls(
            requirements=(
                EvidenceRequirementSpec(
                    name="answer",
                    goal=str(question or "Answer the question from direct observation.").strip(),
                    role="answer_bearing",
                ),
            ),
            source="runtime_fallback",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "requirements": [
                {
                    "name": spec.name,
                    "goal": spec.goal,
                    "kind": spec.kind,
                    "role": spec.role,
                    "depends_on": list(spec.depends_on),
                    "dependency_type": spec.dependency_type,
                    "temporal_relation": spec.temporal_relation,
                    "temporal_selection": spec.temporal_selection,
                }
                for spec in self.requirements
            ],
        }


def compile_evidence_plan(
    document: WorkingDocument,
    plan: EvidencePlan,
    *,
    question: str,
) -> dict[str, Any]:
    if document.obligations:
        return {
            "source": plan.source,
            "compiled": False,
            "reason": "requirements_already_compiled",
            "requirement_count": len(document.obligations),
        }
    canonical_by_name = {
        spec.name: _canonical_id("requirement", question, index, spec.name, spec.goal)
        for index, spec in enumerate(plan.requirements, start=1)
    }
    unknown_dependencies: list[str] = []
    for spec in plan.requirements:
        requirement_id = canonical_by_name[spec.name]
        dependencies = []
        for dependency in spec.depends_on:
            canonical = canonical_by_name.get(dependency)
            if canonical:
                dependencies.append(canonical)
            else:
                unknown_dependencies.append(f"{spec.name}:{dependency}")
        obligation = EvidenceObligation(
            requirement_id=requirement_id,
            observable_goal=spec.goal,
            evidence_kind=spec.kind,
            temporal_relation=spec.temporal_relation or None,
            depends_on=tuple(dependencies),
            answer_bearing=spec.role in {"answer_bearing", "disambiguation"},
            role=spec.role,
            dependency_type=spec.dependency_type,
        )
        document.obligations[requirement_id] = obligation
        document.obligation_states[requirement_id] = EvidenceObligationState(
            requirement_id=requirement_id
        )
        if spec.dependency_type == "temporal" and dependencies and spec.temporal_relation:
            scope_id = _canonical_id(
                "scope",
                dependencies[0],
                requirement_id,
                spec.temporal_relation,
                spec.temporal_selection,
            )
            document.temporal_scopes[scope_id] = TemporalScope(
                scope_id=scope_id,
                relation=spec.temporal_relation,
                selection=spec.temporal_selection,
                anchor_requirement_id=dependencies[0],
                target_requirement_id=requirement_id,
            )
    document.revision += 1
    return {
        "source": plan.source,
        "compiled": True,
        "requirement_count": len(document.obligations),
        "unknown_dependencies": unknown_dependencies,
        "requirements": [
            {
                "handle": f"R{index}",
                "requirement_id": requirement_id,
                "name": spec.name,
                "role": spec.role,
                "kind": spec.kind,
            }
            for index, (spec, requirement_id) in enumerate(
                zip(plan.requirements, canonical_by_name.values()),
                start=1,
            )
        ],
    }


@dataclass(frozen=True)
class EvidenceItemRef:
    handle: str
    material_handle: str
    attempt_id: str
    interpretation_id: str
    item_id: str
    time_anchor: tuple[float, float]
    item_kind: str
    text: str
    cue_id: str = ""
    refinable: bool = False
    advertise_refinement: bool = False


@dataclass(frozen=True)
class RuntimeEvidenceCatalog:
    requirements: tuple[tuple[str, str], ...]
    materials: tuple[tuple[str, str], ...]
    items: tuple[EvidenceItemRef, ...]
    occurrences: tuple[tuple[str, Mapping[str, Any]], ...]
    scopes: tuple[tuple[str, str], ...]

    @classmethod
    def build(
        cls,
        document: WorkingDocument,
        observations: ObservationLog,
    ) -> "RuntimeEvidenceCatalog":
        requirements = tuple(
            (f"R{index}", requirement_id)
            for index, requirement_id in enumerate(document.obligations, start=1)
        )
        attempt_ids = tuple(
            attempt_id
            for attempt_id in dict.fromkeys(
                str(row.get("attempt_id", "") or "") for row in observations.rows
            )
            if attempt_id
        )
        materials = tuple(
            (f"M{index}", attempt_id)
            for index, attempt_id in enumerate(attempt_ids, start=1)
        )
        material_handles = {attempt_id: handle for handle, attempt_id in materials}
        cues_by_item = {
            str(cue.get("item_id", "") or ""): dict(cue)
            for row in observations.rows
            for cue in tuple(row.get("observation_cues", ()) or ())
            if isinstance(cue, Mapping) and str(cue.get("item_id", "") or "")
        }
        item_rows: list[EvidenceItemRef] = []
        refinable_per_attempt: dict[str, int] = {}
        for row in observations.rows:
            attempt_id = str(row.get("attempt_id", "") or "")
            interpretation_id = str(row.get("interpretation_id", "") or "")
            sampling_config = row.get("sampling_config", {})
            evidence_kind = str(
                sampling_config.get("evidence_kind", "")
                if isinstance(sampling_config, Mapping)
                else ""
            ).casefold()
            for raw_item in tuple(row.get("interpretation_items", ()) or ()):
                if not isinstance(raw_item, Mapping):
                    continue
                item_id = str(raw_item.get("item_id", "") or "")
                raw_anchor = tuple(raw_item.get("time_anchor", ()) or ())
                if not item_id or len(raw_anchor) != 2:
                    continue
                anchor = tuple(sorted((float(raw_anchor[0]), float(raw_anchor[1]))))
                cue = cues_by_item.get(item_id, {})
                point_item = abs(anchor[0] - anchor[1]) <= 1e-6
                item_kind = str(
                    raw_item.get("item_kind", "observation") or "observation"
                ).casefold()
                refinable = bool(point_item and cue)
                advertise_refinement = bool(
                    refinable
                    and (
                        evidence_kind in {"transient_event", "ui_text", "text_exact"}
                        or item_kind in {"event", "ui_label", "ui_description", "text"}
                    )
                    and refinable_per_attempt.get(attempt_id, 0) < 6
                )
                if advertise_refinement:
                    refinable_per_attempt[attempt_id] = refinable_per_attempt.get(attempt_id, 0) + 1
                item_rows.append(
                    EvidenceItemRef(
                        handle=f"E{len(item_rows) + 1}",
                        material_handle=material_handles.get(attempt_id, ""),
                        attempt_id=attempt_id,
                        interpretation_id=interpretation_id,
                        item_id=item_id,
                        time_anchor=anchor,
                        item_kind=item_kind,
                        text=str(raw_item.get("text", "") or "").strip(),
                        cue_id=str(cue.get("cue_id", "") or ""),
                        refinable=refinable,
                        advertise_refinement=advertise_refinement,
                    )
                )
        occurrence_values: list[Mapping[str, Any]] = []
        seen_occurrences: set[str] = set()
        for row in observations.rows:
            config = row.get("sampling_config", {})
            occurrence_set = config.get("occurrence_set") if isinstance(config, Mapping) else None
            if not isinstance(occurrence_set, Mapping):
                continue
            for candidate in tuple(occurrence_set.get("candidates", ()) or ()):
                if not isinstance(candidate, Mapping):
                    continue
                occurrence_id = str(candidate.get("occurrence_id", "") or "")
                if not occurrence_id or occurrence_id in seen_occurrences:
                    continue
                seen_occurrences.add(occurrence_id)
                occurrence_values.append(
                    {
                        **dict(candidate),
                        "attempt_id": str(row.get("attempt_id", "") or ""),
                    }
                )
        occurrences = tuple(
            (f"O{index}", candidate)
            for index, candidate in enumerate(occurrence_values, start=1)
        )
        scopes = tuple(
            (f"S{index}", scope_id)
            for index, scope_id in enumerate(document.temporal_scopes, start=1)
        )
        return cls(requirements, materials, tuple(item_rows), occurrences, scopes)

    def resolve_requirement(self, value: str) -> str:
        return _resolve_handle(value, self.requirements)

    def resolve_scope(self, value: str) -> str:
        return _resolve_handle(value, self.scopes)

    def resolve_occurrence(self, value: str) -> Mapping[str, Any] | None:
        normalized = str(value or "").strip()
        for handle, candidate in self.occurrences:
            if normalized in {handle, str(candidate.get("occurrence_id", "") or "")}:
                return candidate
        return None

    def resolve_item(self, value: str) -> EvidenceItemRef | None:
        normalized = str(value or "").strip()
        return next(
            (
                item
                for item in self.items
                if normalized in {item.handle, item.item_id}
            ),
            None,
        )

    def item_by_id(self, item_id: str) -> EvidenceItemRef | None:
        return self.resolve_item(item_id)

    def single_open_answer_requirement(self, document: WorkingDocument) -> str:
        values = tuple(
            requirement_id
            for _, requirement_id in self.requirements
            if document.obligations[requirement_id].answer_bearing
            and document.obligation_states[requirement_id].status
            not in {"supported", "satisfied", "unresolved"}
        )
        return values[0] if len(values) == 1 else ""

    def render(
        self,
        document: WorkingDocument,
        *,
        feedback: Mapping[str, Any] | None = None,
        max_items: int = 36,
        max_occurrences: int = 12,
    ) -> str:
        lines = ["REQUIREMENTS"]
        for handle, requirement_id in self.requirements:
            obligation = document.obligations[requirement_id]
            state = document.obligation_states[requirement_id]
            dependency_handles = [
                candidate_handle
                for candidate_handle, candidate_id in self.requirements
                if candidate_id in obligation.depends_on
            ]
            dependency = (
                f" depends={obligation.dependency_type}:{','.join(dependency_handles)}"
                if dependency_handles
                else ""
            )
            lines.append(
                f"{handle} [{obligation.role},{obligation.evidence_kind}] {state.status.upper()}{dependency} "
                f"{obligation.observable_goal}"
            )
        lines.append("CANDIDATES")
        if not self.occurrences:
            lines.append("none")
        for handle, candidate in self.occurrences[:max_occurrences]:
            lines.append(
                f"{handle} {list(candidate.get('time_range', ()) or ())} "
                f"hits={int(candidate.get('hit_count', 0) or 0)}"
            )
        lines.append("EVIDENCE")
        if not self.items:
            lines.append("none")
        for item in self.items[-max(1, int(max_items)):]:
            anchor = f"{item.time_anchor[0]:.3f}" if item.time_anchor[0] == item.time_anchor[1] else f"{item.time_anchor[0]:.3f}-{item.time_anchor[1]:.3f}"
            refinable = " refinable" if item.advertise_refinement else ""
            lines.append(
                f"{item.handle} [{item.material_handle} {anchor} {item.item_kind}{refinable}] "
                f"{_compact(item.text, 220)}"
            )
        if feedback:
            lines.extend(
                (
                    "FEEDBACK",
                    json.dumps(
                        _compact_feedback(feedback),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )
        return "\n".join(lines)


def advance_requirement_state(
    document: WorkingDocument,
    requirement_id: str,
    status: str,
    *,
    attempt_ids: Sequence[str] = (),
    item_ids: Sequence[str] = (),
    residual_uncertainty: str = "",
) -> None:
    if requirement_id not in document.obligations:
        return
    current = document.obligation_states.get(
        requirement_id,
        EvidenceObligationState(requirement_id=requirement_id),
    )
    order = {
        "open": 0,
        "candidate_found": 1,
        "observed": 2,
        "contested": 2,
        "supported": 3,
        "satisfied": 3,
        "unresolved": 3,
    }
    normalized = str(status or "open").casefold()
    if normalized != "unresolved" and order.get(normalized, 0) < order.get(current.status, 0):
        normalized = current.status
    document.obligation_states[requirement_id] = replace(
        current,
        status=normalized,
        supporting_attempt_ids=tuple(
            dict.fromkeys((*current.supporting_attempt_ids, *(str(item) for item in attempt_ids if str(item))))
        ),
        supporting_item_ids=tuple(
            dict.fromkeys((*current.supporting_item_ids, *(str(item) for item in item_ids if str(item))))
        ),
        residual_uncertainty=str(residual_uncertainty or current.residual_uncertainty),
    )


def _resolve_handle(value: str, rows: Sequence[tuple[str, str]]) -> str:
    normalized = str(value or "").strip()
    return next(
        (canonical for handle, canonical in rows if normalized in {handle, canonical}),
        "",
    )


def _unique_requirement_names(
    specs: Sequence[EvidenceRequirementSpec],
) -> tuple[EvidenceRequirementSpec, ...]:
    counts: dict[str, int] = {}
    normalized = []
    for spec in specs:
        counts[spec.name] = counts.get(spec.name, 0) + 1
        suffix = counts[spec.name]
        normalized.append(
            spec if suffix == 1 else replace(spec, name=f"{spec.name}_{suffix}")
        )
    return tuple(normalized)


def _canonical_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _compact(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _compact_feedback(value: Mapping[str, Any]) -> dict[str, Any]:
    compact = {
        key: value.get(key)
        for key in (
            "type",
            "action",
            "reason",
            "requested_tasks",
            "requested_task_count",
            "completed_tasks",
            "new_observation_interpretations",
            "returned_observation_count",
        )
        if value.get(key) is not None
    }
    errors = tuple(value.get("errors", ()) or ())
    if errors:
        compact["error_codes"] = [
            str(error.get("code", "error") if isinstance(error, Mapping) else str(error).split(":", 1)[0])
            for error in errors[:6]
        ]
    outcomes = tuple(value.get("outcomes", ()) or ())
    if outcomes:
        compact["outcomes"] = [
            {
                "query_id": str(outcome.get("query_id", "") or ""),
                "status": str(outcome.get("status", "") or ""),
                "reused": bool(outcome.get("reused", False)),
            }
            for outcome in outcomes[:8]
            if isinstance(outcome, Mapping)
        ]
    return compact
