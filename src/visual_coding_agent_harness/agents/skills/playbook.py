"""Operator-keyed investigation playbooks for planner guidance."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from ...contracts import TargetRegistry


@dataclass(frozen=True)
class StopDiagnostic:
    reason_code: str
    unmet_shape: str
    repair_hint: str


@dataclass(frozen=True)
class Playbook:
    skill_name: str
    answer_operator: str
    decomposition: str
    evidence_shape_target: tuple[str, ...]
    investigation_hints: tuple[str, ...] = ()
    unsafe_final_conditions: tuple[str, ...] = ()
    stop_diagnostics: tuple[StopDiagnostic, ...] = ()
    suggested_actions: tuple[str, ...] = ()
    forbidden_actions: tuple[str, ...] = ()


def render_playbook_block(
    playbook: Playbook,
    *,
    registry: TargetRegistry | None = None,
    option_labels: Sequence[str] = (),
    central_subjects: Sequence[str] = (),
    max_chars: int = 4000,
) -> str:
    """Render a compact SKILL.md-like prompt block for the active investigation playbook."""
    lines = [
        f"## Skill Playbook: {playbook.skill_name}",
        f"Operator: {playbook.answer_operator}",
    ]
    if central_subjects:
        lines.append("Central subjects: " + ", ".join(_bounded_items(central_subjects, limit=4)))
    if playbook.suggested_actions:
        lines.append("Suggested actions: " + ", ".join(_bounded_items(playbook.suggested_actions, limit=12)))
    if playbook.forbidden_actions:
        lines.append("Forbidden actions: " + ", ".join(_bounded_items(playbook.forbidden_actions, limit=8)))
    target_lines = _registry_target_lines(registry)
    option_lines = _option_mapping_lines(registry=registry, option_labels=option_labels)
    if target_lines:
        lines.extend(["", "### Registered Targets", *target_lines])
    if option_lines:
        lines.extend(["", "### Option Target Map", *option_lines])
    lines.extend(["", "### Decomposition", _clip_text(playbook.decomposition, 1200)])
    if playbook.evidence_shape_target:
        lines.extend(["", "### Evidence Shape Required To Stop"])
        lines.extend(f"- {item}" for item in _bounded_items(playbook.evidence_shape_target, limit=8))
    if playbook.investigation_hints:
        lines.extend(["", "### Investigation Hints"])
        lines.extend(f"- {item}" for item in _bounded_items(playbook.investigation_hints, limit=5))
    if playbook.unsafe_final_conditions:
        lines.extend(["", "### Unsafe Final Conditions"])
        lines.extend(f"- {item}" for item in _bounded_items(playbook.unsafe_final_conditions, limit=5))
    if playbook.stop_diagnostics:
        lines.extend(["", "### Stop Diagnostics"])
        for diagnostic in playbook.stop_diagnostics[:6]:
            lines.append(f"- {diagnostic.reason_code} -> {diagnostic.repair_hint}")
    rendered = "\n".join(lines).strip()
    return _clip_text(rendered, max_chars)


def playbook_for_operator(
    operator: str,
    *,
    question: str,
    options: Sequence[str],
    registry: TargetRegistry | None,
    route: str | None = None,
) -> Playbook | None:
    del question, options, registry, route
    builders = {
        "select_present": _select_present_playbook,
        "select_absent": _select_absent_playbook,
        "causal_bind": _causal_bind_playbook,
        "universal_intersection": _universal_intersection_playbook,
        "ordered_projection": _ordered_projection_playbook,
        "main_arc": _main_arc_playbook,
    }
    builder = builders.get(str(operator or "").strip())
    return builder() if builder is not None else None


def with_suggested_actions(playbook: Playbook | None, actions: Sequence[str]) -> Playbook | None:
    if playbook is None:
        return None
    return replace(playbook, suggested_actions=tuple(_bounded_items(actions, limit=32)))


def _select_present_playbook() -> Playbook:
    return Playbook(
        skill_name="grounded_factual_qa",
        answer_operator="select_present",
        decomposition=(
            "For each option, create one affirmable target. The selected answer is the option with supported "
            "answer-grade evidence."
        ),
        evidence_shape_target=(
            "candidate option has at least one supported binding",
            "evidence is answer-grade, not navigation-only",
            "if two options share atoms, use discriminative aliases",
        ),
        investigation_hints=(
            "Localize likely evidence before asking a visual or transcript tool to bind the target.",
            "Prefer direct support for the option claim over broad topical overlap.",
        ),
        unsafe_final_conditions=(
            "only navigation/search evidence exists",
            "top candidates share unresolved atoms",
        ),
        stop_diagnostics=(
            StopDiagnostic("candidate_binding_missing", "candidate lacks supported binding", "probe candidate target"),
            StopDiagnostic("top2_ambiguous", "top candidates remain tied", "probe discriminative atom"),
            StopDiagnostic("navigation_only", "support is navigation-only", "localize then read or bind answer-grade evidence"),
        ),
    )


def _select_absent_playbook() -> Playbook:
    return Playbook(
        skill_name="complement_absence_qa",
        answer_operator="select_absent",
        decomposition=(
            "Create one affirmable target per option. The answer is the one option whose target remains unsupported, "
            "after all competitors are confirmed present."
        ),
        evidence_shape_target=(
            "all competitors present",
            "candidate absent after real probe",
            "exactly one absent candidate",
        ),
        investigation_hints=(
            "Confirm competitors first; do not infer absence from silence.",
            "Weak search overlap is not presence evidence.",
            "If the candidate itself receives positive support, it cannot be the absent answer.",
        ),
        unsafe_final_conditions=(
            "only the selected candidate was probed",
            "multiple options are still unconfirmed",
            "support is navigation-only",
        ),
        stop_diagnostics=(
            StopDiagnostic("competitor_presence_missing", "competitor option lacks positive support", "probe missing competitor"),
            StopDiagnostic("multiple_absent_candidates", "more than one option remains unsupported", "probe the unconfirmed options"),
            StopDiagnostic("candidate_has_positive_support", "absent candidate has positive support", "reject absent candidate"),
        ),
    )


def _causal_bind_playbook() -> Playbook:
    return Playbook(
        skill_name="causal_asr_qa",
        answer_operator="causal_bind",
        decomposition=(
            "Identify the asked effect or relationship. Create one causal target per option. Use transcript or ASR "
            "binding to decide which cause is narrated as the reason."
        ),
        evidence_shape_target=(
            "selected option has binding-sourced support",
            "topic overlap alone is insufficient",
            "if two causes bind, use primacy or tie-breaking cues",
        ),
        investigation_hints=(
            "Reason and why questions need a cause binding, not just co-occurrence.",
            "Negative wording inside a reason question is still causal unless the answer asks for an absent item.",
            "Use ASR or transcript binding when narrator wording carries the reason.",
        ),
        unsafe_final_conditions=(
            "only topical overlap supports the cause",
            "cause target is not bound to an observation",
        ),
        stop_diagnostics=(
            StopDiagnostic("causal_binding_missing", "cause lacks binding evidence", "run bind_asr_claim for option cause targets"),
            StopDiagnostic("topic_overlap_only", "support is topical overlap only", "bind transcript cue instead of using overlap"),
            StopDiagnostic("two_causes_supported", "multiple causes bind", "inspect narrator primacy markers"),
        ),
        suggested_actions=("search_segments", "read_segment_detail", "bind_asr_claim", "verify_ledger_answer"),
    )


def _universal_intersection_playbook() -> Playbook:
    return Playbook(
        skill_name="universal_set_qa",
        answer_operator="universal_intersection",
        decomposition=(
            "Identify evidence groups from segment or topic structure. Probe each group. Candidate answer must appear "
            "in every relevant group."
        ),
        evidence_shape_target=(
            "candidate supported in all visited groups",
            "visited groups cover the relevant case or scene set",
            "single group support is not complete",
        ),
        investigation_hints=(
            "Track group coverage explicitly before finalizing.",
            "A strong single-window fact is insufficient for all/every questions.",
        ),
        unsafe_final_conditions=(
            "only one group was checked",
            "some relevant groups are unvisited",
        ),
        stop_diagnostics=(
            StopDiagnostic("group_unvisited", "a relevant group is unvisited", "probe highest-priority unvisited group"),
            StopDiagnostic("coverage_incomplete", "selected candidate lacks full group coverage", "inspect missing group"),
            StopDiagnostic("single_group_only", "only one group supports the answer", "collect another group"),
        ),
    )


def _ordered_projection_playbook() -> Playbook:
    return Playbook(
        skill_name="timeline_ordering",
        answer_operator="ordered_projection",
        decomposition=(
            "Split each option into ordered tokens. Register ordered item targets. Locate earliest mention or "
            "observation for each item independently. Project observed order to the unique matching option."
        ),
        evidence_shape_target=(
            "each item has unique earliest position",
            "observed order matches exactly one option",
            "if partial order only, continue probing missing items",
        ),
        investigation_hints=(
            "Locate each ordered item independently.",
            "Use narrower timestamps when two positions are ambiguous.",
        ),
        unsafe_final_conditions=(
            "one contiguous span is the only evidence",
            "ordered item positions are missing or ambiguous",
        ),
        stop_diagnostics=(
            StopDiagnostic("ordered_item_missing", "ordered item lacks evidence", "probe missing item"),
            StopDiagnostic("order_position_ambiguous", "item order position is ambiguous", "inspect narrower timestamp/window"),
            StopDiagnostic("multiple_order_options_match", "multiple options match partial order", "collect another item position"),
        ),
    )


def _main_arc_playbook() -> Playbook:
    return Playbook(
        skill_name="main_idea",
        answer_operator="main_arc",
        decomposition=(
            "Use global_gist only as a seed. Create per-option theme targets and central-subject aliases. Compare "
            "broad distinct-segment coverage between top candidates. Select the option that dominates the whole-video arc."
        ),
        evidence_shape_target=(
            "selected option has whole-video coverage",
            "selected option has distinct-segment dominance over runner-up",
            "global_gist alone is insufficient",
            "local subtopic cannot beat broader main arc",
        ),
        investigation_hints=(
            "global_gist is not an option vote",
            "compare top-2 option coverage before final",
            "broad coverage beats local subtopic salience",
        ),
        unsafe_final_conditions=(
            "global_gist-only answer",
            "runner-up breadth is not checked",
            "selected arc is only a local subtopic",
        ),
        stop_diagnostics=(
            StopDiagnostic("global_hint_only", "only a global hint exists", "run target_coverage grouped by option"),
            StopDiagnostic("insufficient_breadth", "top candidates lack breadth evidence", "probe more segments for top candidates"),
            StopDiagnostic("arc_not_dominant", "selected arc does not dominate runner-up", "compare top-2 option coverage"),
            StopDiagnostic("no_option_projection", "coverage cannot project to an option", "recompile option theme targets"),
        ),
    )


def _registry_target_lines(registry: TargetRegistry | None) -> list[str]:
    targets_by_id = getattr(registry, "targets_by_id", None)
    if not isinstance(targets_by_id, Mapping):
        return []
    lines = []
    for target_id in sorted(str(key) for key in targets_by_id):
        target = targets_by_id.get(target_id)
        text = str(getattr(target, "canonical_text", "")).strip()
        if text:
            lines.append(f"- {target_id}: {text}")
    return lines[:12]


def _option_mapping_lines(*, registry: TargetRegistry | None, option_labels: Sequence[str]) -> list[str]:
    options_by_id = getattr(registry, "options_by_id", None)
    if isinstance(options_by_id, Mapping) and options_by_id:
        lines = []
        for option_id in sorted(str(key) for key in options_by_id):
            option = options_by_id.get(option_id)
            targets = list(getattr(option, "target_sequence", ()) or ())
            raw = str(getattr(option, "raw_option_text", "") or "").strip()
            target_text = ", ".join(str(item) for item in targets) or "(none)"
            suffix = f" ({raw})" if raw else ""
            lines.append(f"- {option_id} -> {target_text}{suffix}")
        return lines[:8]
    lines = []
    for option in option_labels[:8]:
        option_text = str(option).strip()
        if option_text:
            lines.append(f"- {option_text}")
    return lines


def _bounded_items(values: Sequence[Any], *, limit: int) -> list[str]:
    result = []
    for value in values:
        text = str(value).strip()
        if text:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."
