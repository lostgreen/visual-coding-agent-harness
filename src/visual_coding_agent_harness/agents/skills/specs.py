"""Declarative v4 skill specs and a tiny compiler to interpreter programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from string import Formatter
from typing import Any, FrozenSet, Mapping, Sequence

from ..question_policy import classify_question_route


@dataclass(frozen=True)
class SkillTrigger:
    route: str
    markers: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class SkillStep:
    step: str
    op: str
    args: Mapping[str, Any] = field(default_factory=dict)
    assign: str = ""
    foreach: str = ""
    on_fail: str = ""


@dataclass(frozen=True)
class SkillSpec:
    name: str
    version: int
    trigger: SkillTrigger
    input_slots: Sequence[str]
    procedure: Sequence[SkillStep]
    sufficiency: Sequence[str]
    verifier_checks: Sequence[str]
    recovery: Mapping[str, Any] = field(default_factory=dict)
    exemplars: Sequence[str] = field(default_factory=tuple)
    self_check: Sequence[str] = field(default_factory=tuple)
    allowed_actions: FrozenSet[str] = field(default_factory=frozenset)

    def prompt_context(self) -> str:
        lines = [f"Skill: {self.name}@v{self.version}", "Procedure:"]
        for step in self.procedure:
            foreach = f" foreach={step.foreach}" if step.foreach else ""
            assign = f" assign={step.assign}" if step.assign else ""
            lines.append(f"- {step.step}: {step.op}{foreach}{assign}")
        lines.append("Sufficiency:")
        lines.extend(f"- {item}" for item in self.sufficiency)
        lines.append("Verifier checks:")
        lines.extend(f"- {item}" for item in self.verifier_checks)
        if self.exemplars:
            lines.append("Exemplars:")
            lines.extend(f"- {item}" for item in self.exemplars)
        return "\n".join(lines)


class SkillRegistry:
    def __init__(self, skills: Sequence[SkillSpec] = ()) -> None:
        self._skills = {skill.name: skill for skill in skills}

    def register(self, skill: SkillSpec) -> None:
        if skill.name in self._skills:
            raise ValueError(f"Skill already registered: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillSpec:
        return self._skills[name]

    def list(self) -> Sequence[SkillSpec]:
        return tuple(self._skills.values())


def skill_catalog_prompt(
    *,
    registry: SkillRegistry | None = None,
    exhausted_tools: frozenset[str] | None = None,
) -> str:
    resolved_registry = registry or builtin_skill_registry()
    blocked = frozenset(exhausted_tools or ())
    lines = ["Available skills:"]
    for skill in resolved_registry.list():
        marker_text = ", ".join(skill.trigger.markers) if skill.trigger.markers else "(none)"
        remaining = sorted(action for action in skill.allowed_actions if action not in blocked)
        spent = sorted(action for action in skill.allowed_actions if action in blocked)
        suffix = f" ({', '.join(f'{tool}=exhausted' for tool in spent)})" if spent else ""
        lines.append(
            f"- {skill.name}@v{skill.version}: route={skill.trigger.route}; markers={marker_text}; "
            f"allowed_actions={', '.join(remaining) or '(none)'}{suffix}; "
            f"sufficiency={'; '.join(skill.sufficiency)}"
        )
    return "\n".join(lines)


def allowed_actions_for_skill(skill_id: str) -> frozenset[str]:
    """Return allowed tool actions for a built-in skill id or short skill name."""
    normalized = str(skill_id).strip()
    if not normalized:
        return frozenset()
    short_name = normalized.split("@", 1)[0]
    for skill in builtin_skill_registry().list():
        if skill.name == short_name or f"{skill.name}@v{skill.version}" == normalized:
            return frozenset(skill.allowed_actions)
    return frozenset()


def builtin_skill_registry() -> SkillRegistry:
    return SkillRegistry(
        [
            SkillSpec(
                name="main_idea",
                version=1,
                trigger=SkillTrigger(
                    route="gist_global",
                    markers=("main idea", "overall", "summary", "mainly about", "synopsis"),
                ),
                input_slots=("question", "options", "video_id", "duration_sec"),
                procedure=(
                    SkillStep(
                        step="global_seed_0",
                        op="global_gist",
                        args={
                            "video_path": "{video_id}",
                            "question": "{question}",
                            "duration_sec": "{duration_sec}",
                            "seed": 0,
                        },
                        assign="g1",
                    ),
                    SkillStep(
                        step="decide",
                        op="answer_agent",
                        args={"evidence": "evidence_table_v2()", "route": "gist_global"},
                        assign="decision",
                    ),
                ),
                sufficiency=("whole_video_coverage_evidence", "localized_or_indexed_fact_support"),
                verifier_checks=("selected_option_has_structured_support", "main_idea_coverage_floor_holds"),
                recovery={"ambiguous": {"action": "escalate", "skill": "grounded_factual_qa"}},
                exemplars=(
                    "Q: what is the video mainly about -> sparse global topic hint -> local/indexed coverage facts -> answer",
                ),
                self_check=("global_gist is not an option vote", "decision cites coverage evidence"),
                allowed_actions=frozenset(
                    {"global_gist", "query_context", "vision_read", "target_coverage", "read_segment_detail", "search_segments"}
                    | {"verify_ledger_answer"}
                ),
            ),
            SkillSpec(
                name="mutex_fact_qa",
                version=1,
                trigger=SkillTrigger(route="needle_local", markers=("option", "neither", "true")),
                input_slots=("question", "options", "video_id", "option_x_text", "option_y_text", "mutex_windows"),
                procedure=(
                    SkillStep(
                        step="locate_mutex_window",
                        op="ground_question",
                        args={"query": "{option_x_text} OR {option_y_text}"},
                        assign="mutex_windows",
                        on_fail="widen_query",
                    ),
                    SkillStep(
                        step="read_mutex_once",
                        op="vision_read",
                        foreach="mutex_windows",
                        args={
                            "window": "{candidate}",
                            "ask_for": (
                                "In this window, is option X (`{option_x_text}`) true, "
                                "OR option Y (`{option_y_text}`) true, OR NEITHER true? "
                                "Cite only visible frames. If no visible evidence supports either, return NEITHER."
                            ),
                        },
                        assign="mutex_fact[{candidate}]",
                    ),
                    SkillStep(
                        step="decide",
                        op="answer_agent",
                        args={"evidence": "evidence_table_v2()", "route": "needle_local"},
                        assign="decision",
                    ),
                ),
                sufficiency=("single_window_mutex_read_supports_one_option_or_neither",),
                verifier_checks=("selected_option_has_structured_support", "no_unaddressed_conflict"),
                recovery={"insufficient": {"action": "need_more_evidence", "target": "mutex distinguishing window"}},
                self_check=("one vision_read per mutex window",),
                allowed_actions=frozenset(
                    {
                        "ground_question",
                        "query_context",
                        "vision_read",
                        "target_coverage",
                        "read_segment_detail",
                        "locate_targets_in_segment",
                        "verify_segment_anchors",
                        "search_segments",
                    }
                    | {"verify_ledger_answer"}
                ),
            ),
            SkillSpec(
                name="grounded_factual_qa",
                version=1,
                trigger=SkillTrigger(route="needle_local", markers=("which", "what", "where", "who")),
                input_slots=("question", "options", "video_id", "target_fact"),
                procedure=(
                    SkillStep(
                        step="locate",
                        op="ground_question",
                        args={"query": "{target_fact}"},
                        assign="cand",
                        on_fail="widen_query",
                    ),
                    SkillStep(
                        step="read",
                        op="vision_read",
                        foreach="candidates",
                        args={"window": "{candidate}", "ask_for": "{target_fact}"},
                        assign="fact[{candidate}]",
                    ),
                    SkillStep(
                        step="decide",
                        op="answer_agent",
                        args={"evidence": "evidence_table_v2()", "route": "needle_local"},
                        assign="decision",
                    ),
                ),
                sufficiency=("distinguishing_fact_exists",),
                verifier_checks=(
                    "selected_option_has_structured_support",
                    "no_decisive_weak_grounding",
                    "no_unaddressed_conflict",
                ),
                recovery={"insufficient": {"action": "need_more_evidence", "target": "distinguishing fact window"}},
                self_check=("decision.citations all visually_confirmed",),
                allowed_actions=frozenset(
                    {
                        "ground_question",
                        "query_context",
                        "vision_read",
                        "target_coverage",
                        "read_segment_detail",
                        "locate_targets_in_segment",
                        "verify_segment_anchors",
                        "search_segments",
                    }
                    | {"verify_ledger_answer"}
                ),
            ),
            SkillSpec(
                name="timeline_ordering",
                version=1,
                trigger=SkillTrigger(
                    route="temporal_order",
                    markers=("before", "after", "first", "last", "then", "order", "sequence"),
                ),
                input_slots=("question", "options", "video_id", "events"),
                procedure=(
                    SkillStep(
                        step="caption_coarse_segments",
                        op="caption_segment",
                        foreach="segments",
                        args={
                            "question": (
                                "Openly describe the segment's actual visible/narrated events, artworks, objects, "
                                "people, scene changes, and onscreen text in presentation order."
                            )
                        },
                        assign="caption[{segment}]",
                    ),
                    SkillStep(
                        step="read_first_timestamp",
                        op="vision_read",
                        foreach="events",
                        args={
                            "window": "caption_match[{event}]",
                            "ask_for": "At what timestamp (precise, in seconds) does '{event}' first appear?",
                        },
                        assign="fact[{event}]",
                    ),
                    SkillStep(step="assemble", op="read_timeline_sorted", assign="timeline"),
                    SkillStep(
                        step="decide",
                        op="answer_agent",
                        args={"evidence": "evidence_table_v2()", "route": "temporal_order"},
                        assign="decision",
                    ),
                ),
                sufficiency=(
                    "every_event_has_confirmed_timestamp",
                    "observed_order_matches_one_option",
                    "single_scene_subwindow_vision_read_present",
                ),
                verifier_checks=(
                    "temporal_order_consistent",
                    "no_unconfirmed_event_in_selected_option",
                    "selected_option_has_structured_support",
                    "single_scene_constraint_satisfied_when_applicable",
                ),
                recovery={
                    "missing_event": {"action": "need_more_evidence", "target": "missing event window"},
                    "conflict": {"action": "need_more_evidence", "target": "conflicting event timestamps"},
                },
                exemplars=(
                    "caption coarse segments -> read first timestamps -> sort timeline -> compare to option sequences",
                ),
                self_check=("decision.option != null", "decision.citations all confirmed"),
                allowed_actions=frozenset(
                    {
                        "caption_segment",
                        "query_context",
                        "vision_read",
                        "read_timeline_sorted",
                        "target_coverage",
                        "read_segment_detail",
                        "locate_targets_in_segment",
                        "verify_segment_anchors",
                        "search_segments",
                    }
                    | {"verify_ledger_answer"}
                ),
            ),
        ]
    )


def select_skill(question: str, *, registry: SkillRegistry | None = None, route: str | None = None) -> SkillSpec:
    resolved_registry = registry or builtin_skill_registry()
    resolved_route = route or classify_question_route(question)
    lowered = question.lower()
    scored = []
    for skill in resolved_registry.list():
        route_score = 10 if skill.trigger.route == resolved_route else 0
        marker_score = sum(1 for marker in skill.trigger.markers if marker.lower() in lowered)
        if route_score or marker_score:
            scored.append((route_score + marker_score, skill.name, skill))
    if not scored:
        return resolved_registry.get("grounded_factual_qa")
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2]


def compile_skill_program(skill: SkillSpec, *, slots: Mapping[str, Any]) -> list[dict[str, Any]]:
    program = []
    for step in skill.procedure:
        compiled: dict[str, Any] = {
            "tool": step.op,
            "args": _format_value(step.args, slots),
        }
        if step.assign:
            compiled["assign"] = _format_value(step.assign, slots)
        if step.foreach:
            compiled["foreach"] = step.foreach
        if step.on_fail:
            compiled["on_fail"] = step.on_fail
        program.append(compiled)
    return program


def _format_value(value: Any, slots: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return _format_template(value, slots)
    if isinstance(value, Mapping):
        return {key: _format_value(child, slots) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_format_value(child, slots) for child in value]
    return value


def _format_template(template: str, slots: Mapping[str, Any]) -> Any:
    fields = [field_name for _, field_name, _, _ in Formatter().parse(template) if field_name]
    if not fields:
        return template
    if template == "{" + fields[0] + "}" and len(fields) == 1:
        return slots.get(fields[0], template)
    values = {name: slots.get(name, "{" + name + "}") for name in fields}
    return template.format(**values)
