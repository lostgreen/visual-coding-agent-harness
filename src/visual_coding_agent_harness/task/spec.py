"""Generic task contracts for option-level long-video QA."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from ..answer_operators import AnswerOperator, derive_answer_operator, normalize_answer_operator
from ..contracts import TargetRegistry

AnswerFormat = Literal["mcq", "short_answer", "freeform"]

_OPTION_PREFIX_RE = re.compile(r"^\s*([A-Za-z])[\).\:\-\s]+(.+?)\s*$")


@dataclass(frozen=True)
class OptionSpec:
    label: str
    text: str
    required_targets: Sequence[str] = field(default_factory=tuple)
    target_sequence: Sequence[str] = field(default_factory=tuple)
    theme_targets: Sequence[str] = field(default_factory=tuple)
    forbidden_targets: Sequence[str] = field(default_factory=tuple)
    mutex_group: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", str(self.label).strip().upper()[:1])
        object.__setattr__(self, "text", str(self.text or "").strip())
        object.__setattr__(self, "required_targets", _tuple_of_strings(self.required_targets))
        object.__setattr__(self, "target_sequence", _tuple_of_strings(self.target_sequence))
        object.__setattr__(self, "theme_targets", _tuple_of_strings(self.theme_targets))
        object.__setattr__(self, "forbidden_targets", _tuple_of_strings(self.forbidden_targets))
        object.__setattr__(self, "mutex_group", str(self.mutex_group or ""))


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    question: str
    answer_format: AnswerFormat
    route: str
    options: Sequence[OptionSpec | Sequence[Any]]
    target_registry: TargetRegistry | None = None
    answer_operator: AnswerOperator = "select_present"

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", str(self.task_id or "task"))
        object.__setattr__(self, "question", str(self.question or ""))
        object.__setattr__(self, "route", str(self.route or ""))
        object.__setattr__(self, "options", tuple(_normalize_option(option) for option in self.options))
        object.__setattr__(
            self,
            "answer_operator",
            normalize_answer_operator(
                self.answer_operator,
                question=self.question,
                route=self.route,
                options=tuple(option.text for option in self.options),
            ),
        )

    def normalized(self) -> "TaskSpec":
        """Return a copy-like normalized task.

        The constructor already normalizes options; this method makes tests and
        call sites that build tuple-shaped options explicit without exposing
        normalization internals.
        """

        return self


def build_task_spec(
    *,
    task_id: str,
    question: str,
    options: Sequence[str | Mapping[str, Any]],
    route: str,
    target_registry: TargetRegistry | None = None,
    answer_format: AnswerFormat = "mcq",
    answer_operator: str | None = None,
) -> TaskSpec:
    """Build a generic task spec from option text and a target registry."""

    compiled_options: list[OptionSpec] = []
    for index, raw_option in enumerate(options):
        label, text = _option_label_and_text(raw_option, index=index)
        matched_targets = _matched_target_refs(text, target_registry)
        target_refs = tuple(target_id for target_id, _position in matched_targets)
        if _route_is_ordered(route):
            target_sequence = target_refs
            required_targets = target_refs
            theme_targets: tuple[str, ...] = ()
        elif _route_is_main_idea(route):
            target_sequence = ()
            required_targets = target_refs
            theme_targets = target_refs
        else:
            target_sequence = ()
            required_targets = target_refs
            theme_targets = ()
        compiled_options.append(
            OptionSpec(
                label=label,
                text=text,
                required_targets=required_targets,
                target_sequence=target_sequence,
                theme_targets=theme_targets,
            )
        )

    return TaskSpec(
        task_id=task_id,
        question=question,
        answer_format=answer_format,
        route=route,
        options=tuple(compiled_options),
        target_registry=target_registry,
        answer_operator=normalize_answer_operator(
            answer_operator,
            question=question,
            route=route,
            options=tuple(option.text for option in compiled_options),
        ),
    )


def _normalize_option(option: OptionSpec | Sequence[Any]) -> OptionSpec:
    if isinstance(option, OptionSpec):
        return option
    items = list(option)
    label = items[0] if len(items) > 0 else ""
    text = items[1] if len(items) > 1 else ""
    required_targets = items[2] if len(items) > 2 else ()
    target_sequence = items[3] if len(items) > 3 else ()
    theme_targets = items[4] if len(items) > 4 else ()
    return OptionSpec(
        label=str(label),
        text=str(text),
        required_targets=required_targets,
        target_sequence=target_sequence,
        theme_targets=theme_targets,
    )


def _option_label_and_text(raw_option: str | Mapping[str, Any], *, index: int) -> tuple[str, str]:
    if isinstance(raw_option, Mapping):
        label = str(raw_option.get("label") or raw_option.get("option_id") or _index_label(index))
        text = str(raw_option.get("text") or raw_option.get("raw_option_text") or "")
        return label.strip().upper()[:1], text.strip()

    raw_text = str(raw_option or "").strip()
    match = _OPTION_PREFIX_RE.match(raw_text)
    if match:
        return match.group(1).upper(), match.group(2).strip()
    return _index_label(index), raw_text


def _matched_target_refs(text: str, registry: TargetRegistry | None) -> tuple[tuple[str, int], ...]:
    if registry is None:
        return ()
    haystack = str(text or "").casefold()
    matches: list[tuple[str, int]] = []
    for target_id, target in registry.targets_by_id.items():
        positions = [
            haystack.find(candidate.casefold())
            for candidate in (target.canonical_text, *target.aliases)
            if str(candidate or "").strip()
        ]
        positions = [position for position in positions if position >= 0]
        if positions:
            matches.append((str(target_id), min(positions)))
    return tuple(sorted(matches, key=lambda item: (item[1], item[0])))


def _route_is_ordered(route: str) -> bool:
    route_text = str(route or "").casefold()
    return any(marker in route_text for marker in ("order", "sequence", "timeline", "temporal"))


def _route_is_main_idea(route: str) -> bool:
    route_text = str(route or "").casefold()
    return any(marker in route_text for marker in ("main_idea", "gist", "synopsis", "summary"))


def _tuple_of_strings(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in values if str(value or "").strip())


def _index_label(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if 0 <= index < len(alphabet):
        return alphabet[index]
    return str(index + 1)
