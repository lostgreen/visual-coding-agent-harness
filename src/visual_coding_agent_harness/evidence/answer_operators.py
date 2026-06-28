"""Generic answer-operator detection for option-level video QA."""

from __future__ import annotations

import re
from typing import Literal, Sequence

AnswerOperator = Literal[
    "select_present",
    "select_absent",
    "causal_bind",
    "universal_intersection",
    "ordered_projection",
    "main_arc",
]
ModalityRoute = Literal["visual_primary", "asr_primary", "ocr_primary", "multimodal"]

ALLOWED_ANSWER_OPERATORS = frozenset(AnswerOperator.__args__)  # type: ignore[attr-defined]
ALLOWED_MODALITY_ROUTES = frozenset(ModalityRoute.__args__)  # type: ignore[attr-defined]

_MAIN_ARC_RE = re.compile(r"\b(main\s+idea|mainly\s+about|mostly\s+about|overall|summary|summari[sz]e|theme)\b", re.I)
_ORDER_RE = re.compile(r"\b(order|ordered|sequence|successively|successive|first|then|before|after|timeline)\b", re.I)
_UNIVERSAL_RE = re.compile(r"\b(all|every|each|universally|across\s+all|in\s+all|both)\b", re.I)
_CAUSAL_RE = re.compile(r"\b(why|reason|because|cause|primary\s+reason|according\s+to[^?]{0,80}\bwhy)\b", re.I)
_ABSENT_RE = re.compile(
    r"\b("
    r"which\s+is\s+not|what\s+is\s+not|not\s+true|not\s+described|not\s+seen|not\s+shown|"
    r"did\s+not\s+see|does\s+not\s+appear|except|absent|missing"
    r")\b",
    re.I,
)
_ASR_RE = re.compile(
    r"\b("
    r"say|said|mention|state|narrator|narration|explain|discuss|talk\s+about|"
    r"according\s+to|describe|tell|tells|told|claim"
    r")\b",
    re.I,
)
_OCR_RE = re.compile(r"\b(text|sign|label|caption|subtitle|banner|written|reads?)\b", re.I)
_VISUAL_RE = re.compile(
    r"\b("
    r"color|wear|wearing|show|appear|visible|scene|how\s+many|count|"
    r"point|gesture|holds?|holding|left|right|next\s+to"
    r")\b",
    re.I,
)
_BIOGRAPHY_RE = re.compile(
    r"\b(life|background|born|grew\s+up|childhood|career|journey|biography|history\s+of)\b",
    re.I,
)

ROUTE_TO_MODALITIES: dict[ModalityRoute, tuple[str, ...]] = {
    "visual_primary": ("visual", "ocr", "index"),
    "asr_primary": ("asr", "index", "visual"),
    "ocr_primary": ("ocr", "visual", "index"),
    "multimodal": ("visual", "asr", "ocr", "index"),
}


def derive_answer_operator(question: str, *, route: str, options: Sequence[str]) -> AnswerOperator:
    """Classify the answer logic using only generic question-form markers."""

    text = _normalize(question)
    route_text = _normalize(route)
    if route_text == "gist_global" or _MAIN_ARC_RE.search(text):
        return "main_arc"
    if ("temporal" in route_text or _ORDER_RE.search(text)) and _options_look_ordered(options):
        return "ordered_projection"
    if _UNIVERSAL_RE.search(text):
        return "universal_intersection"
    if _CAUSAL_RE.search(text):
        return "causal_bind"
    if _ABSENT_RE.search(text):
        return "select_absent"
    return "select_present"


def normalize_answer_operator(value: str | None, *, question: str = "", route: str = "", options: Sequence[str] = ()) -> AnswerOperator:
    text = str(value or "").strip()
    if text in ALLOWED_ANSWER_OPERATORS:
        return text  # type: ignore[return-value]
    return derive_answer_operator(question, route=route, options=options)


def derive_modality_route(question: str, *, operator: AnswerOperator) -> ModalityRoute:
    """Choose a primary evidence route from generic question-form markers."""

    text = _normalize(question)
    if operator == "main_arc":
        return "asr_primary"
    if _OCR_RE.search(text):
        return "ocr_primary"
    if _BIOGRAPHY_RE.search(text) or _ASR_RE.search(text):
        return "asr_primary"
    if _VISUAL_RE.search(text):
        return "visual_primary"
    return "multimodal"


def _options_look_ordered(options: Sequence[str]) -> bool:
    if not options:
        return True
    ordered_count = 0
    for option in options:
        text = _normalize(option)
        if "->" in text or re.search(r"\b(?:then|before|after|first|second|third|finally)\b", text):
            ordered_count += 1
    return ordered_count >= 2


def _normalize(value: str) -> str:
    return " ".join(str(value or "").split()).casefold()
