"""Contracts and conservative builders for ordered transcript evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence
import re

from .targets import OptionSpec, TargetSpec


@dataclass(frozen=True)
class OrderedTranscriptItem:
    target_ref: str
    span_start_char: int
    span_end_char: int
    mention_start_sec: float | None
    mention_end_sec: float | None
    matched_text: str


@dataclass(frozen=True)
class OrderedTranscriptSequence:
    evidence_id: str
    obs_id: str
    segment_id: str
    ordered_target_refs: tuple[str, ...]
    items: tuple[OrderedTranscriptItem, ...]
    snippet: str
    subject: str | None
    context: str | None
    status: Literal["supported", "ambiguous", "rejected"]
    source: Literal["indexed_transcript"]
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "obs_id": self.obs_id,
            "segment_id": self.segment_id,
            "ordered_target_refs": list(self.ordered_target_refs),
            "items": [asdict(item) for item in self.items],
            "snippet": self.snippet,
            "subject": self.subject,
            "context": self.context,
            "status": self.status,
            "source": self.source,
            "confidence": self.confidence,
        }


def build_ordered_transcript_sequence(
    *,
    text: str,
    targets: Sequence[TargetSpec],
    segment_id: str,
    obs_id: str = "",
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> OrderedTranscriptSequence | None:
    """Create answer-grade order evidence from one contiguous transcript span.

    The builder is intentionally conservative: every target must appear exactly
    once, the mentions must live in a compact contiguous list span, and obvious
    comparison/catalogue contexts are rejected.
    """

    snippet_text = _compact_text(text)
    if not snippet_text or len(targets) < 2:
        return None

    items: list[OrderedTranscriptItem] = []
    missing = False
    duplicate = False
    negated = False
    for target in targets:
        matches = _target_matches(snippet_text, target)
        if not matches:
            missing = True
            continue
        if len(matches) > 1:
            duplicate = True
        match = matches[0]
        if _is_negated(snippet_text, match.start()):
            negated = True
        items.append(
            OrderedTranscriptItem(
                target_ref=target.target_id,
                span_start_char=int(match.start()),
                span_end_char=int(match.end()),
                mention_start_sec=start_sec,
                mention_end_sec=end_sec,
                matched_text=match.group(0),
            )
        )

    ordered_items = tuple(sorted(items, key=lambda item: item.span_start_char))
    ordered_refs = tuple(item.target_ref for item in ordered_items)
    list_snippet = _sequence_snippet(snippet_text, ordered_items)
    subject = _common_subject(targets)
    evidence_id = f"seq_{_safe_id(obs_id or segment_id or 'transcript')}"

    if negated or _is_unrelated_enumeration_context(snippet_text, ordered_items):
        status: Literal["supported", "ambiguous", "rejected"] = "rejected"
        confidence = 0.2
    elif missing or duplicate or len(ordered_items) != len(targets) or not _is_contiguous_list_span(snippet_text, ordered_items):
        status = "ambiguous"
        confidence = 0.5
    else:
        status = "supported"
        confidence = 0.94

    return OrderedTranscriptSequence(
        evidence_id=evidence_id,
        obs_id=str(obs_id),
        segment_id=str(segment_id),
        ordered_target_refs=ordered_refs,
        items=ordered_items,
        snippet=list_snippet,
        subject=subject,
        context="continuous_asr_enumeration" if status == "supported" else "indexed_transcript_span",
        status=status,
        source="indexed_transcript",
        confidence=confidence,
    )


def ordered_sequence_exact_option(
    sequence: OrderedTranscriptSequence,
    options: Sequence[OptionSpec] | Mapping[str, OptionSpec],
) -> str | None:
    """Return the only option whose target sequence exactly matches evidence."""

    if sequence.status != "supported":
        return None
    option_values = options.values() if isinstance(options, Mapping) else options
    matches = [
        str(option.option_id).strip().upper()[:1]
        for option in option_values
        if tuple(str(ref) for ref in option.target_sequence) == sequence.ordered_target_refs
    ]
    unique = sorted({match for match in matches if match})
    return unique[0] if len(unique) == 1 else None


def _target_matches(text: str, target: TargetSpec) -> list[re.Match[str]]:
    matches: list[re.Match[str]] = []
    seen: set[tuple[int, int]] = set()
    for alias in _target_aliases(target):
        for match in re.finditer(_phrase_pattern(alias), text, flags=re.IGNORECASE):
            key = (int(match.start()), int(match.end()))
            if key in seen:
                continue
            seen.add(key)
            matches.append(match)
    return sorted(matches, key=lambda match: (match.start(), match.end()))


def _target_aliases(target: TargetSpec) -> list[str]:
    aliases = [target.canonical_text, *list(target.aliases)]
    return _unique_nonempty(aliases)


def _phrase_pattern(phrase: str) -> str:
    tokens = [re.escape(token) for token in re.findall(r"[A-Za-z0-9]+", str(phrase or ""))]
    if not tokens:
        return r"$^"
    return r"\b" + r"[\W_]+".join(tokens) + r"\b"


def _is_negated(text: str, match_start: int) -> bool:
    window = text[max(0, int(match_start) - 64) : int(match_start)].lower()
    return bool(re.search(r"\b(?:never|not|no|without|did not|didn't)\b", window))


def _is_contiguous_list_span(text: str, items: Sequence[OrderedTranscriptItem]) -> bool:
    if len(items) < 2:
        return False
    start = min(item.span_start_char for item in items)
    end = max(item.span_end_char for item in items)
    span = text[start:end]
    if len(span) > 700:
        return False
    for left, right in zip(items, items[1:]):
        bridge = text[left.span_end_char : right.span_start_char]
        if re.search(r"[.!?]\s+[A-Z]", bridge):
            return False
    separator_count = len(re.findall(r",|;|:|\band\b|\bthen\b|[\"“”]", span, flags=re.IGNORECASE))
    return separator_count >= max(1, len(items) - 2)


def _is_unrelated_enumeration_context(text: str, items: Sequence[OrderedTranscriptItem]) -> bool:
    if not items:
        return False
    start = max(0, min(item.span_start_char for item in items) - 120)
    end = min(len(text), max(item.span_end_char for item in items) + 120)
    window = text[start:end].lower()
    return bool(
        re.search(
            r"\b(?:compare|comparison|catalogue|catalog|unrelated|random|not\s+the\s+(?:order|sequence)|separate\s+examples)\b",
            window,
        )
    )


def _sequence_snippet(text: str, items: Sequence[OrderedTranscriptItem]) -> str:
    if not items:
        return _compact_text(text)[:240]
    start = max(0, min(item.span_start_char for item in items) - 80)
    end = min(len(text), max(item.span_end_char for item in items) + 80)
    return _compact_text(text[start:end])[:500]


def _common_subject(targets: Sequence[TargetSpec]) -> str | None:
    subjects = {str(target.subject).strip() for target in targets if str(target.subject or "").strip()}
    return next(iter(subjects)) if len(subjects) == 1 else None


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "")).strip("_")
    return safe or "transcript"


def _compact_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _unique_nonempty(values: Sequence[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
