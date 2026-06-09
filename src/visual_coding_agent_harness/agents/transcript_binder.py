"""Conservative transcript-to-target evidence binding."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

from ..contracts import ClaimRelation, EvidenceBinding, RelationBinding, TargetSpec


@dataclass(frozen=True)
class TranscriptBindingResult:
    evidence_bindings: Sequence[EvidenceBinding] = field(default_factory=tuple)
    relation_bindings: Sequence[RelationBinding] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_bindings": [_binding_dict(binding) for binding in self.evidence_bindings],
            "relation_bindings": [asdict(binding) for binding in self.relation_bindings],
        }


class TranscriptEvidenceBinder:
    """Bind indexed transcript snippets to target contracts without treating hits as proof."""

    def bind(
        self,
        *,
        text: str,
        targets: Sequence[TargetSpec],
        relations: Sequence[ClaimRelation] = (),
        obs_id: str = "",
        segment_id: str = "",
        start_sec: float | None = None,
        source: str = "asr",
    ) -> TranscriptBindingResult:
        snippet = _compact_text(text)
        target_matches = {target.target_id: _first_target_match(snippet, target) for target in targets}
        evidence_bindings = tuple(
            self._bind_target(
                snippet=snippet,
                target=target,
                match=target_matches[target.target_id],
                obs_id=obs_id,
                segment_id=segment_id,
                start_sec=start_sec,
                source=source,
            )
            for target in targets
        )
        bindings_by_target = {binding.target_id: binding for binding in evidence_bindings}
        relation_bindings = tuple(
            self._bind_relation(
                snippet=snippet,
                relation=relation,
                bindings_by_target=bindings_by_target,
                target_matches=target_matches,
                obs_id=obs_id,
                start_sec=start_sec,
                source=source,
            )
            for relation in relations
        )
        return TranscriptBindingResult(
            evidence_bindings=evidence_bindings,
            relation_bindings=relation_bindings,
        )

    def _bind_target(
        self,
        *,
        snippet: str,
        target: TargetSpec,
        match: re.Match[str] | None,
        obs_id: str,
        segment_id: str,
        start_sec: float | None,
        source: str,
    ) -> EvidenceBinding:
        subject_supported = _subject_supported(snippet, target.subject)
        status = "supported" if match is not None and subject_supported and not _is_negated(snippet, match.start()) else "rejected"
        if match is not None and _is_artwork_subject_context(snippet, target=target, match_start=match.start()):
            status = "ambiguous"
        if match is not None and target.subject and not subject_supported:
            status = "ambiguous"
        relation = str(target.relation or "present")
        evidence_id = _evidence_id(segment_id=segment_id, target_id=target.target_id)
        return EvidenceBinding(
            evidence_id=evidence_id,
            obs_id=str(obs_id),
            target_id=target.target_id,
            subject=str(target.subject or ""),
            relation=relation,
            status=status,
            mention_timestamp_sec=start_sec,
            source=source,
            snippet=snippet,
            claim_modality=target.modality_hint,
        )

    def _bind_relation(
        self,
        *,
        snippet: str,
        relation: ClaimRelation,
        bindings_by_target: Mapping[str, EvidenceBinding],
        target_matches: Mapping[str, re.Match[str] | None],
        obs_id: str,
        start_sec: float | None,
        source: str,
    ) -> RelationBinding:
        source_binding = bindings_by_target.get(relation.source_target_id)
        destination_binding = bindings_by_target.get(relation.destination_target_id)
        status = "rejected"
        if (
            source_binding is not None
            and destination_binding is not None
            and source_binding.status == "supported"
            and destination_binding.status == "supported"
        ):
            source_match = target_matches.get(source_binding.target_id)
            destination_match = target_matches.get(destination_binding.target_id)
            source_pos = source_match.start() if source_match is not None else None
            destination_pos = destination_match.start() if destination_match is not None else None
            if source_pos is not None and destination_pos is not None:
                if str(relation.kind).strip().lower() in {"before", "precedes", "then"}:
                    status = _before_relation_status(
                        snippet=snippet,
                        source_pos=source_pos,
                        source_end=source_match.end(),
                        destination_pos=destination_pos,
                        destination_end=destination_match.end(),
                    )
                else:
                    status = "supported"
        return RelationBinding(
            binding_id=f"rel_bind_{relation.relation_id}",
            obs_id=str(obs_id),
            relation_id=relation.relation_id,
            status=status,
            source=source,
            snippet=snippet,
            mention_timestamp_sec=start_sec,
        )


def _binding_dict(binding: EvidenceBinding) -> dict[str, object]:
    payload = asdict(binding)
    payload["claim_modality"] = str(binding.claim_modality.value)
    return payload


def _compact_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _first_target_match(snippet: str, target: TargetSpec) -> re.Match[str] | None:
    for alias in _target_aliases(target):
        match = re.search(_phrase_pattern(alias), snippet, flags=re.IGNORECASE)
        if match:
            return match
    return None


def _target_aliases(target: TargetSpec) -> list[str]:
    aliases = [target.canonical_text, *list(target.aliases)]
    built_ins = {
        "upper class": ["upper class", "upper-class", "upper echelons", "high society", "upper"],
        "humble background": ["humble background", "humble origins", "modest background"],
        "successful": ["successful", "becoming successful"],
    }.get(target.canonical_text.lower().strip(), [])
    return _unique_nonempty([*aliases, *built_ins])


def _phrase_pattern(phrase: str) -> str:
    escaped = re.escape(phrase.strip())
    escaped = escaped.replace(r"\ ", r"[\s-]+")
    return rf"(?<!\w){escaped}(?!\w)"


def _subject_supported(snippet: str, subject: str | None) -> bool:
    subject_text = str(subject or "").strip()
    if not subject_text:
        return True
    if subject_text.lower() in {"he", "she", "they", "him", "her"}:
        return True
    return re.search(_phrase_pattern(subject_text), snippet, flags=re.IGNORECASE) is not None


def _is_negated(snippet: str, match_start: int) -> bool:
    window = snippet[max(0, match_start - 48):match_start].lower()
    return bool(re.search(r"\b(?:never|not|no|without|failed to|did not|didn't)\b", window))


def _is_artwork_subject_context(snippet: str, *, target: TargetSpec, match_start: int) -> bool:
    subject = str(target.subject or "").strip()
    if not subject or subject.lower() in {"he", "she", "they", "him", "her"}:
        return False
    before = snippet[max(0, match_start - 96):match_start].lower()
    if not re.search(r"\b(?:paint(?:ed|ing)?|depict(?:s|ed|ing)?|portray(?:s|ed|ing)?|represent(?:s|ed|ing)?)\b", before):
        return False
    object_tail = snippet[match_start:match_start + 80].lower()
    return bool(re.search(r"\b(?:man|woman|figure|subject|painting|artwork|portrait)\b", object_tail))


def _before_relation_status(
    *,
    snippet: str,
    source_pos: int,
    source_end: int,
    destination_pos: int,
    destination_end: int,
) -> str:
    if source_pos >= destination_pos:
        return "rejected"
    relation_span = snippet[source_pos:destination_end].lower()
    between = snippet[source_end:destination_pos].lower()
    if re.search(r"\b(?:only\s+)?after\b", between):
        return "rejected"
    if re.search(
        r"\b(?:then|later|subsequently|before|rose\s+through|to\s+reach|from\b.+\bto\b|moved\s+into|withdrew)\b",
        relation_span,
    ):
        return "supported"
    return "ambiguous"


def _evidence_id(*, segment_id: str, target_id: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9_]+", "_", str(segment_id or "segment")).strip("_") or "segment"
    target = re.sub(r"[^A-Za-z0-9_]+", "_", str(target_id or "target")).strip("_") or "target"
    return f"ev_bind_{segment}_{target}"


def _unique_nonempty(values: Sequence[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
