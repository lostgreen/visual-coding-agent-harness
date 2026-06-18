"""Minimal final-answer citation integrity gate."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

from ..memory import excerpt_hash, normalized_text


@dataclass(frozen=True)
class FinalIntegrityDecision:
    accepted: bool
    selected_option: str
    rejection_reason: str | None
    rejection_hint: str
    cited_memory_ids: tuple[str, ...]
    cited_observation_ids: tuple[str, ...]
    warnings: tuple[str, ...]


def evaluate_final_integrity(
    *,
    selected_option: str,
    citations: Sequence[str],
    workspace: Any,
    reason: str = "",
) -> FinalIntegrityDecision:
    """Check final citation integrity without judging semantic support."""

    del reason
    option_id = str(selected_option or "").strip()
    option_ids = _integrity_option_ids(workspace)
    if option_ids and option_id not in option_ids:
        return _integrity_reject(
            option_id,
            "invalid_option",
            "Choose one of the legal option ids.",
            cited_memory_ids=(),
            cited_observation_ids=(),
        )

    citation_ids = tuple(str(item).strip() for item in citations if str(item).strip())
    if not citation_ids:
        return _integrity_reject(
            option_id,
            "missing_citation",
            "Final answers must cite at least one memory id.",
            cited_memory_ids=(),
            cited_observation_ids=(),
        )

    cited_memory_ids: list[str] = []
    cited_observation_ids: list[str] = []
    warnings: list[str] = []
    anchor_by_id = workspace.read_produced_anchors_by_id()
    for citation in citation_ids:
        if citation.startswith("mem_"):
            memory = workspace.get_memory(citation)
            if memory is None:
                return _integrity_reject(
                    option_id,
                    "dangling_memory_id",
                    f"Unknown memory citation: {citation}.",
                    cited_memory_ids=tuple(cited_memory_ids),
                    cited_observation_ids=tuple(cited_observation_ids),
                )
            if memory.superseded_by:
                return _integrity_reject(
                    option_id,
                    "superseded_memory",
                    f"Memory {citation} was superseded by {memory.superseded_by}.",
                    cited_memory_ids=tuple(cited_memory_ids),
                    cited_observation_ids=tuple(cited_observation_ids),
                )
            for anchor in memory.anchors:
                produced = anchor_by_id.get(anchor.anchor_id)
                if produced is None:
                    return _integrity_reject(
                        option_id,
                        "dangling_anchor_id",
                        f"Memory {citation} cites unknown anchor {anchor.anchor_id}.",
                        cited_memory_ids=tuple(cited_memory_ids),
                        cited_observation_ids=tuple(cited_observation_ids),
                    )
                if anchor.excerpt and normalized_text(anchor.excerpt) not in normalized_text(produced.excerpt):
                    return _integrity_reject(
                        option_id,
                        "anchor_excerpt_mismatch",
                        f"Memory {citation} excerpt is not present in anchor {anchor.anchor_id}.",
                        cited_memory_ids=tuple(cited_memory_ids),
                        cited_observation_ids=tuple(cited_observation_ids),
                    )
                if anchor.excerpt_hash and anchor.excerpt_hash != excerpt_hash(anchor.excerpt):
                    return _integrity_reject(
                        option_id,
                        "anchor_excerpt_hash_mismatch",
                        f"Memory {citation} anchor hash does not match its excerpt.",
                        cited_memory_ids=tuple(cited_memory_ids),
                        cited_observation_ids=tuple(cited_observation_ids),
                    )
            cited_memory_ids.append(citation)
            continue
        if citation.startswith("obs_"):
            if not _allow_raw_observation_final_citation():
                return _integrity_reject(
                    option_id,
                    "raw_observation_citation_without_memory",
                    f"Final answers must cite planner memory, not raw observation {citation}.",
                    cited_memory_ids=tuple(cited_memory_ids),
                    cited_observation_ids=tuple(cited_observation_ids),
                )
            if workspace.get_observation(citation) is None:
                return _integrity_reject(
                    option_id,
                    "dangling_observation_id",
                    f"Unknown observation citation: {citation}.",
                    cited_memory_ids=tuple(cited_memory_ids),
                    cited_observation_ids=tuple(cited_observation_ids),
                )
            cited_observation_ids.append(citation)
            warnings.append("raw_observation_citation_without_memory")
            continue
        return _integrity_reject(
            option_id,
            "unparseable_citation",
            f"Citation must be a memory id or observation id: {citation}.",
            cited_memory_ids=tuple(cited_memory_ids),
            cited_observation_ids=tuple(cited_observation_ids),
        )

    warnings.extend(_integrity_warnings(workspace=workspace, selected_option=option_id))
    return FinalIntegrityDecision(
        accepted=True,
        selected_option=option_id,
        rejection_reason=None,
        rejection_hint="",
        cited_memory_ids=tuple(cited_memory_ids),
        cited_observation_ids=tuple(cited_observation_ids),
        warnings=tuple(warnings),
    )


def _integrity_option_ids(workspace: Any) -> set[str]:
    method = getattr(workspace, "_known_option_ids", None)
    if callable(method):
        return set(method())
    registry = getattr(workspace, "target_registry", None)
    options_by_id = getattr(registry, "options_by_id", {})
    if isinstance(options_by_id, dict):
        return {str(key) for key in options_by_id.keys()}
    return set()


def _allow_raw_observation_final_citation() -> bool:
    return os.environ.get("HARNESS_ALLOW_RAW_OBS_FINAL_CITATION", "").strip().lower() in {"1", "true", "yes", "on"}


def _integrity_warnings(*, workspace: Any, selected_option: str) -> tuple[str, ...]:
    warnings: list[str] = []
    for entry in workspace.memory_entries():
        if entry.kind == "conflict" and entry.supports_option == selected_option and not entry.superseded_by:
            warnings.append(f"unresolved_conflict_memory:{entry.entry_id}")
    return tuple(warnings)


def _integrity_reject(
    selected_option: str,
    reason: str,
    hint: str,
    *,
    cited_memory_ids: Sequence[str],
    cited_observation_ids: Sequence[str],
) -> FinalIntegrityDecision:
    return FinalIntegrityDecision(
        accepted=False,
        selected_option=selected_option,
        rejection_reason=reason,
        rejection_hint=hint,
        cited_memory_ids=tuple(cited_memory_ids),
        cited_observation_ids=tuple(cited_observation_ids),
        warnings=(),
    )
