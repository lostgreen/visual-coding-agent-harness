"""Workspace-backed timeline tools."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from ..core.registry import ToolRegistry, tool
from ..workspace import EvidenceWorkspace


def build_timeline_registry(*, workspace: Optional[EvidenceWorkspace] = None) -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="append_to_timeline", description="Append an observed event to the workspace timeline.")
    def append_to_timeline(
        obs_id: str,
        entity: str,
        observed_at_sec: float | None = None,
        window: Sequence[float] = (),
        confidence_signal: str = "",
        claim: str = "",
    ) -> Mapping[str, object]:
        if workspace is None:
            return {
                "claim": "No workspace is attached; timeline was not updated.",
                "confidence": 0.0,
                "regions": [],
                "limitations": "Workspace-backed tool requires an EvidenceWorkspace.",
            }
        row = workspace.append_to_timeline(
            obs_id=obs_id,
            entity=entity,
            observed_at_sec=observed_at_sec,
            window=window or None,
            confidence_signal=confidence_signal,
            claim=claim,
        )
        return {
            "claim": f"Timeline appended for {row['entity']}.",
            "confidence": 1.0,
            "regions": [row],
            "limitations": "Cheap workspace write; does not inspect video frames.",
        }

    registry.register(append_to_timeline)
    return registry
