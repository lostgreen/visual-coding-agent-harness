"""Evidence workspace for visual tool results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Observation:
    observation_id: str
    tool: str
    claim: str
    confidence: float
    input_artifacts: Sequence[str] = field(default_factory=list)
    regions: Sequence[Mapping[str, Any]] = field(default_factory=list)
    limitations: str = ""
    raw_output: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""


class EvidenceWorkspace:
    """Persist artifacts, observations, trace events, and an answer-facing ledger."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def create(cls, base_dir: Path, run_id: str) -> "EvidenceWorkspace":
        root = base_dir / "runs" / run_id
        for child in [
            root,
            root / "input",
            root / "artifacts",
            root / "artifacts" / "frames",
            root / "artifacts" / "clips",
            root / "artifacts" / "crops",
            root / "artifacts" / "masks",
        ]:
            child.mkdir(parents=True, exist_ok=True)

        for filename in ["observations.jsonl", "trace.jsonl"]:
            (root / filename).touch(exist_ok=True)

        ledger = root / "ledger.md"
        if not ledger.exists():
            ledger.write_text("# Evidence Ledger\n\n", encoding="utf-8")

        return cls(root=root)

    def write_observation(
        self,
        *,
        tool_name: str,
        claim: str,
        confidence: float,
        input_artifacts: Sequence[str] = (),
        regions: Sequence[Mapping[str, Any]] = (),
        limitations: str = "",
        raw_output: Mapping[str, Any] | None = None,
    ) -> Observation:
        observation = Observation(
            observation_id=self._next_observation_id(),
            tool=tool_name,
            claim=claim,
            confidence=confidence,
            input_artifacts=list(input_artifacts),
            regions=list(regions),
            limitations=limitations,
            raw_output=dict(raw_output or {}),
            created_at=_utc_now(),
        )
        self._append_jsonl("observations.jsonl", asdict(observation))
        return observation

    def write_trace_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self._append_jsonl(
            "trace.jsonl",
            {
                "type": event_type,
                "created_at": _utc_now(),
                "payload": dict(payload),
            },
        )

    def write_ledger_entry(self, observation: Observation) -> None:
        artifacts = ", ".join(observation.input_artifacts) or "-"
        limitation = observation.limitations or "-"
        line = (
            f"- `{observation.observation_id}` | tool: `{observation.tool}` | "
            f"confidence: {observation.confidence:.2f} | artifacts: {artifacts} | "
            f"claim: {observation.claim} | limitations: {limitation}\n"
        )
        with (self.root / "ledger.md").open("a", encoding="utf-8") as handle:
            handle.write(line)

    def _append_jsonl(self, filename: str, payload: Mapping[str, Any]) -> None:
        with (self.root / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
            handle.write("\n")

    def _next_observation_id(self) -> str:
        existing = 0
        observations = self.root / "observations.jsonl"
        if observations.exists():
            with observations.open("r", encoding="utf-8") as handle:
                existing = sum(1 for line in handle if line.strip())
        return f"obs_{existing + 1:04d}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
