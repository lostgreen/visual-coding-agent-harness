"""Evidence workspace for visual tool results."""

from __future__ import annotations

import json
import re
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

    VISUAL_EVIDENCE_TOOLS = {"caption_segment", "qa_segment", "inspect_segment"}
    NAVIGATION_TOOLS = {"video_ls", "search_segments", "read_segment", "expand_window", "zoom"}

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

    def compact_ledger_text(
        self,
        *,
        max_working_observations: int = 4,
        max_visual_evidence: int = 8,
    ) -> str:
        """Return a bounded answer-facing context derived from the raw ledger trace."""

        ledger_path = self.root / "ledger.md"
        if not ledger_path.exists():
            return ""
        raw_ledger = ledger_path.read_text(encoding="utf-8")
        entries = _parse_ledger_entries(raw_ledger)
        if not entries:
            return raw_ledger

        visual_entries = [
            entry for entry in entries if str(entry.get("tool", "")) in self.VISUAL_EVIDENCE_TOOLS
        ][-max_visual_evidence:]
        navigation_entries = [
            entry for entry in entries if str(entry.get("tool", "")) in self.NAVIGATION_TOOLS
        ]
        working_entries = entries[-max_working_observations:] if max_working_observations > 0 else []

        sections = ["# Compact Evidence Context", ""]
        sections.append("## Long-Term Visual Evidence")
        if visual_entries:
            sections.extend(_format_compact_entry(entry) for entry in visual_entries)
        else:
            sections.append("(none)")

        sections.extend(["", "## Navigation Summary"])
        if navigation_entries:
            sections.extend(
                f"- {entry['observation_id']}: {entry.get('tool', 'unknown')}"
                for entry in navigation_entries
            )
        else:
            sections.append("(none)")

        sections.extend(["", "## Short-Term Working Buffer"])
        if working_entries:
            sections.extend(_format_rawish_entry(entry) for entry in working_entries)
        else:
            sections.append("(none)")
        sections.append("")
        return "\n".join(sections)

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


def _parse_ledger_entries(ledger_text: str) -> list[Mapping[str, Any]]:
    entries = []
    for line in ledger_text.splitlines():
        obs_match = re.search(r"`(obs_[0-9]{4})`", line)
        if not obs_match:
            continue
        tool_match = re.search(r"tool:\s*`?([A-Za-z0-9_]+)`?", line)
        confidence_match = re.search(r"confidence:\s*([0-9.]+)", line)
        artifacts_match = re.search(r"artifacts:\s*(.*?)\s*\|\s*claim:", line)
        claim_match = re.search(r"claim:\s*(.*?)\s*\|\s*limitations:", line)
        limitation_match = re.search(r"limitations:\s*(.*)$", line)
        entries.append(
            {
                "observation_id": obs_match.group(1),
                "tool": tool_match.group(1) if tool_match else "unknown",
                "confidence": confidence_match.group(1) if confidence_match else "",
                "artifacts": artifacts_match.group(1).strip() if artifacts_match else "-",
                "claim": claim_match.group(1).strip() if claim_match else "",
                "limitations": limitation_match.group(1).strip() if limitation_match else "-",
            }
        )
    return entries


def _format_compact_entry(entry: Mapping[str, Any]) -> str:
    confidence = f" | confidence: {entry['confidence']}" if entry.get("confidence") else ""
    limitations = entry.get("limitations") or "-"
    return (
        f"- `{entry['observation_id']}` | tool: `{entry.get('tool', 'unknown')}`{confidence} | "
        f"claim: {entry.get('claim', '')} | limitations: {limitations}"
    )


def _format_rawish_entry(entry: Mapping[str, Any]) -> str:
    artifacts = entry.get("artifacts") or "-"
    limitations = entry.get("limitations") or "-"
    confidence = entry.get("confidence") or "0.00"
    return (
        f"- `{entry['observation_id']}` | tool: `{entry.get('tool', 'unknown')}` | "
        f"confidence: {confidence} | artifacts: {artifacts} | "
        f"claim: {entry.get('claim', '')} | limitations: {limitations}"
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
