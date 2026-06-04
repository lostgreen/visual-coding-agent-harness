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
    ANSWER_EVIDENCE_TOOLS = VISUAL_EVIDENCE_TOOLS | {
        "caption_image",
        "caption_region",
        "ocr_region",
        "qa_region",
        "inspect_region",
        "verify_local_claim",
    }

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

    def evidence_table(self, *, question: str, options: Sequence[str] = ()) -> dict[str, Any]:
        """Return an option-grouped answer evidence table for arbitration."""

        option_map = _option_letter_map(options)
        groups: dict[str, list[dict[str, Any]]] = {letter: [] for letter in option_map}
        groups.setdefault("unassigned", [])
        rows = []

        for observation in self._read_observation_dicts():
            tool_name = str(observation.get("tool", ""))
            if tool_name in self.NAVIGATION_TOOLS:
                continue
            if self.ANSWER_EVIDENCE_TOOLS and tool_name not in self.ANSWER_EVIDENCE_TOOLS:
                continue

            raw_output = observation.get("raw_output", {})
            if not isinstance(raw_output, Mapping):
                raw_output = {}
            supported_option = _normalize_supported_option(
                raw_output.get("supported_option")
                or raw_output.get("supported_option_letter")
                or raw_output.get("answer_option")
                or _first_item(raw_output.get("supported_options"))
                or _supported_option_from_claim(str(observation.get("claim", ""))),
                option_map=option_map,
            )
            group_key = supported_option or "unassigned"
            groups.setdefault(group_key, [])

            row = {
                "obs_id": str(observation.get("observation_id", "")),
                "time_range": _observation_time_range(observation),
                "tool": tool_name,
                "supported_option": supported_option,
                "event_label": _observation_event_label(
                    raw_output=raw_output,
                    claim=str(observation.get("claim", "")),
                ),
                "claim": str(observation.get("claim", "")),
                "confidence": float(observation.get("confidence", 0.0) or 0.0),
                "grounding_quality": _grounding_quality(
                    raw_output=raw_output,
                    limitations=str(observation.get("limitations", "")),
                ),
                "limitations": str(observation.get("limitations", "")),
                "artifact": _first_item(observation.get("input_artifacts")) or "",
            }
            rows.append(row)
            groups[group_key].append(row)

        sorted_groups = {key: _sort_evidence_rows(value) for key, value in groups.items()}
        sorted_rows = [
            row
            for key in sorted(sorted_groups, key=_option_sort_key)
            for row in sorted_groups[key]
        ]
        return {
            "question": question,
            "options": list(options),
            "groups": sorted_groups,
            "rows": sorted_rows,
        }

    def _append_jsonl(self, filename: str, payload: Mapping[str, Any]) -> None:
        with (self.root / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
            handle.write("\n")

    def _read_observation_dicts(self) -> list[dict[str, Any]]:
        observations_path = self.root / "observations.jsonl"
        if not observations_path.exists():
            return []
        observations = []
        with observations_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    observations.append(payload)
        return observations

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


def _option_letter_map(options: Sequence[str]) -> dict[str, str]:
    mapping = {}
    for index, option in enumerate(options):
        text = str(option).strip()
        match = re.match(r"^([A-Za-z])(?:[\.)]\s*|\s+|$)", text)
        letter = match.group(1).upper() if match else chr(ord("A") + index)
        mapping[letter] = text
    return mapping


def _normalize_supported_option(value: Any, *, option_map: Mapping[str, str]) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    letter_match = re.search(r"\b(?:option\s*)?([A-Za-z])\b", text, flags=re.IGNORECASE)
    if letter_match:
        letter = letter_match.group(1).upper()
        if not option_map or letter in option_map:
            return letter
    for letter, option_text in option_map.items():
        if text == option_text or text.lower() in option_text.lower() or option_text.lower() in text.lower():
            return letter
    return None


def _supported_option_from_claim(claim: str) -> str | None:
    match = re.search(r"\b(?:supports?|matches|chooses?|answer(?:s)?|option)\s+(?:option\s*)?([A-Za-z])\b", claim)
    if match:
        return match.group(1).upper()
    return None


def _first_item(value: Any) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[0] if value else None
    return value


def _observation_time_range(observation: Mapping[str, Any]) -> list[float] | None:
    raw_output = observation.get("raw_output", {})
    if isinstance(raw_output, Mapping):
        time_range = raw_output.get("time_range")
        if isinstance(time_range, Sequence) and not isinstance(time_range, (str, bytes)) and len(time_range) >= 2:
            return [float(time_range[0]), float(time_range[1])]
        start = raw_output.get("start_sec")
        end = raw_output.get("end_sec")
        if start is not None and end is not None:
            return [float(start), float(end)]

    regions = observation.get("regions", [])
    if isinstance(regions, Sequence) and not isinstance(regions, (str, bytes)):
        for region in regions:
            if not isinstance(region, Mapping):
                continue
            start = region.get("start_sec")
            end = region.get("end_sec")
            if start is not None and end is not None:
                return [float(start), float(end)]
    return None


def _observation_event_label(*, raw_output: Mapping[str, Any], claim: str) -> str:
    for key in ["event_label", "event", "event_name", "sequence_item", "visible_event"]:
        value = _first_item(raw_output.get(key))
        if value is not None and str(value).strip():
            return str(value).strip()
    match = re.search(r"\bevent:\s*([^.;|]+)", claim, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _grounding_quality(*, raw_output: Mapping[str, Any], limitations: str) -> str:
    explicit = str(raw_output.get("grounding_quality", "")).strip().lower()
    if explicit in {"visually_confirmed", "inferred", "external_knowledge", "weak"}:
        return explicit

    text = limitations.lower()
    if "external knowledge" in text or "outside knowledge" in text or "world knowledge" in text:
        return "external_knowledge"
    if "infer" in text or "guess" in text or "deduc" in text or "assum" in text:
        return "inferred"
    weak_markers = [
        "not directly visible",
        "not visible",
        "lacks explicit",
        "lack explicit",
        "no explicit",
        "ambiguous",
        "unclear",
        "low resolution",
        "blur",
        "limited",
        "weak",
    ]
    if any(marker in text for marker in weak_markers):
        return "weak"
    return "visually_confirmed"


def _sort_evidence_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grounding_rank = {
        "visually_confirmed": 0,
        "inferred": 1,
        "weak": 2,
        "external_knowledge": 3,
    }
    return sorted(
        rows,
        key=lambda row: (
            grounding_rank.get(str(row.get("grounding_quality", "weak")), 9),
            -float(row.get("confidence", 0.0) or 0.0),
            str(row.get("obs_id", "")),
        ),
    )


def _option_sort_key(option: str) -> tuple[int, str]:
    if option == "unassigned":
        return (1, option)
    return (0, option)


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
