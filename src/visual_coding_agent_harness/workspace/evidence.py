"""Append-only evidence ledger for multi_v3 investigations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from visual_coding_agent_harness.contracts.report import Finding


class EvidenceLedger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, finding: Finding) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(finding.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    def extend(self, findings: Iterable[Finding]) -> None:
        for finding in findings:
            self.append(finding)

    def read_all(self) -> list[Finding]:
        if not self.path.exists():
            return []
        findings = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            findings.append(Finding.from_dict(json.loads(line)))
        return findings


__all__ = ["EvidenceLedger"]
