"""Sidecar workspace for parallel multi_v3 investigators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from visual_coding_agent_harness.contracts.query import ScopedQuery
from visual_coding_agent_harness.contracts.report import CandidateShot, Finding, InvestigationReport

from .evidence import EvidenceLedger


class InvestigatorWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.ledger = EvidenceLedger(self.root / "evidence_ledger.jsonl")

    def record_request(self, query: ScopedQuery) -> None:
        query_dir = self._query_dir(query.query_id)
        self._write_json(query_dir / "request.json", query.to_dict())
        self._trace("request_recorded", {"query_id": query.query_id})

    def record_explore(self, query_id: str, candidates: Sequence[CandidateShot]) -> None:
        query_dir = self._query_dir(query_id)
        self._write_json(query_dir / "explore.json", {"query_id": query_id, "candidates": [item.to_dict() for item in candidates]})
        self._trace("explore_recorded", {"query_id": query_id, "candidate_count": len(candidates)})

    def record_verify(self, query_id: str, shot_id: str, findings: Sequence[Finding]) -> None:
        query_dir = self._query_dir(query_id)
        path = query_dir / f"verify_{shot_id}.json"
        self._write_json(path, {"query_id": query_id, "shot_id": shot_id, "findings": [item.to_dict() for item in findings]})
        self._trace("verify_recorded", {"query_id": query_id, "shot_id": shot_id, "finding_count": len(findings)})

    def record_report(self, report: InvestigationReport) -> None:
        query_dir = self._query_dir(report.query_id)
        self._write_json(query_dir / "report.json", report.to_dict())
        self.ledger.extend(report.findings)
        self._merge_coverage(explored=report.explored_shots, verified=report.verified_shots)
        self._trace("report_recorded", {"query_id": report.query_id, "status": report.status})

    def _query_dir(self, query_id: str) -> Path:
        path = self.root / "queries" / query_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _merge_coverage(self, *, explored: Sequence[str], verified: Sequence[str]) -> None:
        path = self.root / "coverage.json"
        current = {"explored_shots": [], "verified_shots": []}
        if path.exists():
            current.update(json.loads(path.read_text(encoding="utf-8")))
        current["explored_shots"] = sorted(set(str(item) for item in current.get("explored_shots", [])) | set(explored))
        current["verified_shots"] = sorted(set(str(item) for item in current.get("verified_shots", [])) | set(verified))
        self._write_json(path, current)

    def _trace(self, event: str, payload: dict[str, object]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "trace.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, "payload": payload}, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
