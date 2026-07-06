from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from vcah.index import build_cold_index
from vcah.memory import AgentMemory, EvidenceStore, TraceStore
from vcah.model import ModelClient
from vcah.tools import AgentTools
from vcah.types import (
    Answer,
    ClaimVerdict,
    EvidenceRecord,
    InvestigatorOutputEmpty,
    InvestigatorOutputInvalid,
    QueryClaim,
    ToolAction,
    ToolResult,
    validate_investigator_input,
    validate_investigator_output,
    validate_reasoner_claims,
    verify_final_answer,
)
from vcah.video import probe_duration


class VideoAgent:
    def __init__(self, *, model: ModelClient | None = None, max_steps: int = 8) -> None:
        self.model = model or ModelClient()
        self.max_steps = max(1, int(max_steps))

    def ask(
        self,
        video_path: str,
        question: str,
        *,
        run_dir: Path,
        duration_sec: float | None = None,
        asr_cues: Sequence[Any] = (),
        ocr_lines: Sequence[Any] = (),
        range_detector: Any = None,
        keyframe_sampler: Any = None,
        index_mode: str = "fast",
    ) -> Answer:
        run_dir = Path(run_dir)
        run_artifacts = run_dir / "run"
        run_artifacts.mkdir(parents=True, exist_ok=True)
        duration = float(duration_sec) if duration_sec is not None else probe_duration(video_path)
        index = build_cold_index(
            video_path,
            duration_sec=duration,
            run_dir=run_dir,
            model=self.model,
            asr_cues=asr_cues,
            ocr_lines=ocr_lines,
            range_detector=range_detector,
            keyframe_sampler=keyframe_sampler,
            index_mode=index_mode,
        )
        memory = AgentMemory.empty(run_artifacts / "memory.json")
        memory.last_query = question
        evidence = EvidenceStore.empty(run_artifacts / "evidence.jsonl")
        trace = TraceStore(run_artifacts / "trace.jsonl")
        tools = AgentTools(index, memory, evidence, run_artifacts)

        for _step in range(self.max_steps):
            digest = index.timeline_digest(
                query=memory.last_query,
                open_claims=[claim.text for claim in memory.open_claims()],
                visited_beats=memory.visited_beats,
            )
            action = self.model.controller(question, digest, memory.digest(), evidence.digest())
            if not isinstance(action, ToolAction):
                action = ToolAction.from_mapping(action)
            validate_reasoner_claims(action.claims, options=_question_options(question))
            if action.type.startswith("search") and action.query:
                memory.last_query = action.query
            result = tools.run(replace(action, claims=()))
            new_evidence = _new_evidence(evidence, result)
            verdicts: tuple[ClaimVerdict, ...] = ()
            if action.claims:
                verdicts = tuple(self.model.verify(tuple(QueryClaim.from_claim(claim) for claim in action.claims), new_evidence))
                _validate_verdict_citations(evidence, verdicts)
                memory.update_ledger(action.claims, verdicts)
            if action.type == "answer":
                verification = _verify_answer_citations(evidence, action, question, memory.claim_ledger)
                result = replace(
                    result,
                    payload={
                        **dict(result.payload),
                        "final_verification": verification,
                        "investigator_received_hypothesis": bool(verification.get("investigator_received_hypothesis")),
                    },
                )
            memory.record_result(result)
            trace.append(action, result, new_evidence, verdicts)
            memory.save()
            if action.type == "answer":
                if result.payload.get("final_verification", {}).get("passed"):
                    return self._write_answer(run_artifacts, Answer(action.answer, action.citations, run_dir))
                return self._write_answer(run_artifacts, Answer("Insufficient verified evidence.", (), run_dir))

        return self._write_answer(run_artifacts, Answer("Insufficient verified evidence.", (), run_dir))

    def _write_answer(self, run_artifacts: Path, answer: Answer) -> Answer:
        (run_artifacts / "answer.json").write_text(
            json.dumps({"answer": answer.answer, "citations": list(answer.citations)}, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return answer


def _verify_answer_citations(
    evidence: EvidenceStore,
    action: ToolAction,
    question: str = "",
    claim_ledger: dict[str, tuple[object, object]] | None = None,
) -> dict[str, object]:
    if action.investigator_payload:
        try:
            validate_investigator_input(action.investigator_payload)
        except InvestigatorOutputInvalid:
            return {
                "passed": False,
                "reason": "investigator_input_contains_hypothesis",
                "investigator_received_hypothesis": True,
            }
    citations_valid = evidence.valid(action.citations)
    requires_table = bool(action.selected) or _looks_like_mcq(question)
    if not citations_valid:
        if not action.citations:
            return {"passed": False, "reason": "missing_citations", "citations_valid": False}
        return {"passed": False, "reason": "unknown_citations", "citations_valid": False}
    selected = action.selected or _selected_from_answer(action.answer)
    if selected and claim_ledger and any(getattr(claim, "option", "") for claim, _verdict in claim_ledger.values()):
        verification = verify_final_answer(question, {}, selected, claim_ledger=claim_ledger)  # type: ignore[arg-type]
        return {
            **verification,
            "citations_valid": True,
            "selected": selected,
            "investigator_received_hypothesis": False,
        }
    if requires_table and not action.evidence_table:
        return {"passed": False, "reason": "missing_evidence_table", "citations_valid": True}
    if action.evidence_table:
        options = _evidence_options(question, action.evidence_table, action.selected)
        try:
            validate_investigator_output(action.evidence_table, options=options)
        except (InvestigatorOutputEmpty, InvestigatorOutputInvalid) as exc:
            return {"passed": False, "reason": "invalid_evidence_table", "detail": str(exc), "citations_valid": True}
        verification = verify_final_answer(question, action.evidence_table, selected)
        return {
            **verification,
            "citations_valid": True,
            "selected": selected,
            "investigator_received_hypothesis": False,
        }
    return {"passed": True, "reason": None, "citations_valid": True, "investigator_received_hypothesis": False}


def _looks_like_mcq(question: str) -> bool:
    return bool(re.search(r"(?m)(^|\n)\s*[A-H][\).:]\s+\S+", question))


def _evidence_options(question: str, evidence_table: object, selected: str) -> tuple[str, ...]:
    labels = _question_options(question)
    if labels:
        return labels
    if isinstance(evidence_table, dict):
        table_labels = tuple(sorted(key for key in evidence_table if re.fullmatch(r"[A-H]", str(key))))
        if table_labels:
            return table_labels
    selected = selected.strip().upper()
    return (selected,) if selected else ("A", "B", "C", "D")


def _selected_from_answer(answer: str) -> str:
    match = re.search(r"\b([A-H])\b", answer.upper())
    return match.group(1) if match else ""


def _question_options(question: str) -> tuple[str, ...]:
    return tuple(sorted(set(re.findall(r"(?m)(?:^|\n)\s*([A-H])[\).:]\s+\S+", question))))


def _new_evidence(evidence: EvidenceStore, result: ToolResult) -> tuple[EvidenceRecord, ...]:
    if result.n_new <= 0:
        return ()
    return tuple(evidence.records[-result.n_new :])


def _validate_verdict_citations(evidence: EvidenceStore, verdicts: tuple[ClaimVerdict, ...]) -> None:
    for verdict in verdicts:
        if verdict.status == "unknown" and not verdict.citations:
            continue
        if not evidence.valid(verdict.citations):
            raise InvestigatorOutputInvalid(f"Verifier returned invalid citations for {verdict.claim_id}")
