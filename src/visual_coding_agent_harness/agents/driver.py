"""Parallel dispatch driver for the multi_v3 loop."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import re
from typing import Any, Mapping, Sequence

from visual_coding_agent_harness.contracts.evidence import EvidenceRecord
from visual_coding_agent_harness.contracts.query import ScopedQuery
from visual_coding_agent_harness.contracts.report import DigestItem, InvestigationReport
from visual_coding_agent_harness.workspace.investigator_ws import InvestigatorWorkspace
from visual_coding_agent_harness.workspace.investigator_ws import digest_reports
from visual_coding_agent_harness.workspace.video_workspace import VideoWorkspace


@dataclass(frozen=True)
class WorkspaceRunResult:
    answer: str
    citations: tuple[str, ...] = ()
    confidence: str = ""
    rounds: int = 0
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CounterCheckHit:
    option_id: str
    beat_id: str
    score: float
    verbatim: str


@dataclass(frozen=True)
class CounterCheckResult:
    status: str
    hits: tuple[CounterCheckHit, ...] = ()

    def __bool__(self) -> bool:
        return self.status == "needs_verify" and bool(self.hits)

    def __iter__(self):
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, index: int) -> CounterCheckHit:
        return self.hits[index]


class MultiV3Driver:
    def __init__(
        self,
        *,
        reasoner,
        investigator,
        workspace: InvestigatorWorkspace,
        max_rounds: int,
        max_concurrency: int = 4,
        valid_scene_ids: Sequence[str] = (),
        video_workspace: VideoWorkspace | None = None,
    ) -> None:
        self.reasoner = reasoner
        self.investigator = investigator
        self.workspace = workspace
        self.max_rounds = max(1, int(max_rounds))
        self.max_concurrency = max(1, int(max_concurrency))
        self.valid_scene_ids = {str(scene_id) for scene_id in valid_scene_ids if str(scene_id)}
        self.video_workspace = video_workspace

    def run(
        self,
        *,
        question: str,
        options: Mapping[str, str],
        index_context: str,
        overview_image_path: str = "",
    ) -> WorkspaceRunResult:
        previous_digest: tuple[DigestItem, ...] = ()
        last_reports: tuple[InvestigationReport, ...] = ()
        counter_check_blocks = 0
        for round_number in range(1, self.max_rounds + 1):
            decision = self.reasoner.decide(
                question=question,
                options=options,
                index_context=index_context,
                overview_image_path=overview_image_path,
                previous_digest=previous_digest,
                round_number=round_number,
            )
            if decision.action == "answer":
                citations = self._valid_citations(decision.citations, previous_digest=previous_digest)
                if not citations:
                    self.workspace.record_warning(
                        "answer_missing_citation",
                        {"proposed_answer": decision.answer, "citations": list(decision.citations)},
                    )
                    previous_digest = previous_digest + (
                        DigestItem(
                            query_id=f"warning_round_{round_number}",
                            goal_id="citation_gate",
                            status="partial",
                            summary="Previous answer lacked a valid evidence citation.",
                        ),
                    )
                    continue
                counter_result = (
                    counter_check_mcq(
                        workspace=self.video_workspace,
                        question=question,
                        options=options,
                        proposed_answer=decision.answer,
                        existing_evidence=tuple(self.workspace.evidence_records.read_all()),
                    )
                    if self.video_workspace is not None
                    else CounterCheckResult(status="clear")
                )
                if counter_result.status == "needs_verify" and counter_check_blocks < 1:
                    counter_check_blocks += 1
                    self.workspace.record_warning(
                        "counter_check_hits",
                        {
                            "status": counter_result.status,
                            "hits": [hit.__dict__ for hit in counter_result.hits],
                            "proposed_answer": decision.answer,
                        },
                    )
                    previous_digest = previous_digest + (
                        DigestItem(
                            query_id=f"counter_round_{round_number}",
                            goal_id="counter_check",
                            status="partial",
                            summary="Counter-check found competing option text; verify the competing option before answering.",
                        ),
                    )
                    continue
                if counter_result.status == "needs_verify":
                    self.workspace.record_warning(
                        "counter_check_repeated",
                        {
                            "status": counter_result.status,
                            "hits": [hit.__dict__ for hit in counter_result.hits],
                            "proposed_answer": decision.answer,
                        },
                    )
                return self._answer_result(decision, rounds=round_number, citations=citations)
            queries = self._filter_queries(decision.queries)
            if not queries:
                continue
            last_reports = self._dispatch_queries(queries)
            previous_digest = digest_reports(last_reports, query_goal_ids={query.query_id: query.goal_id for query in queries})
        return WorkspaceRunResult(
            answer="need_more_evidence",
            citations=tuple(citation for item in previous_digest for citation in item.citation_ids),
            confidence="low",
            rounds=self.max_rounds,
            metadata={
                "status": "need_more_evidence",
                "strategy": "multi_v3",
                "report_count": len(last_reports),
            },
        )

    def _dispatch_queries(self, queries: Sequence[ScopedQuery]) -> tuple[InvestigationReport, ...]:
        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(queries))) as executor:
            return tuple(executor.map(self.investigator.run, queries))

    def _answer_result(self, decision, *, rounds: int, citations: Sequence[str] | None = None) -> WorkspaceRunResult:
        valid_citations = tuple(citations) if citations is not None else self._valid_citations(decision.citations)
        return WorkspaceRunResult(
            answer=decision.answer,
            citations=valid_citations,
            confidence=decision.confidence,
            rounds=rounds,
            metadata={
                "status": "final" if decision.answer else "need_more_evidence",
                "strategy": "multi_v3",
                "rationale": decision.rationale,
                "goals": [goal.to_dict() for goal in decision.goals],
            },
        )

    def _valid_citations(self, citations: Sequence[str], previous_digest: Sequence[DigestItem] = ()) -> tuple[str, ...]:
        requested = tuple(str(item) for item in citations if str(item))
        if not requested:
            return ()
        valid = set()
        del previous_digest
        for record in self.workspace.evidence_records.read_all():
            valid.add(record.evidence_id)
        return tuple(item for item in requested if item in valid)

    def _filter_queries(self, queries: Sequence[ScopedQuery]) -> tuple[ScopedQuery, ...]:
        queries = tuple(queries)
        if not self.valid_scene_ids:
            return queries
        filtered_queries: list[ScopedQuery] = []
        dropped: list[dict[str, object]] = []
        trimmed: list[dict[str, object]] = []
        for query in queries:
            requested = tuple(str(scene_id) for scene_id in query.scope.scene_ids if str(scene_id))
            valid = tuple(scene_id for scene_id in requested if scene_id in self.valid_scene_ids)
            invalid = tuple(scene_id for scene_id in requested if scene_id not in self.valid_scene_ids)
            if not valid:
                dropped.append({"query_id": query.query_id, "invalid_scene_ids": list(invalid)})
                continue
            if invalid:
                trimmed.append({"query_id": query.query_id, "invalid_scene_ids": list(invalid), "kept_scene_ids": list(valid)})
                scope = type(query.scope)(
                    chapter_ids=valid,
                    time_range=query.scope.time_range,
                    entity_hints=query.scope.entity_hints,
                    modality_hint=query.scope.modality_hint,
                )
                filtered_queries.append(replace(query, scope=scope))
            else:
                filtered_queries.append(query)
        if dropped or trimmed:
            self.workspace.record_warning(
                "invalid_scene_ids_filtered",
                {"dropped": dropped, "trimmed": trimmed, "valid_scene_ids": sorted(self.valid_scene_ids)},
            )
        return tuple(filtered_queries)


def counter_check_mcq(
    *,
    workspace: VideoWorkspace,
    question: str,
    options: Mapping[str, str],
    proposed_answer: str,
    existing_evidence: Sequence[EvidenceRecord] = (),
) -> CounterCheckResult:
    del question
    hits: list[CounterCheckHit] = []
    option_text_by_id = {str(option_id): str(option_text) for option_id, option_text in options.items()}
    for option_id, option_text in options.items():
        if str(option_id) == str(proposed_answer):
            continue
        queries = [f'"{option_text}"', *_keywords(option_text)[:3]]
        seen: set[tuple[str, str]] = set()
        for query in queries:
            for hit in workspace.search_text(query):
                key = (str(option_id), hit.beat_id)
                if key in seen:
                    continue
                seen.add(key)
                beat = workspace.get_beat(hit.beat_id)
                verbatim = beat.asr_verbatim or " ".join(beat.ocr_verbatim)
                hits.append(CounterCheckHit(option_id=str(option_id), beat_id=hit.beat_id, score=hit.score, verbatim=verbatim))
    ordered = tuple(sorted(hits, key=lambda item: (-item.score, item.option_id, item.beat_id)))
    if not ordered:
        return CounterCheckResult(status="clear")
    unresolved = tuple(
        hit
        for hit in ordered
        if not _refuted_by_existing_evidence(hit, option_text_by_id.get(hit.option_id, ""), existing_evidence)
    )
    if not unresolved:
        return CounterCheckResult(status="refuted_by_existing_evidence")
    return CounterCheckResult(status="needs_verify", hits=unresolved)


def _refuted_by_existing_evidence(
    hit: CounterCheckHit,
    option_text: str,
    existing_evidence: Sequence[EvidenceRecord],
) -> bool:
    option_text_norm = str(option_text or "").casefold()
    keywords = _keywords(option_text_norm)
    for record in existing_evidence:
        if record.stance != "refutes":
            continue
        if record.beat_id and record.beat_id != hit.beat_id:
            continue
        text = f"{record.claim} {record.verbatim}".casefold()
        if hit.option_id.casefold() in text or (option_text_norm and option_text_norm in text):
            return True
        if keywords and all(keyword in text for keyword in keywords[:2]):
            return True
    return False


def _keywords(text: str) -> tuple[str, ...]:
    stop = {"the", "and", "that", "this", "with", "from", "which", "what", "when", "where", "jacket"}
    tokens = []
    seen = set()
    for match in re.finditer(r"\w+", str(text or "").casefold()):
        token = match.group(0)
        if len(token) <= 2 or token in stop or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tuple(tokens)
