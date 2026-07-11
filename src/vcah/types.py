from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable, Literal, Mapping, Sequence


@dataclass(frozen=True)
class Frame:
    frame_id: str
    time_sec: float
    path: str
    ocr_text: str = ""


@dataclass(frozen=True)
class Beat:
    beat_id: str
    chapter_id: str
    start_sec: float
    end_sec: float
    keyframe_path: str
    asr_text: str = ""
    ocr_text: tuple[str, ...] = ()
    frame_paths: tuple[str, ...] = ()
    frame_times: tuple[float, ...] = ()
    asr_cues: tuple[Mapping[str, Any], ...] = ()
    ocr_cues: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if float(self.end_sec) < float(self.start_sec):
            raise ValueError("Beat end_sec must be greater than or equal to start_sec")
        object.__setattr__(self, "start_sec", float(self.start_sec))
        object.__setattr__(self, "end_sec", float(self.end_sec))
        object.__setattr__(self, "ocr_text", tuple(str(item) for item in self.ocr_text if str(item).strip()))
        object.__setattr__(self, "frame_paths", tuple(str(item) for item in self.frame_paths if str(item).strip()))
        object.__setattr__(self, "frame_times", tuple(float(item) for item in self.frame_times))
        object.__setattr__(self, "asr_cues", tuple(_cue_mapping(item) for item in self.asr_cues))
        object.__setattr__(self, "ocr_cues", tuple(_cue_mapping(item) for item in self.ocr_cues))


@dataclass(frozen=True)
class Chapter:
    chapter_id: str
    start_sec: float
    end_sec: float
    beat_ids: tuple[str, ...]
    thumb_path: str = ""

    def __post_init__(self) -> None:
        if float(self.end_sec) < float(self.start_sec):
            raise ValueError("Chapter end_sec must be greater than or equal to start_sec")
        object.__setattr__(self, "start_sec", float(self.start_sec))
        object.__setattr__(self, "end_sec", float(self.end_sec))
        object.__setattr__(self, "beat_ids", tuple(str(item) for item in self.beat_ids if str(item).strip()))


@dataclass(frozen=True)
class Hit:
    beat_id: str
    score: float
    modality: Literal["text", "visual"]


@dataclass(frozen=True)
class IndexDiagnostics:
    duration_sec: float
    chapter_count: int
    beat_count: int
    median_beat_sec: float
    max_beat_sec: float
    visual_index_dim: int
    visual_embedding_norm_mean: float
    embedding_backend: str
    index_mode: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoverageSegment:
    request_id: str
    start_sec: float
    end_sec: float
    modality: str = ""
    coverage: float = 1.0

    def __post_init__(self) -> None:
        start = float(self.start_sec)
        end = float(self.end_sec)
        if end < start:
            raise ValueError("CoverageSegment end_sec must be greater than or equal to start_sec")
        object.__setattr__(self, "request_id", str(self.request_id or "").strip())
        object.__setattr__(self, "start_sec", start)
        object.__setattr__(self, "end_sec", end)
        object.__setattr__(self, "modality", str(self.modality or "").strip())
        object.__setattr__(self, "coverage", max(0.0, min(1.0, float(self.coverage))))


@dataclass(frozen=True)
class ClaimContract:
    required_scope: Literal["local", "window", "multi_window", "full_video"] = "window"
    quantifier: Literal[
        "none",
        "existential",
        "universal",
        "distinct_count",
        "total_count",
        "scalar_quantity",
        "order",
        "comparison",
    ] = "none"
    observation_target: Literal["text", "entity", "object", "event", "action", "relation", "attribute"] = "text"
    aggregation: Literal["none", "deduplicate", "count", "accumulate", "order", "compare", "summarize"] = "none"
    required_observability: tuple[Literal["asr", "ocr", "visual"], ...] = ()
    observability_mode: Literal["all", "any"] = "all"
    measurement_unit: str = ""
    boundary_hint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_observability",
            tuple(item for item in self.required_observability if item in {"asr", "ocr", "visual"}),
        )
        if self.observability_mode not in {"all", "any"}:
            object.__setattr__(self, "observability_mode", "all")
        object.__setattr__(self, "measurement_unit", str(self.measurement_unit or "").strip().casefold())
        object.__setattr__(self, "boundary_hint", str(self.boundary_hint or "").strip())


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    beat_id: str
    start_sec: float | None
    end_sec: float | None
    modality: Literal["asr", "ocr", "visual", "frame", "derived"]
    pointer: str
    verbatim: str
    claim: str = ""
    frame_refs: tuple[str, ...] = ()
    attestation_model: str = ""
    temporal_scope: Literal["local_frame", "window", "multi_window", "full_video", "workspace"] = "window"
    evidence_kind: Literal[
        "quote",
        "visual_observation",
        "entity_observation",
        "event_observation",
        "claim_verification",
        "navigation_hint",
        "aggregate",
        "summary",
    ] = "quote"
    observation_polarity: Literal["positive", "negative", "unknown"] = "unknown"
    sampling_coverage: Literal["sparse", "dense", "exact", "complete_for_manifest", "unknown"] = "unknown"
    parent_evidence_ids: tuple[str, ...] = ()
    request_ids: tuple[str, ...] = ()
    coverage_manifest: tuple[CoverageSegment, ...] = ()
    task_id: str = ""
    observation_id: str = ""
    sampling_fps: float = 0.0
    confidence: float = 0.0
    source_lineage: tuple[Mapping[str, Any], ...] = ()
    entity_ids: tuple[str, ...] = ()
    operation_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        modality = "visual" if self.modality == "frame" else self.modality
        object.__setattr__(self, "modality", modality)
        start = None if self.start_sec is None else float(self.start_sec)
        end = None if self.end_sec is None else float(self.end_sec)
        if start is not None and end is not None and end < start:
            raise ValueError("EvidenceRecord end_sec must be greater than or equal to start_sec")
        object.__setattr__(self, "start_sec", start)
        object.__setattr__(self, "end_sec", end)
        object.__setattr__(self, "frame_refs", tuple(str(item) for item in self.frame_refs if str(item).strip()))
        object.__setattr__(
            self,
            "parent_evidence_ids",
            tuple(str(item) for item in self.parent_evidence_ids if str(item).strip()),
        )
        object.__setattr__(self, "request_ids", tuple(str(item) for item in self.request_ids if str(item).strip()))
        object.__setattr__(
            self,
            "coverage_manifest",
            tuple(_coverage_segment(item) for item in self.coverage_manifest),
        )
        object.__setattr__(self, "task_id", str(self.task_id or "").strip())
        object.__setattr__(self, "observation_id", str(self.observation_id or "").strip())
        object.__setattr__(self, "sampling_fps", max(0.0, float(self.sampling_fps or 0.0)))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence or 0.0))))
        object.__setattr__(self, "source_lineage", tuple(dict(item) for item in self.source_lineage))
        object.__setattr__(self, "entity_ids", tuple(str(item) for item in self.entity_ids if str(item).strip()))
        object.__setattr__(self, "operation_metadata", dict(self.operation_metadata or {}))
        if modality == "visual" and self.evidence_kind == "quote":
            object.__setattr__(self, "evidence_kind", "visual_observation")
        if (
            modality in {"asr", "ocr"}
            and self.evidence_kind == "quote"
            and self.sampling_coverage == "unknown"
            and _manifest_covers_modality(self.coverage_manifest, modality)
        ):
            object.__setattr__(self, "sampling_coverage", "complete_for_manifest")
        if modality == "derived" and not self.parent_evidence_ids:
            raise ValueError("Derived evidence requires parent_evidence_ids")
        if modality == "derived" and not self.coverage_manifest:
            raise ValueError("Derived evidence requires coverage_manifest")


IMAGE_PATH_PATTERN = re.compile(r"\.(?:jpe?g|png|webp|bmp)(?:$|\b)", re.IGNORECASE)


def is_path_only_visual_evidence(record: EvidenceRecord) -> bool:
    if record.modality != "visual":
        return False
    text = str(record.verbatim or "").strip()
    if not text:
        return True
    if not str(record.attestation_model or "").strip() and IMAGE_PATH_PATTERN.search(text):
        return True
    return _looks_like_image_path_only(text)


def _looks_like_image_path_only(text: str) -> bool:
    tokens = [token.strip(" ,;:()[]{}<>\"'") for token in str(text or "").split()]
    tokens = [token for token in tokens if token]
    return bool(tokens) and all(IMAGE_PATH_PATTERN.search(token) for token in tokens)


@dataclass(frozen=True)
class Claim:
    claim_id: str
    option: str
    text: str
    polarity: Literal["assert", "negate"] = "assert"
    contract: ClaimContract | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", str(self.claim_id).strip())
        object.__setattr__(self, "option", str(self.option or "").strip().upper())
        object.__setattr__(self, "text", str(self.text or "").strip())
        object.__setattr__(self, "polarity", "negate" if self.polarity == "negate" else "assert")
        if self.contract is not None and not isinstance(self.contract, ClaimContract):
            object.__setattr__(self, "contract", _claim_contract(self.contract))
        validate_investigator_input({"claim": self.text})


@dataclass(frozen=True)
class QueryClaim:
    claim_id: str
    text: str

    @classmethod
    def from_claim(cls, claim: Claim) -> "QueryClaim":
        return cls(claim.claim_id, sanitize_query_claim_text(claim.text))


@dataclass(frozen=True)
class ClaimVerdict:
    claim_id: str
    status: Literal["supported", "contradicted", "unknown"]
    support_evidence_ids: tuple[str, ...] = ()
    contradict_evidence_ids: tuple[str, ...] = ()
    entailment_kind: Literal["direct", "derived", "absence", "proxy", "none"] = "none"
    capability_checks: tuple[str, ...] = ()
    reason: str | None = None
    source: Literal["verifier"] = "verifier"

    def __post_init__(self) -> None:
        status = str(self.status or "unknown")
        if status not in {"supported", "contradicted", "unknown"}:
            status = "unknown"
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "support_evidence_ids",
            tuple(str(item) for item in self.support_evidence_ids if str(item).strip()),
        )
        object.__setattr__(
            self,
            "contradict_evidence_ids",
            tuple(str(item) for item in self.contradict_evidence_ids if str(item).strip()),
        )
        object.__setattr__(
            self,
            "capability_checks",
            tuple(str(item) for item in self.capability_checks if str(item).strip()),
        )
        object.__setattr__(self, "source", "verifier")

    @property
    def citations(self) -> tuple[str, ...]:
        return self.support_evidence_ids if self.status == "supported" else self.contradict_evidence_ids


@dataclass(frozen=True)
class Window:
    start_sec: float
    end_sec: float
    request_id: str = ""

    def __post_init__(self) -> None:
        start = float(self.start_sec)
        end = float(self.end_sec)
        if end < start:
            raise ValueError("Window end_sec must be greater than or equal to start_sec")
        object.__setattr__(self, "start_sec", start)
        object.__setattr__(self, "end_sec", end)
        object.__setattr__(self, "request_id", str(self.request_id or "").strip())


@dataclass(frozen=True)
class WindowRequest:
    request_id: str
    start_sec: float
    end_sec: float
    modalities: tuple[Literal["asr", "ocr", "frames"], ...] = ()

    def __post_init__(self) -> None:
        start = float(self.start_sec)
        end = float(self.end_sec)
        if end < start:
            raise ValueError("WindowRequest end_sec must be greater than or equal to start_sec")
        object.__setattr__(self, "request_id", str(self.request_id or "").strip())
        object.__setattr__(self, "start_sec", start)
        object.__setattr__(self, "end_sec", end)
        object.__setattr__(
            self,
            "modalities",
            tuple(item for item in self.modalities if item in {"asr", "ocr", "frames"}),
        )


@dataclass(frozen=True)
class WindowCoverage:
    requested: Window
    actuals: tuple[Window, ...]
    coverage: float
    passed: bool


@dataclass(frozen=True)
class SelectionPolicy:
    mode: Literal["choose_supported", "choose_contradicted", "choose_best_score"] = "choose_supported"

    def __post_init__(self) -> None:
        if self.mode not in {"choose_supported", "choose_contradicted", "choose_best_score"}:
            object.__setattr__(self, "mode", "choose_supported")


def window_overlap_ratio(requested: Window, actuals: Iterable[Window]) -> float:
    requested_len = requested.end_sec - requested.start_sec
    if requested_len <= 0:
        return 0.0
    intervals: list[tuple[float, float]] = []
    for actual in actuals:
        overlap_start = max(requested.start_sec, actual.start_sec)
        overlap_end = min(requested.end_sec, actual.end_sec)
        if overlap_end > overlap_start:
            intervals.append((overlap_start, overlap_end))
    intervals.sort()
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    covered = sum(end - start for start, end in merged)
    return min(1.0, covered / requested_len)


@dataclass(frozen=True)
class ToolAction:
    type: str
    query: str = ""
    beat_id: str = ""
    beat_ids: tuple[str, ...] = ()
    windows: tuple[Window, ...] = ()
    modalities: tuple[Literal["asr", "ocr", "frames"], ...] = ()
    selected: str = ""
    evidence_table: Mapping[str, Any] = field(default_factory=dict)
    investigator_payload: Mapping[str, Any] = field(default_factory=dict)
    answer: str = ""
    citations: tuple[str, ...] = ()
    claims: tuple[Claim, ...] = ()
    raw_window_count: int = 0
    parsed_window_count: int = 0
    window_parse_errors: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ToolAction":
        citations = payload.get("citations") or ()
        if isinstance(citations, str):
            citations = (citations,)
        beat_ids = payload.get("beat_ids") or ()
        if isinstance(beat_ids, str):
            beat_ids = (beat_ids,)
        windows, raw_window_count, window_parse_errors = _parse_windows_with_report(payload)
        modalities = payload.get("modalities") or ()
        if isinstance(modalities, str):
            modalities = (modalities,)
        return cls(
            type=str(payload.get("type") or payload.get("tool") or ""),
            query=str(payload.get("query") or ""),
            beat_id=str(payload.get("beat_id") or ""),
            beat_ids=tuple(str(item) for item in beat_ids),
            windows=windows,
            modalities=tuple(_normalize_modality(item) for item in modalities if _normalize_modality(item)),
            selected=str(payload.get("selected") or payload.get("option") or ""),
            evidence_table=_mapping(payload.get("evidence_table")),
            investigator_payload=_mapping(payload.get("investigator_payload")),
            answer=str(payload.get("answer") or ""),
            citations=tuple(str(item) for item in citations),
            claims=_parse_claims(payload.get("claims") or ()),
            raw_window_count=raw_window_count,
            parsed_window_count=len(windows),
            window_parse_errors=window_parse_errors,
        )


@dataclass(frozen=True)
class ToolResult:
    tool: str
    beat_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    text: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    n_new: int = 0


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: tuple[str, ...]
    run_dir: Path | None = None


class InvestigatorOutputEmpty(ValueError):
    pass


class InvestigatorOutputInvalid(ValueError):
    pass


HYPOTHESIS_KEYS = {
    "hypothesis",
    "likely_answer",
    "predicted_option",
    "reasoner_guess",
    "candidate_answer",
    "initial_answer",
    "answer_hypothesis",
}
HYPOTHESIS_PATTERNS = (
    re.compile(r"\blikely answer\b", re.IGNORECASE),
    re.compile(r"\banswer hypothesis\b", re.IGNORECASE),
    re.compile(r"\bhypothesis\s*:", re.IGNORECASE),
    re.compile(r"\boption\s+[A-H]\b", re.IGNORECASE),
    re.compile(r"选项\s*[A-H]", re.IGNORECASE),
    re.compile(r"\bverify option\s+[A-H]\b", re.IGNORECASE),
)

OPTION_JUDGMENT_PATTERNS = (
    re.compile(r"\b(option|answer)\s+[A-H]\s+(?:is\s+)?(?:supported|contradicted|correct|incorrect)\b", re.IGNORECASE),
    re.compile(r"\b(?:supports?|contradicts?|refutes?)\s+option\s+[A-H]\b", re.IGNORECASE),
    re.compile(r"\btherefore\s+(?:the\s+)?answer\s+(?:is\s+)?[A-H]\b", re.IGNORECASE),
)


def investigator_input_has_hypothesis(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in HYPOTHESIS_KEYS:
                return True
            if investigator_input_has_hypothesis(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(investigator_input_has_hypothesis(item) for item in value)
    if isinstance(value, str):
        return any(pattern.search(value) for pattern in HYPOTHESIS_PATTERNS)
    return False


def validate_investigator_input(payload: Mapping[str, Any]) -> None:
    if investigator_input_has_hypothesis(payload):
        raise InvestigatorOutputInvalid("Investigator input contains answer hypothesis")


def sanitize_query_claim_text(text: str) -> str:
    sanitized = str(text or "")
    sanitized = re.sub(r"\b(?:verify|support|contradict|refute)\s+option\s+[A-H]\b", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\b(?:option|answer)\s+[A-H]\b", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\b(?:likely|predicted|selected|correct|incorrect)\s+(?:answer|option)\b", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def contains_option_judgment(text: str) -> bool:
    return any(pattern.search(str(text or "")) for pattern in OPTION_JUDGMENT_PATTERNS)


AGGREGATE_CLAIM_PATTERN = re.compile(r"\b(count|total|all|every|never)\b", re.IGNORECASE)
CLAIM_ANCHOR_PATTERN = re.compile(r"\b(?:ev_\d{4,}|bt\d{5,}|beat\s+bt\d{5,})\b", re.IGNORECASE)


def validate_reasoner_claims(claims: Sequence[Claim], *, options: Sequence[str] = ()) -> None:
    option_labels = tuple(str(option).strip().upper() for option in options if str(option).strip())
    counts = {option: 0 for option in option_labels}
    for claim in claims:
        validate_investigator_input({"claim": claim.text})
        if AGGREGATE_CLAIM_PATTERN.search(claim.text) and not CLAIM_ANCHOR_PATTERN.search(claim.text):
            raise InvestigatorOutputInvalid("Aggregate-sensitive claim must cite an evidence_id or beat_id anchor")
        if claim.option:
            counts.setdefault(claim.option, 0)
            counts[claim.option] += 1
    if len(counts) >= 2 and max(counts.values()) - min(counts.values()) > 1:
        raise InvestigatorOutputInvalid("Reasoner claim counts differ by more than one across options")


def validate_evidence_record(record: EvidenceRecord) -> None:
    if not str(record.evidence_id or "").strip():
        raise InvestigatorOutputInvalid("EvidenceRecord requires evidence_id")
    if not str(record.verbatim or "").strip():
        raise InvestigatorOutputInvalid("EvidenceRecord.verbatim must be non-empty")
    if contains_option_judgment(record.verbatim):
        raise InvestigatorOutputInvalid("EvidenceRecord contains option-level judgment")
    validate_investigator_input({"evidence": record.verbatim})
    if is_path_only_visual_evidence(record):
        raise InvestigatorOutputInvalid("Path-only visual evidence is not an observation")


def validate_investigator_output(output: Mapping[str, Any] | None, *, options: Sequence[str] = ("A", "B", "C", "D")) -> None:
    if not output:
        raise InvestigatorOutputEmpty("Investigator output is empty")
    if "evidence" in output:
        raw_evidence = output.get("evidence")
        if not isinstance(raw_evidence, Sequence) or isinstance(raw_evidence, (str, bytes)):
            raise InvestigatorOutputInvalid("Investigator evidence must be a list")
        if not raw_evidence:
            raise InvestigatorOutputEmpty("Investigator evidence is empty")
        for item in raw_evidence:
            record = item if isinstance(item, EvidenceRecord) else _evidence_record(item)
            validate_evidence_record(record)
        forbidden = {"support", "contradict", "status", "entailment_kind", "option_score", "likely_answer"}
        if any(key in output for key in forbidden):
            raise InvestigatorOutputInvalid("Investigator output mixes evidence with option-level judgments")
        return
    if any(str(key).strip().upper() in {str(option).strip().upper() for option in options} for key in output):
        raise InvestigatorOutputInvalid("Legacy option-level investigator output is not allowed")
    valid_statuses = {"supported", "contradicted", "unknown"}
    for option in options:
        item = output.get(option)
        if not isinstance(item, Mapping):
            raise InvestigatorOutputInvalid(f"Missing evidence table entry for option {option}")
        if item.get("status") not in valid_statuses:
            raise InvestigatorOutputInvalid(f"Invalid evidence status for option {option}")
        for bucket in ("support", "contradict"):
            entries = item.get(bucket)
            if not isinstance(entries, list):
                raise InvestigatorOutputInvalid(f"Option {option} {bucket} must be a list")


def verify_final_answer(
    question: str,
    evidence_table: Mapping[str, Any],
    selected: str,
    *,
    claim_ledger: Mapping[str, tuple[Claim, ClaimVerdict]] | None = None,
    threshold: float = 0.34,
    selection_policy: SelectionPolicy | None = None,
) -> dict[str, Any]:
    if claim_ledger:
        policy = selection_policy or _selection_policy_for_question(question)
        return verify_claim_ledger_answer(claim_ledger, selected, threshold=threshold, selection_policy=policy)
    selected = str(selected).strip().upper()
    if not selected:
        return {"passed": False, "reason": "missing_selected_option"}
    entry = evidence_table.get(selected)
    if not isinstance(entry, Mapping):
        return {"passed": False, "reason": "missing_selected_option_evidence"}
    polarity = "negative" if _is_negative_question(question) else "positive"
    expected = "contradicted" if polarity == "negative" else "supported"
    if entry.get("status") != expected:
        return {"passed": False, "reason": f"selected_option_not_{expected}", "polarity": polarity}
    return {"passed": True, "reason": None, "polarity": polarity}


def score_claim_ledger(claim_ledger: Mapping[str, tuple[Claim, ClaimVerdict]]) -> dict[str, float]:
    return _score_claim_ledger(claim_ledger, mode="choose_supported")


def _score_claim_ledger(
    claim_ledger: Mapping[str, tuple[Claim, ClaimVerdict]],
    *,
    mode: Literal["choose_supported", "choose_contradicted", "choose_best_score"],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for claim, verdict in claim_ledger.values():
        if not claim.option:
            continue
        score = _claim_sign(claim, verdict)
        if mode == "choose_contradicted":
            score = -score
        elif mode == "choose_best_score":
            score = abs(score)
        totals[claim.option] = totals.get(claim.option, 0.0) + score
        counts[claim.option] = counts.get(claim.option, 0) + 1
    return {option: totals.get(option, 0.0) / max(1, counts.get(option, 0)) for option in sorted(counts)}


def verify_claim_ledger_answer(
    claim_ledger: Mapping[str, tuple[Claim, ClaimVerdict]],
    selected: str,
    *,
    threshold: float = 0.34,
    selection_policy: SelectionPolicy | None = None,
) -> dict[str, Any]:
    selected = str(selected).strip().upper()
    if not selected:
        return {"passed": False, "reason": "missing_selected_option"}
    policy = selection_policy or SelectionPolicy()
    scores = _score_claim_ledger(claim_ledger, mode=policy.mode)
    if not scores:
        return {"passed": False, "reason": "missing_claim_ledger", "scores": scores}
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    winner, top_score = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    tied = len(ordered) > 1 and ordered[1][1] == top_score
    margin = top_score - runner_up
    if tied or margin < float(threshold):
        return {
            "passed": False,
            "reason": "insufficient_verified_evidence",
            "winner": winner,
            "selected": selected,
            "scores": scores,
            "margin": margin,
            "threshold": float(threshold),
            "selection_policy": policy.mode,
        }
    if selected != winner:
        return {
            "passed": False,
            "reason": "selected_option_not_top_scoring",
            "winner": winner,
            "selected": selected,
            "scores": scores,
            "margin": margin,
            "threshold": float(threshold),
            "selection_policy": policy.mode,
        }
    return {
        "passed": True,
        "reason": None,
        "winner": winner,
        "selected": selected,
        "scores": scores,
        "margin": margin,
        "threshold": float(threshold),
        "selection_policy": policy.mode,
    }


def to_jsonable(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: to_jsonable(item)
            for key, item in asdict(value).items()
            if not (key == "request_id" and not str(item or "").strip())
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
            if not (str(key) == "request_id" and not str(item or "").strip())
        }
    return value


def _parse_windows(payload: Mapping[str, Any]) -> tuple[Window, ...]:
    windows, _raw_count, _errors = _parse_windows_with_report(payload)
    return windows


def _parse_windows_with_report(payload: Mapping[str, Any]) -> tuple[tuple[Window, ...], int, tuple[str, ...]]:
    raw_windows = payload.get("windows") or payload.get("inspect_windows") or ()
    windows: list[Window] = []
    errors: list[str] = []
    raw_count = 0
    if payload.get("start_sec") is not None or payload.get("end_sec") is not None:
        raw_count += 1
        try:
            windows.append(
                Window(
                    _parse_seconds(payload.get("start_sec")),
                    _parse_seconds(payload.get("end_sec")),
                    str(payload.get("request_id") or ""),
                )
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"inline:{exc}")
    if isinstance(raw_windows, Mapping):
        raw_windows = (raw_windows,)
    for index, item in enumerate(raw_windows, start=1):
        raw_count += 1
        if not isinstance(item, Mapping):
            errors.append(f"window_{index}:not_mapping")
            continue
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        if start is None or end is None:
            errors.append(f"window_{index}:missing_start_or_end")
            continue
        try:
            windows.append(Window(_parse_seconds(start), _parse_seconds(end), str(item.get("request_id") or item.get("id") or "")))
        except (TypeError, ValueError) as exc:
            errors.append(f"window_{index}:{exc}")
    return tuple(windows), raw_count, tuple(errors)


def _parse_claims(value: Any) -> tuple[Claim, ...]:
    if isinstance(value, Mapping):
        value = (value,)
    claims = []
    for item in value:
        if isinstance(item, Claim):
            claims.append(item)
        elif isinstance(item, Mapping):
            claims.append(
                Claim(
                    claim_id=str(item.get("claim_id") or item.get("id") or ""),
                    option=str(item.get("option") or ""),
                    text=str(item.get("text") or item.get("claim") or ""),
                    polarity="negate" if item.get("polarity") == "negate" else "assert",
                    contract=_claim_contract(item.get("contract")) if item.get("contract") else None,
                )
            )
    return tuple(claims)


def _parse_seconds(value: Any) -> float:
    if value is None:
        raise ValueError("Window start_sec/end_sec is required")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" not in text:
        return float(text)
    parts = [float(part) for part in text.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    raise ValueError(f"Invalid time value: {value!r}")


def _normalize_modality(value: object) -> Literal["asr", "ocr", "frames"] | None:
    text = str(value).casefold()
    if text in {"asr", "ocr", "frames"}:
        return text  # type: ignore[return-value]
    if text == "frame":
        return "frames"
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _coverage_segment(value: Any) -> CoverageSegment:
    if isinstance(value, CoverageSegment):
        return value
    if isinstance(value, Mapping):
        return CoverageSegment(
            request_id=str(value.get("request_id") or ""),
            start_sec=float(value.get("start_sec", value.get("start", 0.0)) or 0.0),
            end_sec=float(value.get("end_sec", value.get("end", value.get("start_sec", 0.0))) or 0.0),
            modality=str(value.get("modality") or ""),
            coverage=float(value.get("coverage", 1.0) or 0.0),
        )
    raise ValueError("Invalid coverage segment")


def _claim_contract(value: Any) -> ClaimContract:
    if isinstance(value, ClaimContract):
        return value
    payload = _mapping(value)
    observability = payload.get("required_observability") or ()
    if isinstance(observability, str):
        observability = (observability,)
    return ClaimContract(
        required_scope=str(payload.get("required_scope") or "window"),  # type: ignore[arg-type]
        quantifier=str(payload.get("quantifier") or "none"),  # type: ignore[arg-type]
        observation_target=str(payload.get("observation_target") or "text"),  # type: ignore[arg-type]
        aggregation=str(payload.get("aggregation") or "none"),  # type: ignore[arg-type]
        required_observability=tuple(observability),  # type: ignore[arg-type]
        observability_mode=str(payload.get("observability_mode") or "all"),  # type: ignore[arg-type]
        measurement_unit=str(payload.get("measurement_unit") or ""),
        boundary_hint=str(payload.get("boundary_hint") or ""),
    )


def _evidence_record(value: Any) -> EvidenceRecord:
    if isinstance(value, EvidenceRecord):
        return value
    payload = _mapping(value)
    allowed = {
        "attestation_model",
        "beat_id",
        "claim",
        "confidence",
        "coverage_manifest",
        "end_sec",
        "entity_ids",
        "evidence_id",
        "evidence_kind",
        "frame_refs",
        "id",
        "modality",
        "observation_polarity",
        "observation_id",
        "operation_metadata",
        "parent_evidence_ids",
        "pointer",
        "request_ids",
        "sampling_coverage",
        "sampling_fps",
        "source_lineage",
        "start_sec",
        "temporal_scope",
        "task_id",
        "text",
        "verbatim",
    }
    unknown = sorted(str(key) for key in payload if str(key) not in allowed)
    if unknown:
        raise InvestigatorOutputInvalid(f"Unknown evidence item keys: {', '.join(unknown)}")
    return EvidenceRecord(
        evidence_id=str(payload.get("evidence_id") or payload.get("id") or ""),
        beat_id=str(payload.get("beat_id") or ""),
        start_sec=payload.get("start_sec"),
        end_sec=payload.get("end_sec"),
        modality=str(payload.get("modality") or "visual"),  # type: ignore[arg-type]
        pointer=str(payload.get("pointer") or ""),
        verbatim=str(payload.get("verbatim") or payload.get("text") or ""),
        claim=str(payload.get("claim") or ""),
        frame_refs=tuple(payload.get("frame_refs") or ()),
        attestation_model=str(payload.get("attestation_model") or ""),
        temporal_scope=str(payload.get("temporal_scope") or "window"),  # type: ignore[arg-type]
        evidence_kind=str(payload.get("evidence_kind") or "quote"),  # type: ignore[arg-type]
        observation_polarity=str(payload.get("observation_polarity") or "unknown"),  # type: ignore[arg-type]
        sampling_coverage=str(payload.get("sampling_coverage") or "unknown"),  # type: ignore[arg-type]
        parent_evidence_ids=tuple(payload.get("parent_evidence_ids") or ()),
        request_ids=tuple(payload.get("request_ids") or ()),
        coverage_manifest=tuple(payload.get("coverage_manifest") or ()),
        task_id=str(payload.get("task_id") or ""),
        observation_id=str(payload.get("observation_id") or ""),
        sampling_fps=float(payload.get("sampling_fps", 0.0) or 0.0),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        source_lineage=tuple(payload.get("source_lineage") or ()),
        entity_ids=tuple(payload.get("entity_ids") or ()),
        operation_metadata=dict(payload.get("operation_metadata") or {}),
    )


def _cue_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return {
            "start_sec": float(value.get("start_sec", value.get("start", value.get("time_sec", value.get("time", 0.0)))) or 0.0),
            "end_sec": float(value.get("end_sec", value.get("end", value.get("time_sec", value.get("time", 0.0)))) or 0.0),
            "text": str(value.get("text", "") or ""),
        }
    return _mapping(value)


def _is_negative_question(question: str) -> bool:
    return bool(re.search(r"\b(not correct|incorrect|false|not true|except)\b", question, re.IGNORECASE))


def _selection_policy_for_question(question: str) -> SelectionPolicy:
    return SelectionPolicy("choose_contradicted" if _is_negative_question(question) else "choose_supported")


def _claim_sign(claim: Claim, verdict: ClaimVerdict) -> float:
    if verdict.status == "unknown":
        return 0.0
    supported = verdict.status == "supported"
    assertive = claim.polarity == "assert"
    if supported and assertive:
        return 1.0
    if (not supported) and assertive:
        return -1.0
    if supported and not assertive:
        return -1.0
    return 1.0


def _manifest_covers_modality(segments: Sequence[CoverageSegment], modality: str) -> bool:
    return any(segment.modality == modality and segment.coverage > 0 for segment in segments)
