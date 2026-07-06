from __future__ import annotations

import hashlib
import os
from typing import Any, Mapping, Sequence

import numpy as np

from vcah.types import ClaimVerdict, EvidenceRecord, QueryClaim, ToolAction, is_path_only_visual_evidence


ATTESTATION_PROMPT = (
    "Return a JSON list of atomic visual observations. Each item must be one sentence or less, "
    "describe only visible facts such as people, text, objects, actions, or scene details, and avoid inference."
)


class ModelClient:
    """Thin model wrapper configured by environment variables."""

    embedding_dim = 8

    def __init__(self) -> None:
        self.controller_model = os.getenv("VCAH_CONTROLLER_MODEL", "local-scripted")
        self.vision_model = os.getenv("VCAH_VISION_MODEL", "local-placeholder")
        self.verifier_model = os.getenv("VCAH_VERIFIER_MODEL", self.controller_model)
        self.embed_model = os.getenv("VCAH_EMBED_MODEL", "local-hash")
        self.transcribe_model = os.getenv("VCAH_TRANSCRIBE_MODEL", "none")
        self.allow_placeholder_visual = os.getenv("VCAH_ALLOW_PLACEHOLDER_VISUAL", "").casefold() in {"1", "true", "yes"}
        self.last_verify_claims: tuple[QueryClaim, ...] = ()
        self.last_verify_evidence: tuple[EvidenceRecord, ...] = ()

    def controller(self, question: str, index_digest: str, memory_digest: str, evidence_digest: str) -> ToolAction:
        del index_digest
        if not memory_digest:
            return ToolAction(type="search_text", query=question)
        if "ev_" in evidence_digest:
            first_id = evidence_digest.split()[0]
            return ToolAction(type="answer", answer="See verified evidence.", citations=(first_id,))
        return ToolAction(type="focus_clip", beat_id="")

    def vision(self, image_paths: Sequence[str], prompt: str) -> str:
        return "\n".join(self.attest(image_paths, prompt))

    def attest(self, image_paths: Sequence[str], prompt: str) -> tuple[str, ...]:
        del image_paths, prompt
        return ()

    def verify(self, query_claims: Sequence[QueryClaim], evidence: Sequence[EvidenceRecord]) -> tuple[ClaimVerdict, ...]:
        self.last_verify_claims = tuple(query_claims)
        self.last_verify_evidence = tuple(evidence)
        verdicts = []
        for claim in query_claims:
            claim_tokens = set(_tokens(claim.text))
            best: EvidenceRecord | None = None
            best_overlap = 0
            for record in evidence:
                if is_path_only_visual_evidence(record):
                    continue
                overlap = len(claim_tokens & set(_tokens(record.verbatim)))
                if overlap > best_overlap:
                    best = record
                    best_overlap = overlap
            if best is not None and best_overlap >= max(1, int(len(claim_tokens) * 0.35)):
                verdicts.append(ClaimVerdict(claim.claim_id, "supported", (best.evidence_id,)))
            else:
                verdicts.append(ClaimVerdict(claim.claim_id, "unknown", ()))
        return tuple(verdicts)

    def embed_text(self, queries: Sequence[str]) -> np.ndarray:
        return np.asarray([_hash_embedding(query, self.embedding_dim) for query in queries], dtype=np.float32)

    def embed_image(self, paths: Sequence[str]) -> np.ndarray:
        return np.asarray([_hash_embedding(path, self.embedding_dim) for path in paths], dtype=np.float32)

    def transcribe(self, video_path: str) -> tuple[Mapping[str, Any], ...]:
        del video_path
        return ()


class ScriptedModel(ModelClient):
    def __init__(self, actions: Sequence[Mapping[str, Any]] = ()) -> None:
        super().__init__()
        self._actions = [ToolAction.from_mapping(action) for action in actions]
        self._cursor = 0

    def controller(self, question: str, index_digest: str, memory_digest: str, evidence_digest: str) -> ToolAction:
        del question, index_digest, memory_digest, evidence_digest
        if self._cursor >= len(self._actions):
            return ToolAction(type="answer", answer="Insufficient verified evidence.", citations=())
        action = self._actions[self._cursor]
        self._cursor += 1
        return action


def _hash_embedding(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    values = [float(digest[index] - 127) for index in range(max(1, int(dim)))]
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [value / norm for value in values]


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in str(text or "").casefold().replace(".", " ").split() if token)
