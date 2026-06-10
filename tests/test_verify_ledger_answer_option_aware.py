from __future__ import annotations

from pathlib import Path

from visual_coding_agent_harness.contracts import ClaimModality, ClaimRelation, OptionSpec, TargetRegistry, TargetSpec
from visual_coding_agent_harness.tools.verification import build_verification_registry
from visual_coding_agent_harness.workspace import EvidenceWorkspace


def test_bare_mcq_letter_without_option_evidence_is_not_supported() -> None:
    registry = build_verification_registry()

    result = registry.execute(
        "verify_ledger_answer",
        {
            "answer": "B",
            "ledger_text": "No answer-grade observation supports the selected option.",
            "candidate_options": ["A. alpha process", "B. beta outcome"],
            "requires_visual_evidence": False,
            "min_score": 0.0,
        },
    )

    region = result["regions"][0]
    gate = region["evidence_gate"]
    assert region["verdict"] == "insufficient"
    assert region["support_score"] == 0.0
    assert gate["resolved_answer"]["claim_text"] == "beta outcome"
    assert "unsupported_selected_option" in gate["reason_codes"]


def test_bare_mcq_letter_scores_resolved_candidate_option_text() -> None:
    registry = build_verification_registry()

    result = registry.execute(
        "verify_ledger_answer",
        {
            "answer": "B",
            "ledger_text": "`obs_0001` tool: `qa_segment` claim: the beta outcome is directly supported.",
            "candidate_options": ["A. alpha process", "B. beta outcome"],
            "requires_visual_evidence": False,
        },
    )

    region = result["regions"][0]
    gate = region["evidence_gate"]
    assert region["verdict"] == "supported"
    assert region["supported_terms"] == ["beta", "outcome"]
    assert gate["resolved_answer"]["option_id"] == "B"
    assert gate["reason_codes"] == []


def test_letter_answer_resolves_registry_option_text_when_candidates_absent(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, run_id="registry_option_text")
    workspace.target_registry = TargetRegistry.from_specs(
        targets=[TargetSpec("T1", "beta outcome", modality_hint=ClaimModality.NARRATED_FACT)],
        options=[OptionSpec("B", target_sequence=("T1",), raw_option_text="beta outcome")],
    )
    registry = build_verification_registry(workspace=workspace)

    result = registry.execute(
        "verify_ledger_answer",
        {
            "answer": "B",
            "ledger_text": "`obs_0001` tool: `qa_segment` claim: the beta outcome is stated.",
            "requires_visual_evidence": False,
        },
    )

    region = result["regions"][0]
    gate = region["evidence_gate"]
    assert region["verdict"] == "supported"
    assert gate["resolved_answer"]["target_refs"] == ["T1"]
    assert region["supported_terms"] == ["beta", "outcome"]


def test_transcript_bindings_satisfy_letter_answer_with_target_refs(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, run_id="transcript_binding_verify")
    workspace.target_registry = TargetRegistry.from_specs(
        targets=[
            TargetSpec("T1", "first narrated step", modality_hint=ClaimModality.NARRATED_FACT),
            TargetSpec("T2", "second narrated step", modality_hint=ClaimModality.NARRATED_FACT),
        ],
        options=[
            OptionSpec(
                "B",
                target_sequence=("T1", "T2"),
                required_relations=("R1",),
                raw_option_text="first narrated step then second narrated step",
                option_kind="sequence",
            )
        ],
        relations=[ClaimRelation("R1", "before", "T1", "T2")],
    )
    for target_ref in ("T1", "T2"):
        workspace.write_evidence_row(
            {
                "obs_id": "obs_0001",
                "tool": "transcript_evidence_binder",
                "segment_id": "seg_0001",
                "event_label": target_ref,
                "claim": f"Transcript binding supports {target_ref}.",
                "confidence": 0.88,
                "grounding_quality": "indexed_transcript",
                "evidence_binding": {
                    "evidence_id": f"bind_{target_ref}",
                    "target_id": target_ref,
                    "status": "supported",
                    "claim_modality": "narrated_fact",
                    "relation_bindings": [
                        {
                            "relation_id": "R1",
                            "status": "supported",
                            "ordered_target_refs": ["T1", "T2"],
                            "evidence_ids": ["bind_T1", "bind_T2"],
                            "modality": "narrated_fact",
                        }
                    ],
                },
            }
        )
    registry = build_verification_registry(workspace=workspace)

    result = registry.execute(
        "verify_ledger_answer",
        {
            "answer": "B",
            "target_refs": ["T1", "T2"],
            "requires_visual_evidence": True,
        },
    )

    region = result["regions"][0]
    gate = region["evidence_gate"]
    assert region["verdict"] == "supported"
    assert gate["structured_support"]["supported_target_refs"] == ["T1", "T2"]
    assert gate["structured_support"]["supported_relation_refs"] == ["R1"]
    assert gate["reason_codes"] == []


def test_invalid_citation_feedback_uses_reason_code() -> None:
    registry = build_verification_registry()

    result = registry.execute(
        "verify_ledger_answer",
        {
            "answer": "alpha process",
            "ledger_text": "`obs_0001` tool: `qa_segment` claim: alpha process is supported.",
            "required_citations": ["not_an_obs"],
            "requires_visual_evidence": False,
        },
    )

    gate = result["regions"][0]["evidence_gate"]
    assert result["regions"][0]["verdict"] == "insufficient"
    assert "invalid_citation_id" in gate["reason_codes"]
