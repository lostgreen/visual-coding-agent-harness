---
name: grounded_factual_qa
version: 1
description: Localized factual question answering for visible or directly inspectable facts.
when_to_use: Use when the question asks which, what, where, or who and needs localized anchor-backed memory.
default_claim_modality: visual_fact
recovery_rules:
  insufficient:
    action: need_more_evidence
    target: distinguishing fact window
  ambiguous:
    action: ask_answer_agent_to_abstain
    target: unresolved factual ambiguity
self_check:
  citations: final citations are mem_ ids
  anchors: cited memory entries have real anchors
  modalities: use visual anchors when the selected claim is visually observable; use ASR/OCR/caption anchors when the selected claim is narrated or textual
---
# Grounded Factual QA

Use this playbook for localized factual questions that are not mainly timeline, biography, or mixed ASR/visual reasoning.

Planner playbook:
- Localize the target fact with search or navigation tools before reading expensive details.
- Ask visual readers for neutral factual descriptions, not option votes.
- Final answers need memory citations backed by real anchors for the selected claim.

Final check:
- final citations are mem_ ids
- cited memory entries have real anchors
- use visual anchors when the selected claim is visually observable; use ASR/OCR/caption anchors when the selected claim is narrated or textual
