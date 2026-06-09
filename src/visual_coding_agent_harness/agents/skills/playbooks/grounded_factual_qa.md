---
name: grounded_factual_qa
version: 1
default_claim_modality: visual_fact
recovery_rules:
  insufficient:
    action: need_more_evidence
    target: distinguishing fact window
  ambiguous:
    action: ask_answer_agent_to_abstain
    target: unresolved factual ambiguity
---
# Grounded Factual QA

Use this playbook for localized factual questions that are not mainly timeline, biography, or mixed ASR/visual reasoning.

Planner playbook:
- Localize the target fact with search or navigation tools before reading expensive details.
- Ask visual readers for neutral factual descriptions, not option votes.
- Final answers need cited answer-grade evidence for the selected claim.
