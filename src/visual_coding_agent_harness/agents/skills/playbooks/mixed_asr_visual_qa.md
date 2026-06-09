---
name: mixed_asr_visual_qa
version: 1
default_claim_modality: mixed
recovery_rules:
  missing_modality:
    action: need_more_evidence
    target: missing ASR or visual corroboration
  mismatch:
    action: resolve_conflict
    target: ASR visual mismatch
---
# Mixed ASR Visual QA

Use this playbook when the answer depends on both what is said and what is visible, such as a narrator naming a thing while the screen shows its visual state.

Planner playbook:
- Gather the transcript claim and the corresponding visible anchor.
- Keep ASR and visual observations separate in evidence until AnswerAgent verifies consistency.
- If one modality is absent, continue with the available modality only when it directly answers the question.
