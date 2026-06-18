---
name: narration_timeline_qa
version: 1
description: Timeline or biography-style questions whose evidence is mainly narrated or spoken.
when_to_use: Use when the question asks for narrated sequence, life story, reason, or background information.
default_claim_modality: narrated_fact
recovery_rules:
  missing_asr:
    action: use_visual_fallback
    target: narrated event anchors
  conflict:
    action: need_more_evidence
    target: transcript and visual timeline conflict
---
# Narration Timeline QA

Use this playbook when the question asks for a narrated life story, biography, reason, or timeline that is likely carried by ASR or voiceover. Treat transcript anchors as the primary source for narrated facts and do not require a visual gate for every narrated claim.

Planner playbook:
- First look for ASR or transcript segments that state the life stage, motive, or sequence directly.
- Use read_segment_detail for narrated biography and life-order claims, then write memory from exact transcript anchors.
- Finalize narrated life-order options only after memory cites the required transcript anchor chain.
- Do not visually verify abstract narrated facts as if they were visible objects.
- Use visual reads to anchor named scenes, people, artworks, or events when the transcript alone leaves the timeline ambiguous.
- Keep the final answer grounded in cited narrated facts, with visual citations used as corroboration rather than a mandatory gate.
