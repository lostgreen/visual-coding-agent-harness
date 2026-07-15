# VideoMME Contract and Verification Debug State

## Goal

Improve the general multi-round Agent architecture using the errors10 failure traces without encoding case-specific answers.

## Current Evidence

- Current branch: `codex/m3-slim-video-agent`.
- errors10 had 5 grounded answers with only 1 correct; 445-2 repeated the same rejected answer four times; 521-2 repeatedly recast equivalent conditions.
- Contract compilation previously classified 445-2 as multi-window rather than full-video and 744-1 as a window-level existential claim.
- Model-driven Investigator reports already emit structured per-condition results; the unconditional partial behavior is limited to deterministic or cached reuse paths.
- Existing repair infrastructure already covers source coverage, event candidates, identity anchors, entity candidates, scores, and spatial relations; the missing link was contract activation and rejected-answer control flow.

## Latest Change

- Count, sequence, state-transition, narrative-inference, epistemic-option, attribute-transition, and cross-window identity questions now compile to stricter contracts.
- Conditions are aligned across rounds by stable semantic identity and completion evaluates the active condition set using accumulated historical evidence.
- Discriminative questions fail closed on missing answer audits and require independent `verify_claim` evidence.
- Repeated rejected submissions are converted into coverage, event-enumeration, or contrastive repair tasks.
- Verification: local full suite `285 passed`.

## Current Run

- No post-change remote VLM run has been executed yet.
- Diagnostic group: `videomme_v2_errors10_dynamic_v1`.
- Disjoint guard group: `videomme_long_regression50_v1`.

## Stale Evidence

- Pre-change errors10 results remain valid as the baseline only; they do not measure the upgraded code.

## Next Actions

1. Review and commit the local framework changes.
2. Rerun errors10 in a fresh output root and compare false-grounded, correct-but-forced, repeated submissions, and cost.
3. Run the disjoint regression50 group and report accuracy plus grounded coverage/precision.
4. Add tracklet-based identity linking only if 445-4 remains blocked after the stricter contract and independent verification path.
