# VideoMME Contract and Verification Debug State

## Goal

Improve the general multi-round Agent architecture using the errors10 failure traces without encoding case-specific answers.

## Current Evidence

- Current branch: `codex/m3-slim-video-agent`.
- errors10 had 5 grounded answers with only 1 correct; 445-2 repeated the same rejected answer four times; 521-2 repeatedly recast equivalent conditions.
- Contract compilation previously classified 445-2 as multi-window rather than full-video and 744-1 as a window-level existential claim.
- Model-driven Investigator reports already emit structured per-condition results; the unconditional partial behavior is limited to deterministic or cached reuse paths.
- Existing repair infrastructure already covers source coverage, event candidates, identity anchors, entity candidates, scores, and spatial relations; the missing link was contract activation and rejected-answer control flow.
- Post-change KML errors10: accuracy stayed 2/10, false-grounded fell 4 -> 0, grounded coverage fell 5/10 -> 0, and average investigations rose 11.1 -> 16.9.

## Latest Change

- Count, sequence, state-transition, narrative-inference, epistemic-option, attribute-transition, and cross-window identity questions now compile to stricter contracts.
- Conditions are aligned across rounds by stable semantic identity and completion evaluates the active condition set using accumulated historical evidence.
- Discriminative questions fail closed on missing answer audits and require independent `verify_claim` evidence.
- Repeated rejected submissions are converted into coverage, event-enumeration, or contrastive repair tasks.
- Verification: local full suite `285 passed`.

## Current Run

- Completed KML diagnostic group: `videomme_v2_errors10_dynamic_v1`.
- Output: `/m2v_intern/xuboshen/zgw/VideoAgent/videomme_v2_errors10_contract_v2_86118df`.
- Improvements: 441-3 and 468-3 became correct; all four baseline false-grounded cases became forced choice.
- Regressions: 445-3 and 521-2 became incorrect; no case remained grounded.
- Disjoint guard group: `videomme_long_regression50_v1`.

## Stale Evidence

- Pre-change errors10 results remain valid as the baseline only; they do not measure the upgraded code.

## Next Actions

1. Recover grounded recall without weakening the fail-closed audit.
2. Separate localized distinct-object counts from full-video event enumeration and recheck 521-2.
3. Preserve strong earlier candidates across repair rounds and recheck 445-3.
4. Improve event discovery for 445-2, then rerun errors10.
5. Run regression50 only after the diagnostic regressions are resolved.
