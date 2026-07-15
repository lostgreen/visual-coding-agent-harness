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
- Root-cause review of that run found grounded readiness was all-or-nothing, fine-grained contracts stayed on a fixed 0.5fps grid, repeated-window contradictions were not arbitrated, and forced choice had no evidence-strength calibration.

## Latest Change

- Count, sequence, state-transition, narrative-inference, epistemic-option, attribute-transition, and cross-window identity questions now compile to stricter contracts.
- Conditions are aligned across rounds by stable semantic identity and completion evaluates the active condition set using accumulated historical evidence.
- Discriminative questions fail closed on missing answer audits and require independent `verify_claim` evidence.
- Repeated rejected submissions are converted into coverage, event-enumeration, or contrastive repair tasks.
- Verification: local full suite `285 passed`.
- Follow-up iteration adds strict/partial grounding grades; partial requires >=60% satisfied critical conditions, >=75% scope coverage, visual retrieval, no contradiction, and the existing discriminative answer audit.
- Commit `896aa58` temporarily mapped identity/order/count contracts to fps floors; that design was rejected because it made the framework perform semantic sampling decisions.
- Forced count choices now rank independent claim checks, countable entity witnesses, or canonical event candidates. Wrong-window positives do not count as target support; zero target-positive evidence deterministically penalizes stronger numerical assertions.
- Current follow-up removes every contract-to-fps mapping. Reasoner visual tasks now declare `sampling_floor_fps` plus `temporal_resolution_rationale` from expected evidence dynamics; missing declarations default to 0.5fps and are traced.
- Investigator governance is contract-agnostic: explicit `not_found`, confidence below 0.7, or structured-slot conflict triggers a maximum three-attempt fps/phase ladder. Negative/conflicted observations are not reused as positive support, terminal absence is resolution-qualified, and unresolved slots block grounded readiness.
- Sampling stability telemetry now reports fps, upshifts, trigger causes, negative-to-positive conversions, conflicts, and unspecified floors.
- Current local verification: `PYTHONPATH=src:. pytest -q` -> `299 passed`.

## Current Run

- Completed KML diagnostic group: `videomme_v2_errors10_dynamic_v1`.
- Output: `/m2v_intern/xuboshen/zgw/VideoAgent/videomme_v2_errors10_contract_v2_86118df`.
- Improvements: 441-3 and 468-3 became correct; all four baseline false-grounded cases became forced choice.
- Regressions: 445-3 and 521-2 became incorrect; no case remained grounded.
- Disjoint guard group: `videomme_long_regression50_v1`.

## Stale Evidence

- Pre-change errors10 results remain valid as the baseline only; they do not measure the upgraded code.
- `/m2v_intern/xuboshen/zgw/VideoAgent/videomme_v2_errors10_contract_v2_86118df` is also stale for the new graded-grounding/sampling/calibration iteration.

## Next Actions

1. Run the full local suite, commit/push the three-layer sampling iteration, and create a clean KML worktree because the existing remote checkout is dirty.
2. Run focused KML tests, then rerun errors10 in small batches with fresh output paths and multiple seeds.
3. Compare accuracy, grounded strict/partial counts, false-grounded count, sampling fps/phase use, and per-case investigation count against contract_v2.
4. Inspect only compact fingerprints for 468-3, 521-2, and 445-3 before deciding another code change.
5. Run regression50 only after the diagnostic regressions are resolved.
