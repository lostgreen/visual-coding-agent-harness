# MGER Phase 5R Frozen Reproducibility Audit (2026-08-05)

## Decision

Gate R1 is **PASS**: the current runtime reproduced all historical per-task
frame counts and timestamp digests exactly, 551/551 frames across 10/10 cases.

Gate R2 is **DISTRIBUTIONALLY COMPARABLE, NOT TRAJECTORY-EQUIVALENT**. The
historical/current root-level frame IQRs overlap, all nine reported R2
metrics have overlapping ranges, and all 10 per-case frame ranges overlap.
However, the compatibility prompt is not bit-for-bit historical, and case 0072
shows a consistent first-action shift across all roots. This is sufficient to
replace the 55.1 single point with `FrozenBehaviorReferenceV2`, but not to claim
identical prompts, trajectories, or deterministic behavior.

P5R-5 / official Gate-0 was **NOT RUN**. No official `gpt-5` configuration at
temperature 0 is available, and the plan forbids substituting
or relabeling GPT-5.5. No Minimal Controller, Binder, new evidence state,
retrieval-policy promotion, 40-case method comparison, or judge claim was made.

## Frozen Scope

- Cohort: the fixed 10-case MMLifelong set from revision `74f012d`.
- Historical controller: real detached checkout
  `74f012dfc1f3a3e29541ab6d21cb261c937c702a`.
- Current controller: Frozen compatibility path inherited from `87d575d`, run
  at `b9ebd2ac5b8caf3c98f2bcc3397e9e55ffa1d385` with Phase5R
  replay/provenance instrumentation.
- Reasoner and Investigator: `pa/gmn-2.5-pr`.
- Budgets: 4 semantic rounds, 12 investigations, 4 tasks per round.
- Caption substrate: `hybrid` index, `adaptive` query strategy, digest
  `fb25cc6e8c5c4f855d86b724882b14502d3ff17d0670e1c0dc61e52a3bd08b96`.
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` at revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Live sequence: `old1, new1, old2, new2, old3, new3`, three workers per root.
- Runtime only: no judge calls are part of Gate R1 or R2.

## Gate R1: Mechanical Determinism

Gate R1 is PASS. Recorded historical decisions were replayed without Reasoner
or Investigator model calls through the current Frozen runtime. All 10 cases
matched exact per-task frame counts and normalized timestamp digests.

| Case | Historical frames | Replay frames | Result |
| --- | ---: | ---: | --- |
| 0031 | 72 | 72 | PASS |
| 0038 | 40 | 40 | PASS |
| 0072 | 19 | 19 | PASS |
| 0097 | 84 | 84 | PASS |
| 0108 | 32 | 32 | PASS |
| 0117 | 32 | 32 | PASS |
| 0119 | 16 | 16 | PASS |
| 0146 | 104 | 104 | PASS |
| 0165 | 56 | 56 | PASS |
| 0190 | 96 | 96 | PASS |
| **Total** | **551** | **551** | **10/10 PASS** |

The focused precheck also passed 0072, 0097, and 0119 at 119/119 frames.

### Replay Defect Found And Fixed

An early replay implementation raised the configured semantic-round budget to
the number of recorded decisions. Case 0165 exposed this setup error: it
dispatched historical round-5 task `r1_t5`, which the original 4-round runtime
had recorded during finalization but never executed, adding 96 false frames.
Commit `8b757e4` preserves the historical semantic budget and lets the normal
finalization path consume later decisions without dispatching their tasks. The
formal R1 artifacts were regenerated from scratch after the fix; all earlier
budget-raised roots are excluded.

## Gate R2: Behavioral Distribution

All six roots completed 10/10 cases with no failed cases. The machine reference
passed case-ID, case-count, root-count, within-root, within-arm, cross-arm
configuration, declared-revision, and embedded-current-revision checks.

One pre-formal current root at `e33a74c` is excluded. Its Phase5R reporter
assumed every case materialized a visual-frame manifest and failed valid
frame-free paths in 0038/0072. Commit `b9ebd2a` made missing manifests an
explicit zero-visual state; all three formal current roots were then rerun from
empty directories at that single commit.

| Arm / root | Mean frames | Requested windows | Actual windows | Requested duration | Caption | ASR | Rounds | Answer rate | ObservedCaseRate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical 1 | 118.7 | 1.7 | 1.7 | 610.21 | 1.3 | 0.0 | 5.1 | 0.9 | 1.0 |
| Current 1 | 73.2 | 1.9 | 1.3 | 196.37 | 1.4 | 0.0 | 5.5 | 0.9 | 0.9 |
| Historical 2 | 80.6 | 2.0 | 1.8 | 94.00 | 1.1 | 0.3 | 5.7 | 0.9 | 1.0 |
| Current 2 | 117.8 | 2.0 | 1.7 | 1295.50 | 1.3 | 0.1 | 4.6 | 0.8 | 0.9 |
| Historical 3 | 92.3 | 1.4 | 1.3 | 150.39 | 1.4 | 0.2 | 5.5 | 0.9 | 0.9 |
| Current 3 | 60.1 | 1.9 | 1.2 | 110.60 | 1.5 | 0.1 | 5.4 | 0.9 | 1.0 |

| Root-mean metric | Historical median [IQR] | Current median [IQR] | Current - historical | IQR overlap | Range overlap |
| --- | ---: | ---: | ---: | --- | --- |
| Visual frames | 92.3 [86.45, 105.50] | 73.2 [66.65, 95.50] | -19.1 | yes | yes |
| Requested windows | 1.7 [1.55, 1.85] | 1.9 [1.90, 1.95] | +0.2 | no | yes |
| Actual windows | 1.7 [1.50, 1.75] | 1.3 [1.25, 1.50] | -0.4 | yes | yes |
| Requested duration | 150.39 [122.20, 380.30] | 196.37 [153.49, 745.94] | +45.98 | yes | yes |
| Caption searches | 1.3 [1.20, 1.35] | 1.4 [1.35, 1.45] | +0.1 | yes | yes |
| ASR searches | 0.2 [0.10, 0.25] | 0.1 [0.05, 0.10] | -0.1 | yes | yes |
| Semantic rounds | 5.5 [5.30, 5.60] | 5.4 [5.00, 5.45] | -0.1 | yes | yes |
| Answer rate | 0.9 [0.90, 0.90] | 0.9 [0.85, 0.90] | 0.0 | yes | yes |
| ObservedCaseRate | 1.0 [0.95, 1.00] | 0.9 [0.90, 0.95] | -0.1 | yes | yes |

The current median frame cost is 20.7% below the historical-current-env
median, but the distributions are not separated: historical roots span
80.6-118.7 and current roots span 60.1-117.8. Current per-case median deltas
are mixed (six lower, three higher, one equal), and all 10 per-case ranges
overlap. This does not support a uniform compatibility-cost shift.

| Case | Historical frames (3 roots) | Current frames (3 roots) | Median delta | Range overlap |
| --- | --- | --- | ---: | --- |
| 0031 | 60, 80, 96 | 70, 70, 96 | -10 | yes |
| 0038 | 40, 90, 90 | 28, 108, 198 | +18 | yes |
| 0072 | 96, 184, 192 | 0, 66, 97 | -118 | yes |
| 0097 | 64, 96, 116 | 0, 64, 96 | -32 | yes |
| 0108 | 51, 85, 320 | 3, 29, 288 | -56 | yes |
| 0117 | 8, 80, 96 | 32, 32, 40 | -48 | yes |
| 0119 | 12, 126, 192 | 52, 85, 288 | -41 | yes |
| 0146 | 96, 108, 126 | 108, 108, 108 | 0 | yes |
| 0165 | 0, 15, 43 | 12, 19, 56 | +4 | yes |
| 0190 | 56, 66, 232 | 70, 96, 192 | +30 | yes |

This is descriptive evidence from three roots per arm, not an equivalence test
or statistical-significance claim.

## Frame-Cost Decomposition

The focus cases confirm that frame changes come from action selection and
window materialization, not a sampling-runtime multiplier. Values below are
the marginal medians across each arm's three roots.

| Case | Arm | Frames | Requested / actual windows | Duration | Requested / effective FPS | Cap hits | Reinspect | Caption / ASR | Rounds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0072 | Historical | 184 | 2 / 2 | 156 | 2.00 / 1.17 | 2 | 0 | 1 / 0 | 6 |
| 0072 | Current | 66 | 3 / 1 | 217 | 2.00 / 1.29 | 1 | 0 | 1 / 0 | 4 |
| 0097 | Historical | 96 | 2 / 1 | 138 | 1.00 / 0.75 | 1 | 0 | 1 / 0 | 6 |
| 0097 | Current | 64 | 2 / 1 | 158 | 1.25 / 0.33 | 1 | 0 | 2 / 0 | 6 |
| 0119 | Historical | 126 | 1 / 2 | 200 | 0.75 / 0.48 | 2 | 0 | 1 / 1 | 6 |
| 0119 | Current | 85 | 2 / 2 | 62 | 1.25 / 1.25 | 2 | 0 | 2 / 0 | 6 |

For 0072, the median reduction is driven by fewer materialized visual windows
and fewer cap hits despite more requested windows. For 0097, current roots use
more caption search and lower effective visual FPS. For 0119, current roots
request more visual windows but the median requested duration is shorter; one
current root is a wide-window outlier (4112 seconds requested, 288 frames).

## Decision-Trace Divergence

Exact live trace equality is absent both within and across arms:

- Historical within-arm: 0/30 exact root pairs.
- Current within-arm: 0/30 exact root pairs.
- Cross-arm: 0/90 exact pairs.
- Every pair's first exact-fingerprint divergence is round 1.

Therefore zero cross-arm exact digests cannot by itself diagnose compatibility
drift; the unseeded controller already has zero exact repeatability within the
same implementation. The protocol-independent traces still reveal one stable
surface difference. For 0072, all historical roots start with caption search
near virtual time 47031, while all current roots start with ASR near 4200. The
first Reasoner prompts are deterministic within each arm but differ across
arms: historical length/hash `22275 / a5d6b8b3...`, current `22379 /
c5f9a9d1...`. Cases 0097 and 0119 retain caption-search-first behavior in both
arms and diverge later.

The practical conclusion is narrower than bit-for-bit compatibility: current
Frozen is suitable as a contemporaneous distributional baseline, while exact
trajectory controls should continue to use the real historical checkout or
first repair prompt-surface parity, especially for 0072.

## Provenance And Limitations

The current path records runner commit, requested model settings, provider
request IDs, seed support, temperature/top-p, caption and input digests, prompt
digest, frame-cache/source-manifest digests, and environment fingerprint. For
the unmodified historical checkout, equivalent available fields are rebuilt
from its immutable run configs and interaction traces. The provider does not
return a pinned deployment revision, so the controller remains stochastic and
service-version-unpinned. All roots explicitly record `temperature=null`,
`top_p=null`, `requested_seed=null`, and seed support `unknown`. Provider
request-ID counts were historical `71/80/75` and current `76/70/72`; IDs remain
inside the machine artifact rather than this report. The six roots are
therefore same-day, interleaved, multi-root controls rather than claims of
bit-for-bit model determinism.

## Official GPT-5 And Gate-0

The KML configuration inventory contains a GPT-5.5 deployment but no official
`gpt-5, temperature=0` evaluator. The plan explicitly forbids relabeling or
substituting GPT-5.5. Gate-0 remains conditionally blocked and was not called;
this does not invalidate the completed runtime R1/R2 audit.

## Verification

- Latest local full suite before final report: 349 passed.
- Remote clean-worktree full suite at reference-generator commit `81abf3a`:
  349 passed, 88 deprecation warnings.
- Focused behavior/replay tests after final provenance normalization: 8 passed.
- Ruff on all changed Python files and formatting checks on the new behavior
  report/test files: passed.
- `git diff --check`: passed.

## Artifacts

- Focus-3 Gate R1:
  `/home/xuboshen/zgw/mger_runs/phase5r-gate-r1-focus3-8b757e4-r1-20260805.json`
- All-10 Gate R1:
  `/home/xuboshen/zgw/mger_runs/phase5r-gate-r1-all10-8b757e4-r1-20260805.json`
- Formal replay root:
  `/home/xuboshen/zgw/mger_runs/cases10-phase5r-replay-8b757e4-r1-20260805`
- FrozenBehaviorReferenceV2:
  `/home/xuboshen/zgw/mger_runs/FrozenBehaviorReferenceV2-81abf3a-20260805.json`
  (387349 bytes, SHA-256
  `a6f2e63895feb8520b8a91713f3aca7ee1944cfef501a9c8ff49b33c5117dddc`).
  Generator commit: `81abf3a`.
- Historical live roots:
  `/home/xuboshen/zgw/mger_runs/cases10-phase5r-old74f012d-live-r{1,2,3}-20260805`.
- Current Frozen live roots:
  `/home/xuboshen/zgw/mger_runs/cases10-phase5r-newb9ebd2a-frozen-live-r{1,2,3}-20260805`.
