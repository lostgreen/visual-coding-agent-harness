# Phase 3 B4/B5: sampled cues and interpretation provenance

- Commit: `fd10572`
- Cases: `0072`, `0097`, `0146`
- Runtime root: `/home/xuboshen/zgw/mger_runs/cases3-b45-fd10572-r1-20260804`
- Reasoner / Investigator: `pa/gmn-2.5-pr`
- Development judge: `pa/gmn-2.5-pr` (diagnostic only)

## Runtime results

| Case | Answer | Reference valid | Frames | Observation claims | Item-bound claims | Cues | Verified | Child-refined |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0072 | no | no | 14 | 1 | 1 | 6 | 1 | 0 |
| 0097 | yes | no | 96 | 2 | 2 | 9 | 0 | 0 |
| 0146 | yes | no | 96 | 0 | 0 | 18 | 0 | 0 |

- Observation-claim item binding: `3/3` (`100%`).
- Dangling interpretation items: `0`.
- False refinements: `0`.
- Silent task drops: `0`.
- Development judge: all three responses parsed without retry; all scored `0.0`.

## Acceptance

- ObservationCue objects were emitted only for point items whose timestamps exactly matched sampled frame times.
- Observation claims that were created used an exact `(attempt_id, interpretation_id, item_id)` foreign-key triple.
- Case `0072` completed same-frame cue verification with a real verification item.
- Candidate parents remained ineligible as final material support.

## Decision

Keep B4/B5. They close provenance and free-timestamp failure modes, but this batch did not reach a verified child refinement and did not improve correctness. B6/B7 therefore make transient-event child refinement and typed evidence closure explicit final-answer requirements instead of relying on prompt compliance alone.
