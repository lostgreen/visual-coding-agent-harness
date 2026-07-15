# errors10 three-layer sampling KML evaluation

## Setup

- Code: `aef2161` (`Move sampling policy into three-layer control`)
- KML worktree: `/home/xuboshen/zgw/visual-coding-agent-harness-aef2161`
- Valid output: `/m2v_intern/xuboshen/zgw/VideoAgent/videomme_v2_errors10_three_layer_aef2161_seed20260707`
- Dataset/runner: VideoMME-v2 via `tools/run_videomme_v2_eval.py --method agent`
- Group: `videomme_v2_errors10_dynamic_v1`
- Seed: `20260707`; budget: 6 rounds, 20 investigations, 4 workers
- Local verification: 299 tests passed; KML focused verification: 166 tests passed

Two earlier directories named `errors10_three_layer_aef2161_seed20260707_batch1/2` used the legacy Video-MME runner and case-ID collisions. They are invalid for this evaluation and excluded from every metric below.

## Results

| Case | Contract v2 | Three-layer | Correct | Mode | Investigations |
|---|---|---|---:|---|---:|
| 441-2 | F | F | no | forced | 12 |
| 441-3 | B | B | yes | forced | 20 |
| 441-4 | D | G | no | forced | 13 |
| 445-2 | G | H | no | forced | 18 |
| 445-3 | D | E | no | forced | 11 |
| 445-4 | A | C | no | forced | 13 |
| 468-2 | D | A | no | forced | 16 |
| 468-3 | G | D | no | forced | 10 |
| 521-2 | E | F | yes | partial-grounded | 20 |
| 744-1 | A | A | no | forced | 18 |

Aggregate comparison with contract v2:

- Accuracy: `2/10 -> 2/10`.
- Grounded coverage: `0/10 -> 1/10`; the grounded answer is correct.
- False grounded: `0 -> 0`.
- Average investigations: `16.9 -> 15.1`.
- Correct recovery: 521-2 changed from wrong forced `E` to correct partial-grounded `F`.
- Regression: 468-3 changed from correct forced `G` to wrong forced `D`.

## Sampling telemetry

- Final evidence fps histogram: `0.5: 15`, `1.0: 38`, `2.0: 86`.
- Reasoner floor unspecified: 39 of 139 visual evidence records; declaration coverage is about 72% and misses the 90% target.
- Adaptive upshifts: 103.
- Trigger distribution: `not_found: 48`, `structured_slot_conflict: 2`.
- Negative-to-positive conversions: 2.
- Conflicted final observations: 3; none produced a grounded answer.

## Interpretation

The responsibility split is functioning: Reasoner-selected floors are visible in telemetry, low-resolution negative evidence triggers a type-agnostic ladder, conflicts block grounding, and 521-2 recovers through partial grounding without a false-grounded regression. The architecture change is therefore useful even though aggregate accuracy is flat.

The adaptive ladder is not yet cost-effective. Only 2 of 48 negative triggers converted to positive evidence, while 103 upshifts were issued. Several cases reached high investigation counts because Reasoner dispatched new windows after terminal qualified absence. The next iteration should improve window/phase choice and use terminal absence in planning before increasing the retry budget.

One seed is insufficient for the stability KPI. In particular, 468-3 regressed and 445-3 still drifted. Run focused multi-seed checks on 468-3, 521-2, and 445-3 before regression50.
