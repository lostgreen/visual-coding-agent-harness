# errors10 contract-v2 KML evaluation

## Setup

- Code: `86118df` (`Strengthen VideoMME contract verification`)
- KML output: `/m2v_intern/xuboshen/zgw/VideoAgent/videomme_v2_errors10_contract_v2_86118df`
- Group: `videomme_v2_errors10_dynamic_v1`
- Reasoner: `mmu-0-2-openai-swedencentral-gpt-5.5`
- Investigator: `pa/gmn-2.5-pr`
- Budget: 6 rounds, 20 investigations, 4 case workers
- Cold indexes: copied from the earlier KML errors10 workspace; evidence, traces, and summaries were rerun.
- Baseline: the user-provided `errors10_thumbnails_batch1/2` summaries. The older KML output root had later, different summaries and is not the comparison baseline.

## Results

| Case | Baseline | Contract v2 | Correct change | Investigations |
|---|---|---|---|---:|
| 441-2 | F, grounded | F, forced | wrong -> wrong | 4 -> 19 |
| 441-3 | H, forced | B, forced | wrong -> correct | 19 -> 19 |
| 441-4 | G, grounded | D, forced | wrong -> wrong | 13 -> 19 |
| 445-2 | G, forced | G, forced | wrong -> wrong | 6 -> 17 |
| 445-3 | B, grounded | D, forced | correct -> wrong | 9 -> 15 |
| 445-4 | F, forced | A, forced | wrong -> wrong | 15 -> 14 |
| 468-2 | A, forced | D, forced | wrong -> wrong | 13 -> 16 |
| 468-3 | C, grounded | G, forced | wrong -> correct | 10 -> 13 |
| 521-2 | F, forced | E, forced | correct -> wrong | 12 -> 20 |
| 744-1 | D, grounded | A, forced | wrong -> wrong | 10 -> 17 |

Aggregate comparison:

- Accuracy: 2/10 -> 2/10.
- False grounded: 4 -> 0.
- Grounded coverage: 5/10 -> 0/10.
- Average investigations: 11.1 -> 16.9.
- Correct gains: 441-3 and 468-3.
- Regressions: 445-3 and 521-2.

## Interpretation

The stricter contracts removed every false-grounded result, and the epistemic and attribute-transition changes recovered two answers. The framework over-corrected, however: every case ended as forced choice, average investigation cost increased by 52%, and two previously correct cases regressed.

445-2 demonstrates that the control-flow repair works but event recall remains weak. It triggered mandatory full-source coverage and repeated-submission repair, reducing repeated answer gates while still finding only the wrong count.

The next iteration should preserve the fail-closed audit behavior while recovering grounded recall:

1. Separate full-video occurrence enumeration from localized distinct-object counting so 521-2 does not spend the full budget diluting a strong same-frame count.
2. Make successful independent `verify_claim` evidence a sufficient grounded path instead of allowing every comparison case to reach finalization as insufficient.
3. Preserve and compare strong earlier candidates across contract repairs to reduce 445-3-style answer drift.
4. Improve event candidate discovery before adding more gate strictness for 445-2.
5. Re-run this diagnostic group, then use the disjoint regression50 group before accepting the next change.
