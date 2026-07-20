# errors10 Reasoner-authority KML evaluation

## Setup

- Code: `55d50be` (`fix: show the complete investigation decision schema`)
- KML worktree: `/home/xuboshen/zgw/visual-coding-agent-harness-55d50be`
- Output: `/home/xuboshen/outputs/vcah-55d50be/videomme-v2-errors10-gpt55-gemini25pro`
- Dataset: `/ytech_m2v5_hdd/workspace/kling_mm/Datasets/Video-MME-v2`
- Runner: `/home/xuboshen/zgw/visual-coding-agent-harness-55d50be/tools/run_videomme_v2_eval.py --method agent`
- Group: `videomme_v2_errors10_dynamic_v1`
- Reasoner: `mmu-0-2-openai-swedencentral-gpt-5.5`
- Investigator: `pa/gmn-2.5-pr`
- Budget: 6 investigation rounds, 20 investigations, 4 case workers
- Python: `/home/xuboshen/Anaconda/envs/VLMEvalKit/bin/python`
- Exit status: 0

## Results

| Case | Answer | Correct | Reference valid | Investigations | Visual calls | ASR calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 441-2 | D | yes | yes | 5 | 0 | 5 |
| 441-3 | H | no | yes | 8 | 3 | 5 |
| 441-4 | F | no | no | 4 | 4 | 0 |
| 445-2 | B | no | yes | 9 | 7 | 2 |
| 445-3 | D | no | yes | 4 | 2 | 2 |
| 445-4 | H | yes | yes | 5 | 5 | 0 |
| 468-2 | D | no | yes | 4 | 1 | 3 |
| 468-3 | H | no | no | 4 | 0 | 4 |
| 521-2 | F | yes | yes | 7 | 4 | 3 |
| 744-1 | A | no | no | 9 | 8 | 1 |

Aggregate:

- Accuracy: 3/10 (30%).
- Reference-valid answers: 7/10.
- Investigation count: 59 total, 5.9 average.
- Investigator execution: 34 Gemini visual observations and 25 ASR searches.
- Visual transport: 2,544 images requested, 2,544 attached, 0 dropped.
- Model calls: 70 GPT-5.5 Reasoner workspace calls and 34 Gemini Investigator calls.
- Serialization recovery: 16 successful Reasoner JSON-repair calls, 0 repair failures.
- API health: 0 retries; all 120 model completions ended with `finish_reason=stop`.

The group summary and all ten per-case summaries agree on case count and
correctness. Every per-case summary records the required Reasoner and
Investigator model IDs.

## Comparison

The earlier three-layer run at `aef2161` scored 2/10 with 15.1 average
investigations. This run scored 3/10 with 5.9 average investigations. The
contract-v2 run at `86118df` also scored 2/10 with 16.9 average investigations.
These are directional comparisons only because the current runner does not
expose the earlier run's fixed seed.

The main remaining runtime signals are the three reference-invalid final
answers (`441-4`, `468-3`, and `744-1`) and the 16 JSON-repair calls. They are
more actionable than increasing the investigation budget: all three weak cases
already reached the forced-finalization round, while the run used only 29.5% of
the nominal investigation budget.
