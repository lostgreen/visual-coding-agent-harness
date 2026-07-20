# errors10 orchestration evaluation

## Setup

- Code: `bf91e33` (`fix: preserve context through final repair`)
- KML worktree: `/home/xuboshen/zgw/visual-coding-agent-harness-55d50be`
- Output: `/home/xuboshen/outputs/vcah-bf91e33/videomme-v2-errors10-gpt55-gemini25pro`
- Dataset: `/ytech_m2v5_hdd/workspace/kling_mm/Datasets/Video-MME-v2`
- Group: `videomme_v2_errors10_dynamic_v1`
- Reasoner: `mmu-0-2-openai-swedencentral-gpt-5.5`
- Investigator: `pa/gmn-2.5-pr`
- Budget: 6 investigation rounds, 20 investigations, 4 case workers
- Python: `/home/xuboshen/Anaconda/envs/VLMEvalKit/bin/python`
- Exit status: 0

## Results

| Case | Answer | Gold | Correct | Reference valid | Rounds | Investigations |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 441-2 | H | D | no | yes | 8 | 8 |
| 441-3 | none | B | no | no | 8 | 2 |
| 441-4 | none | C | no | no | 8 | 4 |
| 445-2 | none | E | no | no | 8 | 5 |
| 445-3 | none | B | no | no | 8 | 2 |
| 445-4 | none | H | no | no | 8 | 6 |
| 468-2 | none | G | no | no | 8 | 3 |
| 468-3 | G | G | yes | yes | 8 | 7 |
| 521-2 | F | F | yes | yes | 4 | 7 |
| 744-1 | none | G | no | no | 8 | 1 |

Aggregate:

- Accuracy: 2/10 (20%).
- Reference-valid answers: 3/10; no invalid candidate was returned as a final answer.
- Investigation count: 45 total, 4.5 average.
- Investigator execution: 31 Gemini visual observations and 31 ASR searches.
- Non-consuming ASR: 12 zero-hit searches and 5 duplicate searches.
- Visual transport: 2,421 images requested, 2,421 attached, 0 dropped.
- Model calls: 76 GPT-5.5 workspace calls, 6 GPT-5.5 JSON-repair calls, and 31 Gemini calls.
- API health: 0 retries; all 113 completions ended with `finish_reason=stop`.

## Gate behavior

Every case that reached forced finalization used the first final call to read
existing observations. The second and last call returned only `answer` or
`update_workspace`; none attempted another investigation or observation read.
Rejected candidates and their residual uncertainty were present in the repair
context.

The retained gate intentionally rejected answers whose own
`residual_uncertainty` admitted an option/evidence mismatch. This lowers answer
coverage relative to the 3/10 `55d50be` run, which returned three
reference-invalid candidates. The independent support-audit experiment was
discarded after scoring 0/10 and rejecting the stable correct `521-2` case.
