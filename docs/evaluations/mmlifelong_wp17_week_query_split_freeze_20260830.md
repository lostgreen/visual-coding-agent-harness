# MM-Lifelong WP17 Week Query Split Freeze

- Split implementation commit: `f5976b5`
- Selection seed: `20260830`
- Decision: `WP17_WEEK_QUERY_SPLIT_FROZEN`
- Structural gate: `true`
- Model calls: `0`

## Data Availability Audit

- Week annotations: 200 query rows.
- Week videos: available; 6,266 video files, approximately 102.8 GB.
- Month annotations: available.
- Month videos: unavailable on the current KML machine.
- Consequence: Month is not a dependency of the current SMC mechanism path.

## Frozen Roles

| Partition | Count | Method selection | Claim scope |
|---|---:|---:|---|
| Week-dev | 60 | allowed | cross-domain method development |
| Week-holdout | 140 | forbidden | final query-level holdout after method freeze |

Both partitions share the same Week video corpus. The holdout is not eligible for an unseen-video claim.

Selection uses only `case_id`, `question_type`, and fixed-seed SHA256 ranking. Persisted case rows contain only `case_id`, `question_type`, and `case_sha256`. Question, options, answer, clue interval, and model-output fields are absent.

## Question-Type Allocation

| Type | Universe | Development | Holdout |
|---|---:|---:|---:|
| Attribute Recognition | 8 | 2 | 6 |
| Causal Reasoning | 17 | 5 | 12 |
| Counting | 59 | 18 | 41 |
| Entity Recognition | 23 | 7 | 16 |
| Event Recognition | 17 | 5 | 12 |
| Event Tracking | 6 | 2 | 4 |
| Hallucination Detection | 10 | 3 | 7 |
| Language Content Recall | 15 | 4 | 11 |
| Social Interaction | 9 | 3 | 6 |
| Temporal Reasoning | 36 | 11 | 25 |

## Artifact Integrity

- `week_dev60.json`: `5e2ef838da5060db33da94ce1775ba839c1031ffa9fb382d40afef8ff9ac94b8`
- `week_holdout140.json`: `da0b64ae463ce70cc1a7a70dcc208edb8e4f32d9f889325de9a61f861de51303`
- `week_query_split_protocol.json`: `1ef1a858c69311ab06720af4f98c075c85aeca60189c64ee956f840235608c96`

Remote artifact root:

`/m2v_intern/xuboshen/zgw/mger_runs/mmlifelong-wp17-week-query-split-f5976b5-20260830`

The split was frozen before any Week SMC construction or QA outcome was observed.
