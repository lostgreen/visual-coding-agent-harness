# MM-Lifelong Adapter

The adapter converts a dataset row into two records:

- `RuntimeQuestionV1` contains the question, options, public grouping fields,
  and non-reference source identity.
- `EvaluationRecordV1` contains the reference answer, official clue intervals,
  and evaluator metadata.

Prepared case workspaces store these as `case.json` and
`evaluation_case.json`. Only `case.json` is copied into an agent run.
