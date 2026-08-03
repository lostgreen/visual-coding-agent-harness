# Material-grounded localization iteration - 2026-08-03

## Goal

Implement the first mechanical part of the MGER roadmap: keep Caption locator
hits as source-material candidates, expose competing occurrences, distinguish
new passage material from changed query interpretation, and report the resulting
novelty and cost signals without changing answer ownership or retrieval defaults.

## Current case evidence

- The paired bundle remains diagnostic evidence from trace commit `4b7e740`,
  not a result for the current runtime.
- Case 0038's old Caption Top-k covers roughly 17749-18047 seconds while the
  official clue is at 19950-19952 seconds. Occurrence clustering can expose
  competition among returned clusters, but it cannot recover a missing correct
  candidate. A fresh paired run must not call this case fixed unless the correct
  occurrence enters the candidate set and the answer is correct or explicitly
  ambiguous.
- Case 0097's adaptive trace returned 21 Caption hits with 11 unique passages.
  The new accounting reports this material novelty directly. Conservative budget
  reuse activates only when a changed query returns no new passage IDs, so the old
  trace may still spend budget when each query contributes one new but irrelevant
  passage.
- The earlier promotion rejection remains current evidence: accuracy 0.25 to
  0.50, reference-valid 0.75 to 0.50, and visual-frame cost 1.748x.

## Implemented

- `CaptionOccurrenceSetV1` clusters hits by source identity and a bounded time
  gap. It emits candidate-only occurrence IDs, ranges, passage IDs, and an
  ambiguity status; it never selects the semantically correct occurrence.
- Caption indexes preserve source-segment metadata, and the investigator binds
  hits to source video IDs from the virtual manifest.
- Caption material attempts are identified by the returned passage source
  pointers rather than query wording. A different query over the same material
  becomes another interpretation under the same `attempt_id`.
- Same-scope searches with zero novel passage IDs are retained for audit but do
  not consume investigation budget. Query-level reuse remains unchanged.
- Mechanical status tells the Reasoner when separate occurrence clusters remain
  pending and requires identity comparison before promotion to answer support.
- Case and aggregate metrics now expose Caption material attempts, passage
  novelty/deduplication, result-set reuse, occurrence candidate count, unique
  visual material attempts, and visual interpretation/reinterpretation counts.

## Verification

- Focused localization, runtime-status, metrics, and summary checks: 53 passed.
- Full local suite: 249 passed.
- Ruff lint, `python -m py_compile`, and `git diff --check` pass.
- The original DOCX rendered cleanly across 12 pages before its requirements were
  mapped into this iteration.

## Constraints and next actions

1. Re-run the latest frozen baseline before comparing behavior; the uploaded
   code archive matches the current `src/` and `tests/` and is not a new patch.
2. Add an entity-preserving and multilingual candidate-recall ablation on the
   frozen Caption index. Keep it opt-in until 0038/0097-style paired evidence
   improves occurrence/clue recall without reference or cost regression.
3. Run paired 0038 and 0097 checks with at least two independent roots. Require
   0038 to become correct or explicit ambiguous, and require 0097 passage novelty
   and visual-frame cost to improve without clue-coverage regression.
4. Only after candidate recall is demonstrated should occurrence status become
   an answer-time grounding gate; this iteration is observability and scheduling,
   not semantic verification.

## Artifact paths

- Roadmap source: `/Users/lostgreen/Downloads/VCAH_LongVideo_Paper_Objective_and_Roadmap_20260803.docx`
- Paired case bundle: `/Users/lostgreen/Downloads/vcah_cases4_paired_replay_405cb66_analyzed_1f495cc_20260803.zip`
