# 2026-06-09 Planner-Owned Grounding Refactor Handoff

## Current goal

Replace the previous VideoMME 605/611/612-oriented semantic shortcuts with a general Planner-owned grounding architecture:

- Planner/Grounder owns semantic interpretation, target/relation/option claim definitions, route/modality choice, and final option decision.
- Framework owns deterministic surface parsing, structural validation, stable target/relation IDs, registry freezing, tool execution, provenance, and protocol gates.
- Evidence tools observe and bind facts; they do not choose options or create benchmark-specific target semantics.

## Current evidence

- Current local branch: `codex/agent-ownership-context-redesign`.
- Latest local verification: `PYTHONPATH=src:. pytest -q` => `490 passed in 1.72s`.
- Whitespace check: `git diff --check` => clean.
- The previous KML run based on older code is stale for this refactor:
  - `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_final_closure_d9f99c7_3demo_20260609_233313_pyenv`

## Files changed

- Added generic grounding phase:
  - `src/visual_coding_agent_harness/agents/grounding/contracts.py`
  - `src/visual_coding_agent_harness/agents/grounding/validator.py`
  - `src/visual_coding_agent_harness/agents/grounding/compiler.py`
  - `src/visual_coding_agent_harness/agents/grounding/planner.py`
- Wired `AgentBudget.planner_owned_grounding` and VideoMME runner flag/default.
- Removed/demoted benchmark-specific runtime semantics from:
  - `agents/question_policy.py`
  - `agents/iterative_agent.py`
  - `agents/answer_agent.py`
  - `agents/open_questions.py`
  - `agents/prompt_stack.py`
  - `agents/transcript_binder.py`
  - `tools/navigation.py`
  - `tools/inspector.py`

## Important decisions

- No pre-Planner semantic target registry is created from option text.
- `target_coverage` seeding only runs when a frozen `TargetRegistry` exists.
- Ordered transcript/navigation rows are option-neutral and expose sequence bindings rather than `supported_option`.
- Deterministic temporal/main-idea option takeover is removed; AnswerAgent is the verifier.
- Generic forced visual fallback is disabled by default instead of silently appending visual reads after navigation-only rounds.
- Benchmark examples and hard-coded life-journey / artwork / rise-stability-fall vocabulary are removed from runtime prompts and tests now guard against reintroduction.

## Tests added or updated

- `tests/test_grounding_plan.py`: validates GroundingPlan contracts, compiler, planner retry, and fallback.
- `tests/test_runtime_source_cleanliness.py`: prevents benchmark semantic constants from re-entering runtime source.
- Existing policy/navigation/answer/iterative tests were updated to assert the new ownership boundary instead of old deterministic shortcuts.

## Next actions

1. Commit and push this refactor when ready.
2. Launch a fresh KML 3-demo run only from the new pushed commit; previous KML paths should not be used as evidence for this version.
3. Inspect only compact remote summaries: result count, terminal status, top failure fingerprint, and output artifact paths.
4. If a demo fails, first check whether GroundingPlan output was valid/frozen before debugging tool routing or AnswerAgent verification.
