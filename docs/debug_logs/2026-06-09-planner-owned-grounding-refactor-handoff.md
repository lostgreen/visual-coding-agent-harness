# 2026-06-09 Planner-Owned Grounding Refactor Handoff

## Current goal

Replace the previous VideoMME 605/611/612-oriented semantic shortcuts with a general Planner-owned grounding architecture:

- Planner/Grounder owns semantic interpretation, target/relation/option claim definitions, route/modality choice, and final option decision.
- Framework owns deterministic surface parsing, structural validation, stable target/relation IDs, registry freezing, tool execution, provenance, and protocol gates.
- Evidence tools observe and bind facts; they do not choose options or create benchmark-specific target semantics.

## Current evidence

- Current local branch: `codex/agent-ownership-context-redesign`.
- Latest local verification: `PYTHONPATH=src:. pytest -q` => `502 passed in 1.99s`.
- Whitespace check: `git diff --check` => clean.
- The previous KML run based on older code is stale for this refactor:
  - `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_final_closure_d9f99c7_3demo_20260609_233313_pyenv`
- The interrupted KML run from commit `43dddd2` is stale and was stopped after it blocked in per-segment ffmpeg re-encoding:
  - `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_grounding_43dddd2_3demo_20260610_100525_pyenv`
  - latest failure fingerprint: child `ffmpeg` was re-encoding a 300s segment clip for `611-2` with `libx264`; the main python process was killed and remained only as a defunct zombie under the launcher shell.

## Files changed

- Added generic grounding phase:
  - `src/visual_coding_agent_harness/agents/grounding/contracts.py`
  - `src/visual_coding_agent_harness/agents/grounding/validator.py`
  - `src/visual_coding_agent_harness/agents/grounding/compiler.py`
  - `src/visual_coding_agent_harness/agents/grounding/planner.py`
- Wired `AgentBudget.planner_owned_grounding` and VideoMME runner flag/default.
- Wired compiled grounding runtime policy into `IterativeVisualAgent`:
  - `route`
  - `recommended_skill_id`
  - `acceptable_evidence_sources`
  - `unresolved_ambiguities`
  - framework-owned `raw_options`
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
- Grounding now runs before exploration target hints; when a frozen registry exists, target hints are derived from registry canonical claims.
- GroundingPlan route and recommended skill now become the effective runtime route/skill; keyword classifiers are fallback only when grounding is unavailable.
- Validator rejects invalid route/claim/relation/modality/polarity/evidence-source values and enforces exact raw option set/text.
- Compiler preserves claim kind, polarity, acceptable evidence sources, unresolved ambiguities, raw options, route, and recommended skill.
- `target_coverage` seeding only runs when a frozen `TargetRegistry` exists.
- Ordered transcript/navigation rows are option-neutral and expose sequence bindings rather than `supported_option`.
- Deterministic temporal/main-idea option takeover is removed; AnswerAgent is the verifier.
- Generic forced visual fallback is disabled by default instead of silently appending visual reads after navigation-only rounds.
- Benchmark examples and hard-coded life-journey / artwork / rise-stability-fall vocabulary are removed from runtime prompts and tests now guard against reintroduction.
- VideoMME long-video visual reads now use a run-level precomputed 2fps frame cache:
  - runner builds `run_root/frame_cache/<video_id>_2fps` before non-direct strategies;
  - scene-index visual captioning, global/query context, segment caption/QA, inspector, vision_read, enrichment, and anchor verification all prefer sampled frame paths over per-call mp4 clips;
  - Qwen-VL backend serializes frame requests as multi-image inputs;
  - old physical clip extraction remains as a compatibility fallback only when no frame sampler is available.

## Tests added or updated

- `tests/test_grounding_plan.py`: validates GroundingPlan contracts, enum/option fidelity, compiler metadata, planner retry, and fallback.
- `tests/test_iterative_agent.py`: validates GroundingPlan-owned route/skill/target hints in the runtime planner prompt.
- `tests/test_runtime_source_cleanliness.py`: prevents benchmark semantic constants from re-entering runtime source.
- `tests/test_frame_cache.py`: validates 2fps frame-cache command construction and window-limited uniform sampling.
- `tests/test_qwen_vl_backend.py`: validates frame-backed requests become ordered image inputs.
- `tests/test_caption_qa_tools.py`, `tests/test_scene_index_builder.py`, and `tests/test_eval_runner.py`: validate frame cache is preferred over physical clips across runtime tools and VideoMME runner preprocessing.
- Existing policy/navigation/answer/iterative tests were updated to assert the new ownership boundary instead of old deterministic shortcuts.

## Next actions

1. Commit and push the frame-cache fix, then launch a fresh KML 3-demo run from that commit; previous KML paths should not be used as evidence for this version.
2. Inspect only compact remote summaries: result count, terminal status, top failure fingerprint, and output artifact paths.
3. If a demo fails, first check whether GroundingPlan output was valid/frozen before debugging tool routing or AnswerAgent verification.
4. Remaining design cleanup: convert AnswerAgent auto-final paths into verifier recommendations, remove dormant forced-visual branches, and add a controlled GroundingPlan amendment path.
