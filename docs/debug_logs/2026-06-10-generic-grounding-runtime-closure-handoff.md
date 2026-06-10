# Generic Grounding Runtime Closure Handoff

## Current Goal

Implement the generic runtime closure plan on isolated branch `codex/generic-grounding-runtime-closure`, without adding case-specific logic for the three VideoMME anchors.

## Current Evidence

- Branch base: `78148fe`.
- First closure commit: `43367db`.
- Runtime-closure commit: `143062b`.
- Remote regression on `43367db` completed at
  `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_generic_grounding_43367db_20260610_121447`.
- `43367db` remote failure fingerprint, now stale for current code:
  - `605-1`: `final_rejected`, zero evidence chains.
  - `611-2`: `route_repair_exhausted`, zero evidence chains.
  - `612-1`: `max_rounds_reached`, zero evidence chains.
  - All three bootstraps emitted `grounding_validation_failed`, then planner prompts rendered `No target_refs are registered for this run`.
- `143062b` remote no-text-planner run is stale:
  `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_grounding_runtime_143062b_20260610_171510`.
  All cases terminaled at `grounding_bootstrap_failed` because `planner_model_path` was empty.
- `143062b` text-planner run, current failure fingerprint before this note:
  `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_grounding_runtime_143062b_textplanner_20260610_180956`.
  `605-1` and `611-2` showed `grounding_bootstrap_failed` after two grounding attempts with `findings=[]`.
  This points to JSON parse/extraction failure or non-JSON model output, not schema validation.
- Current full local verification after parse/diagnostic hardening:
  `PYTHONPATH=src:. pytest -q` -> `544 passed`.
- Anti-specialization scanner is active and passed:
  `PYTHONPATH=src:. pytest -q tests/test_no_case_specific_logic.py` -> `3 passed`.

## Important Decisions

- `GroundingPlan` now requires planner-owned `central_subjects`.
- Each `GroundingOption` now requires planner-owned `option_kind`.
- `OptionSpec` stores `raw_option_text` and `option_kind` while preserving existing ref-based `target_sequence`.
- `target_refs` are exact `T<n>` registry refs only.
- `additional_targets` is discovery-only and rejected for bound-target tools.
- Final gate has closed reason codes and returns `final_rejected` on rejected reserved final rounds instead of silently falling through to `max_rounds_reached`.
- `ground_question` is routed through the text backend when a routed text backend is available.
- For planner-owned MCQ/QA grounding, bootstrap failure is now terminal and structured as `grounding_bootstrap_failed`; the agent no longer silently falls back to natural-language targets.
- `read_segment_detail(promote_answer_evidence=True)` uses registry targets when explicit `target_refs` are omitted and suppresses raw option-text promotion.
- Workspace post-observation hooks promote registry-backed ASR/OCR evidence into answer-facing evidence rows before the next planner round.
- `verify_ledger_answer("B")` resolves the letter to the corresponding option/registry claim before support scoring.
- Main-idea final gating requires per-option coverage and rejects single-option chases with `no_per_option_coverage`.
- Ordered-list extraction records source spans, excludes forward references, and no longer synthesizes observed timestamps for text-only spans.
- `eval_videomme` supports YAML config profiles and writes `resolved_config.json` under the run root.
- Grounding bootstrap now distinguishes `grounding_parse_failed` from `grounding_validation_failed`.
- Grounding prompt now explicitly requests JSON-only output, names allowed routes, evidence sources, option kinds, and skill ids.
- Grounding JSON extraction now skips prose/schema-note braces and takes the first balanced candidate that actually parses as JSON.
- Bootstrap trace now records only compact grounding raw-output diagnostics: `raw_text_chars` and a 500-character excerpt.

## Files Changed

- Grounding contracts/compiler/validator/prompt:
  `src/visual_coding_agent_harness/agents/grounding/{contracts,compiler,validator,planner}.py`
- Runtime/final gate:
  `src/visual_coding_agent_harness/agents/{contracts,final_gate,iterative_agent,prompt_stack}.py`
  `src/visual_coding_agent_harness/agents/skills/policy_constants.py`
- Tool protocol:
  `src/visual_coding_agent_harness/tools/{navigation,inspector}.py`
- Runtime target contract:
  `src/visual_coding_agent_harness/contracts/targets.py`
- Tests:
  `tests/test_{grounding_plan,final_gate,no_case_specific_logic,iterative_agent,route_validator,video_navigation,caption_qa_tools}.py`
  `tests/_anti_specialization_blacklist.txt`
- Follow-up closure files:
  `src/visual_coding_agent_harness/backends/routed.py`
  `src/visual_coding_agent_harness/tools/verification.py`
  `src/visual_coding_agent_harness/workspace.py`
  `src/visual_coding_agent_harness/agents/grounding/lexicon.py`
  `src/visual_coding_agent_harness/evals/videomme/runner.py`
  `configs/videomme_agent_v2_three_case.yaml`
  `tests/test_{grounding_bootstrap,promote_answer_evidence,verify_ledger_answer_option_aware,main_idea_option_evaluation,ordered_list_extraction}.py`

## Constraints

- Do not commit `plans/2026-06-10-video-agent-loop-fixes-plan.md` unless explicitly requested; it pre-existed as an untracked user plan file.
- Do not add runtime branches on case id, video id, ground-truth answer, option-specific answer, or anchor-case entity/phrase.
- Older logs/runs before this branch are stale for code correctness; current evidence is local tests only.

## Next Actions

1. Commit and push the grounding parse/diagnostic hardening on `codex/generic-grounding-runtime-closure`.
2. Sync KML to the pushed commit with `git pull --ff-only`.
3. Start a new YAML-backed VideoMME three-case run with the text planner model path set.
4. Treat the two `143062b` run roots above as stale unless comparing regressions.
5. Inspect only compact KML summaries: bootstrap status, prompt target_refs presence, evidence_chain_count, final_decision, and per-case trajectory paths.
