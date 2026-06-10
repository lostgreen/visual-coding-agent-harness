# Generic Grounding Runtime Closure Handoff

## Current Goal

Implement the generic runtime closure plan on isolated branch `codex/generic-grounding-runtime-closure`, without adding case-specific logic for the three VideoMME anchors.

## Current Evidence

- Branch base: `78148fe`.
- Current full local verification: `PYTHONPATH=src:. pytest -q` -> `518 passed`.
- Anti-specialization scanner is active and passed in the full suite.

## Important Decisions

- `GroundingPlan` now requires planner-owned `central_subjects`.
- Each `GroundingOption` now requires planner-owned `option_kind`.
- `OptionSpec` stores `raw_option_text` and `option_kind` while preserving existing ref-based `target_sequence`.
- `target_refs` are exact `T<n>` registry refs only.
- `additional_targets` is discovery-only and rejected for bound-target tools.
- Final gate has closed reason codes and returns `final_rejected` on rejected reserved final rounds instead of silently falling through to `max_rounds_reached`.

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

## Constraints

- Do not commit `plans/2026-06-10-video-agent-loop-fixes-plan.md` unless explicitly requested; it pre-existed as an untracked user plan file.
- Do not add runtime branches on case id, video id, ground-truth answer, option-specific answer, or anchor-case entity/phrase.
- Older logs/runs before this branch are stale for code correctness; current evidence is local tests only.

## Next Actions

1. Review final diff one more time.
2. Commit the branch.
3. Push `codex/generic-grounding-runtime-closure`.
4. If requested, sync KML and launch a fresh run from this branch.
