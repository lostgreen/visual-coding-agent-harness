# Agentic Video Exploration Harness Design

> Date: 2026-07-10
> Status: proposed design for review
> Scope: upgrade the active `VirtualVideoWorkspace` multi-round path and align it with MM-Lifelong-style evaluation

## 1. Product Story

The harness should treat a long video as an unfamiliar environment rather than as a document that must be fully converted into text before answering a question.

The intended workflow resembles how Claude Code or Codex works in an unfamiliar repository:

1. Start from a compact map of the environment.
2. Form a question-specific hypothesis and exploration plan.
3. Open a relevant area.
4. Inspect raw local material.
5. Record durable, source-grounded observations.
6. Reconcile observations with prior evidence.
7. Revisit, narrow, compare, or exclude areas as needed.
8. Answer only when the evidence workspace satisfies the question's proof requirements.

The system must not depend on a prebuilt semantic text index, full-video captions, query-independent entity extraction, or top-k retrieval candidates. Timestamped ASR is allowed when it already exists, but it is raw sensory material: it becomes visible only through Investigator actions over a selected segment or time window.

The central design principle is:

> Runtime memory is a product of Agent exploration, not a precomputed substitute for watching the video.

## 2. Why The Current Mechanism Needs To Change

The current virtual-video path has the right outer shape but insufficient state and evidence semantics.

### 2.1 What already works

- `VirtualVideoWorkspace` preserves source-to-virtual lineage.
- Reasoner starts from segment overview images instead of retrieval-selected beats.
- Investigator publicly exposes only `open_segment` and `inspect_window`.
- Low-fps frames are cached globally and higher-fps frames are sampled on demand.
- The driver supports multiple Reasoner-Investigator rounds and a task budget.
- The slim agent path already contains reusable primitives:
  - `EvidenceRecord`
  - `ClaimContract`
  - `CoverageSegment`
  - claim capability gates
  - entity observations and identity relations
  - derived evidence requirements

### 2.2 Failure patterns observed in the long VideoMME cases

The easy contiguous cases reached 4/6 accuracy, while the interleaved six-hour cases reached 1/4. The trajectories exposed structural failures rather than isolated prompt errors:

1. **Source-scope pollution**: evidence from distractor videos was counted as if it belonged to the question source.
2. **Virtual/source-time confusion**: source-relative clues such as "39-43 minutes" were interpreted as virtual concatenation time.
3. **Lost chronology**: phrases such as "the latter part" could not be resolved after source chunks were shuffled.
4. **Repeated exploration**: the Agent revisited wrong windows because it had no explicit coverage and exclusion ledger.
5. **Unstructured entity evidence**: person-counting answers relied on inconsistent natural-language descriptions instead of identity observations and deduplication relations.
6. **Weak citation gate**: a visual evidence ID was sufficient even when source scope, temporal coverage, and aggregate proof were missing.
7. **Sampling truncation**: a 64-frame window could be generated, but only the first 16 frames were sent to the VLM, omitting later parts of the window.
8. **Unused coarse observation**: low-fps preview frames were materialized but not used in evidence reasoning.
9. **No observation reuse**: repeated requests could resample the same area without gaining new information.
10. **First-ASR-hit bias**: `_choose_window_from_segment_packet` returns the first lexical cue that matches task terms, systematically favoring early segment content instead of comparing all plausible cue clusters.
11. **Unbounded exact entity lower bound**: `count_entity_bounds` uses an exact maximum-clique search whose exponential cost becomes unsafe when a long-horizon question yields dozens of observations.

### 2.3 Current data-model split

The repository currently has two evidence systems:

- The virtual path uses `InvestigationEvidence`, primarily a free-form summary with frames and lineage.
- The slim path uses structured `EvidenceRecord`, claim contracts, coverage manifests, and capability-aware verification.

The upgrade must unify the virtual path with the structured evidence system. A third evidence model would make consistency and verification harder.

## 3. Goals

### 3.1 Primary goals

1. Make the Agent explore an unknown long video without a prebuilt semantic text index.
2. Preserve a two-tool Investigator interface:
   - `open_segment`
   - `inspect_window`
3. Build query-time hot memory from verified multimodal observations.
4. Prevent repeated, source-invalid, future-leaking, or temporally inconsistent exploration.
5. Support local questions and long-horizon aggregation questions with the same evidence substrate.
6. Require explicit parent evidence and coverage for count, ordering, comparison, habit, and full-video claims.
7. Evaluate both final-answer quality and the evidence-seeking trajectory on MM-Lifelong.

### 3.2 Non-goals

- No full-video semantic caption database before a question starts.
- No mandatory text, visual, or multimodal embedding index.
- No query-independent person, object, or event graph construction.
- No physical concatenation of source videos.
- No full-video high-fps cache.
- No expansion of the public Investigator tool surface beyond two actions.
- No attempt to solve identity recognition with a large training pipeline in this milestone.

## 4. Cold Substrate Versus Hot Workspace

The architecture must make this distinction explicit.

### 4.1 Allowed cold substrate

The following artifacts may exist before a question starts because they are mechanical, query-agnostic, and source-preserving:

- source video paths and durations
- segment order and source offsets
- virtual-time mapping
- physical or wall-clock timestamps when the dataset provides them
- query-time cutoff metadata
- raw timestamped ASR cues
- an optional deterministic token-to-cue map for exact lexical ASR lookup
- low-fps frame cache
- segment overview contact sheets
- beat contact sheets used only for navigation
- frame and ASR lineage manifests

These artifacts describe where information can be inspected. They do not claim what the video means.

The lexical ASR map is equivalent to `grep` over raw subtitle files. It may normalize case and tokenize text, but it must not use embeddings, generated synonyms, semantic expansion, or summaries. Implementations may scan the cue file directly instead of materializing an inverted map; both must return the same cue IDs and lineage.

### 4.2 Forbidden cold semantics

The following must not be precomputed for benchmark runs:

- full-video captions or summaries
- per-segment semantic summaries generated by an MLLM
- text embeddings or query retrieval indices
- semantic ASR retrieval indices
- extracted entities, events, relationships, habits, or counts
- question-specific candidate windows
- target or distractor labels exposed to the Agent
- benchmark clue intervals, answers, reasons, or gold evidence

### 4.3 Query-time hot workspace

During one question, the Agent may create and persist:

- opened and inspected segment/window records
- raw multimodal observation evidence
- rejected or exhausted regions
- claims and their verification status
- entity and event observations
- identity and temporal relations
- conflicts and unresolved questions
- derived count, order, comparison, and summary evidence
- candidate answers grounded in verified claims

In benchmark cold-start mode, this semantic workspace is isolated per question. A later question over the same video does not receive the previous question's hot memory. A separate warm-memory experiment may reuse it, but results must be reported separately.

## 5. Target Architecture

```text
VirtualVideoManifest
  - observation timeline
  - source timeline
  - wall-clock timeline
  - raw ASR references
  - low-fps frame references
          |
          v
Initial Segment Map (<= 40 overview images)
          |
          v
Query Compiler
  - ClaimContract
  - source scope
  - temporal interpretation
  - completion requirements
          |
          v
Reasoner
  - reads workspace dashboard
  - proposes InvestigationTasks
          |
          v
Investigator inner loop
  open_segment
    -> choose coarse area
  inspect_window low-fps + local ASR
    -> narrow or increase fps when needed
  inspect_window detail
    -> emit atomic observations
          |
          v
Evidence Commit + Workspace Reconciliation
  - validate lineage and query cutoff
  - merge coverage
  - detect duplicate observations
  - update entity/event relations
  - derive aggregate evidence when justified
          |
          v
Completion Gate
  - enough modalities?
  - enough source scope?
  - enough temporal coverage?
  - aggregation parents complete?
          |
     yes / no
      |     |
   answer  repair task -> next round
```

## 6. Unified Time And Source Model

MM-Lifelong and EgoLife require more than virtual playback time. Every segment, frame, ASR cue, inspection request, and evidence record must support three clocks.

```python
@dataclass(frozen=True)
class TemporalRef:
    virtual_start_sec: float
    virtual_end_sec: float
    source_start_sec: float
    source_end_sec: float
    wall_clock_start: str | None = None
    wall_clock_end: str | None = None
    day_index: int | None = None
```

### 6.1 Clock semantics

- `virtual_time`: cumulative playable observation time in the workspace.
- `source_time`: position inside the original source video.
- `wall_clock_time`: physical chronology across days, including unobserved gaps.

### 6.2 Query cutoff

Datasets such as EgoLife and EgoMemReason ask questions at a particular time. The workspace must expose a `query_cutoff` and reject any observation whose wall-clock or source chronology occurs after it.

### 6.3 Source scope

Every question receives an explicit scope policy:

```python
@dataclass(frozen=True)
class SourceScope:
    allowed_source_video_ids: tuple[str, ...] = ()
    allowed_segment_ids: tuple[str, ...] = ()
    before_query_cutoff: bool = True
    scope_kind: str = "workspace"
```

For synthetic smoke cases, the evaluator may know which source is the target, but the Agent must infer the relevant source from observations. Once the Reasoner adopts a source hypothesis, evidence outside that scope must be marked as out-of-scope rather than silently combined.

## 7. Query Compiler And Proof Contract

Before exploration starts, the question is compiled into a proof contract. This is not retrieval: it describes what kind of evidence would be sufficient.

The existing `ClaimContract` should remain the core vocabulary:

- `required_scope`: local, window, multi-window, full-video
- `quantifier`: existential, universal, distinct count, total count, order, comparison
- `observation_target`: text, entity, object, event, action, relation, attribute
- `aggregation`: none, deduplicate, count, order, compare, summarize
- `required_observability`: ASR, OCR, visual
- `observability_mode`: all or any

The compiler adds:

- source-scope expectations
- temporal phrases and their clock interpretation
- query cutoff
- expected number of evidence regions when known from question semantics, not gold labels
- explicit stop conditions
- a version number and revision reason when the contract changes

Examples:

| Question | Contract |
|---|---|
| "What number is on the jersey?" | window, attribute, visual/OCR |
| "How many distinct scholars comment on Napoleon?" | full-video, distinct_count, entity, deduplicate, visual + ASR |
| "Which event happened first?" | multi-window, order, event, visual/ASR |
| "What does the wearer usually do after coffee?" | full-video, summarize, event pattern, multi-window |

The compiler may be model-assisted, but the output is validated against a finite schema. It must never receive clue intervals or answers.

Because schema validity does not imply semantic correctness, the contract is a revisable hypothesis rather than an immutable oracle:

- every contract starts at `version=1`
- Reasoner may propose a revision after evidence or repeated gate failures
- a revision records the old contract, new contract, reason, and triggering evidence IDs
- at most two contract revisions are allowed by default
- after two failed completion repairs, the run produces both a grounded refusal and an ungated best candidate rather than looping indefinitely

For the focused development set, a small manually reviewed gold-contract file is maintained only for evaluator diagnostics. It is never loaded by Agent code. Contract precision, scope error, modality error, and unnecessary-overconstraint rate are reported separately.

## 8. Investigation Task Contract

`InvestigationTask` should evolve from a loose goal string into a verifiable request.

```python
@dataclass(frozen=True)
class InvestigationTask:
    query_id: str
    goal: str
    claim_ids: tuple[str, ...]
    segment_id: str = ""
    time_range: tuple[float, float] | None = None
    modality_hint: tuple[str, ...] = ()
    expected_observation: str = ""
    source_scope: SourceScope | None = None
    exploration_intent: str = "discover"
    completion_condition: str = ""
    priority: float = 0.0
```

Allowed exploration intents include:

- `discover`: look for a new relevant event or entity.
- `verify`: inspect an already suspected fact at higher detail.
- `compare`: collect evidence needed to relate two observations.
- `exclude`: test whether a plausible region can be ruled out.
- `complete_coverage`: inspect an uncovered interval required by an aggregate claim.

Reasoner tasks should not contain answer-option judgments. They describe evidence needs, not likely answers.

## 9. Two-Tool Investigator Protocol

The public tool surface remains fixed.

### 9.1 `open_segment`

Purpose: navigation and orientation, not final evidence.

It returns a bounded packet:

- segment ID and three-clock interval
- source lineage
- beat thumbnail pages
- ASR density and bounded local excerpts
- prior exploration state for this segment
- uncovered and previously inspected ranges

It must not return a generated full-segment semantic summary. Raw ASR excerpts are permitted, but output is capped and paged to prevent `open_segment` from becoming a hidden full-text retrieval call.

The method may accept navigation parameters while preserving the tool name:

```python
open_segment(segment_id=None, page=0, page_size=12, asr_query="")
```

When `asr_query` is non-empty, `open_segment` may perform a global exact lexical scan over raw ASR cues. This does not add a third public tool. The response contains paginated cue matches, source/time lineage, and the number of cues searched. It does not contain generated summaries or semantic matches.

Global lexical ASR lookup follows four rules:

1. Results are navigation hints or direct ASR quotes, never visual evidence.
2. Joint-modality contracts still require a subsequent visual `inspect_window` around the cue.
3. All matches are ranked or paged; selection must not return only the first chronological cue.
4. The trace records the query, searched cue count, returned cue IDs, and pagination state.

Matching cues are grouped into temporal clusters. Investigator chooses among clusters using the visual page, source/day coverage, task contract, and prior evidence yield. Chronological order may be a tie-breaker but never the primary selection rule.

### 9.2 `inspect_window`

Purpose: produce evidence-bearing observations from a selected time range.

```python
inspect_window(
    start_sec,
    end_sec,
    fps=0.5,
    max_frames=64,
    modalities=("visual", "asr"),
    observation_goal="...",
)
```

Requirements:

- Frames uniformly cover the complete requested window.
- If the VLM image budget is lower than the sampled frame count, the final VLM subset is also uniformly distributed across the complete window. It must not use `frames[:16]`.
- Low-fps preview frames and local ASR are actually supplied to the Investigator's first observation call.
- Higher fps is requested only after the preview identifies a narrower unresolved region or the task contract requires motion, OCR, or fine attributes.
- A repeated window reuses existing observations unless the request changes modality, fps, crop, or evidence goal.
- Each invocation has a unique observation call ID even when the same query is retried.
- Cross-segment windows are split into lineage-preserving source windows.

### 9.3 Observation equivalence and cache semantics

Frame reuse and semantic-observation reuse are different operations.

#### Frame materialization reuse

Raw frame files may be reused whenever an existing source-time sample set covers the requested interval at an equal or higher effective fps. A canonical materialization key uses:

- source video ID
- source interval aligned to a five-second grid
- fps tier
- crop or resolution tier

Near-identical floating-point windows therefore map to stable cache regions. The returned manifest still reports the exact requested and actual covered intervals.

#### Semantic observation reuse

An existing evidence-bearing observation is equivalent only when all conditions hold:

- same source video
- source-time IoU at least `0.8`
- existing modalities are a superset of requested modalities
- existing effective fps is at least the requested fps tier
- normalized observation-goal fingerprint matches
- no newer conflicting evidence requires reinspection

When the time range matches but the observation goal differs, frames are reused while the VLM observation is rerun and a new evidence record is created.

Duplicate-inspection metrics operate on source-time IoU and capability equivalence, not exact `(start, end)` floating-point equality.

### 9.4 Investigator inner loop

One Reasoner task may use up to three internal observation steps without consuming additional task budget:

1. `open_segment` for orientation.
2. coarse `inspect_window` using low fps and local ASR.
3. optional narrower `inspect_window` at 1 or 2 fps.

The Investigator may return:

- satisfied with observations
- partially satisfied with a suggested follow-up
- exhausted with explicit searched coverage
- invalid because the requested time or source scope cannot be resolved

## 10. Evidence Model

### 10.1 Atomic evidence

The virtual path should emit the existing `EvidenceRecord` rather than `InvestigationEvidence`.

An atomic record states only what was observed:

```text
"A gray-haired man in a white shirt appears in a talking-head shot while the ASR discusses Napoleon's military reforms."
```

It must not state:

```text
"This proves option B, three scholars."
```

Required additions to `EvidenceRecord` are lineage and observation identity:

- `task_id`
- `observation_id`
- `segment_id`
- `source_video_id`
- `source_time_range`
- `wall_clock_range`
- `entity_ids` and `event_ids` when assigned after reconciliation

The evidence store remains append-only. Corrections create superseding or conflicting records instead of mutating historical observations.

### 10.2 Derived evidence

Aggregate claims cannot cite a set of unrelated atomic observations directly. The workspace must produce a derived `EvidenceRecord` with:

- `modality="derived"`
- `evidence_kind="aggregate"`
- parent evidence IDs
- coverage manifest
- deterministic operation metadata
- explicit uncertainty or count bounds

Supported MVP operations:

- entity deduplication and count bounds
- event occurrence counting
- temporal ordering
- state comparison
- multi-window summary with parent closure

The existing `count_entity_bounds` is the starting point for distinct-person questions. Unknown identity relations produce lower and upper bounds instead of a false exact count.

Entity reconciliation uses a scale guard:

- up to 16 reconciled entity groups: exact maximum-clique lower bound
- above 16 groups: deterministic greedy clique lower bound plus the existing conservative upper bound

Derived count evidence records `algorithm="exact"` or `algorithm="greedy_lower_bound"`. An approximate lower bound cannot support an exact count unless it equals a separately justified upper bound.

### 10.3 Negative evidence

"Not seen" is not global evidence unless coverage supports the absence claim. Sparse frames may rule out a local frame but cannot refute a multi-window or full-video claim. The existing capability gate already encodes this principle and should be used by the virtual path.

## 11. Exploration Ledger

The Agent needs a durable map of where it has looked and what was learned.

```python
@dataclass(frozen=True)
class ExplorationVisit:
    visit_id: str
    task_id: str
    segment_id: str
    virtual_range: tuple[float, float]
    source_range: tuple[float, float]
    wall_clock_range: tuple[str | None, str | None]
    modalities: tuple[str, ...]
    sampling_fps: float
    status: str
    evidence_ids: tuple[str, ...]
    evidence_yield: int
    exclusion_reason: str = ""
```

The ledger derives per-segment state:

- unseen
- overview_opened
- partially_inspected
- exhausted_for_goal
- excluded_for_goal
- conflict_requires_revisit

It also reports:

- covered ranges per modality
- duplicate inspection ratio
- evidence yield per visit
- unresolved gaps required by claim contracts
- sources and days already checked
- explicit exclusion reasons

The Reasoner may revisit an interval only when at least one condition changes:

- a different claim or observation goal
- a new modality
- a higher-detail sampling plan
- a conflict that requires verification
- an incomplete aggregate coverage requirement

## 12. Evidence Workspace Dashboard

The Reasoner should no longer receive only the last few natural-language summaries. It receives a structured dashboard rendered from persistent artifacts.

```text
Question contract
Budget and round
Source and temporal hypotheses
Segment familiarity map
Coverage gaps
Verified atomic evidence
Derived evidence
Entity/event tables
Claim ledger
Conflicts
Open questions
Recommended repair requirements from the completion gate
```

The dashboard must remain bounded:

- initial segment overview images are sent once, or reused through conversation state when supported
- later rounds receive IDs and compact structured summaries
- detailed frames remain behind Investigator reports
- old evidence is referenced by ID rather than repeatedly copied in full

The default Reasoner dashboard text budget is 8,000 model tokens, excluding image payloads. If the projection exceeds the budget, records are kept in this priority order:

1. completion-gate repair requirements
2. unresolved questions and contract revisions
3. conflicts and unknown entity relations
4. uncovered source/day ranges required by the contract
5. newly verified evidence and derived evidence
6. older verified summaries
7. exhausted or low-yield visit detail

Truncated sections retain counts and stable IDs so the Reasoner can request a focused view in a later round. The full append-only artifacts remain on disk.

This makes the workspace analogous to a coding Agent's file changes, diagnostics, and task ledger rather than an ever-growing chat transcript.

## 13. Multi-Round Control Loop

The upgraded driver uses a fixed state transition:

```text
compile query
  -> reason
  -> accept tasks
  -> investigate
  -> commit atomic evidence
  -> reconcile workspace
  -> derive aggregates
  -> verify claims
  -> completion gate
  -> answer or generate repair requirements
```

### 13.1 Task budget

- `max_investigations` counts accepted Reasoner tasks.
- Investigator internal observation steps do not consume task budget.
- Frames, VLM calls, ASR characters, and wall time are recorded separately as cost.

### 13.2 Stop behavior

The Reasoner may answer early only when the completion gate passes. Reaching max rounds does not silently relax evidence requirements.

Every run produces two explicitly separated outputs:

- `grounded_answer`: returned only when the completion gate passes; otherwise `Insufficient verified evidence`
- `forced_answer`: the best candidate answer from the same workspace state, produced without further tool calls and marked `ungated=true`

The forced answer exists for answer-accuracy comparability with baselines such as ReMA, which force finalization and permit a best-effort answer. It can never be presented as grounded output, and its citations remain subject to ordinary existence and lineage checks even though proof completeness may be missing.

Primary system reporting emphasizes grounded accuracy and grounding quality. `forced_answer_accuracy` is reported separately for direct comparison with permissive baselines.

### 13.3 Repair loop

When a final answer fails verification, the gate returns machine-readable repair requirements, for example:

```json
{
  "reason": "aggregation_or_coverage_missing",
  "required_action": "complete_coverage",
  "uncovered_sources": ["day4", "day5"],
  "required_modalities": ["visual", "asr"]
}
```

These requirements become the next round's highest-priority open questions.

The same completion failure may trigger at most two repair rounds for the same contract version. A third failure closes the grounded path, emits unmet requirements, and produces the ungated forced answer. A contract revision resets the repair counter but is capped as defined in Section 7.

## 14. Final Evidence Gate

The current `_citations_are_visual` check is insufficient. Final verification must check:

1. All citations exist.
2. Every cited visual record contains an attested observation, not only frame paths.
3. Evidence occurs before the query cutoff.
4. Evidence belongs to the adopted source scope.
5. Temporal references are consistent across virtual, source, and wall-clock time.
6. Required modalities are present according to `ClaimContract`.
7. Required temporal scope is satisfied.
8. Aggregate claims cite derived evidence.
9. Derived evidence has valid parent closure and sufficient coverage.
10. Exact entity counts have equal lower and upper bounds.
11. Option claims are balanced for multiple-choice questions.
12. The selected answer is supported by cited verified claims.

For open-ended MM-Lifelong questions, final verification operates on answer claims rather than answer options.

## 15. MM-Lifelong Integration

MM-Lifelong is the primary benchmark because it provides real long-horizon timelines and clue-grounded evidence intervals without requiring synthetic distractor concatenation.

### 15.1 Workspace mapping

- Day split: one 23.6-hour continuous gameplay timeline.
- Week split: seven EgoLife day segments totaling 51.9 hours.
- Month split: livestream sessions totaling 105.6 hours across a 51-day physical span.

The adapter maps dataset clips into `VirtualVideoSegment` records while retaining physical gaps in wall-clock metadata. It does not place clue intervals, answers, or temporal certificate labels in `case.json` fields visible to the Agent.

### 15.2 Evaluator-only fields

The evaluator separately stores:

- gold answer
- question type
- temporal certificate
- clue intervals
- dataset split
- source clip IDs

### 15.3 Metrics

Final-answer metrics:

- exact match or normalized answer accuracy
- category accuracy
- temporal-certificate accuracy

Evidence metrics:

- clue interval recall
- clue interval precision
- temporal IoU
- all-required-clues-found rate
- source/day recall
- evidence citation validity
- derived-parent closure validity

For Ref@N-style grounding, the predicted intervals are defined as the source/virtual ranges of evidence cited by the final answer. Exploration visits that were not cited do not enter the predicted interval set.

Two additional interval views are reported but never substituted for final grounding:

- `all_observed_intervals`: all evidence-bearing observations
- `all_exploration_intervals`: every inspected window, including negative and irrelevant visits

The first measures discovery recall; the second is used only for trajectory efficiency and coverage analysis. This avoids penalizing a correct final grounding prediction simply because the Agent explored and rejected other regions.

Agent trajectory metrics:

- accepted tasks
- rounds
- segment opens
- window inspections
- duplicate inspection ratio
- inspected duration divided by total duration
- evidence yield per inspection
- frames and VLM calls
- time to first relevant evidence
- time from first evidence to final answer
- repair-loop count

### 15.4 Recommended evaluation tiers

1. **Focused development set**: 20 questions sampled only from `train@month`, including count, order, state, causal, and language-content tasks.
2. **Long subset**: all questions labeled `Long`.
3. **Ultra-long subset**: all questions labeled `Ultra-Long`.
4. **Full benchmark**: Day, Week, and Month reported separately.
5. **Cold versus warm memory**: optional secondary experiment, never mixed with primary cold-start results.

The 20-question set is for qualitative trajectory inspection, prompt/schema development, and regression tests. It is not used to claim statistically meaningful B3/B4/B5 improvements.

`val@month` is reserved for aggregate development reporting. `test@day` and `test@week` remain OOD test sets. A single Week example may be used as a labeled diagnostic fixture during implementation, but it is permanently excluded from all reported benchmark numbers and identified as test-set-exposed in the repository.

Architecture claims require the complete Long/Ultra-Long evaluation tier with multiple seeds where model sampling is nondeterministic.

## 16. ReMA Baseline Comparison And Research Delta

This design is informed by the official MM-Lifelong baseline, Recursive Multimodal Agent (ReMA):

- Paper: <https://arxiv.org/html/2603.05484v1>
- Official code: <https://github.com/cg1177/Recursive-Multimodal-Agent>

The goal is not to make the control loop more complicated than ReMA. The goal is to keep a similarly simple recursive loop while changing what the Agent is allowed to know before exploration and how observations become trustworthy memory.

### 16.1 What official ReMA actually does

ReMA uses an offline two-phase design.

#### Phase 1: full-video passive perception

Before answering a question, the official implementation:

1. Physically merges each benchmark subset into one chronological MP4.
2. Caches the complete video at 2 fps in LMDB.
3. Splits the full timeline into five-minute clips by default.
4. Samples each clip at 1 fps, capped at 300 frames.
5. Calls an MLLM to generate a generic timestamped caption for every clip.
6. Imports all captions into Mem0 backed by FAISS and `text-embedding-3-large`.
7. Uses GPT-4.1-mini for memory maintenance and LLM reranking.

The paper calls this the Perception Loop and `MemoryManage`. The released code makes the preprocessing dependency explicit: the first run performs full frame caching, clip caption generation, and initial memory construction before the query-time Agent loop starts.

#### Phase 2: query-time recursive control

At question time, the controller has three actions:

- `memory_search_tool`: vector-search prebuilt caption memories, rerank them, summarize them for the query, and write the new summary back into memory.
- `video_inspect_tool`: inspect up to ten requested intervals at 2 fps and write a question-conditioned caption back into memory.
- `finish`: return the answer.

The loop is a straightforward ReAct-style recursion with a maximum iteration count. On the final iteration, the released runner forces `finish` and explicitly allows an answer from general world knowledge when evidence is insufficient.

### 16.2 Strengths we should preserve

ReMA establishes several useful baseline decisions:

1. **Simple controller loop**: reason, call one of a few tools, observe, repeat.
2. **Separate controller and perception models**: the reasoning model does not directly process every frame.
3. **Recursive refinement**: retrieved information can lead to a targeted raw-video inspection.
4. **Uniform temporal sampling**: detailed inspections distribute frames across the requested interval.
5. **Persistent external state**: the controller does not rely only on an ever-growing chat context.
6. **Explicit temporal grounding metric**: answer accuracy is reported together with clue localization.

Our harness should remain comparable at this level:

```text
ReMA: Reason -> MemSearch/MMInspect -> MemoryManage -> Reason/Answer
Ours: Reason -> open_segment/inspect_window -> EvidenceCommit -> Reason/Answer
```

Both loops remain small. The difference lies in preprocessing assumptions, memory semantics, and proof requirements.

### 16.3 Where ReMA is limited for our research question

#### A. It depends on full-video semantic preprocessing

ReMA's global localization ability comes from captions generated for every five-minute clip before the question begins. The paper's ablation shows that two-minute captions outperform five-minute captions, which also means better recall requires more full-video perception calls and a larger semantic index.

This is a strong retrieval baseline, but it does not test whether an Agent can become familiar with an unknown video through query-driven exploration.

#### B. Caption recall is an irreversible bottleneck

If a small visual detail, identity cue, transient object, OCR string, or brief action is omitted from the generic caption, `memory_search_tool` cannot retrieve it. The Agent may inspect raw video only after it somehow obtains a plausible time range from the caption memory or guesses one.

This is precisely the failure mode our project targets: details discarded by query-independent preprocessing should remain recoverable through later exploration.

#### C. Memory is free-form language rather than evidence

Both passive captions and query-time inspection answers are inserted into Mem0 as natural-language memory. Query-conditioned summaries are also written back into the same store. The released implementation does not distinguish:

- raw observation
- model interpretation
- query-conditioned summary
- aggregate conclusion
- contradictory evidence

As memories are merged, reranked, and summarized, a final statement may no longer have a recoverable chain to the frames that originally supported it.

#### D. Memory consolidation may remove rare details

ReMA compresses overlapping or semantically similar memories into a new summary. This controls memory size but makes compression destructive. A detail irrelevant to an earlier query can disappear even though it is essential to a later one.

Our harness must compress the dashboard, not the evidence store. Atomic observations remain immutable and can always be reopened or reinterpreted.

#### E. There is no explicit exploration coverage model

The official controller can repeatedly search similar terms or reinspect overlapping windows. It has no first-class representation of:

- explored versus unexplored ranges
- source/day coverage
- exclusions
- duplicate inspection
- evidence yield
- unresolved gaps required by an aggregate question

The memory vector store answers "what stored text resembles this query?" It does not answer "which necessary parts of the video have not yet been checked?"

#### F. It collapses timeline provenance

The official instructions merge clips into one chronological MP4 and reason mainly with global `HH:MM:SS` offsets. MM-Lifelong distinguishes observational duration from physical temporal span, but the released Agent state has no explicit source-time, virtual-time, and wall-clock-time object carried through every memory record.

This makes source attribution, unobserved gaps, cross-day order, and query-time cutoffs harder to verify.

#### G. It does not enforce structured aggregation

Counting, temporal ordering, state tracking, and social questions are answered from summarized memory. There is no requirement to enumerate parent observations, resolve identities, record uncertainty, or prove coverage before returning an exact conclusion.

#### H. Finalization is permissive

The official loop forces an answer at the last round and permits general world knowledge when video evidence is insufficient. That is useful for maximizing answer rate but unsuitable for an evidence-grounded harness that aims to diagnose exploration failures.

#### I. The released audio path is not central to the implementation

The paper describes Whisper and multimodal inputs, but the released `global_caption.py` and `video_inspect_tool` have their direct audio payload paths commented out. Our benchmark runner should make timestamped ASR handling explicit and traceable instead of relying on an implicit or model-specific audio path.

### 16.4 Our baseline loop should stay equally simple

The main Agent loop should not expose the internal evidence machinery as more controller tools.

```python
for round_id in range(max_rounds):
    decision = reasoner.decide(workspace.dashboard())
    if decision.action == "answer":
        result = completion_gate.verify(decision, workspace)
        if result.passed:
            return decision.answer
        workspace.add_repair_requirements(result.repairs)
        continue

    reports = investigator.run_batch(decision.tasks)
    workspace.commit(reports)
    workspace.reconcile()

return insufficient_verified_evidence(workspace.unmet_requirements())
```

The public control actions remain:

- `investigate`
- `answer`

The public Investigator actions remain:

- `open_segment`
- `inspect_window`

Coverage merging, observation reuse, entity reconciliation, and derived evidence generation are deterministic workspace transitions, not additional actions the Reasoner must learn to call.

### 16.5 The core research advance

The proposed harness replaces ReMA's **precompute, retrieve, and summarize** paradigm with **browse, observe, commit, and verify**.

| Dimension | Official ReMA | Proposed harness |
|---|---|---|
| Initial semantic state | Full-video generic captions | None |
| Unseen-video access | Search caption memory | Browse hierarchical visual map |
| ASR | Part of preprocessing/model path | Raw cues exposed only in opened scope |
| Memory unit | Free-form text memory | Atomic evidence with lineage |
| Memory compression | May replace/merge memories | Immutable evidence, compressed dashboard |
| Localization | Vector retrieval over all clips | Coverage-aware Agent exploration |
| Detailed inspection | Question-conditioned clip caption | Progressive low-to-high-detail observation |
| Time model | Merged global offset | Virtual + source + wall-clock time |
| Counting | Inferred from summaries | Entity observations, relations, count bounds |
| Ordering/comparison | Inferred from summaries | Derived evidence with parent closure |
| Negative findings | Free-form memory | Scope-limited exclusion with coverage |
| Final answer | Controller decides, forced at limit | Contract-aware evidence gate |
| Grounding trace | Extracted intervals | Native citations, parents, and coverage |

The research claim is not that retrieval is unnecessary in every production system. It is narrower:

> For detail-sensitive and multi-clue lifelong questions, preserving access to unexplored raw video and building query-time, provenance-complete evidence can recover information that query-independent semantic preprocessing permanently discards.

### 16.6 Hierarchical browsing without a semantic index

Removing prebuilt captions creates a real localization challenge. The replacement must be a visual and temporal browsing hierarchy, analogous to navigating directories before opening files.

```text
workspace map
  -> physical day/session overview
      -> segment page
          -> beat contact sheets
              -> low-fps window inspection
                  -> narrow high-fps verification
```

This hierarchy is query-agnostic and visual. It may contain raw ASR excerpts only for the currently opened page.

For MM-Lifelong:

- Month starts from the 23 livestream sessions, each with a session overview.
- Week starts from the seven EgoLife days.
- Day starts from chronological gameplay chapters or hour blocks.

When one level exceeds the thumbnail budget, adjacent intervals are grouped into pages. `open_segment` handles paging without adding a new public tool.

### 16.7 Global lexical ASR navigation

Raw ASR can improve navigation without recreating ReMA's global caption index. The correct coding-Agent analogy is repository-wide `grep`: exact lexical lookup over original files is a query-time mechanical operation, not semantic preprocessing.

Allowed behavior:

- Investigator may perform a global exact lexical scan over raw ASR cues before selecting a day or session.
- Matches are returned with complete cue lineage and searched coverage.
- Results are paginated and capped per call.
- The match is a navigation hint or ASR quote, not proof of an associated visual fact.

Forbidden behavior:

- embedding all ASR before the query
- semantic search, synonym expansion, or embedding retrieval across ASR
- generated summaries for every ASR block
- treating an ASR keyword match as visual evidence

This preserves the intended distinction: ASR helps the Agent look, but does not replace looking.

The visual side still has no true `grep` equivalent. Hierarchical contact-sheet browsing and progressive inspection are the proposed mechanism for closing that asymmetry.

### 16.8 Evidence-preserving memory management

Our replacement for ReMA's `MemoryManage` has two layers.

#### Immutable evidence layer

- atomic observations are append-only
- frame and ASR references are retained
- corrections add conflict or supersession links
- derived evidence names all parent evidence
- no summarizer can delete source observations

#### Mutable workspace projection

- compact claim status
- entity/event tables
- coverage summaries
- familiarity state
- unresolved questions
- best current answer candidate

The mutable projection may be regenerated at any time from immutable artifacts. This gives ReMA-like bounded controller context without sacrificing rare details.

### 16.9 Coverage-guided exploration instead of memory retrieval

ReMA's `MemSearch` ranks stored semantic memories. Our Reasoner chooses the next frontier using a bounded workspace dashboard.

The frontier score should combine:

- relevance of the segment overview to the current claim
- whether the source/day is still uncovered
- temporal clue compatibility
- expected information gain
- prior evidence yield
- conflict urgency
- revisit penalty

The first implementation does not require a learned scorer. The Reasoner receives the factors and selects a frontier. Deterministic guards reject redundant requests and expose the next uncovered ranges.

### 16.10 Fair baseline matrix

Evaluation must separate the value of the loop from the value of preprocessing.

| ID | System | Full-video captions | Vector memory | Structured evidence | Coverage ledger |
|---|---|---:|---:|---:|---:|
| B0 | End-to-end sparse frames | no | no | no | no |
| B1 | Official ReMA | yes | yes | no | no |
| B2 | ReMA-style loop over raw segment map, no global ASR grep | no | no | no | no |
| B2-L | B2 + exact global lexical ASR navigation | no | no | no | no |
| B3 | B2-L + immutable evidence workspace | no | no | yes | no |
| B4 | B3 + coverage and observation reuse | no | no | yes | yes |
| B5 | B4 + derived entity/event evidence | no | no | yes | yes |

This matrix answers four separate questions:

1. How much performance comes from full-video caption preprocessing?
2. Can a simple Agent loop localize evidence from a raw visual map?
3. Does provenance-preserving evidence improve final grounding?
4. Do coverage and structured aggregation improve multi-clue questions?
5. How much does global lexical ASR navigation contribute without semantic indexing?

### 16.11 Cost accounting

ReMA pays a large one-time preprocessing cost that can be amortized over multiple questions. Our cold-start Agent pays mostly query-time exploration cost. Both views must be reported.

Metrics must include:

- full-video frames processed before the first question
- preprocessing MLLM calls and tokens
- query-time frames and calls
- memory embedding and reranking calls
- wall-clock latency
- disk footprint
- total cost for 1, 10, and 100 questions over the same video

Primary single-question evaluation includes preprocessing cost. A secondary amortized evaluation reports cost per question as the question count grows.

### 16.12 Expected areas of advantage

The proposed harness is expected to be strongest on:

- OCR and transient fine-detail questions
- identity and distinct-count questions
- clues omitted from generic captions
- questions needing visual and ASR confirmation
- source-specific or query-cutoff questions
- multi-window ordering, comparison, and state tracking
- cases where the first plausible memory hit is incomplete or misleading

ReMA may remain stronger on broad semantic questions where generic captions provide excellent global recall at low query-time cost. The evaluation should report this honestly rather than assume one architecture dominates every category.

### 16.13 Concrete benchmark hypotheses

The MM-Lifelong experiments should test the following hypotheses:

1. **Detail recovery**: B3-B5 recover more gold clue intervals than B1 on attribute, OCR, state-change, and entity questions whose details are absent from passive captions.
2. **Grounding fidelity**: B3-B5 improve clue precision and citation validity even when answer accuracy is similar.
3. **Exploration efficiency**: B4 reduces duplicate inspection and inspected-duration ratio relative to B2/B3.
4. **Aggregation reliability**: B5 improves counting, temporal reasoning, and social-interaction questions over free-form summary memory.
5. **Cold-start trade-off**: B1 is cheaper after sufficient question amortization, while B4/B5 are cheaper and more faithful for one-off, detail-sensitive questions.
6. **Modality necessity**: joint visual-ASR contracts reduce ASR-only shortcuts without reducing evidence recall on language-content questions.
7. **Navigation asymmetry**: B2-L improves Language Content Recall and needle localization over B2, while the remaining gap on visual-only needles measures the cost of having no visual `grep`.

### 16.14 Baseline implementation policy

The official repository should be run as an external reference baseline rather than copied into the active codebase. Its `finish` wrapper is minimally extended to require an answer and final evidence intervals, without changing its caption preprocessing, memory search, or inspection policy. Its outputs are normalized into the same benchmark result schema:

```text
answer
predicted evidence intervals
preprocessing cost
query-time cost
tool trace
```

The active harness should include a thin adapter and evaluator, not a fork of Mem0 or the ReMA implementation. This keeps the comparison auditable and prevents official-baseline behavior from leaking into the proposed architecture.

For both systems, Ref@N uses only final-declared evidence intervals. A shared post-run validator checks interval syntax and lineage but does not see gold clues. Paper-reported ReMA numbers remain cited as external reference values; newly run normalized numbers are labeled separately because the final-output schema is stricter than the unmodified release.

## 17. Artifact Layout

```text
workspace/
  virtual_timeline.json          # cold source/virtual/wall-clock truth
  case.json                      # question-visible case data only
  frame_manifest.jsonl           # cold low-fps frame truth
  asr_virtual_cues.json          # cold raw ASR truth
  segment_overviews/             # cold visual map
  beat_index.json                # cold navigation metadata, no semantic summaries
  observations/
    window_frame_manifest.jsonl  # query-time frame observations
  run/
    query_contract.json
    evidence.jsonl               # atomic and derived EvidenceRecords
    exploration.jsonl            # ExplorationVisit records
    claims.jsonl                 # claims and verdicts
    workspace.json               # bounded derived dashboard state
    trace.jsonl                  # full multi-round interaction trace
    answer.json                  # grounded_answer and forced_answer kept separate
    metrics.json
  evaluator/
    gold.json                    # never loaded by Agent code
    gold_contracts.dev.json      # train@month diagnostics only
```

## 18. Module Boundaries

The implementation should prefer focused modules and migrate incrementally.

### Existing modules to modify

- `src/vcah/virtual_video.py`
  - add wall-clock metadata, query cutoff, and three-clock conversion
- `src/vcah/investigator.py`
  - emit structured `EvidenceRecord`
  - implement observation reuse and the bounded inner loop
- `src/vcah/multiround.py`
  - drive the workspace state machine and completion/repair loop
- `src/vcah/types.py`
  - add source/time lineage fields without weakening existing evidence validation
- `src/vcah/memory.py`
  - support loading/appending run-scoped evidence and claim artifacts
- `src/vcah/verifier.py`
  - add source, cutoff, lineage, and parent-closure gates
- `src/vcah/entities.py`
  - extend entity reconciliation and derived count evidence
- `src/vcah/virtual_index.py`
  - remove generated ASR short summaries from initial Reasoner context
- `tools/build_virtual_trace_viewer.py`
  - render every round, all tasks, coverage, claims, derived evidence, and final gate results

### New focused modules

- `src/vcah/exploration.py`
  - exploration visits, interval merging, duplicate detection, exclusions, frontier gaps
- `src/vcah/workspace_state.py`
  - persistent state projection and bounded Reasoner dashboard
- `src/vcah/query_contracts.py`
  - question-to-contract compilation and schema validation
- `src/vcah/asr_navigation.py`
  - exact lexical cue lookup, pagination, clustering, and lineage
- `src/vcah/aggregation.py`
  - deterministic derived evidence for count, order, comparison, and summaries
- `src/vcah/benchmarks/mm_lifelong.py`
  - dataset loader, workspace builder, and hidden evaluator metadata
- `src/vcah/benchmark_metrics.py`
  - answer, evidence, and trajectory metrics
- `tools/run_mm_lifelong_agent.py`
  - cold-start benchmark runner
- `tools/run_mm_lifelong_rema_baseline.py`
  - invoke or import normalized results from the external official ReMA run

The migration should remove `InvestigationEvidence` after the virtual path writes and consumes `EvidenceRecord` end to end.

## 19. Delivery Phases

### Phase 0: Baseline replay and trace contract

Goal: freeze current behavior and make regressions measurable.

- Add compact replay fixtures for one correct VideoMME trajectory and three representative failures.
- Fix the viewer to render all rounds rather than only the first Reasoner task set.
- Record current repeated-window, source-scope, and evidence-yield metrics.
- Define the normalized output schema for official ReMA and the proposed harness.
- Run official ReMA on a small MM-Lifelong development slice and capture preprocessing plus query-time cost.

Exit criteria:

- Existing runs can be replayed without model calls.
- All rounds and evidence frames are visible in the viewer.
- Baseline metrics are reproducible.
- Official ReMA and the active harness can be compared through the same answer, interval, trace, and cost schema.

### Phase 0.5: Immediate behavior corrections

Goal: repair three confirmed low-cost defects before introducing new state models, then establish a cleaner attribution baseline.

- Replace `high["frames"][:16]` with a uniformly distributed 16-frame subset using the existing frame-selection helper.
- Feed low-fps preview frames and local ASR into the first Investigator VLM call.
- Remove unconditional full-window 2 fps inspection; request a narrower detail window only after preview uncertainty or a fine-detail contract requires it.
- Replace the invalid-output fallback of `segment_start + 60s` with a coverage-aware fallback selected from the best unresolved beat/page rather than the beginning of the segment.
- Replace first-ASR-hit selection with clustered candidate selection.
- Replay the four interleaved six-hour cases before changing evidence types.

Exit criteria:

- VLM frame inputs cover the full selected window.
- Preview observations appear in the trace and influence detail-window selection.
- High-fps sampling is conditional and normally narrower than preview.
- Invalid window-selection output does not default to the first minute.
- The updated four-case result becomes the attribution baseline for Phases 1-7.

### Phase 1: Three-clock timeline and source scope

Goal: prevent source and chronology errors before changing reasoning behavior.

- Add wall-clock metadata and query cutoff.
- Add source-scope validation.
- Add conversion tests for cross-segment and cross-day intervals.

Exit criteria:

- Future evidence is rejected.
- Source-relative and wall-clock-relative questions resolve to the correct source windows.
- Distractor-source evidence cannot satisfy a scoped claim.

### Phase 2: Unified evidence workspace

Goal: replace free-form virtual evidence summaries with structured evidence records.

- Adapt Investigator output to `EvidenceRecord`.
- Persist claims, evidence, and workspace projection.
- Use existing capability gates in the virtual driver.

Exit criteria:

- No final answer cites `InvestigationEvidence`.
- Atomic observations contain source lineage and attestation.
- Path-only or option-judging evidence is rejected.

### Phase 3: Exploration ledger and observation reuse

Goal: make the Agent aware of what it has already inspected.

- Track opened, inspected, exhausted, and excluded ranges.
- Reuse equivalent observations.
- Expose coverage gaps and duplicate ratios to the Reasoner.

Exit criteria:

- Equivalent repeated requests do not resample frames.
- A revisit requires a changed goal, modality, detail level, or conflict.
- Long-case trajectories demonstrate monotonically increasing useful coverage.

### Phase 4: Investigator drill-down loop

Goal: make local observation genuinely progressive.

- Use low-fps preview and ASR in the first VLM observation.
- Let the Investigator choose a narrower second window.
- Uniformly select the final VLM frame subset across the complete window.

Exit criteria:

- A 10-minute inspection cannot silently observe only the first 32 seconds.
- Detail inspection is narrower than preview unless the contract requires broad coverage.
- Empty or irrelevant windows produce explicit exhausted reports, not fabricated evidence.

### Phase 5: Derived evidence and structured reconciliation

Goal: support MM-Lifelong's multi-clue questions.

- Add entity deduplication and count bounds.
- Add event ordering and state comparison.
- Generate derived evidence with parent closure and coverage manifests.

Exit criteria:

- Exact count claims cannot pass from sparse atomic observations.
- Ordering claims cite all required event observations.
- Unknown identity relations produce a count range and a follow-up task.

### Phase 6: Reasoner dashboard and completion repair

Goal: make the multi-round loop workspace-driven.

- Replace the free-form evidence digest with a structured dashboard.
- Add contract-aware completion checks.
- Convert final-gate failures into repair tasks.
- Add contract revision history, per-version repair counters, and forced-answer finalization.
- Enforce the 8,000-token dashboard projection budget and priority order.

Exit criteria:

- Reasoner can explain what remains unverified without seeing raw hidden evaluator data.
- Repeated wrong-window loops are reduced in replay tests.
- Final answers satisfy source, modality, scope, and aggregate gates.
- Grounded and forced answers are both emitted and scored separately.
- Contract revisions and dashboard truncation are visible in the trace.

### Phase 7: MM-Lifelong adapter and evaluation

Goal: evaluate on real long-horizon data.

- Build Day, Week, and Month workspaces.
- Keep evaluator artifacts isolated.
- Build the 20-question focused set exclusively from `train@month`.
- Mark any Week diagnostic fixture as test-exposed and exclude it from reports.
- Run `val@month`, then Long and Ultra-Long tiers with multiple seeds where needed.

Exit criteria:

- Agent prompts contain no answer or clue interval leakage.
- Evidence metrics are computed independently from answer accuracy.
- At least one successful case requires multiple disjoint clue intervals.
- Reports include accuracy, clue recall, inspected-duration ratio, duplicate ratio, and cost.
- Ref@N uses final cited evidence intervals only.
- Grounded accuracy, forced-answer accuracy, and contract-error metrics remain separate.

## 20. Testing Strategy

### Unit tests

- three-clock source/virtual/wall-clock conversion
- query-cutoff enforcement
- source-scope compatibility
- interval union, gap detection, and duplicate inspection detection
- observation cache key behavior
- frame-cache reuse across five-second-quantized near-identical windows
- semantic-observation reuse at source-time IoU `>=0.8`
- same frames but different observation goals trigger a new VLM observation
- uniformly distributed VLM frame subset selection
- global lexical ASR lookup returns all paginated cue clusters rather than the first cue
- invalid model window output uses an unresolved coverage frontier rather than segment start
- atomic evidence validation
- derived evidence parent closure
- entity count lower/upper bounds
- entity count switches to conservative approximate mode above 16 groups
- event ordering from wall-clock references
- completion gate repair requirements
- contract revision limit and per-version repair counters
- dashboard priority truncation at an 8,000-token budget

### Integration tests

- fake Reasoner and Investigator complete a multi-window count question
- source distractor evidence is excluded
- a repeated task reuses prior evidence
- low-fps preview triggers a narrower high-fps inspection
- a final answer is rejected until derived evidence exists
- max-round finalization emits grounded refusal plus a separately marked forced candidate
- replay of prior VideoMME failure trajectories follows new repair paths
- MM-Lifelong adapter hides evaluator-only fields
- normalized ReMA result import cannot expose gold clue intervals to the Agent

### Remote tests

- one focused MM-Lifelong Week case on KML
- one Month Ultra-Long multi-clue case
- one visual + ASR case
- one entity count/dedup case
- one event ordering case
- the same focused cases through official ReMA for side-by-side metrics

Remote monitoring should use compact status and metrics summaries rather than streaming raw logs.

## 21. Risks And Mitigations

### Runtime memory becomes another text index

Mitigation: primary benchmark mode uses per-question cold starts. Runtime notes must cite observations and cannot exist before the question.

### Global ASR lookup becomes a semantic retrieval shortcut

Mitigation: permit only exact lexical/token lookup over raw cues, return complete lineage and searched coverage, forbid generated expansion, and require visual confirmation whenever the contract includes visual observability.

### ASR dominates exploration

Mitigation: raw ASR is visible only inside opened segments or inspected windows. Joint-modality contracts require visual evidence before finalization.

### Reasoner dashboard grows without bound

Mitigation: persist full records on disk but render bounded projections, IDs, coverage summaries, and unresolved items.

### Entity reconciliation is unreliable

Mitigation: store explicit `same`, `different`, and `unknown` relations; report count bounds; ask for targeted comparison evidence instead of forcing a merge; switch from exact maximum clique to a conservative greedy lower bound above 16 groups.

### Cost becomes excessive

Mitigation: reuse observations, cap initial thumbnails, use low-fps previews, count internal costs separately, and measure inspected-duration ratio.

### Comparison against ReMA is unfair

Mitigation: report preprocessing and query-time cost separately, use the official code as an external baseline, align controller/perception backbones where possible, and publish both single-question and amortized multi-question results.

### Benchmark annotations contain malformed intervals

Mitigation: sanitize evaluator inputs with reversible normalization, record all corrections, and never silently alter Agent-visible timelines.

## 22. Definition Of Done

The upgraded harness is successful when:

1. A benchmark run starts without semantic captions, summaries, or retrieval embeddings.
2. Reasoner initially sees no beat-level evidence or full ASR transcript.
3. All semantic memory is created after Agent actions.
4. Investigator exposes only `open_segment` and `inspect_window`.
5. Every cited fact has source, temporal, and observation lineage.
6. Aggregate answers require derived evidence with parent closure and coverage.
7. The Agent does not repeatedly inspect equivalent windows without a stated reason.
8. Source-time, virtual-time, and wall-clock-time remain distinguishable.
9. MM-Lifelong clue recall and trajectory efficiency are reported alongside answer accuracy.
10. The trace viewer can reconstruct every Reasoner and Investigator round, including images, ASR excerpts, evidence commits, coverage changes, and final-gate decisions.
11. Official ReMA and each proposed ablation are evaluated through the same answer, grounding, and cost schema.
12. Results identify where query-time exploration wins and where precomputed caption memory remains more efficient.
13. Near-identical floating-point windows reuse frame materialization, while semantic evidence is reused only under explicit source-IoU, capability, and goal-equivalence rules.
14. Reported Ref@N intervals come only from final cited evidence, never from all exploration visits.
15. Every benchmark run reports both grounded and ungated forced answers without mixing their accuracy metrics.

## 23. Recommended First Milestone

The first implementation milestone should include Phase 0.5 and stop after Phase 4, proving the exploration substrate before adding complex reasoning:

```text
three-clock timeline
  + immediate sampling/preview/window-fallback corrections
  + unified EvidenceRecord
  + exploration ledger
  + observation reuse
  + real low-to-high-fps drill-down
  + all-round trace viewer
```

Acceptance should use:

- the existing correct `606-3` trajectory
- the source-polluted `698-3` trajectory
- the source-time-confused `701-3` trajectory
- the remaining interleaved six-hour case used in the current four-case suite
- one MM-Lifelong Week question with public clue intervals
- the same MM-Lifelong question through official ReMA

The Week question is a diagnostic-only fixture and is excluded from all reported benchmark numbers. Quantitative development uses `train@month` and aggregate reporting begins on `val@month`.

Only after this milestone shows correct source handling, reduced duplicate exploration, recoverable evidence traces, and a fair side-by-side ReMA comparison should the project add entity/event aggregation and run the broader MM-Lifelong tiers.
