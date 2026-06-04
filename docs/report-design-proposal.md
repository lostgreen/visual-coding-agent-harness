# Visual Coding-Agent Multimodal Harness: Design Proposal

Date: 2026-06-02

## 0. One-Sentence Goal

Build a visual-oriented coding-agent-style harness for multimodal agents: the main MLLM does not answer from one-shot image/video input, but actively plans, calls visual tools, delegates local visual subtasks, maintains an evidence workspace, verifies evidence-answer alignment, and turns tool traces into later training data.

This project starts training-free. The first milestone is an executable harness that can collect high-quality perception-reasoning-action trajectories. Training then uses those trajectories to improve tool selection, visual action grounding, verifier behavior, and stopping policy.

## 1. Why Claude Code Is the Right Analogy

Claude Code's core idea is not that the prompt is magic. The model has agency, but the engineering value comes from the harness around it. A coding agent gets stronger because it is placed inside a managed work environment: filesystem, shell, editor tools, browser, permission policy, context compaction, task state, subagents, and trace persistence.

The minimal loop is stable:

```text
user request
  -> model(messages, tools)
  -> tool_use
  -> execute tool
  -> tool_result
  -> append result to messages
  -> repeat until no tool_use
```

Almost everything else is a harness layer:

| Claude Code mechanism | What it manages | Multimodal migration |
| --- | --- | --- |
| Agent loop | Model-tool interaction | Keep the same loop, but replace code tools with visual tools. |
| Tool registry / dispatch map | Tool schema, handler, validation | Register visual foundation models and visual operations as typed tools. |
| Workspace | Files, diffs, generated artifacts | Evidence workspace with frames, crops, masks, OCR patches, observations, ledgers. |
| Context compact | Prevents command/file outputs from flooding context | Large images/videos and long tool outputs stay on disk; context sees compact observations. |
| Subagents | Fresh context for local investigations | Visual workers inspect time windows, regions, OCR, objects, relations, or external evidence. |
| Task DAG | Long-running structured goals | Decompose a visual QA/search task into dependent and parallel subtasks. |
| Permissions/hooks | Cost, safety, approval, side effects | Gate high-cost tools, web search, private media, or destructive artifact writes. |
| Stream/event protocol | Unified status/tool/text events | Unified visual event stream: tool call, artifact written, observation, verifier result, usage. |

The key design lesson is: do not build a fixed prompt chain. Build a constrained executable environment where the model can decide what to inspect next, while the harness controls interfaces, evidence, budget, provenance, and verification.

## 2. Research Motivation

Long-video QA, multi-image reasoning, OCR-heavy visual understanding, GUI understanding, geolocalization, and open-world visual search often fail for the same reason: the model lacks a reliable visual operating environment.

Common failure modes:

- The model answers from a coarse global impression without inspecting the relevant region or time.
- A caption, retrieval snippet, or previous summary becomes the answer authority even when visual evidence is weak.
- Long tool traces pollute context, so the model remembers the wrong evidence.
- The model knows it should zoom, crop, OCR, seek, or compare, but does not execute the correct action.
- Multiple visual tools exist, but their outputs are unstructured and cannot be composed or audited.

The proposed harness turns these into engineering targets: tool access, visual action grounding, evidence persistence, context compression, subtask isolation, and answer verification.

## 3. Literature Anchors

The current note already points to the right paper cluster. We use them as design anchors, not as a finished literature review.

| Paper/source | Key idea for this project |
| --- | --- |
| Learn Claude Code | Agent capability emerges from a stable loop plus harness layers: tools, context, task system, subagents, memory, protocols. |
| Open Design Claude runtime | Adapter pattern: preserve native agent loops, parse tool streams into unified events, persist artifacts and tool traces. |
| Visual ChatGPT | ChatGPT as main scheduler; Prompt Manager wraps 22 visual foundation models as callable tools with state and file-name management. |
| VisProg | Training-free visual reasoning via executable visual programs, module registry, interpreter state, and visual rationale. |
| ReTool-Video | Rich video tool library with base/meta tools, recursive grounding from high-level intents to executable visual operations. |
| ParaVT | Parallel temporal crop calls and shared-weight subagents reduce sequential context corruption; training must handle format collapse and tool-use shortcuts. |
| VideoThinker | Generate tool trajectories in caption space, filter them, then replace caption zoom outputs with real frame tokens for tool-use training. |
| VideoSEAL | Decouple planner and inspector; final answer authority should come from evidence inspection, not noisy planner trace. |
| Agentic-MME | Evaluate process, not only final answers: whether tools are invoked, invoked correctly, and used efficiently. |
| Pixel Reasoner / Walk the Talk | Visual operations such as zoom-in/select-frame are reasoning primitives; training must close the gap between textual reasoning and visual action execution. |
| XSkill / IMPACT-CYCLE | Experience, skills, claim-level correction, provenance, and memory can be extracted from runs for reuse and supervision. |

## 4. Positioning Against ReTool-Video, ParaVT, and VisProg

The project should not be framed as "adding more visual tools." Existing work already shows that tool calling is useful. Our stronger position is that visual tool calls need coding-agent-style governance: workspace state, context compaction, evidence ledgers, subagent isolation, verifier control, and trace-to-training conversion.

| Work | Main focus | Difference in our project |
| --- | --- | --- |
| VisProg | LLM generates a visual program; an interpreter calls predefined vision/knowledge/logic modules. | VisProg is closer to one-shot program execution. We use a persistent multi-turn agent loop with workspace, trace, ledger, subagents, verifier, and context compact. |
| ReTool-Video | Large base/meta video tool library and recursive grounding from high-level intents to executable tool chains. | ReTool is strong in tool inventory and recursive resolution. We focus on evidence workspace, answer authority, trace governance, and collaboration protocols. |
| ParaVT | Parallel temporal crop calls and RL for stable long-video tool use. | ParaVT focuses on parallel crop-video. We generalize parallelism into specialist visual workers: temporal, spatial, OCR/UI, entity, action, ledger, and verifier. |

The one-line distinction is:

> VisProg writes visual programs; ReTool-Video recursively calls richer video tools; ParaVT trains parallel video tool use; we make multimodal agents work like Claude Code.

## 5. System Thesis

The central claim should not be "more tools make MLLMs stronger." The sharper claim is:

> Multimodal agents need coding-agent-style harnesses. Visual foundation models become useful for complex tasks only when they are wrapped in typed tools, grounded observations, artifact provenance, context compaction, subagent isolation, and evidence-aware verification.

This creates three implementation lines:

1. Multi-tool visual-oriented harness.
2. Collaborative workflow among main agent, visual subagents, tools, evidence ledger, and verifier.
3. Training-free behavior observation followed by tool-call training and behavior improvement.

## 6. Overall Design Choice: Text-Only Main Agent

For P0, the Main Agent should be a text-only planner/controller. It does not directly see the full image or video. It chooses tools, reads structured observations, updates the task DAG and evidence ledger, asks the verifier whether evidence is sufficient, and answers with citations.

```text
Text-only Main Agent
  -> choose visual tools
  -> read structured observations
  -> update task DAG and ledger
  -> request verification
  -> answer with cited evidence
```

This is close to VisProg's idea that the LLM can coordinate visual modules without directly seeing pixels, but our runtime is closer to Claude Code's loop. The benefits are cleaner trajectories, easier error attribution, and a stronger training signal: every visual fact must come through a tool call and become an observation.

Important ablations:

| Setting | Meaning | Purpose |
| --- | --- | --- |
| Text-only Main Agent | Main Agent sees only observations/ledger, not raw pixels. | Test whether the harness alone can support complex visual reasoning. |
| Multimodal Main Agent | Main Agent can directly inspect selected images/frames. | Test whether direct visual access helps or contaminates evidence trails. |
| Hybrid Main Agent | Default text-only; selected artifacts become visible only when verifier requests. | Balance evidence control and visual intuition. |

## 7. Line 1: Multi-Tool Visual-Oriented Harness

### 7.1 Architecture

```text
User task
  -> Main MLLM agent
  -> Visual task planner
  -> Tool registry
  -> Tool executor
  -> Evidence workspace
  -> Evidence ledger
  -> Answer agent
  -> Verifier
  -> final answer or follow-up tool call
```

The main agent sees a compact workspace summary and tool descriptions. It does not directly receive full-resolution images, full videos, raw OCR dumps, or long captions unless a tool result is small enough and useful enough. Large outputs are stored as artifacts.

### 7.2 Evidence Workspace

The evidence workspace is the visual equivalent of a coding agent's project directory.

```text
visual-coding-agent-harness/
  runs/<run_id>/
    input/
      task.json
      media/
    artifacts/
      frames/
      clips/
      crops/
      masks/
      ocr_regions/
      retrieved_images/
    observations.jsonl
    ledger.md
    tasks.json
    trace.jsonl
    answer.json
```

Each layer has a role:

| File/path | Role |
| --- | --- |
| `artifacts/` | Stores visual outputs that should not enter the prompt directly. |
| `observations.jsonl` | Structured observation from every tool call. |
| `ledger.md` | Compact answer-ready evidence table. |
| `tasks.json` | Subtask DAG with status, owner, dependencies, and blocked reason. |
| `trace.jsonl` | Full tool-use trajectory for later analysis/training. |
| `answer.json` | Final answer, cited evidence, confidence, unresolved conditions. |

### 7.3 Tool Registry

Every tool should have:

- Name and natural-language description.
- JSON schema for arguments.
- Input media type and output media type.
- Cost class: cheap, medium, expensive.
- Concurrency safety: parallel-safe or exclusive.
- Observation formatter.
- Provenance fields.
- Failure formatter.

Proposed P0 tools:

| Tool | Purpose | Input | Output |
| --- | --- | --- | --- |
| `sample_frames` | Coarse video exploration | video path, sampling strategy | frame artifact paths with timestamps |
| `seek_clip` | Extract a target time window | video path, start/end | clip path, sampled keyframes |
| `crop_region` | Inspect a spatial region | image/frame path, bbox | crop artifact path |
| `zoom_region` | Magnify small visual details | image/frame path, bbox, scale | zoomed crop path |
| `ocr_region` | Read text, UI, signs, documents | image/crop path | text spans with bbox/confidence |
| `caption_image` | Get a coarse semantic description | image/crop path | short caption with uncertainty |
| `inspect_region` / `verify_local_claim` | Check a local visual fact rather than answer the whole task | image/crop path, local question or claim | artifact-linked local observation |
| `compare_frames` | Check temporal or before/after changes | frame paths | change observations |
| `detect_objects` | Find candidate objects/entities | image path, classes optional | boxes, labels, confidence |
| `write_observation` | Let agent record a claim with support | claim, sources | ledger candidate |
| `summarize_evidence` | Compact observations into ledger | observation ids | updated ledger |
| `verify_answer` | Check answer-evidence consistency | answer, ledger | pass/fail, missing evidence |

P1/P2 tools:

| Tool class | Candidate tools/foundation models |
| --- | --- |
| Segmentation | SAM/SAM2-style mask proposal, object mask refinement. |
| Detection/open-vocabulary | Grounding DINO, OWL-style detector, Florence-style detection, YOLO for fast baselines. |
| OCR/document | PaddleOCR, EasyOCR, TrOCR, document layout parser. |
| Video tracking | SAM2 video propagation, ByteTrack/DeepSORT, optical flow. |
| Temporal retrieval | CLIP/SigLIP frame embeddings, caption index, moment retriever, temporal grounding model. |
| Depth/pose/geometry | Depth estimator, pose estimator, 3D/4D spatial tools for dynamic scenes. |
| External knowledge | Web/entity/product/logo/place lookup, only when grounded to visual evidence. |
| Memory/skills | Load task-specific protocols, known failure modes, dataset rules, previously successful tool patterns. |

### 7.4 Why Use OCR, Caption, or Local VQA Tools If MLLMs Already Have Them?

These tools are not valuable because the Main Agent lacks all perception. Their value is that they turn implicit perception into explicit, localized, auditable operations.

| Direct MLLM perception | Toolized visual operation |
| --- | --- |
| Model sees an image/video and emits free-form text. | Harness specifies time, region, and local question before calling a tool. |
| Evidence source is ambiguous. | Output includes frame, timestamp, bbox, artifact path, and confidence. |
| Failure is hard to diagnose. | Failure can be attributed to retrieval, crop, OCR/VQA, ledger, or answer generation. |
| Long visual contexts pollute the prompt. | Observations are compacted into an evidence ledger. |
| Weak training signal. | Each tool call and argument can be labeled for SFT/preference/RL. |

Thus `inspect_region` should be implemented as a local evidence-checking operator. It may use the same MLLM, a smaller VLM, a stronger inspector VLM, a specialized classifier, or an ensemble. The important boundary is role separation: the Main Agent decides where to look and what to ask; the local tool returns a grounded observation.

### 7.5 Observation Format

Tool results must not be free-form text only. Each observation should link a claim to visual evidence.

```json
{
  "observation_id": "obs_0042",
  "run_id": "run_20260602_001",
  "tool": "ocr_region",
  "input_artifacts": ["artifacts/crops/frame_01240_box_3.png"],
  "output_artifacts": [],
  "time_range": [41.2, 41.2],
  "regions": [{"frame": "frame_01240.jpg", "bbox": [312, 180, 498, 330]}],
  "claim": "The sign reads 'EXIT'.",
  "raw_output": "EXIT",
  "confidence": 0.91,
  "supports_question": true,
  "limitations": "Text is partially blurred; verifier may request another frame."
}
```

This is the migration of code-agent artifact discipline into visual reasoning. The answer should cite `observation_id`, frame/time, and region, not an ungrounded caption.

### 7.6 Context Management

The harness should apply three levels of compaction:

1. Tool-result micro-compact: after a few turns, replace verbose tool results with `observation_id`, artifact path, and one-line summary.
2. Evidence ledger compact: periodically summarize relevant observations into a claim-support table.
3. Trace archive: keep complete raw traces in `trace.jsonl` for training, not active context.

This directly targets the VideoSEAL-style evidence misalignment problem: the final answer should not be generated from a long noisy planner trace.

## 8. Ultra-Long Video Scheduling

Ultra-long videos should be treated like large codebases: index first, retrieve candidate evidence, inspect locally, update the ledger, verify sufficiency, then refine or answer.

### 8.1 Initial Video Indexing

Initial shot/scene segmentation should use traditional or lightweight dedicated methods, such as TransNetV2, PySceneDetect, ffmpeg scene filters, or embedding-change scores. This is an indexing layer, not the core reasoning contribution.

P0 should combine:

| Segmenting signal | Role |
| --- | --- |
| Uniform windows | Guaranteed coverage, e.g. 30s or 60s windows. |
| Shot boundary detection | Useful for edited videos with clear cuts. |
| Adaptive keyframe sampling | Representative and high-change frames inside each unit. |
| OCR/transcript/change hints | Useful for screen recordings, lectures, meetings, and UI videos. |

Index pipeline:

```text
video
  -> uniform temporal windows
  -> shot boundary detector
  -> motion/change score
  -> sparse keyframes
  -> sparse OCR / caption / embedding
  -> merged timeline_index.json
```

### 8.2 Query-Driven Scheduling Loop

```text
Index -> Retrieve -> Parallel Inspect -> Ledger -> Verify -> Refine/Answer
```

Concrete steps:

1. Build or load the video index.
2. Main Agent parses the question into evidence goals.
3. `retrieve_moments(query, top_k)` returns candidate timeline units.
4. Parallel visual workers inspect candidate moments.
5. Workers write local observations.
6. Ledger Agent updates the evidence ledger.
7. Verifier checks whether evidence is sufficient.
8. If insufficient, expand top-k, narrow the window, dense sample, crop/zoom, OCR, or track.
9. If sufficient, Answer Agent responds with cited evidence.

This generalizes ParaVT: instead of one-turn parallel crop-video, the harness uses persistent indexing, query-driven retrieval, specialist workers, evidence ledger, and verifier-controlled refinement.

## 9. Line 2: Collaborative Workflow

### 9.1 Roles

| Role | Responsibility | Context boundary |
| --- | --- | --- |
| Main Agent | Understand task, create subtasks, select tools, manage budget, decide when to answer. | Sees question, tool registry, task DAG, compact ledger. |
| Temporal Worker | Locate relevant video windows. | Sees video metadata, sampled frames, query-specific temporal objective. |
| Spatial Worker | Inspect objects, regions, spatial relations. | Sees selected frames/crops only. |
| OCR/UI Worker | Read text, screens, diagrams, documents, signs. | Sees OCR-relevant crops and question. |
| Entity/Knowledge Worker | Link visually grounded entities to external knowledge. | Sees visual evidence and query; must cite external sources. |
| Ledger Agent | Convert observations into concise evidence claims. | Sees observations, not full raw trace. |
| Answer Agent | Produce final answer from evidence. | Sees ledger, question, candidate constraints. |
| Verifier | Check answer support, missing evidence, temporal grounding, and overuse of tools. | Sees answer, ledger, selected artifacts if needed. |

Subagents matter because visual tasks are context-heavy. A worker can inspect many frames or crops, but the parent should receive only structured observations and a short recommendation.

### 9.2 Workflow

1. Parse the user question into answer type, required visual evidence, possible temporal/spatial targets, and risk flags.
2. Create a task DAG:
   - coarse exploration,
   - target localization,
   - local evidence extraction,
   - evidence ledger update,
   - answer generation,
   - verification.
3. Run cheap tools first:
   - frame sampling,
   - coarse caption,
   - rough OCR,
   - low-cost retrieval.
4. Spawn visual subagents for independent evidence checks:
   - different time windows,
   - different image regions,
   - different candidate entities,
   - different answer hypotheses.
5. Write all tool outputs into `observations.jsonl`.
6. Compact observations into `ledger.md`.
7. Generate an answer only from the ledger.
8. Verifier audits:
   - Does the answer cite concrete evidence?
   - Did we inspect the right time window or region?
   - Are any answer claims unsupported?
   - Is there evidence misalignment: correct-looking answer but wrong/weak trace?
   - Did the agent overuse expensive tools?
9. If verification fails, return a targeted follow-up action, not a broad restart.

### 9.3 Protocols Between Agents

Worker requests should be structured:

```json
{
  "task_id": "t_temporal_02",
  "role": "temporal_worker",
  "objective": "Find the time window where the person picks up the red cup.",
  "allowed_tools": ["sample_frames", "seek_clip", "compare_frames", "caption_image"],
  "input_artifacts": ["input/media/video.mp4"],
  "budget": {"max_tool_calls": 8, "max_seconds": 90},
  "return_format": "observations_json"
}
```

Worker response:

```json
{
  "task_id": "t_temporal_02",
  "status": "completed",
  "observations": ["obs_0011", "obs_0012"],
  "recommended_next_actions": [
    "crop frame_00480 around the table to confirm cup color"
  ],
  "confidence": 0.72,
  "missing_conditions": [
    "Need local crop because sampled frame is low resolution"
  ]
}
```

This is the multimodal version of team protocols: structured request-response with ids, not ad hoc agent chatter.

### 9.4 Example End-to-End Case

Question: "In the video, what does the person do immediately after reading the sign?"

Workflow:

1. Main Agent identifies two required evidence pieces: sign-reading moment and immediate next action.
2. Temporal Worker samples frames and finds candidate sign-viewing intervals.
3. OCR Worker crops the sign and confirms readable text.
4. Spatial/Action Worker inspects frames after the sign moment and describes the action.
5. Ledger Agent writes:
   - `obs_07`: sign text visible at 00:41.2, crop path.
   - `obs_11`: person turns right and opens door at 00:43.0-00:45.5, frame paths.
6. Answer Agent says: "After reading the sign, the person turns right and opens the door."
7. Verifier checks if the cited time order is correct. If not, asks for `compare_frames` around 00:41-00:46.

## 10. Line 3: Training-Free First, Then Tool-Use Training

### 10.1 Stage A: Training-Free Harness

Goal: observe model behavior before changing model weights.

What to collect:

- Which tools the model calls.
- Tool arguments: frame ranges, boxes, questions, classes.
- Whether the visual action hits the right evidence.
- Whether the model stops too early or overuses tools.
- Whether final answer is supported by evidence.
- Failure recovery after verifier feedback.

Training-free baselines:

| Baseline | Purpose |
| --- | --- |
| Direct MLLM | Measures one-shot visual reasoning. |
| Caption-only agent | Tests whether textification creates evidence shortcuts. |
| Few-tool agent | Tests minimal zoom/crop/retrieve hypothesis. |
| Multi-tool no ledger | Tests whether many tools without harness create noise. |
| Full visual coding-agent harness | Tests registry + workspace + subagents + compact + verifier. |

### 10.2 Stage B: Trajectory Filtering

Not every run should become training data. The harness should label each trajectory:

| Label | Meaning |
| --- | --- |
| `successful_grounded` | Correct answer, correct evidence, efficient enough. |
| `answer_correct_evidence_wrong` | Looks correct but trace is unsupported; VideoSEAL-style misalignment. |
| `tool_format_failure` | Bad JSON/tool call format; ParaVT-style format collapse signal. |
| `wrong_visual_action` | Crop/seek/zoom misses the target. |
| `under_tool_use` | Model answers without needed visual inspection. |
| `over_tool_use` | Too many irrelevant or expensive tools. |
| `recovered_by_verifier` | Initial answer failed but targeted follow-up fixed it. |

These labels become training/evaluation signals.

### 10.3 Stage C: Tool-Call Training Targets

Start with smaller components before full MLLM RL.

| Component | Training objective | Data source |
| --- | --- | --- |
| Tool selection policy | Choose when to call crop/OCR/seek/detect/verify. | Successful and failed trajectories. |
| Tool argument generator | Produce correct bbox, timestamp, class list, local question. | Tool calls paired with verifier/evidence labels. |
| Observation summarizer | Convert raw tool output into structured claims. | `observations.jsonl` and accepted ledger rows. |
| Verifier | Detect unsupported claims and missing visual evidence. | Pass/fail verifier judgments and human audits. |
| Stopping policy | Decide when evidence is sufficient. | Successful vs premature answer traces. |
| Recovery policy | Choose targeted follow-up after failure. | Recovered trajectories. |

### 10.4 Stage D: SFT, Preference, and RL

Possible training path:

1. SFT cold start:
   - Use successful grounded trajectories.
   - Teach schema-following and basic tool-use format.
2. Step-wise preference tuning:
   - Prefer tool calls that inspect the correct evidence with lower cost.
   - Penalize irrelevant tools, wrong crops, and unsupported answer moves.
3. Verifier-guided improvement:
   - Generate candidate tool trajectories.
   - Verifier scores evidence support and missing conditions.
4. RL / GRPO-style training:
   - Reward answer accuracy, evidence groundedness, tool format validity, visual action correctness, and cost efficiency.
   - Avoid ParaVT-style shortcuts where the model learns to skip tools or emits parseable but useless tool calls.

Reward sketch:

```text
R = answer_accuracy
  + evidence_support
  + temporal_grounding
  + visual_action_hit
  + format_validity
  - unsupported_claim_penalty
  - unnecessary_tool_cost
  - premature_stop_penalty
```

The important point is that tool-use training should not optimize final answer alone. It must optimize the process.

## 11. Evaluation Plan

### 11.1 Metrics

| Metric | What it checks |
| --- | --- |
| Answer accuracy | Final QA correctness. |
| Evidence support | Whether answer claims are backed by specific observations. |
| Temporal grounding | Whether the agent inspected the correct video interval. |
| Spatial grounding | Whether crop/zoom/detect hit the correct region. |
| Tool-use precision | Whether called tools were necessary/relevant. |
| Tool-use recall | Whether needed tools were actually used. |
| Recovery success | Whether verifier feedback fixes failures. |
| Cost/latency | Tool calls, wall time, expensive model/tool usage. |
| Context pressure | Whether long traces degrade answer support. |
| Overthinking | Process redundancy compared with human or oracle trajectories. |

### 11.2 Ablations

| Ablation | Question |
| --- | --- |
| Full trace vs compact ledger | Does compact evidence reduce hallucination/context pollution? |
| No verifier | How much unsupported-answer rate increases? |
| No subagents | Does shared context hurt long-video/multi-image tasks? |
| Sequential vs parallel temporal workers | Does ParaVT-style parallelism reduce error propagation? |
| Text-only observation vs artifact-linked observation | Is visual provenance necessary? |
| Few tools vs full registry | Are more tools helpful only with harness controls? |
| Rule router vs model-chosen tools | Does coding-agent-style autonomy help? |
| No cost budget | Does the model overuse expensive tools? |

### 11.3 Candidate Benchmarks

Use a mix of small local cases and public benchmarks:

- Long-video QA / Video-MME-like tasks for temporal evidence.
- TextVQA / InfographicsVQA / document-image QA for OCR and crop.
- V* / complex counting for zoom and local inspection.
- Agentic-MME-style process verification for tool correctness and efficiency.
- Custom 20-100 case harness benchmark with annotated evidence frames/regions.

## 12. Implementation Milestones

### Milestone 0: Project Skeleton

Create the project directory, design docs, and future code layout.

```text
visual-coding-agent-harness/
  docs/
  src/
  experiments/
  artifacts/
```

### Milestone 1: Minimal Visual Harness

Implement:

- `agent_loop.py`
- `registry.py`
- `workspace.py`
- `events.py`
- P0 tools: sample, crop, zoom, OCR, caption/VQA placeholder.
- `trace.jsonl`, `observations.jsonl`, `ledger.md` writing.

Deliverable: one image case and one short-video case run end to end.

### Milestone 2: Evidence Ledger + Verifier

Implement:

- Structured observation formatter.
- Ledger compaction.
- Answer-with-citations format.
- Verifier tool that emits pass/fail and targeted follow-up.

Deliverable: compare direct answer vs harness answer on 20 cases.

### Milestone 3: Visual Subagents

Implement:

- Worker request/response schema.
- Temporal worker.
- Spatial worker.
- OCR worker.
- Parallel execution for independent subtasks.

Deliverable: long-video tasks with temporal parallelism and compact parent context.

### Milestone 4: Dataset and Trace Analysis

Implement:

- Trajectory labeling.
- Tool-use metrics.
- Failure taxonomy.
- Export format for SFT/preference data.

Deliverable: first trajectory dataset report.

### Milestone 5: Tool-Use Training Prototype

Start with one trainable component:

- Tool selection classifier,
- verifier,
- observation summarizer,
- or small SFT for tool-call format.

Deliverable: show behavior improvement on held-out cases.

## 13. Immediate Development Choice

For the first implementation, choose a conservative stack:

- Python harness.
- Local artifact workspace.
- JSON schema-like tool definitions.
- Simple image/video tools with OpenCV/PIL/moviepy.
- OCR via available local OCR package or placeholder interface.
- MLLM/API adapter kept abstract, so we can run Claude/OpenAI/Qwen-style backends later.

The first goal is not model performance. The first goal is to make every visual action executable, logged, compacted, and auditable.

## 14. Summary for Presentation

This project migrates the design philosophy of Claude Code into multimodal reasoning. Claude Code shows that a strong agent is not just a model; it is a model inside a managed executable environment. For multimodal tasks, that environment should be a visual evidence workspace with tool registry, artifact provenance, context compaction, visual subagents, and answer verification.

The proposal has three concrete parts:

1. Build a multi-tool visual harness where visual foundation models become typed, composable, provenance-aware tools.
2. Build a collaborative workflow where a main agent, visual workers, ledger, answer agent, and verifier cooperate through structured protocols.
3. Run training-free first to observe natural tool behavior, then convert successful, failed, and recovered traces into SFT/preference/RL data for tool-call and behavior improvement.

The expected contribution is not "we added more tools." It is a harness that makes tools usable, evidence auditable, and multimodal agent behavior trainable.
