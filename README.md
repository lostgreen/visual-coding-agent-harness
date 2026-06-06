# Visual Coding-Agent Harness

This repository is a research prototype for a multimodal coding-agent-style harness.

P0 borrows the useful shape of VisProg: a text planner emits a visual program, and an interpreter executes registered visual modules. The added harness layer records tool calls, structured observations, trace events, and an evidence ledger so later versions can support long-video reasoning, subagents, verification, and tool-use training.

## Run Tests

```bash
PYTHONPATH=src python3 -m unittest discover tests
```

## Run Demo

```bash
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
from visual_coding_agent_harness.demo import run_demo

result = run_demo(Path("."), run_id="demo")
print(result)
PY
```

The demo writes:

- `runs/demo/observations.jsonl`
- `runs/demo/trace.jsonl`
- `runs/demo/ledger.md`

## Project Layout

Package code lives under `src/visual_coding_agent_harness/`. Evaluation
runners, metrics, schemas, and ablation helpers live under
`src/visual_coding_agent_harness/evals/`; command modules live under
`src/visual_coding_agent_harness/cli/`.

`runs/` is reserved for generated run artifacts. The legacy Python files in
`runs/` and `scripts/` are compatibility wrappers for older commands.
Prefer the package CLIs for new usage:

```bash
PYTHONPATH=src python3 -m visual_coding_agent_harness.cli.eval_videomme --help
PYTHONPATH=src python3 -m visual_coding_agent_harness.cli.run_ablation --help
PYTHONPATH=src python3 -m visual_coding_agent_harness.cli.generate_ablation_report --help
PYTHONPATH=src python3 -m visual_coding_agent_harness.cli.audit_trajectory --help
```

## P0 Scope

- No external visual models.
- No GPU requirement.
- No full long-video pipeline.
- Dummy tools preserve the schema expected from real VLM, OCR, and verifier backends.

## Tool Protocol

Agents should call tools through `ToolRequest` and consume `ToolResult`.

```python
from visual_coding_agent_harness.protocol import ToolRequest
from visual_coding_agent_harness.tools.traditional import build_traditional_registry

registry = build_traditional_registry()
results = registry.execute_batch([
    ToolRequest(
        tool="crop_region",
        arguments={
            "image_path": "input/frame.png",
            "bbox": [0, 0, 500, 500],
            "output_path": "runs/demo/artifacts/crops/frame_crop.png",
        },
        request_id="crop_1",
        caller="spatial_worker",
    )
])
```

Traditional tools currently include:

- `crop_region`, `zoom_region`, `threshold_image`, `enhance_image`, `edge_detect`
- `sample_frames`, `extract_clip`

Image tools use Pillow. Video tools build and execute `ffmpeg` commands; they raise a clear tool error when `ffmpeg` is not installed.

## VLM Agent Smoke

The first model-backed path keeps one foundation model instance shared between
the main visual agent and its VLM tools. This is the baseline for comparing
direct VLM answering against tool-use behavior without changing model size.

The main agent receives a tool-use prompt with a fixed tool catalog:

- `caption_video(video_path, question, nframes=8, max_pixels=151200)`
- `qa_video(video_path, question, nframes=8, max_pixels=151200)`
- `caption_image(image_path, question)`
- `qa_image(image_path, question)`
- `caption_region(image_path, bbox, question)`
- `qa_region(image_path, bbox, question)`

Caption/QA tools use structured prompts that force visible-evidence-only
answers, uncertainty when evidence is weak, and no invented identities/text.
Region tools first crop the requested normalized `[0,1000]` bbox into
`runs/<run_id>/artifacts/crops/`, then send that crop to the shared VLM backend.

Input schema:

```json
{
  "question": "What is the video mainly about?",
  "media_path": "/path/to/video.mp4",
  "media_type": "video",
  "tool_policy": "required"
}
```

The model planner is asked to emit only:

```json
{
  "answer": "...",
  "program": [
    {
      "tool": "caption_video",
      "args": {
        "video_path": "/path/to/video.mp4",
        "question": "What is the video mainly about?"
      },
      "assign": "caption"
    }
  ]
}
```

The harness normalizes media paths to the caller-provided input path, executes
the program, and returns:

```json
{
  "input": {
    "question": "What is the video mainly about?",
    "media_path": "/path/to/video.mp4",
    "media_type": "video",
    "tool_policy": "required"
  },
  "output": {
    "answer": "...",
    "program": [],
    "observation_ids": ["obs_0001"],
    "assignments": {"caption": "obs_0001"}
  },
  "debug": {
    "planner_text": "raw planner response"
  }
}
```

```bash
python -m visual_coding_agent_harness.cli.vlm_smoke \
  --model-path /m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct \
  --media-path /path/to/video.mp4 \
  --question "What happens in the video?" \
  --base-dir . \
  --run-id qwen3_vl_4b_smoke
```

The run writes the same harness artifacts under `runs/<run_id>/`:

- `observations.jsonl`
- `trace.jsonl`
- `ledger.md`

## Iterative Long-Video Agent

The iterative agent is the first autonomous exploration flow. It starts from a
scene index, asks the main model to choose a segment-level tool call, writes the
tool result into the evidence ledger, and replans from the updated ledger until
the model returns a final answer or the round budget is exhausted. By default
the planner is text-only: it receives the question, scene index, inspected
segment list, and ledger, while tools are responsible for inspecting video.

Autonomous exploration policy:

- The planner sees `Already inspected segments` and `Uninspected segment candidates`.
- The harness binds `segment_id` to the real `video_path/start_sec/end_sec`.
- Each round is capped by `AgentBudget.max_tool_calls_per_round`, default `1`.
- Repeated segment requests are automatically redirected to the next uninspected segment.
- If the planner emits no usable tool call, the harness falls back to `caption_segment` on the next uninspected segment.
- Final answers must cite observation ids from `ledger.md`.

Current P1 tools:

- `video_ls(query="", max_segments=16, top_k=5)`
- `search_segments(query, top_k=5, modalities=[])`
- `read_segment(segment_id)`
- `expand_window(segment_id, before_sec=30, after_sec=30)`
- `caption_segments(segment_ids=[], question, nframes=8, max_pixels=151200, fps=0.0, max_segments=3)`
- `ingest_segment_metadata(segment_id, low_fps_caption="", asr_text="", ocr_text="", entities=[])`
- `caption_segment(video_path, segment_id, start_sec, end_sec, question, nframes=8)`
- `qa_segment(video_path, segment_id, start_sec, end_sec, question, nframes=8)`
- Segment tools also accept `max_pixels=151200` and `fps=0.0` for controlled
  video sampling.
- `summarize_ledger_evidence(max_claims=5)`
- `verify_ledger_answer(answer, min_score=0.6)`

The navigation tools are the video equivalent of repository `ls`, `rg`, and
local file reads. `video_ls` is the map-first entry point: it returns modality
coverage, a bounded timeline outline, candidate segments for an optional query,
and recommended next tools. The planner can search or read indexed metadata
before asking visual tools to inspect pixels. For segment VLM tools, the planner
only needs to emit `segment_id`; the harness binds the real `video_path`,
`start_sec`, `end_sec`, and default `nframes` from `SceneIndex`.

`caption_segments` and `ingest_segment_metadata` are VideoMap enrichment tools:
they make `video_ls` behave like a coding-agent workspace index that improves as
the agent reads more evidence. `verify_ledger_answer` is a lightweight lexical
verifier over `ledger.md`; it is intentionally conservative and should later be
augmented with model-based entailment and temporal consistency checks.

Planner output for another exploration round:

```json
{
  "status": "continue",
  "rationale": "Need focused evidence from the likely segment.",
  "program": [
    {
      "tool": "caption_segment",
      "args": {
        "segment_id": "seg_0002",
        "question": "What is discussed in this segment?"
      },
      "assign": "focused_caption"
    }
  ]
}
```

Planner output for final answer:

```json
{
  "status": "final",
  "answer": "...",
  "citations": ["obs_0001"],
  "confidence": 0.78
}
```

Run with Qwen3-VL:

```bash
PYTHONPATH=src python3 -m visual_coding_agent_harness.cli.iterative_smoke \
  --model-path /m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct \
  --media-path /path/to/video.mp4 \
  --question "What is the video mainly about?" \
  --duration-sec 600 \
  --window-sec 30 \
  --max-rounds 4 \
  --extract-clips \
  --base-dir . \
  --run-id qwen3_vl_iterative_smoke
```

With `--extract-clips`, segment tools write short videos into
`runs/<run_id>/artifacts/clips/` before calling the VLM backend. Without it,
the tools pass segment time bounds through metadata and prompt text only.

## Direct vs Map-First Description Comparison

Use this runner to test whether the harness helps a long-video description task:

```bash
PYTHONPATH=src python3 -m visual_coding_agent_harness.cli.description_comparison \
  --model-path /m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct \
  --media-path /path/to/long_video.mp4 \
  --question "Describe the video." \
  --duration-sec 3169.06 \
  --window-sec 300 \
  --max-rounds 4 \
  --direct-nframes 64 \
  --max-pixels 151200 \
  --extract-clips \
  --base-dir . \
  --run-id qwen_description_compare
```

The comparison loads one shared Qwen backend and records two strategies:

- `direct_full_video`: one direct VLM description request over the video.
- `map_first_explore`: text-only planner calls `video_ls`, reads the ledger,
  then refines candidate segments with local tools.

The summary is written to `runs/<run_id>/comparison.json`; exploration evidence
is written to `runs/<run_id>_explore/ledger.md`, `observations.jsonl`, and
`trace.jsonl`.

Current smoke observation: `60s` extracted clips are practical for a first loop;
`300s` clips are CPU-heavy and should be avoided until we add better frame
sampling or low-fps VideoMap captions. With no upfront captions/ASR/OCR index,
`video_ls` falls back to timeline anchors, so the next research iteration should
enrich VideoMap before planning.
