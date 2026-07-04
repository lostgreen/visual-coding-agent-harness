# VCAH

A small evidence-seeking video agent.

Build a cheap cold video index, run an evidence-seeking video agent over it, and answer only with verified evidence citations.

## Run

```bash
python main.py --video demo.mp4 --question "What happens in this video?"
python main.py --videomme-root data/videomme --case 601-1 --run-dir runs/601-1
```

## How It Works

1. Build a cold index: chapters, beats, ASR/OCR text index, visual index, diagnostics.
2. Run a small agent loop: observe, act, memorize.
3. Store observations in memory.
4. Store verified citations in `evidence.jsonl`.
5. Answer only with valid evidence ids.

The current `ModelClient` is a local smoke-test controller. It can exercise the
tool loop and artifact contract, but real long-video evaluation needs a
tool-calling LLM controller and a non-placeholder embedding model.

Visual-only observations are kept in memory until they are verified. The
evidence gate accepts only `EvidenceRecord` ids backed by raw ASR/OCR or a
future verified frame record.

## Files

```text
main.py              single command entry
src/vcah/agent.py    observe -> act -> memorize loop
src/vcah/index.py    cold index
src/vcah/tools.py    search/open/focus/answer tools
src/vcah/memory.py   memory, evidence, trace stores
src/vcah/video.py    frame extraction and timeline grids
src/vcah/model.py    thin model wrapper
src/vcah/types.py    dataclasses
src/vcah/evals.py    VideoMME single-case smoke
```

## Artifacts

```text
runs/601-1/
  cold_index/
    index.json
    diagnostics.json
    visual_index.npz
    timeline.jpg
  run/
    memory.json
    evidence.jsonl
    trace.jsonl
    answer.json
```
