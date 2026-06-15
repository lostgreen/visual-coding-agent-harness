#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python}"
MODEL_PATH="${PLANNER_MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-9b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --dtype auto \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enforce-eager \
  --skip-mm-profiling \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --disable-log-stats
