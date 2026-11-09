#!/usr/bin/env bash
# vLLM's OpenAI API on :8001 (the gym keeps :8000). Same command on a GCP VM and an
# Insomnia GPU node. Extra args go to `vllm serve`
# (e.g. --enable-prefix-caching).
set -euo pipefail
VLLM=0.33.0
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null ||
  curl -LsSf https://astral.sh/uv/install.sh | sh
# hermes = Qwen2.5's tool-call format; the agent needs it
exec uvx --python 3.12 --from "vllm==$VLLM" vllm serve \
  "$MODEL" --port 8001 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes "$@"
