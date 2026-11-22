#!/usr/bin/env bash
# Qwen3-8B (4-bit, 4.6 GB) on an Apple-silicon laptop:
# mlx_lm's OpenAI API on :8001, where serve.sh puts vLLM. For
# trying the agents without a GPU or API credits, not for the
# study's numbers. Extra args go to `mlx_lm.server`.
set -euo pipefail
MLX_LM=0.32.2
MODEL=${MODEL:-mlx-community/Qwen3-8B-4bit}
# Qwen3 thinks before it answers unless told not to; the agents
# want the answer only. The server keeps 10 conversations' KV
# caches by default: 7 GB after ten turns of one episode, and
# the GPU runs out of memory on an 18 GB laptop. Two, at 8 bits
# (which serves one request at a time), are 3 GB at 20 turns.
exec uvx --from "mlx-lm==$MLX_LM" mlx_lm.server \
  --model "$MODEL" --port 8001 --max-tokens 2048 \
  --chat-template-args '{"enable_thinking": false}' \
  --prompt-cache-size 2 --kv-bits 8 "$@"
