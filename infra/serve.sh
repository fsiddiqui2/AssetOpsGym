#!/usr/bin/env bash
# vLLM's OpenAI API on :8001 (the gym keeps :8000). Same command on a GCP VM and an
# Insomnia GPU node. Extra args go to `vllm serve`
# (e.g. --enable-prefix-caching).
set -euo pipefail
VLLM=0.33.0
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
# how vLLM is run: its PyPI build, unless the host needs another
# (insomnia.sbatch sets VLLM_RUN)
RUN=${VLLM_RUN:-uvx --python 3.12 --from vllm==$VLLM}
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null ||
  curl -LsSf https://astral.sh/uv/install.sh | sh
# With GPU_LOG set, a line a second of
# `epoch, index, util %, MiB used, MiB total` for
# profiling.read_gpu_log (rollouts.py --gpu-log).
if [ -n "${GPU_LOG:-}" ]; then
  Q=index,utilization.gpu,memory.used,memory.total
  ( while sleep 1; do
      nvidia-smi --query-gpu=$Q --format=csv,noheader,nounits |
        sed "s/^/$(date +%s), /"
    done >> "$GPU_LOG" ) &
  trap "kill $! 2>/dev/null" EXIT
fi
# hermes = Qwen2.5's tool-call format; the agent needs it
$RUN vllm serve \
  "$MODEL" --port 8001 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes "$@"
