#!/bin/bash
# vLLM on one Insomnia GPU. Once, from the login node (downloads
# are slow on compute nodes and should not hold a GPU), with
# UV_CACHE_DIR, UV_PROJECT_ENVIRONMENT and HF_HOME exported to
# scratch (home has a 50 GB quota):
#   uv sync -p 3.12 --project infra/insomnia
#   uv run --project infra/insomnia python -c "from \
#     huggingface_hub import snapshot_download as d; \
#     d('Qwen/Qwen2.5-3B-Instruct')"
# Then from the repo root, with the same variables plus
# HF_HUB_OFFLINE=1 and VLLM_API_KEY (the port is open to the
# cluster) in --export:
#   sbatch --account=<group> infra/insomnia.sbatch
# MODEL picks another model; Qwen3 also needs
#   --default-chat-template-kwargs '{"enable_thinking": false}'
# or it thinks before every answer.
# Arguments after the script go to `vllm serve`, e.g.
#   ... infra/insomnia.sbatch --chat-template infra/qwen25_tools.jinja
# (Qwen2.5's template with its tool-call example in single
# braces; the 3B model can copy the doubled ones, and the call
# then does not parse).
# then from your laptop (node from `squeue --me -h -o %N`):
#   ssh -L 8001:<node>:8001 <uni>@insomnia.rcs.columbia.edu
#SBATCH --job-name=vllm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=vllm-%j.log
cd "$SLURM_SUBMIT_DIR"
# the driver here cannot load vLLM's PyPI (CUDA 13) build
export VLLM_RUN="uv run --project infra/insomnia"
# the nodes' nvcc is too old for FlashInfer's JIT sampler
export VLLM_USE_FLASHINFER_SAMPLER=0
LOCAL=/local/$USER  # node-local disk for vLLM's compile caches
if mkdir -p "$LOCAL" 2>/dev/null; then
  export VLLM_CACHE_ROOT=$LOCAL/vllm-cache
  export TRITON_CACHE_DIR=$LOCAL/triton-cache
  export TORCHINDUCTOR_CACHE_DIR=$LOCAL/inductor-cache
fi
export GPU_LOG=${GPU_LOG:-gpu-$SLURM_JOB_ID.csv}
exec infra/serve.sh "$@"
