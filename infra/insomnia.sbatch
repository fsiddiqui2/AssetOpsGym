#!/bin/bash
# vLLM on one Insomnia GPU. From the repo root:
#   sbatch --account=<group> infra/insomnia.sbatch
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
exec infra/serve.sh "$@"
