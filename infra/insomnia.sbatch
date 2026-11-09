#!/bin/bash
# vLLM on one Insomnia GPU. From the repo root:
#   sbatch --account=<group> infra/insomnia.sbatch
# then from your laptop (node from `squeue --me -h -o %N`):
#   ssh -L 8000:<node>:8000 <uni>@insomnia.rcs.columbia.edu
#SBATCH --job-name=vllm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=vllm-%j.log
exec "$SLURM_SUBMIT_DIR/infra/serve.sh"
