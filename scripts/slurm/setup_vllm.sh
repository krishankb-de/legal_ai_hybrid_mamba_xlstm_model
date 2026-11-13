#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node (x86 .venv -> "Exec format error"); gx13v1: faulty GPU
#SBATCH --constraint=GLB_SCRATCH   # the uv cache lives on /sc/scratch, mounted only on these nodes
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --job-name=setup_vllm
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# setup_vllm.sh -- build .venv-vllm, the vLLM server's own environment, from envs/vllm/uv.lock.
# vLLM pins its own torch, so it never enters .venv; jobs start `vllm serve` from .venv-vllm and
# talk to it over HTTP (scripts/slurm/lib/vllm_server.sh). A CPU job: vllm_smoke.sh checks the GPU path.
# Submit from your workstation:  ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/setup_vllm.sh"
# The log ends with the vLLM and torch versions and "SETUP VLLM DONE".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export UV_CACHE_DIR="${SCRATCH_ROOT}/.uv-cache"
export UV_LINK_MODE=copy   # cache (scratch) and .venv-vllm (home) are different filesystems
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-vllm"   # beside .venv, not envs/vllm/.venv
export PYTHONUNBUFFERED=1
mkdir -p logs "$UV_CACHE_DIR"

echo "host: $(hostname)  arch: $(uname -m)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
if ! command -v uv &> /dev/null; then export PATH="$HOME/.local/bin:$PATH"; fi
command -v uv &> /dev/null || { echo "FATAL: no uv (setup_env.sh installs it)"; exit 1; }
echo "uv: $(uv --version)"

uv sync --locked --project envs/vllm
.venv-vllm/bin/python -c 'from importlib.metadata import version; import torch
print("vllm", version("vllm"), "torch", torch.__version__, "cuda", torch.version.cuda)'
echo "SETUP VLLM DONE"
