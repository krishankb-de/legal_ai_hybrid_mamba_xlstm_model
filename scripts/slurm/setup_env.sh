#!/usr/bin/env bash
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node (x86 .venv -> "Exec format error"); gx13v1: faulty GPU
#SBATCH --constraint=GLB_SCRATCH   # the uv cache lives on /sc/scratch, mounted only on these nodes
#SBATCH --mem=32G   # job 2588677: uv sync peaked at 14.1 GB of 16G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=setup_env
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# setup_env.sh -- build the cluster .venv from uv.lock (plan P4-E, port map §11.3, decision 18).
# Submit from the Mac:  ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/setup_env.sh"
# torch 2.11.0+cu128 comes from the lock (Linux resolves the cu128 index); uv installs the
# project editable. The uv cache goes to scratch: home has a 200 GiB quota.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export UV_CACHE_DIR="${SCRATCH_ROOT}/.uv-cache"
export UV_LINK_MODE=copy   # cache (scratch) and .venv (home) are different filesystems
export PYTHONUNBUFFERED=1
mkdir -p logs "$UV_CACHE_DIR"

echo "host: $(hostname)  arch: $(uname -m)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"

if ! command -v uv &> /dev/null; then
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env"
  fi
fi
echo "uv: $(uv --version)"

uv sync --locked
.venv/bin/python scripts/check_env.py
# The LER scrub's own environment (decision 22; flair needs transformers < 5), from its own lock:
# the P4-L scrub array runs scripts/scrub_ner_worker.py in it.
uv sync --locked --project envs/scrub
envs/scrub/.venv/bin/python scripts/check_env.py --scrub
echo "SETUP_ENV DONE"
