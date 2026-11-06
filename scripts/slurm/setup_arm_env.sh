#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --nodelist=ga03   # the ARM GPU node (aarch64, 2x L40) this environment is for
#SBATCH --exclude=gx17v1,gx13v1   # gx13v1: faulty GPU
#SBATCH --constraint=GLB_SCRATCH   # the uv cache lives on /sc/scratch, mounted only on these nodes
#SBATCH --gpus=1   # the CUDA check at the end
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --job-name=setup_arm_env
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# setup_arm_env.sh -- build .venv-arm, the inference environment of the ARM GPU node, from
# envs/arm/uv.lock. The x86 .venv and uv cannot run there, so this job installs an aarch64 uv of the
# same version under $SCRATCH_ROOT/.uv-aarch64. Jobs that run on that node pick .venv-arm and put the
# repository on PYTHONPATH (eval_retrieval.sh does).
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/setup_arm_env.sh"
# The log ends with torch, CUDA and the GPU's name, then "SETUP ARM ENV DONE".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
UV_VERSION=0.12.19   # the uv that wrote envs/arm/uv.lock
UV_DIR="${SCRATCH_ROOT}/.uv-aarch64"
export UV_CACHE_DIR="${SCRATCH_ROOT}/.uv-cache"
export UV_LINK_MODE=copy   # cache (scratch) and .venv-arm (home) are different filesystems
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-arm"   # beside .venv, not envs/arm/.venv
export PYTHONUNBUFFERED=1
mkdir -p logs "$UV_CACHE_DIR"

echo "host: $(hostname)  arch: $(uname -m)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
[[ "$(uname -m)" == aarch64 ]] || { echo "FATAL: not an aarch64 node"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

if [[ ! -x "$UV_DIR/uv" ]]; then
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_INSTALL_DIR="$UV_DIR" UV_NO_MODIFY_PATH=1 sh
fi
echo "uv: $("$UV_DIR/uv" --version)"

"$UV_DIR/uv" sync --locked --project envs/arm
PYTHONPATH="$PWD" .venv-arm/bin/python scripts/eval_retrieval.py --help > /dev/null
PYTHONPATH="$PWD" .venv-arm/bin/python -c 'import torch
x = torch.ones(256, 256, device="cuda")
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), float((x @ x)[0, 0]))'
echo "SETUP ARM ENV DONE"
