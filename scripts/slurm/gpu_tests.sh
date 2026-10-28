#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # the Inductor cache lives on /sc/scratch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=gpu_tests
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# gpu_tests.sh -- the `cuda` test layer on one H100 (plan P4-J1, §8.1), plus check_env.py.
# Submit from the Mac:  ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/gpu_tests.sh"
# The log ends in "GPU TESTS PASSED" only if every step exits 0.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# Environment block, port map §11.2
SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# one fresh cache per job: a shared Inductor cache once timed two shapes identically (port map §14)
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/.torchinductor/gpu_tests_${SLURM_JOB_ID:-local}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"   # uv, for check_env.py (job 2588703)
mkdir -p logs "$TORCHINDUCTOR_CACHE_DIR"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0), f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')"
nvidia-smi

FAILED=()
step() {  # name, command...
  local name="$1"; shift
  echo; echo "== $name"
  if "$@"; then echo "[PASS] $name"; else echo "[FAIL] $name"; FAILED+=("$name"); fi
}

step "environment == uv.lock" .venv/bin/python scripts/check_env.py
step "pytest -m cuda" .venv/bin/python -m pytest tests/ -m cuda -v -rA -p no:cacheprovider

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "GPU TESTS FAILED: ${FAILED[*]}"
  exit 1
fi
echo "GPU TESTS PASSED"
