#!/usr/bin/env bash
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # HF_HOME and caches live on /sc/scratch
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=preflight
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# preflight.sh -- CPU checks before any GPU job (plan P4-I, port map §11.1).
# Submit from the Mac:  ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/preflight.sh"
# Every step runs; the log ends in "PRE-FLIGHT PASSED" only if all of them exit 0.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/.torchinductor"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"   # uv (setup_env installs it there); batch jobs lack it (job 2588703)
mkdir -p logs

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"

FAILED=()
step() {  # name, command...
  local name="$1"; shift
  echo; echo "== $name"
  if "$@"; then echo "[PASS] $name"; else echo "[FAIL] $name"; FAILED+=("$name"); fi
}

step "environment == uv.lock" .venv/bin/python scripts/check_env.py
step "screen arms verify --full (ARCH line per arm, parameter bands)" .venv/bin/python scripts/screen_arms.py verify --full
step "pytest (not cuda, not slow, not reference)" \
  .venv/bin/python -m pytest tests/ -m "not cuda and not slow and not reference" -q -p no:cacheprovider

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "PRE-FLIGHT FAILED: ${FAILED[*]}"
  exit 1
fi
echo "PRE-FLIGHT PASSED"
