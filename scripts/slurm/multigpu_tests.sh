#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # the Inductor cache lives on /sc/scratch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --job-name=multigpu_tests
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# multigpu_tests.sh -- the `multigpu` test layer on two H100 of one node (plan P4-J3, §8.1): the
# DDP step of tests/test_multigpu.py, gradients identical across ranks.
# Submit from the Mac:  ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch --gpus=2 scripts/slurm/multigpu_tests.sh"
# (`--gpus=2` on the sbatch line as well: port map §11.1.) The log ends in "MULTIGPU TESTS PASSED"
# only if every step exits 0.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

NUM_GPUS="${NUM_GPUS:-2}"
# Environment block, port map §11.2
SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/.torchinductor/multigpu_tests_${SLURM_JOB_ID:-local}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"   # uv, for check_env.py (job 2588703)
mkdir -p logs "$TORCHINDUCTOR_CACHE_DIR"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  NUM_GPUS: $NUM_GPUS"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
# A directive in this file is not overridden by an environment variable (port map §11.1): fail
# fast when the node shows fewer GPUs than the run needs, instead of skipping every test.
.venv/bin/python -c "import torch, sys; n = torch.cuda.device_count(); print('GPUs:', n, [torch.cuda.get_device_name(i) for i in range(n)]); sys.exit(0 if n >= int(sys.argv[1]) else 1)" "$NUM_GPUS"
nvidia-smi

FAILED=()
step() {  # name, command...
  local name="$1"; shift
  echo; echo "== $name"
  if "$@"; then echo "[PASS] $name"; else echo "[FAIL] $name"; FAILED+=("$name"); fi
}

step "environment == uv.lock" .venv/bin/python scripts/check_env.py
step "pytest -m multigpu" .venv/bin/python -m pytest tests/ -m multigpu -v -rA -p no:cacheprovider

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "MULTIGPU TESTS FAILED: ${FAILED[*]}"
  exit 1
fi
echo "MULTIGPU TESTS PASSED"
