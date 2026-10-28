#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # the Inductor cache lives on /sc/scratch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=equivalence
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# equivalence.sh -- rule R1 on one H100 (plan P4-R; verdict P4-S): the operator and model-logit
# equivalence gate of scripts/check_operator_equivalence.py at chunk sizes 64 and 128, compiled,
# at 1e-4; then P2-Y's compile smoke under CUDA (tests/test_gpu.py, the compile tests).
# Submit from the Mac:  ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/equivalence.sh"
# Step 1 is the box's command as written (the default model, ref_hybrid_m3); step 2 runs the same
# gate on the legal base, whose chunk size P4-Q decides. The log ends in "EQUIVALENCE PASSED" only
# if every step exits 0.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# Environment block, port map §11.2
SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# one fresh cache per job (plan R7): a shared Inductor cache once timed two shapes identically
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/.torchinductor/equivalence_${SLURM_JOB_ID:-local}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
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

step "R1 ref_hybrid_m3: chunk 64/128, compiled" \
  .venv/bin/python scripts/check_operator_equivalence.py --device cuda --chunk-sizes 64 128 --compile
step "R1 hybrid_legal_base: chunk 64/128, compiled, L=2048" \
  .venv/bin/python scripts/check_operator_equivalence.py --device cuda --model hybrid_legal_base \
  --chunk-sizes 64 128 --seq-length 2048 --compile
step "compile smoke under CUDA (P2-Y)" \
  .venv/bin/python -m pytest tests/test_gpu.py -m cuda -k compile -v -rA -p no:cacheprovider

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "EQUIVALENCE FAILED: ${FAILED[*]}"
  exit 1
fi
echo "EQUIVALENCE PASSED"
