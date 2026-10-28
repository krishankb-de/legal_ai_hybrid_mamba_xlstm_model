#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # the Inductor cache lives on /sc/scratch
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=diag_compile
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# diag_compile.sh -- per-mixer, per-row compile-vs-eager drift on one H100 (P4-J2 follow-up to job
# 2588779). Diagnostic only: prints a table, exits 0 unless the script itself breaks.
# Submit from the Mac:  ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/diag_compile.sh"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HUB_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/.torchinductor/diag_compile_${SLURM_JOB_ID:-local}"
export PYTHONUNBUFFERED=1
mkdir -p logs "$TORCHINDUCTOR_CACHE_DIR"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

.venv/bin/python scripts/diagnose_compile.py --seq-len 512 --dim 128
