#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node (x86 .venv -> "Exec format error"); gx13v1: faulty GPU
#SBATCH --constraint=GLB_SCRATCH   # outputs/ lives on /sc/scratch
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH --job-name=inspect_ckpt
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# inspect_ckpt.sh -- per-parameter weights and AdamW state of training checkpoints
# (scripts/inspect_checkpoint.py). CPU only. Default: screen attempt 1's
# collapsed runs next to healthy runs of the same configs. RUNS is "<experiment>:<model config>"
# pairs; CKPT (default last.ckpt) the file under each run's checkpoints/.
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/inspect_ckpt.sh"
# The log ends in "INSPECT CKPT DONE".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export PYTHONUNBUFFERED=1
RUNS="${RUNS:-screen_S1_s42:hybrid_legal_base screen_S1_s43:hybrid_legal_base screen_S3_s43:hybrid_legal_ds64 screen_S3_s42:hybrid_legal_ds64}"
read -r -a RUN_LIST <<< "${RUNS}"
mkdir -p logs

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
for pair in "${RUN_LIST[@]}"; do
  run="${pair%%:*}"
  .venv/bin/python scripts/inspect_checkpoint.py --label "$run" --model-config "${pair#*:}" \
    --checkpoint "${SCRATCH_ROOT}/outputs/${run}/checkpoints/${CKPT:-last.ckpt}"
done
echo "INSPECT CKPT DONE"
