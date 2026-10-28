#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node (x86 .venv -> "Exec format error"); gx13v1: faulty GPU
#SBATCH --constraint=GLB_SCRATCH   # outputs/ lives on /sc/scratch
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0-01:00:00
#SBATCH --job-name=screen_metrics
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# screen_metrics.sh -- the screen runs' validation PPL and training curves, read from the TensorBoard
# events under outputs/ on scratch into this log (scripts/screen_metrics.py; P5-D, P5-F). CPU only.
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/screen_metrics.sh"
# PREFIX (default screen_) picks the run directories; WINDOW (default 250) the steps per CURVE row;
# DUMP (a tag regex) and STEPS (A-B) print every value of the matching tags, e.g. for diag_spike.sh:
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && PREFIX=diag_ DUMP='^diag/' STEPS=3000-4000 sbatch scripts/slurm/screen_metrics.sh"
# The log ends in "SCREEN METRICS DONE".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
DUMP_ARGS=()
if [[ -n "${DUMP:-}" ]]; then DUMP_ARGS+=(--dump "$DUMP"); fi
if [[ -n "${STEPS:-}" ]]; then DUMP_ARGS+=(--steps "$STEPS"); fi
.venv/bin/python scripts/screen_metrics.py --outputs "${SCRATCH_ROOT}/outputs" \
  --prefix "${PREFIX:-screen_}" --window "${WINDOW:-250}" "${DUMP_ARGS[@]+"${DUMP_ARGS[@]}"}"
echo "SCREEN METRICS DONE"
