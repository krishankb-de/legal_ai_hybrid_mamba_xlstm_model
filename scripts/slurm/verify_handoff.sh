#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node (x86 .venv -> "Exec format error"); gx13v1: faulty GPU
#SBATCH --constraint=GLB_SCRATCH   # outputs/ lives on /sc/scratch
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH --job-name=verify_handoff
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# verify_handoff.sh -- plan P6-L: each run's last.ckpt against the yaml that builds its model
# (scripts/verify_handoff.py: PASS = 0 missing / 0 unexpected keys; the checkpoint's global_step is
# printed too, so a stage-B last.ckpt shows it reached 18,037). CPU only.
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && RUNS='p6_hybrid_s42_B:hybrid_legal_base p6_transformer_s42_B:transformer_legal_base' sbatch scripts/slurm/verify_handoff.sh"
# RUNS holds "<experiment>:<model config>" entries. Ends with one "HANDOFF <experiment> PASS|FAIL" line
# per run and "VERIFY HANDOFF DONE"; exits 1 if any run failed.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export PYTHONUNBUFFERED=1
read -r -a RUN_LIST <<< "${RUNS:-}"
if [[ "${#RUN_LIST[@]}" -eq 0 ]]; then echo "FATAL: RUNS is empty"; exit 1; fi
mkdir -p logs

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
SUMMARY=()
status=0
for pair in "${RUN_LIST[@]}"; do
  run="${pair%%:*}"
  ckpt="${SCRATCH_ROOT}/outputs/${run}/checkpoints/last.ckpt"
  echo "== ${run}: ${ckpt}"
  if .venv/bin/python scripts/verify_handoff.py "$ckpt" "configs/model/${pair#*:}.yaml"; then
    SUMMARY+=("HANDOFF ${run} PASS")
  else
    SUMMARY+=("HANDOFF ${run} FAIL"); status=1
  fi
done
printf '%s\n' "${SUMMARY[@]}"
echo "VERIFY HANDOFF DONE"
exit "$status"
