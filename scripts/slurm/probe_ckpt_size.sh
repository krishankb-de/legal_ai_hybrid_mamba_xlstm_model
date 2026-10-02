#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # the probe writes its checkpoint to /sc/scratch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --job-name=probe_ckpt
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # pot-hpi-aisc-batch is preemptible: without this a requeue TRUNCATES the log
#SBATCH --requeue
#
# probe_ckpt_size.sh -- how big one hybrid_legal_base checkpoint is (plan P4-N; verdict P4-O): a
# 10-step run on one H100 through train_pretrain_1gpu.sh (so the training wrapper runs once end to
# end before any screen) writing last.ckpt to scratch, then its size, then the probe's outputs are
# deleted here (the login node has no rm). Synthetic rows at the Qwen3 vocabulary and no teacher:
# the teacher is never part of a checkpoint (PretrainLightningModule keeps it out of state_dict).
# Submit from the Mac:  ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/probe_ckpt_size.sh"
# The log ends with "CKPT_BYTES=<n> CKPT_GB=<x>" and "PROBE_CKPT DONE".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export SCRATCH_ROOT
export EXPERIMENT="probe_ckpt_${SLURM_JOB_ID:-local}"
export MODEL_CONFIG=hybrid_legal_base DATASET_CONFIG=synthetic DISTILL_CFG=none
export MAX_STEPS=10 CKPT_EVERY=5 VAL_EVERY=5 SAVE_TOP_K=0 BATCH_SIZE=4 ROW_LEN=2048
export EXTRA_OVERRIDES="dataset.num_rows=64 dataset.min_doc_len=256 dataset.max_doc_len=2048"
OUT="${SCRATCH_ROOT}/outputs/${EXPERIMENT}"

bash scripts/slurm/train_pretrain_1gpu.sh

CKPT="${OUT}/checkpoints/last.ckpt"
[[ -f "$CKPT" ]] || { echo "FATAL: $CKPT was not written"; exit 1; }
ls -la "${OUT}/checkpoints"
BYTES="$(wc -c < "$CKPT" | tr -d ' ')"
echo "CKPT_BYTES=${BYTES} CKPT_GB=$(awk -v b="$BYTES" 'BEGIN { printf "%.3f", b / 1e9 }')"
if [[ "${KEEP_PROBE:-0}" != 1 ]]; then
  rm -rf -- "$OUT"
  echo "probe outputs deleted: $OUT"
fi
echo "PROBE_CKPT DONE"
