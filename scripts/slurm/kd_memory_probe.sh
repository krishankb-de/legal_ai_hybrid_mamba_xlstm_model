#!/usr/bin/env bash
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # HF_HOME (the teachers) and the Inductor cache live on /sc/scratch
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --job-name=kd_probe
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # aisc-batch is preemptible: without this a requeue TRUNCATES the log
#SBATCH --requeue
#
# kd_memory_probe.sh -- does the 8B teacher fit? (plan P4-T; verdict P4-U). Four 20-step 4 x H100
# DDP runs of hybrid_legal_base with online logit KD, through train_pretrain_1gpu.sh:
#   Qwen3-1.7B-Base at 4,096 rows (micro-batch 8 x accum 2) and 8,192 rows (4 x 4),
#   Qwen3-8B-Base   at 4,096 and 8,192, the same shapes (P6's stage A and stage B).
# Every rank prints STEPSTATS (s/step, peak GB) at the end of each run; the summary below collects
# the final lines. An OOM is a result, not an error: the run is marked and the ladder continues.
# P4-U's rule: "8B if peak <= 70 GB at 4,096 rows and step time <= 1.6x the 1.7B step; else 1.7B".
# Synthetic rows at the Qwen3 vocabulary (the cost of KD does not depend on the token values),
# validation and checkpoints off.
# Submit from the Mac:  ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch --gpus=4 scripts/slurm/kd_memory_probe.sh"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export SCRATCH_ROOT
export NUM_GPUS=4 TRAINER_CFG=h100_multi_ddp MODEL_CONFIG=hybrid_legal_base DATASET_CONFIG=synthetic
export MAX_STEPS="${PROBE_STEPS:-20}" SAVE_TOP_K=0
RUN_ID="${SLURM_JOB_ID:-local}"

FAILED=()
probe() {  # name, distill config, row_len, micro-batch, accumulation
  local name="$1"
  export DISTILL_CFG="$2" ROW_LEN="$3" BATCH_SIZE="$4" ACCUM="$5"
  export EXPERIMENT="kd_probe_${RUN_ID}_${name}"
  export EXTRA_OVERRIDES="trainer.limit_val_batches=0 trainer.num_sanity_val_steps=0 trainer.enable_checkpointing=false callbacks.step_stats.every_n_steps=0 callbacks.step_stats.skip_steps=5 dataset.num_rows=256 dataset.min_doc_len=512 dataset.max_doc_len=${ROW_LEN}"
  echo; echo "== probe ${name}: teacher ${DISTILL_CFG}, rows ${ROW_LEN}, micro-batch ${BATCH_SIZE} x accum ${ACCUM} x ${NUM_GPUS} GPUs"
  if bash scripts/slurm/train_pretrain_1gpu.sh; then
    echo "[DONE] ${name}"
  else
    echo "[FAIL] ${name} (OOM or error: see above)"; FAILED+=("$name")
  fi
  rm -rf -- "${SCRATCH_ROOT}/outputs/${EXPERIMENT}"
}

probe t1p7b_L4096 qwen3_1p7b 4096 8 2
probe t1p7b_L8192 qwen3_1p7b 8192 4 4
probe t8b_L4096 qwen3_8b 4096 8 2
probe t8b_L8192 qwen3_8b 8192 4 4

echo
echo "== summary (final STEPSTATS per rank; the verdict reads these lines)"
grep -E '^(== probe|STEPSTATS .* final=1)' "logs/kd_probe_${RUN_ID}.log" 2>/dev/null || echo "(log not found under logs/)"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "KD PROBE RUNS THAT DID NOT FINISH: ${FAILED[*]}"
fi
echo "KD PROBE DONE"
