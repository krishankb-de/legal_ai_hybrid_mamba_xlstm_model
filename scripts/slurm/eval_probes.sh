#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # outputs/, data/ and HF_HOME live on /sc/scratch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --job-name=eval_probes
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
#SBATCH --open-mode=append   # pot-hpi-aisc-batch is preemptible: a requeue re-runs the task into the same log
#SBATCH --requeue
#
# eval_probes.sh -- plan P5-G (and the late-25% PPL of P5-F): one array task per screen arm-seed, on
# that run's last.ckpt. ARM is resolved HERE by `screen_arms.py env` (MODEL_CONFIG and the
# experiment, screen<ATTEMPT>_<arm>_s<seed>), as in screen_array.sh.
#   1. scripts/evaluate_val_ppl.py: the validation rows at ROW_LEN (2,048), overall PPL (= exp of the
#      run's last val/loss) and the late-25% PPL: predicted tokens with >= 1,536 tokens of their own
#      document before them (late_doc, the user's choice of 2026-10-13), plus the row-position variant.
#   2. scripts/evaluate_probes.py at CONTEXT (2,048): MQAR, statute recall and multi-hop, the statutes
#      from the scrubbed gii parts.
# Both write JSON to logs/eval_<array job>/<experiment>.{valppl,probes}.json (logs/ is pulled; analysis/
# is not written from the cluster). Submit from the Mac:
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch --array=0-11 scripts/slurm/eval_probes.sh"
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && ARMS='S5-s42 S5-s43' sbatch --array=0-1 --time=08:00:00 scripts/slurm/eval_probes.sh"
# RUNS="<experiment>:<model config> ..." evaluates those runs' last.ckpt instead of screen arms (P6-G,
# P6-K, P6-M), e.g. RUNS='p6_hybrid_s42_A:hybrid_legal_base' ROW_LEN=4096 PROBES=0 for stage A.
# PROBES=0 skips step 2. VALPPL=0 skips step 1 (a re-run of the probes after its JSON is already in an earlier logs/eval_<job>/;
# job 2663318's probes crashed on gii citation codes after every val-PPL had been written).
# The log ends in "EVAL PROBES DONE <experiment>".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export SCRATCH_ROOT   # configs/dataset/mixture_pretrain.yaml reads the shards under it
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=1   # the Qwen3 tokenizer comes from the fetched cache
export PYTHONUNBUFFERED=1
mkdir -p logs

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -n "${RUNS:-}" ]]; then  # explicit "<experiment>:<model config>" entries (P6 runs, not screen arms)
  read -r -a RUN_LIST <<< "${RUNS}"
  if [[ "${IDX}" -ge "${#RUN_LIST[@]}" ]]; then
    echo "FATAL: array index ${IDX} but only ${#RUN_LIST[@]} runs in RUNS='${RUNS}'"; exit 1
  fi
  EXPERIMENT="${RUN_LIST[$IDX]%%:*}"
  MODEL_CONFIG="${RUN_LIST[$IDX]#*:}"
  ARM="$EXPERIMENT"
else
  ARMS="${ARMS:-S1-s42 S1-s43 S0-s42 S0-s43 S2-s42 S2-s43 S3-s42 S3-s43 S4-s42 S4-s43 S6-s42 S6-s43}"
  read -r -a ARM_LIST <<< "${ARMS}"
  if [[ "${IDX}" -ge "${#ARM_LIST[@]}" ]]; then
    echo "FATAL: array index ${IDX} but only ${#ARM_LIST[@]} arms in ARMS='${ARMS}'"; exit 1
  fi
  ARM="${ARM_LIST[$IDX]}"
  unset EXPERIMENT
  ARM_ENV="$(.venv/bin/python scripts/screen_arms.py env "$ARM")" || { echo "FATAL: cannot resolve arm $ARM"; exit 1; }
  eval "$ARM_ENV"
fi
CKPT="${SCRATCH_ROOT}/outputs/${EXPERIMENT}/checkpoints/last.ckpt"
if [[ ! -f "$CKPT" ]]; then echo "FATAL: no checkpoint $CKPT"; exit 1; fi
shopt -s nullglob
STATUTES=("${SCRATCH_ROOT}"/data/scrubbed/gii.part-*-of-*.jsonl)
shopt -u nullglob
if [[ "${#STATUTES[@]}" -eq 0 ]]; then echo "FATAL: no scrubbed gii parts under ${SCRATCH_ROOT}/data/scrubbed"; exit 1; fi
OUT="logs/eval_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  array: ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
echo "arm: $ARM  config: $MODEL_CONFIG  checkpoint: $CKPT  statutes: ${#STATUTES[@]} gii parts"
if [[ "${VALPPL:-1}" == 1 ]]; then
  .venv/bin/python scripts/evaluate_val_ppl.py --checkpoint "$CKPT" --model-config "$MODEL_CONFIG" \
    --row-len "${ROW_LEN:-2048}" --label "$EXPERIMENT" --out "$OUT/${EXPERIMENT}.valppl.json"
fi
if [[ "${PROBES:-1}" == 1 ]]; then
  .venv/bin/python scripts/evaluate_probes.py --model-config "$MODEL_CONFIG" --checkpoint "$CKPT" \
    --context "${CONTEXT:-2048}" --statutes "${STATUTES[@]}" --out "$OUT/${EXPERIMENT}.probes.json"
fi
echo "EVAL PROBES DONE $EXPERIMENT"
