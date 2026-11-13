#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # data/ and HF_HOME live on /sc/scratch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --job-name=dpo_pairs
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # pot-hpi-aisc-batch is preemptible: without this a requeue TRUNCATES the log
#SBATCH --requeue
#
# build_dpo_pairs.sh -- the SFT model's FSM-constrained samples, verified (mDeBERTa NLI from the fetched
# cache, offline), as DPO/KTO pairs (scripts/build_dpo_pairs.py). One job writes both splits, from the SFT
# builder's train.jsonl and dev.jsonl into $OUT_DIR, named after the SFT run so another model never
# shares them:
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && SFT_CKPT=<sft run>/checkpoints/last.ckpt sbatch scripts/slurm/build_dpo_pairs.sh"
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && HF_BASELINE=1 SFT_CKPT=<baseline sft run>/checkpoints/last.ckpt sbatch scripts/slurm/build_dpo_pairs.sh"
# Levers: SFT_CKPT (required: <sft run>/checkpoints/<file>) SFT_DIR (default $SCRATCH_ROOT/data/sft) SPLITS
# (default "train dev") OUT_DIR (default $SCRATCH_ROOT/data/dpo/<sft run>) MODEL_CONFIG (the SFT run's)
# HF_BASELINE(=1: --hf-model) HF_MODEL N MAX_NEW_TOKENS TEMPERATURE TOP_P SEED.
# Each prompt's samples are appended as they are verified, so a requeue continues where the job stopped.
# A split with no pairs stops the job (exit 1, the reason in the log).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
mkdir -p logs

: "${SFT_CKPT:?set SFT_CKPT to the checkpoint of the SFT run}"
SFT_RUN="$(basename "$(dirname "$(dirname "$SFT_CKPT")")")"
[[ -f "$SFT_CKPT" && "$(basename "$(dirname "$SFT_CKPT")")" == checkpoints && "$SFT_RUN" != . ]] \
  || { echo "FATAL: SFT_CKPT=$SFT_CKPT is not a file <sft run>/checkpoints/<file>"; exit 1; }
SFT_DIR="${SFT_DIR:-${SCRATCH_ROOT}/data/sft}"
OUT_DIR="${OUT_DIR:-${SCRATCH_ROOT}/data/dpo/${SFT_RUN}}"
if [[ "${HF_BASELINE:-0}" == 1 ]]; then
  MODEL_ARGS=(--hf-model "${HF_MODEL:-Qwen/Qwen3-0.6B-Base}")
else
  MODEL_ARGS=(--model-config "${MODEL_CONFIG:-hybrid_legal_base}")
fi

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  restart: ${SLURM_RESTART_COUNT:-0}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
echo "sft run: $SFT_RUN  sft: $SFT_CKPT  model: ${MODEL_ARGS[*]}  data: $SFT_DIR  out: $OUT_DIR"
.venv/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.device_count() >= int(sys.argv[1]) else 1)" 1 \
  || { echo "FATAL: no CUDA device visible"; exit 1; }

for SPLIT in ${SPLITS:-train dev}; do
  .venv/bin/python scripts/build_dpo_pairs.py --sft-data "${SFT_DIR}/${SPLIT}.jsonl" --checkpoint "$SFT_CKPT" \
    "${MODEL_ARGS[@]}" --out "${OUT_DIR}/${SPLIT}.jsonl" --n "${N:-8}" --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
    --temperature "${TEMPERATURE:-1.0}" --top-p "${TOP_P:-0.95}" --seed "${SEED:-0}" --device cuda
done
echo "DPO PAIRS DONE $OUT_DIR"
