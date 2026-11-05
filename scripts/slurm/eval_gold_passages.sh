#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # outputs/, data/ and HF_HOME live on /sc/scratch
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --job-name=eval_gold_passages
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # pot-hpi-aisc-batch is preemptible: a requeue re-runs into the same log
#SBATCH --requeue
#
# eval_gold_passages.sh -- the gold-passage sanity check (scripts/eval_gold_passages.py): each item's
# gold passage among 1, 4 or 9 distractors (easy: other acts; hard: the fused retriever's best chunks
# that are neither gold-relevant nor a copy), greedy under the pointer FSM, then the verifier
# (mDeBERTa NLI). The sets are built once, so every system decodes the same prompts.
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && RUNS='hybrid:42:<experiment>:hybrid_legal_base' sbatch scripts/slurm/eval_gold_passages.sh"
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && SETS=logs/eval_gold_passages_<job>/sets.jsonl RUNS='...' sbatch scripts/slurm/eval_gold_passages.sh"
# A Hugging Face system's model config is hf=<repository>, e.g. 'lora:42:<experiment>:hf=Qwen/Qwen3-8B-Base'.
# RUNS (required): "<system>:<seed>:<experiment>:<model config> ...", each run's last.ckpt under
# $SCRATCH_ROOT/outputs/<experiment>/checkpoints/. SET (default gerlayqa): the items in
# $SCRATCH_ROOT/data/evalsets/<SET>/items.jsonl. INDEX_VERSION (default v1): $SCRATCH_ROOT/data/index/<v>.
# K (default "1 4 9"). SETS: the prompt sets to decode (default logs/eval_gold_passages_<job>/sets.jsonl);
# an existing file is reused (another job's sets: every system sees the same prompts), a missing one
# is built there, with its .counts.json. Outputs in logs/eval_gold_passages_<job>/, per run
# <system>_s<seed>_<experiment>.jsonl (appended as decoded; a requeue resumes it) and .summary.json
# (written last; a requeue skips a run that has one).
# The log has one GOLD line per run, tier and k and ends in "EVAL GOLD PASSAGES DONE".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=1   # the tokenizer, bge-m3 and the NLI model come from the fetched cache
export PYTHONUNBUFFERED=1
mkdir -p logs

if [[ -z "${RUNS:-}" ]]; then echo "FATAL: RUNS='<system>:<seed>:<experiment>:<model config> ...' is required"; exit 1; fi
read -r -a RUN_LIST <<< "${RUNS}"
ITEMS="${SCRATCH_ROOT}/data/evalsets/${SET:-gerlayqa}/items.jsonl"
INDEX="${SCRATCH_ROOT}/data/index/${INDEX_VERSION:-v1}"
read -r -a K_LIST <<< "${K:-1 4 9}"
OUT="${OUT:-logs/eval_gold_passages_${SLURM_JOB_ID:-local}}"   # set OUT to resume a timed-out job in its directory
SETS="${SETS:-$OUT/sets.jsonl}"
mkdir -p "$OUT"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
echo "sets: $SETS  runs: ${RUNS}"
if [[ -f "$SETS" ]]; then
  echo "reusing the prompt sets $SETS"
else
  if [[ ! -f "$ITEMS" ]]; then echo "FATAL: no items $ITEMS"; exit 1; fi
  if [[ ! -f "$INDEX/index.json" ]]; then echo "FATAL: no index $INDEX"; exit 1; fi
  echo "building the prompt sets from $ITEMS and $INDEX, k: ${K_LIST[*]}"
  .venv/bin/python scripts/eval_gold_passages.py build --items "$ITEMS" --index "$INDEX" \
    --out "$SETS" --k "${K_LIST[@]}" --device cuda
fi
for run in "${RUN_LIST[@]}"; do
  IFS=: read -r SYSTEM SEED EXPERIMENT MODEL_CONFIG <<< "$run"
  NAME="${SYSTEM}_s${SEED}_${EXPERIMENT}"   # an SFT and a DPO run of one system and seed stay apart
  if [[ -f "$OUT/${NAME}.summary.json" ]]; then echo "skip $NAME: already scored"; continue; fi
  CKPT="${SCRATCH_ROOT}/outputs/${EXPERIMENT}/checkpoints/last.ckpt"
  if [[ ! -f "$CKPT" ]]; then echo "FATAL: no checkpoint $CKPT"; exit 1; fi
  if [[ "$MODEL_CONFIG" == hf=* ]]; then MODEL_ARGS=(--hf-model "${MODEL_CONFIG#hf=}"); else MODEL_ARGS=(--model-config "$MODEL_CONFIG"); fi
  .venv/bin/python scripts/eval_gold_passages.py decode --sets "$SETS" --system "$SYSTEM" \
    --seed "$SEED" "${MODEL_ARGS[@]}" --checkpoint "$CKPT" --device cuda \
    --out "$OUT/${NAME}.jsonl" --summary "$OUT/${NAME}.summary.json"
done
echo "EVAL GOLD PASSAGES DONE"
