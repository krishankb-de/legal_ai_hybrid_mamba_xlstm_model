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
# $SCRATCH_ROOT/data/evalsets/<SET>/items.jsonl; gerlayqa's and legalquad's are fetched there when
# missing (network; into a temporary directory, moved into place whole). SET=sft_dev: the SFT dev split
# $SCRATCH_ROOT/data/sft/dev.jsonl, each row's gold being the passages its answer cites.
# AS_IS=1 (with SET=sft_dev): one set per SFT row instead, its own prompt as it stands (build --as-is:
# its passages from the index in chunk_ids order, its kind, no retrieval, K unused).
# INDEX_VERSION (default v1): $SCRATCH_ROOT/data/index/<v>.
# DTYPE (default float32; bfloat16 for a large model whose longest prompts do not fit in fp32).
# K (default "1 4 9"). LIMIT: each run decodes at most LIMIT sets not yet decoded (a smoke); its summary
# is written only once every set is, so a later job with the same OUT resumes the run.
# SETS: the prompt sets to decode (default logs/eval_gold_passages_<job>/sets.jsonl);
# an existing file is reused (another job's sets: every system sees the same prompts), a missing one
# is built there, with its .counts.json. Outputs in logs/eval_gold_passages_<job>/, per run
# <system>_s<seed>_<experiment>.jsonl (appended as decoded; a requeue resumes it) and .summary.json
# (written last; a requeue decodes no run that has one). Each run with a summary is then scored beside its
# records: scripts/evaluate_citation.py writes <name>.citation.metrics.jsonl and .citation.summary.json, and
# scripts/audit_truncation.py prints TRUNCATION lines; truncated decodes are a FINDING, an audit failure
# fails the job.
# The log has one GOLD line per run, tier and k (none for AS_IS=1), CITATION and TRUNCATION lines per
# scored run and ends in "EVAL GOLD PASSAGES DONE".
# The ARM GPU node (2x L40) runs it from .venv-arm (setup_arm_env.sh), for items already fetched:
#   sbatch --exclude=gx17v1,gx13v1 --nodelist=ga03 ... scripts/slurm/eval_gold_passages.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=1   # the tokenizer, bge-m3 and the NLI model come from the fetched cache
export PYTHONUNBUFFERED=1
mkdir -p logs
PY=.venv/bin/python
if [[ "$(uname -m)" == aarch64 ]]; then PY=.venv-arm/bin/python; export PYTHONPATH="$PWD"; fi   # the ARM GPU node

if [[ -z "${RUNS:-}" ]]; then echo "FATAL: RUNS='<system>:<seed>:<experiment>:<model config> ...' is required"; exit 1; fi
read -r -a RUN_LIST <<< "${RUNS}"
SET="${SET:-gerlayqa}"
if [[ "${AS_IS:-0}" == 1 && "$SET" != sft_dev ]]; then echo "FATAL: AS_IS=1 decodes SFT rows' own prompts: set SET=sft_dev"; exit 1; fi
if [[ "$SET" == sft_dev ]]; then ITEMS="${SCRATCH_ROOT}/data/sft/dev.jsonl"; else ITEMS="${SCRATCH_ROOT}/data/evalsets/${SET}/items.jsonl"; fi
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
  if [[ ! -f "$ITEMS" && ( "$SET" == gerlayqa || "$SET" == legalquad ) ]]; then
    echo "fetching the $SET items into $(dirname "$ITEMS")"
    mkdir -p "$(dirname "$ITEMS")"
    tmp="$(mktemp -d "${SCRATCH_ROOT}/data/evalsets/.fetch.XXXXXX")"
    HF_HUB_OFFLINE=0 "$PY" -m "lexhybrid.data.evalsets.${SET}" --limit 0 --out "$tmp/set" --manifest-dir "$tmp"
    mv "$tmp/set/info.json" "$tmp/set/items.jsonl" "$(dirname "$ITEMS")/"   # items.jsonl last: it marks a whole fetch
    rm -rf "$tmp"
  fi
  if [[ ! -f "$ITEMS" ]]; then echo "FATAL: no items $ITEMS"; exit 1; fi
  if [[ ! -f "$INDEX/index.json" ]]; then echo "FATAL: no index $INDEX"; exit 1; fi
  if [[ "${AS_IS:-0}" == 1 ]]; then BUILD=(--as-is); else BUILD=(--k "${K_LIST[@]}"); fi
  echo "building the prompt sets from $ITEMS and $INDEX: ${BUILD[*]}"
  "$PY" scripts/eval_gold_passages.py build --items "$ITEMS" --index "$INDEX" \
    --out "$SETS" "${BUILD[@]}" --set-name "$SET" --device cuda
fi
score() {  # <name>: citation metrics and truncation audit of a run, once every set is decoded
  [[ -f "$OUT/$1.summary.json" ]] || return 0   # LIMIT left sets undecoded
  "$PY" scripts/evaluate_citation.py "$OUT/$1.jsonl" \
    --metrics-out "$OUT/$1.citation.metrics.jsonl" --summary-out "$OUT/$1.citation.summary.json"
  local status=0
  "$PY" scripts/audit_truncation.py "$OUT/$1.jsonl" || status=$?
  if [[ $status == 1 ]]; then echo "FINDING: $1 has truncated decodes (TRUNCATION lines above)"
  elif [[ $status != 0 ]]; then echo "FATAL: the truncation audit of $1 failed (exit $status)"; exit "$status"; fi
}
for run in "${RUN_LIST[@]}"; do
  IFS=: read -r SYSTEM SEED EXPERIMENT MODEL_CONFIG <<< "$run"
  NAME="${SYSTEM}_s${SEED}_${EXPERIMENT}"   # an SFT and a DPO run of one system and seed stay apart
  if [[ -f "$OUT/${NAME}.summary.json" ]]; then echo "skip decoding $NAME: already decoded"; score "$NAME"; continue; fi
  CKPT="${SCRATCH_ROOT}/outputs/${EXPERIMENT}/checkpoints/last.ckpt"
  if [[ ! -f "$CKPT" ]]; then echo "FATAL: no checkpoint $CKPT"; exit 1; fi
  if [[ "$MODEL_CONFIG" == hf=* ]]; then MODEL_ARGS=(--hf-model "${MODEL_CONFIG#hf=}"); else MODEL_ARGS=(--model-config "$MODEL_CONFIG"); fi
  "$PY" scripts/eval_gold_passages.py decode --sets "$SETS" --system "$SYSTEM" \
    --seed "$SEED" "${MODEL_ARGS[@]}" --checkpoint "$CKPT" --device cuda ${LIMIT:+--limit "$LIMIT"} ${DTYPE:+--dtype "$DTYPE"} \
    --out "$OUT/${NAME}.jsonl" --summary "$OUT/${NAME}.summary.json"
  score "$NAME"
done
echo "EVAL GOLD PASSAGES DONE"
