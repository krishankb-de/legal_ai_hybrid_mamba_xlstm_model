#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node (x86 .venv -> "Exec format error"); gx13v1: faulty GPU
#SBATCH --constraint=GLB_SCRATCH   # data/raw, the manifests and the HTTP cache live on /sc/scratch
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=4-00:00:00
#SBATCH --job-name=corpus
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
#SBATCH --open-mode=append   # pot-hpi-aisc-batch is preemptible: a requeued task appends to its log
#SBATCH --requeue
#
# build_corpus_array.sh -- the corpus at scale, one CPU array task per collector (plan P4-K;
# verdict P4-L). Submit from the Mac (10 collectors, indices 0-9 of SOURCES):
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch --array=0-9 scripts/slurm/build_corpus_array.sh"
# Writes $SCRATCH_ROOT/data/raw/<src>/<src>.jsonl and $SCRATCH_ROOT/data/manifests/<src>.jsonl,
# both renamed from *.partial only when the collection completes. No --limit (`--limit all`),
# except FineWeb-2: the general-German replay is 30% of a token budget of at most 5B (decision 13),
# so FINEWEB_LIMIT pages (default 2,000,000, ~2B tokens), sampled across dumps.
# At one request per second a legal source takes hours to days. Every response is kept in
# $SCRATCH_ROOT/data/http_cache/<src>: a requeued task -- or the same index resubmitted after a
# TIMEOUT -- replays what it already fetched and carries on. The log ends in
# "CORPUS <src> DONE: <n> documents".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=0   # the collectors read the web and the Hugging Face hub
export HF_DATASETS_OFFLINE=0
export PYTHONUNBUFFERED=1
if [[ -f "$HOME/.hf_token" ]]; then
  HF_TOKEN="$(cat "$HOME/.hf_token")"
  export HF_TOKEN
fi

SOURCES="${SOURCES:-gii rii oldp eurlex ris fedlex bger dip fineweb2_de multilegalpile}"
read -r -a SOURCE_LIST <<< "${SOURCES}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${IDX}" -ge "${#SOURCE_LIST[@]}" ]]; then
  echo "FATAL: array index ${IDX} but only ${#SOURCE_LIST[@]} sources in SOURCES='${SOURCES}'"; exit 1
fi
SRC="${SOURCE_LIST[$IDX]}"
LIMIT=all
if [[ "$SRC" == fineweb2_de ]]; then LIMIT="${FINEWEB_LIMIT:-2000000}"; fi
RAW="${SCRATCH_ROOT}/data/raw/${SRC}"
MANIFESTS="${SCRATCH_ROOT}/data/manifests"
HTTP_CACHE="${SCRATCH_ROOT}/data/http_cache/${SRC}"
mkdir -p logs "$RAW" "$MANIFESTS" "$HTTP_CACHE"

echo "host: $(hostname)  job: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}}_${IDX}  restart: ${SLURM_RESTART_COUNT:-0}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
echo "source: $SRC  limit: $LIMIT  raw: $RAW  cache: $HTTP_CACHE ($(du -sh "$HTTP_CACHE" | cut -f1) so far)"
df -h "$SCRATCH_ROOT" | tail -n 1

.venv/bin/python -m "lexhybrid.data.corpus.collectors.${SRC}" --limit "$LIMIT" \
  --out "$RAW" --manifest-dir "$MANIFESTS" --http-cache "$HTTP_CACHE"

N_DOCS="$(wc -l < "$RAW/${SRC}.jsonl" | tr -d " ")"
echo "raw: $(du -sh "$RAW/${SRC}.jsonl" | cut -f1)  manifest lines: $(wc -l < "$MANIFESTS/${SRC}.jsonl" | tr -d " ")  cache: $(du -sh "$HTTP_CACHE" | cut -f1)"
echo "CORPUS ${SRC} DONE: ${N_DOCS} documents"
