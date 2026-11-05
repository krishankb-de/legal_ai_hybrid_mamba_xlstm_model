#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # data/ and HF_HOME live on /sc/scratch
#SBATCH --mem=192G
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --job-name=build_dense_index
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # pot-hpi-aisc-batch is preemptible: a requeue re-runs into the same log
#SBATCH --requeue
#
# build_dense_index.sh -- the retrieval index over the scrubbed corpus: chunks and BM25
# (scripts/build_index.py --kind bm25, cross-source copies linked from the pack's dedup.jsonl), then
# the bge-m3 vector of every chunk on the GPU (--kind dense). Writes $SCRATCH_ROOT/data/index/$INDEX_VERSION/.
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/build_dense_index.sh"
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && SOURCES='gii rii oldp eurlex' INDEX_VERSION=v2 sbatch scripts/slurm/build_dense_index.sh"
# SOURCES (default "gii rii eurlex": federal statutes, federal court decisions, EU law) picks the
# scrubbed parts. A finished bm25 build in the directory (index.json is written last) is kept, so a
# requeue redoes only the vectors. The log ends with the DENSE line (chunks, bytes, chunks/s) and
# "BUILD DENSE INDEX DONE".
# BM25_ONLY=1 stops after the bm25 build, which needs no GPU, so it need not wait for one:
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && BM25_ONLY=1 sbatch --gpus=0 --job-name=build_bm25_index scripts/slurm/build_dense_index.sh"
# MINUTES stops the encoder after that long; the vectors resume from their last committed batch, so
# an array of short tasks fits the gaps in the queue. Tasks after the finished one exit at once, and
# the last task fails if vectors are still missing, so a job chained afterok does not start:
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && MINUTES=40 sbatch --array=0-3%1 --time=00:45:00 --mem=64G --cpus-per-task=8 --output=logs/build_dense_index_%A_%a.log --error=logs/build_dense_index_%A_%a.log scripts/slurm/build_dense_index.sh"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=1   # bge-m3 comes from the fetched cache
export PYTHONUNBUFFERED=1
mkdir -p logs

OUT="${SCRATCH_ROOT}/data/index/${INDEX_VERSION:-v1}"
DEDUP="${SCRATCH_ROOT}/data/shards/4096/dedup.jsonl"
read -r -a SOURCE_LIST <<< "${SOURCES:-gii rii eurlex}"
DOCS=()
shopt -s nullglob
for source in "${SOURCE_LIST[@]}"; do
  parts=("${SCRATCH_ROOT}/data/scrubbed/${source}".part-*-of-*.jsonl)
  if [[ "${#parts[@]}" -eq 0 ]]; then echo "FATAL: no scrubbed parts for ${source}"; exit 1; fi
  DOCS+=("${parts[@]}")
done
shopt -u nullglob
if [[ ! -f "$DEDUP" ]]; then echo "FATAL: no dedup log $DEDUP"; exit 1; fi

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
echo "sources: ${SOURCE_LIST[*]} (${#DOCS[@]} parts) -> $OUT"
if [[ ! -f "$OUT/index.json" ]]; then
  .venv/bin/python scripts/build_index.py --kind bm25 --docs "${DOCS[@]}" --dedup "$DEDUP" --out "$OUT"
fi
if [[ "${BM25_ONLY:-0}" == 1 ]]; then echo "BUILD BM25 INDEX DONE $OUT"; exit 0; fi
if [[ -f "$OUT/dense/config.json" ]]; then echo "BUILD DENSE INDEX DONE $OUT"; exit 0; fi
rc=0
timeout "${MINUTES:-0}m" .venv/bin/python scripts/build_index.py --kind dense --out "$OUT" --device cuda || rc=$?
if [[ "$rc" == 124 && "${SLURM_ARRAY_TASK_ID:-0}" != "${SLURM_ARRAY_TASK_MAX:-0}" ]]; then
  echo "BUILD DENSE INDEX PARTIAL: the next array task resumes"
  exit 0
fi
if [[ "$rc" != 0 ]]; then exit "$rc"; fi
echo "BUILD DENSE INDEX DONE $OUT"
