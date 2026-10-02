#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node (x86 .venv -> "Exec format error"); gx13v1: faulty GPU
#SBATCH --constraint=GLB_SCRATCH   # data/scrubbed and data/shards live on /sc/scratch
#SBATCH --mem=256G   # dedup signatures + the LSH index of every document of every source
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --job-name=pack_corpus
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # pot-hpi-aisc-batch is preemptible: a requeued run appends to its log
#SBATCH --requeue
#
# pack_corpus.sh -- dedup -> pack at 4,096 and 8,192 over the scrubbed corpus (plan P4-L, second
# step, after scrub_array.sh; verdict P4-M): scripts/build_shards.py streams every source's scrub
# parts (their original order restored), deduplicates all sources together (the commercial-safe copy
# survives), splits each source by document (0.1%, at least 2,000) and packs it at both row lengths
# from one tokenisation, one source per worker process. One source per shard directory: the
# research-only source (multilegalpile) is a shard set of its own, which only the research arm
# admits. Submit from the Mac once every scrub task is COMPLETED:
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/pack_corpus.sh"
# Writes $SCRATCH_ROOT/data/shards/{4096,8192}/<src>/{train,val}/ + meta.json, <row_len>/dedup.jsonl
# and build_summary.json (the numbers of analysis/corpus_manifest.md, P4-M). A requeued run starts
# over (it rewrites every shard). The log ends in "PACK CORPUS DONE".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=1   # the Qwen3 tokenizer comes from the fetched cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false   # one tokenizer per worker process
SHARDS="${SCRATCH_ROOT}/data/shards"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
mkdir -p logs "$SHARDS"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  restart: ${SLURM_RESTART_COUNT:-0}  workers: $WORKERS"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
df -h "$SCRATCH_ROOT" | tail -n 1
echo "scrubbed parts: $(find "${SCRATCH_ROOT}/data/scrubbed" -name '*.jsonl' | wc -l | tr -d ' ')  unfinished: $(find "${SCRATCH_ROOT}/data/scrubbed" -name '*.partial' | wc -l | tr -d ' ')"

.venv/bin/python scripts/build_shards.py --root "$SHARDS" --row-len 4096 8192 \
  --scrubbed "${SCRATCH_ROOT}/data/scrubbed" --raw "${SCRATCH_ROOT}/data/raw" --workers "$WORKERS"

du -sh "$SHARDS"/4096 "$SHARDS"/8192
cat "$SHARDS/build_summary.json"
echo "PACK CORPUS DONE"
