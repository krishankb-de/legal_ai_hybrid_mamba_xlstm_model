#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node (x86 .venv -> "Exec format error"); gx13v1: faulty GPU
#SBATCH --constraint=GLB_SCRATCH   # data/manifests and data/raw live on /sc/scratch
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-04:00:00
#SBATCH --job-name=corpus_counts
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# corpus_counts.sh -- documents and characters per source,
# jurisdiction and licence from the collection manifests, tokens estimated from a stride sample of
# each source (scripts/corpus_counts.py; exact tokens come from pack_corpus.sh). CPU only. Submit:
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/corpus_counts.sh"
# SOURCES and JURISDICTIONS default to German law only. The log ends in
# "CORPUS COUNTS DONE".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=1   # the Qwen3 tokenizer comes from the fetched cache
export PYTHONUNBUFFERED=1
SOURCES="${SOURCES:-gii rii oldp eurlex dip fineweb2_de multilegalpile}"
JURISDICTIONS="${JURISDICTIONS:-DE EU}"
read -r -a SOURCE_LIST <<< "${SOURCES}"
read -r -a JURISDICTION_LIST <<< "${JURISDICTIONS}"
mkdir -p logs

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
echo "sources: ${SOURCES}  jurisdictions: ${JURISDICTIONS}"
.venv/bin/python scripts/corpus_counts.py --manifests "${SCRATCH_ROOT}/data/manifests" \
  --raw "${SCRATCH_ROOT}/data/raw" --sources "${SOURCE_LIST[@]}" --jurisdictions "${JURISDICTION_LIST[@]}" \
  --sample "${SAMPLE:-2000}"
echo "CORPUS COUNTS DONE"
