#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node (x86 .venv -> "Exec format error"); gx13v1: faulty GPU
#SBATCH --constraint=GLB_SCRATCH   # data/ and HF_HOME live on /sc/scratch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --job-name=build_sft_data
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # pot-hpi-aisc-batch is preemptible: a requeue re-runs into the same log
#SBATCH --requeue
#
# build_sft_data.sh -- the SFT data from synthetic rows, optionally with extractive QA items
# (scripts/build_sft_data.py). CPU only.
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && SYNTHETIC=<rows.jsonl> sbatch scripts/slurm/build_sft_data.sh"
# SYNTHETIC (required), QA (optional, needs info.json beside it), OUT (default $SCRATCH_ROOT/data/sft).
# Evaluation questions must never enter training, and the guard only knows what the manifests hold:
# the copies in the repository list a sample of each set. So the guard reads complete manifests in
# $SCRATCH_ROOT/data/evalsets/manifests, fetched here once (network) into a temporary directory and
# moved into place whole, so a killed fetch never leaves a partial list behind.
# The log ends in "BUILD SFT DATA DONE <out>".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export PYTHONUNBUFFERED=1
mkdir -p logs

if [[ -z "${SYNTHETIC:-}" || ! -f "${SYNTHETIC}" ]]; then echo "FATAL: SYNTHETIC='${SYNTHETIC:-}' is not a file"; exit 1; fi
OUT="${OUT:-${SCRATCH_ROOT}/data/sft}"
MANIFESTS="${SCRATCH_ROOT}/data/evalsets/manifests"
mkdir -p "$MANIFESTS"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
for name in gerlerb gerdalir gerlayqa legalquad; do
  if [[ ! -f "$MANIFESTS/evalset_${name}.jsonl" ]]; then
    tmp="$(mktemp -d "${SCRATCH_ROOT}/data/evalsets/.fetch.XXXXXX")"
    HF_HUB_OFFLINE=0 .venv/bin/python -m "lexhybrid.data.evalsets.${name}" --limit 0 --out "$tmp/set" --manifest-dir "$tmp"
    mv "$tmp/evalset_${name}.jsonl" "$MANIFESTS/"
    rm -rf "$tmp"
  fi
done
export HF_HUB_OFFLINE=1   # the Qwen3 tokenizer comes from the fetched cache
.venv/bin/python scripts/build_sft_data.py --synthetic "$SYNTHETIC" ${QA:+--qa "$QA"} --out "$OUT" --manifest-dir "$MANIFESTS"
echo "BUILD SFT DATA DONE $OUT"
