#!/usr/bin/env bash
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --exclude=ga03   # ARM/Grace node; x86 .venv python -> "cannot execute binary file: Exec format error"
#SBATCH --constraint=GLB_SCRATCH   # HF_HOME lives on /sc/scratch, mounted only on these nodes
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=fetch_hf
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # aisc-batch is preemptible; hf download resumes after a requeue
#SBATCH --requeue
#
# fetch_hf.sh -- download every Hugging Face model the plan needs into $HF_HOME (plan P4-G).
# Submit from the Mac:  ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/fetch_hf.sh"
# The only job with HF_HUB_OFFLINE=0; training and eval jobs read the same cache offline.
# Pinned revisions come from the code that loads them (tokenizer.py, scrub_ner_worker.py);
# the others resolve main, and the snapshot listing at the end records the commit fetched.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=0
export PYTHONUNBUFFERED=1
if [[ -f "$HOME/.hf_token" ]]; then
  HF_TOKEN="$(cat "$HOME/.hf_token")"
  export HF_TOKEN
fi
mkdir -p logs "$HF_HOME"

TOKEN_STATE=no
if [[ -n "${HF_TOKEN:-}" ]]; then TOKEN_STATE=yes; fi
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  HF_HOME: $HF_HOME  token: $TOKEN_STATE"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"

QWEN_REV="$(.venv/bin/python -c 'from lexhybrid.data.tokenizer import QWEN3_REVISION as r; print(r)')"
FLAIR_REV="$(sed -n 's/^REVISION = "\([0-9a-f]*\)".*/\1/p' scripts/scrub_ner_worker.py)"
[[ -n "$QWEN_REV" && -n "$FLAIR_REV" ]] || { echo "FATAL: pinned revision not found"; exit 1; }

# repo|revision (decision 14 names the NLI model; decisions 6/8 the Qwen3 tokenizer and teachers)
MODELS=(
  "Qwen/Qwen3-1.7B-Base|$QWEN_REV"
  "Qwen/Qwen3-8B-Base|main"
  "Qwen/Qwen3-0.6B-Base|main"
  "flair/ner-german-legal|$FLAIR_REV"
  "BAAI/bge-m3|main"
  "BAAI/bge-reranker-v2-m3|main"
  "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7|main"
)

FAILED=()
for spec in "${MODELS[@]}"; do
  repo="${spec%%|*}"
  rev="${spec##*|}"
  echo "== $repo @ $rev"
  if ! .venv/bin/hf download "$repo" --revision "$rev" --exclude "onnx/*" --exclude "*.onnx" --exclude "*.onnx_data"; then
    FAILED+=("$repo")
  fi
done

echo "== snapshots"
for spec in "${MODELS[@]}"; do
  repo="${spec%%|*}"
  dir="$HF_HOME/hub/models--${repo//\//--}"
  snaps=""
  for s in "$dir"/snapshots/*; do [[ -e "$s" ]] && snaps+="${s##*/} "; done
  echo "$repo  snapshot: ${snaps:-MISSING}  size: $(du -shL "$dir/snapshots" 2>/dev/null | cut -f1)"  # weights live in hub/blobs; follow the links
done
du -sh "$HF_HOME/hub"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FETCH_HF FAILED: ${FAILED[*]}"
  exit 1
fi
echo "FETCH_HF DONE"
