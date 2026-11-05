#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # outputs/, data/ and HF_HOME live on /sc/scratch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --job-name=eval_system_ablation
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # pot-hpi-aisc-batch is preemptible: a requeue re-runs into the same log
#SBATCH --requeue
#
# eval_system_ablation.sh -- the system ablation on one run's last.ckpt (scripts/eval_system_ablation.py):
# the same questions through A1 (top of the fused-order selection copied), A2 (top of the reranked selection
# copied), A3 (the shipped pipeline), A4_sel (the fused-order selection, A3 without the reranker),
# A4_k5/k10/k16 (fused top k, no selection limits) and A5_* (gold passages, when GOLD is set), then the
# truncation audit over the records (scripts/audit_truncation.py: exit 1, a rate above 0, is reported and
# the job still succeeds; any other failure of the audit fails the job). A requeue resumes the run in the
# same output directory: records already written are kept, a cut last line is redone.
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && EXPERIMENT=<run> QUESTIONS=<questions>.jsonl sbatch scripts/slurm/eval_system_ablation.sh"
# EXPERIMENT and QUESTIONS are required; MODEL_CONFIG (default hybrid_legal_base; hf=<repository> for a Hugging Face system), INDEX_VERSION (v1),
# GOLD (a gold-passage prompt set), SYSTEM (hybrid), SEED (0), N (4), MAX_NEW_TOKENS (256) are optional.
# Records: logs/eval_system_ablation_<job>/<arm>.jsonl and summary.json; one ABLATION line per arm, the
# TRUNCATION lines, then "EVAL SYSTEM ABLATION DONE".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=1   # tokenizer, bge-m3, reranker and NLI model come from the fetched cache
export PYTHONUNBUFFERED=1
mkdir -p logs

: "${EXPERIMENT:?set EXPERIMENT to the run whose last.ckpt is evaluated}"
: "${QUESTIONS:?set QUESTIONS to a question JSONL}"
CKPT="${SCRATCH_ROOT}/outputs/${EXPERIMENT}/checkpoints/last.ckpt"
INDEX="${SCRATCH_ROOT}/data/index/${INDEX_VERSION:-v1}"
for f in "$CKPT" "$QUESTIONS" "$INDEX/index.json" ${GOLD:+"$GOLD"}; do
  if [[ ! -f "$f" ]]; then echo "FATAL: missing $f"; exit 1; fi
done
OUT="${OUT:-logs/eval_system_ablation_${SLURM_JOB_ID:-local}}"   # set OUT to resume a timed-out job in its directory

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
echo "experiment: $EXPERIMENT  questions: $QUESTIONS  index: $INDEX  gold: ${GOLD:-none}"
MODEL_CONFIG="${MODEL_CONFIG:-hybrid_legal_base}"
if [[ "$MODEL_CONFIG" == hf=* ]]; then MODEL_ARGS=(--hf-model "${MODEL_CONFIG#hf=}"); else MODEL_ARGS=(--model-config "$MODEL_CONFIG"); fi
.venv/bin/python scripts/eval_system_ablation.py --questions "$QUESTIONS" --index "$INDEX" \
  "${MODEL_ARGS[@]}" --checkpoint "$CKPT" --system "${SYSTEM:-hybrid}" \
  --seed "${SEED:-0}" --n "${N:-4}" --max-new-tokens "${MAX_NEW_TOKENS:-256}" ${GOLD:+--gold-prompts "$GOLD"} \
  --out "$OUT" --device cuda
shopt -s nullglob
RECORDS=("$OUT"/*.jsonl)
shopt -u nullglob
status=0
if [[ "${#RECORDS[@]}" -gt 0 ]]; then
  .venv/bin/python scripts/audit_truncation.py "${RECORDS[@]}" || status=$?
fi
if [[ "$status" -eq 1 ]]; then echo "TRUNCATION above 0%: see the TRUNCATION lines"; fi
if [[ "$status" -gt 1 ]]; then echo "FATAL: the truncation audit failed (exit $status)"; exit "$status"; fi
echo "EVAL SYSTEM ABLATION DONE $OUT"
