#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # data/ and HF_HOME live on /sc/scratch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=eval_retrieval
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # pot-hpi-aisc-batch is preemptible: a requeue re-runs into the same log
#SBATCH --requeue
#
# eval_retrieval.sh -- retrieval quality on GerLeRB and GerDaLIR over their own corpora
# (scripts/eval_retrieval.py: bm25, dense, fused, reranked; nDCG@10, R@5/10/20, MRR).
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/eval_retrieval.sh"
# SETS (default "gerlerb gerdalir"). A set missing from $SCRATCH_ROOT/data/evalsets/ is fetched in full
# first (network; qrels.jsonl is written last), its manifest to logs/evalsets_<job>/ for the pull.
# FETCH_ONLY=1 stops after the fetch, which needs no GPU, so it need not wait for one:
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && FETCH_ONLY=1 sbatch --gpus=0 --job-name=fetch_evalsets scripts/slurm/eval_retrieval.sh"
# Results: logs/eval_retrieval_<job>/results.json, one RETRIEVAL line per set and arm, then
# "EVAL RETRIEVAL DONE".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export PYTHONUNBUFFERED=1
mkdir -p logs
JOB="${SLURM_JOB_ID:-local}"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
read -r -a SET_LIST <<< "${SETS:-gerlerb gerdalir}"
DIRS=()
for name in "${SET_LIST[@]}"; do
  dir="${SCRATCH_ROOT}/data/evalsets/${name}"
  if [[ ! -f "$dir/qrels.jsonl" ]]; then
    HF_HUB_OFFLINE=0 .venv/bin/python -m "lexhybrid.data.evalsets.${name}" --limit 0 --out "$dir" \
      --manifest-dir "logs/evalsets_${JOB}"
  fi
  DIRS+=(--set-dir "$dir")
done
if [[ "${FETCH_ONLY:-0}" == 1 ]]; then echo "FETCH EVALSETS DONE"; exit 0; fi
export HF_HUB_OFFLINE=1   # bge-m3 and the reranker come from the fetched cache
.venv/bin/python scripts/eval_retrieval.py "${DIRS[@]}" --out "logs/eval_retrieval_${JOB}/results.json" --device cuda
echo "EVAL RETRIEVAL DONE"
