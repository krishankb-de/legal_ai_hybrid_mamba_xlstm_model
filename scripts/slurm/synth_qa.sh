#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # the index, the outputs and HF_HOME live on /sc/scratch
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --job-name=synth_qa
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # pot-hpi-aisc-batch is preemptible: a requeue resumes into the same log
#SBATCH --requeue
#
# synth_qa.sh -- synthetic retrieval-grounded QA for SFT: Qwen/Qwen3-32B, served by vLLM from
# .venv-vllm, writes questions and pointer answers over the index's commercial-safe passages, and
# scripts/generate_synthetic_qa.py (in .venv) keeps the ones the verifier passes. Submit from your workstation:
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch --gpus=2 scripts/slurm/synth_qa.sh"
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && COUNT=50000 SEED=1 INDEX_VERSION=v2 sbatch --gpus=2 scripts/slurm/synth_qa.sh"
# Two H100s, tensor parallel 2: the bf16 weights are ~65 GB, which leaves one 80 GB card ~8 GB of KV
# cache (a handful of 5k-token prompts in flight) beside the NLI model and bge-m3. Split over two
# cards they take ~33 GB each, and the KV cache holds dozens of prompts at once. vLLM takes 85% of
# each card; the NLI model and bge-m3 (~2 GB) run in the rest of GPU 0.
# Needs .venv-vllm (setup_vllm.sh), the index (build_dense_index.sh), and Qwen/Qwen3-32B, bge-m3 and the
# NLI model in $HF_HOME. Writes $SCRATCH_ROOT/data/synthetic_qa/<INDEX_VERSION>_s<SEED>/
# (synthetic_qa.jsonl, stats.json, attempts.jsonl: a requeue resumes after the last attempt); the
# server log is logs/synth_qa_<job>.vllm.log. The log ends in "SYNTH QA DONE".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=1   # the teacher, the NLI model and bge-m3 come from the fetched cache
export PYTHONUNBUFFERED=1
mkdir -p logs

NUM_GPUS="${NUM_GPUS:-2}"
MODEL=Qwen/Qwen3-32B
SEED="${SEED:-0}"
INDEX="${SCRATCH_ROOT}/data/index/${INDEX_VERSION:-v1}"
OUT="${SCRATCH_ROOT}/data/synthetic_qa/${INDEX_VERSION:-v1}_s${SEED}"
VLLM_LOG="logs/synth_qa_${SLURM_JOB_ID:-local}.vllm.log"
# shellcheck source=scripts/slurm/lib/vllm_server.sh
source scripts/slurm/lib/vllm_server.sh

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  restart: ${SLURM_RESTART_COUNT:-0}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
echo "index: $INDEX  out: $OUT  count: ${COUNT:-20000}  seed: $SEED"
[[ -f "$INDEX/index.json" ]] || { echo "FATAL: no index at $INDEX (build_dense_index.sh)"; exit 1; }
.venv/bin/python -c "import sys, torch; n = torch.cuda.device_count(); print('GPUs:', n); sys.exit(n < int(sys.argv[1]))" \
  "$NUM_GPUS" || { echo "FATAL: fewer than ${NUM_GPUS} GPUs"; exit 1; }
vllm_start "$MODEL" --tensor-parallel-size "$NUM_GPUS" --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.85
vllm_wait
.venv/bin/python scripts/generate_synthetic_qa.py --index "$INDEX" --endpoint "${VLLM_URL}/v1" --model "$MODEL" \
  --count "${COUNT:-20000}" --seed "$SEED" --out "$OUT"
echo "SYNTH QA DONE $OUT"
