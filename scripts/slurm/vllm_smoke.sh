#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # HF_HOME lives on /sc/scratch
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=vllm_smoke
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# vllm_smoke.sh -- the vLLM environment end to end on one GPU: serve Qwen/Qwen3-0.6B-Base from
# .venv-vllm (setup_vllm.sh builds it; fetch_hf.sh fetches the model), wait for /health, ask for
# one completion, stop the server. Submit from your workstation:
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/vllm_smoke.sh"
# The server log is logs/vllm_smoke_<job>.vllm.log; the job log ends in "VLLM SMOKE OK".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=1   # the model comes from the fetched cache
mkdir -p logs
MODEL=Qwen/Qwen3-0.6B-Base
VLLM_LOG="logs/vllm_smoke_${SLURM_JOB_ID:-local}.vllm.log"
# shellcheck source=scripts/slurm/lib/vllm_server.sh
source scripts/slurm/lib/vllm_server.sh

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
vllm_start "$MODEL" --max-model-len 2048
vllm_wait
REPLY="$(curl -sf "${VLLM_URL}/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\": \"${MODEL}\", \"prompt\": \"Das Bürgerliche Gesetzbuch regelt\", \"max_tokens\": 16, \"temperature\": 0}")" \
  || { echo "VLLM SMOKE FAILED: no completion"; exit 1; }
echo "completion: $REPLY"
.venv-vllm/bin/python -c 'import json, sys; assert json.loads(sys.argv[1])["choices"][0]["text"].strip()' "$REPLY" \
  || { echo "VLLM SMOKE FAILED: empty completion"; exit 1; }
echo "VLLM SMOKE OK"
