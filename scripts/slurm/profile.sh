#!/usr/bin/env bash
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # the Inductor caches live on /sc/scratch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --job-name=profile
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# profile.sh -- the P4 profiler ladder on one H100 (plan P4-P; verdict P4-Q), port map §11.8:
# one sequence length per PROCESS, a fresh TORCHINDUCTOR_CACHE_DIR per point, the Transformer rows
# first, effective chunk sizes printed and recorded in every row.
# Submit from the Mac:  ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/profile.sh"
#
# Points (bf16, batch 4, random weights):
#   sanity   ref_transformer at 16,384; ref_hybrid_m3 at 16,384 uncompiled chunk 64 and compiled
#            chunk 128 (the reference's 163.9 / 678.2 / 122.9 ms, 7.1 GB -- prediction R4)
#   infer    transformer_legal_base then hybrid_legal_base at 2,048 / 8,192 / 16,384
#   train    both at 2,048 / 4,096 / 8,192, forward+backward with the slab loss, unpacked and packed
#            (a document every DOC_LEN tokens: flex attention on the GPU)
#   chunk    hybrid_legal_base training at 4,096, packed, chunk 64 (the P4-Q rule compares it with
#            the chunk-128 row above: "128 unless it fails R1 or loses > 5% at 4,096")
# CSV/JSON per point go to logs/profile_<jobid>/ (cluster-owned, never touched by the rsync push;
# pulled with `bash scripts/sync_from_cluster.sh logs`). A point that fails is recorded and the
# ladder continues; the log ends in "PROFILE DONE" or lists the failed points.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# Environment block, port map §11.2
SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bf16}"
ITERS="${ITERS:-10}"
DOC_LEN="${DOC_LEN:-1000}"   # packed rows: not a multiple of the 128-token block, like real documents
RUN_ID="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="logs/profile_${RUN_ID}"
INDUCTOR_ROOT="${SCRATCH_ROOT}/.torchinductor/profile_${RUN_ID}"
mkdir -p logs "$OUTPUT_DIR" "$INDUCTOR_ROOT"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  batch: $BATCH_SIZE  dtype: $DTYPE  iters: $ITERS  doc_len: $DOC_LEN"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0), f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')"
nvidia-smi

FAILED=()
point() {  # name, model, seq_len, extra performance_profile.py flags...
  local name="$1" mdl="$2" len="$3"; shift 3
  export TORCHINDUCTOR_CACHE_DIR="${INDUCTOR_ROOT}/${name}_L${len}"
  rm -rf "${TORCHINDUCTOR_CACHE_DIR}"; mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
  echo; echo "== point ${name} L=${len} ($*)"
  if .venv/bin/python scripts/performance_profile.py --sweep --models "${mdl}" --seq-lengths "${len}" \
      --batch_size "${BATCH_SIZE}" --num_iterations "${ITERS}" --dtype "${DTYPE}" \
      --output-dir "${OUTPUT_DIR}/${name}_L${len}" "$@"; then
    echo "[DONE] ${name} L=${len}"
  else
    echo "[FAIL] ${name} L=${len}"; FAILED+=("${name}_L${len}")
  fi
}

# -- sanity against the reference (the Transformer first, port map §11.8)
point sanity_ref_transformer ref_transformer 16384
point sanity_ref_m3_c64 ref_hybrid_m3 16384 --chunk-size 64
point sanity_ref_m3_c128_compiled ref_hybrid_m3 16384 --chunk-size 128 --compile

# -- inference ladder of the legal models
for len in 2048 8192 16384; do point infer_transformer_legal transformer_legal_base "$len"; done
for len in 2048 8192 16384; do point infer_hybrid_legal hybrid_legal_base "$len"; done

# -- training step (slab loss), unpacked then packed
for len in 2048 4096 8192; do
  point train_transformer_legal transformer_legal_base "$len" --backward --loss slab
  point train_transformer_legal_packed transformer_legal_base "$len" --backward --loss slab --doc-len "$DOC_LEN"
done
for len in 2048 4096 8192; do
  point train_hybrid_legal hybrid_legal_base "$len" --backward --loss slab
  point train_hybrid_legal_packed hybrid_legal_base "$len" --backward --loss slab --doc-len "$DOC_LEN"
done

# -- chunk-size rule at the training length
point chunk64_hybrid_legal_packed hybrid_legal_base 4096 --backward --loss slab --doc-len "$DOC_LEN" \
  --chunk-size 64 --mlstm-chunk-size 64

echo
echo "== rows written"
find "$OUTPUT_DIR" -name efficiency_curves.csv | sort
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "PROFILE FAILED POINTS: ${FAILED[*]}"
  exit 1
fi
echo "PROFILE DONE"
