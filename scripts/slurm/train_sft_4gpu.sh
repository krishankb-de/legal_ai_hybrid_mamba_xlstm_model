#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # outputs/, HF_HOME and the SFT data live on /sc/scratch
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --job-name=train_sft_4gpu
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # pot-hpi-aisc-batch is preemptible: without this a requeue TRUNCATES the log
#SBATCH --requeue
#
# train_sft_4gpu.sh -- supervised fine-tuning on 4 x H100 (DDP): the hybrid (scripts/train_sft.py),
# or with HF_BASELINE=1 the Hugging Face baseline (scripts/train_sft_hf.py, sft.hf_model).
# Submit from your workstation, with --gpus=4 on the sbatch line too (a directive is not overridden by an
# environment variable):
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && INIT_FROM=<stage-B run>/checkpoints/last.ckpt LR=4e-5 SEED=42 sbatch --gpus=4 scripts/slurm/train_sft_4gpu.sh"
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && HF_BASELINE=1 LR=4e-5 SEED=42 sbatch --gpus=4 scripts/slurm/train_sft_4gpu.sh"
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && LORA=1 LR=1e-4 SEED=42 sbatch --gpus=4 scripts/slurm/train_sft_4gpu.sh"
#
# Levers (environment; unset ones leave configs/sft/default.yaml's value):
#   INIT_FROM (the hybrid's pretrained last.ckpt, required without HF_BASELINE; weights only)
#   SFT_DATA (sft.data_dir) LR (sft.learning_rate) SEED MODEL_CONFIG HF_BASELINE(=1) EXPERIMENT NUM_GPUS
#   LORA(=1: the pretrained ceiling, rank-16 adapters on a frozen Qwen3-8B-Base, on the baseline's entry point)
#   MAX_STEPS BATCH_SIZE VAL_EVERY CKPT_EVERY SAVE_TOP_K, and EXTRA_OVERRIDES (Hydra arguments,
#   word-split, applied last).
# A requeued run resumes from $OUT/checkpoints/last.ckpt. One SLURM task; Lightning starts the rank
# processes itself (DDPStrategy with LightningEnvironment, as for pretraining).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# Environment block
SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export SCRATCH_ROOT   # configs/sft/default.yaml places output_dir and data_dir under it
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # training jobs read the fetched cache offline
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${SCRATCH_ROOT}/.torchinductor/sft_${SLURM_JOB_ID:-local}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
mkdir -p logs "$TORCHINDUCTOR_CACHE_DIR"

NUM_GPUS="${NUM_GPUS:-4}"
MODEL_CONFIG="${MODEL_CONFIG:-hybrid_legal_base}"
SEED="${SEED:-42}"
LORA="${LORA:-0}"
HF_BASELINE="${HF_BASELINE:-$LORA}"
SAVE_TOP_K="${SAVE_TOP_K:-1}"   # the best dev-loss checkpoint besides last.ckpt
if [[ "$LORA" == 1 ]]; then
  ENTRY=scripts/train_sft_hf.py
  EXPERIMENT="${EXPERIMENT:-sft_lora_qwen3_8b_s${SEED}${LR:+_lr${LR}}}"
elif [[ "$HF_BASELINE" == 1 ]]; then
  ENTRY=scripts/train_sft_hf.py
  EXPERIMENT="${EXPERIMENT:-sft_qwen3_0p6b_s${SEED}${LR:+_lr${LR}}}"
else
  ENTRY=scripts/train_sft.py
  EXPERIMENT="${EXPERIMENT:-sft_${MODEL_CONFIG}_s${SEED}${LR:+_lr${LR}}}"
fi
OUT="${SCRATCH_ROOT}/outputs/${EXPERIMENT}"
CKPT="${OUT}/checkpoints/last.ckpt"

# how full scratch is, before anything else
du -sh "${SCRATCH_ROOT}/outputs" 2>/dev/null || echo "du: ${SCRATCH_ROOT}/outputs does not exist yet"
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  restart: ${SLURM_RESTART_COUNT:-0}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
echo "experiment: $EXPERIMENT  entry: $ENTRY  model: $MODEL_CONFIG  seed: $SEED  lr: ${LR:-default}  NUM_GPUS: $NUM_GPUS"
echo "output_dir: $OUT"
# A directive is not overridden by an environment variable: the run needs the
# GPUs it is configured for, or it stops here.
.venv/bin/python -c "import torch, sys; n = torch.cuda.device_count(); print('GPUs:', n, [torch.cuda.get_device_name(i) for i in range(n)]); sys.exit(0 if n >= int(sys.argv[1]) else 1)" "$NUM_GPUS" \
  || { echo "FATAL: fewer than NUM_GPUS=$NUM_GPUS CUDA devices visible"; exit 1; }
nvidia-smi

ARGS=(
  "model=${MODEL_CONFIG}" "seed=${SEED}" "trainer=h100_multi_ddp" "trainer.devices=${NUM_GPUS}"
  "callbacks.checkpoint.save_top_k=${SAVE_TOP_K}" "experiment_name=${EXPERIMENT}" "output_dir=${OUT}"
)
opt() {  # hydra_key, value: passed only when the value is set
  if [[ -n "$2" ]]; then ARGS+=("$1=$2"); fi
}
opt sft.data_dir "${SFT_DATA:-}"
opt sft.learning_rate "${LR:-}"
opt sft.batch_size "${BATCH_SIZE:-}"
opt trainer.max_steps "${MAX_STEPS:-}"
opt trainer.val_every_opt_steps "${VAL_EVERY:-}"
opt callbacks.checkpoint.every_n_train_steps "${CKPT_EVERY:-}"
if [[ "$LORA" == 1 ]]; then
  ARGS+=("sft.lora_rank=16" "sft.hf_model=Qwen/Qwen3-8B-Base")
fi
if [[ "$HF_BASELINE" != 1 ]]; then
  [[ -f "${INIT_FROM:-}" ]] || { echo "FATAL: INIT_FROM=${INIT_FROM:-} (the pretrained last.ckpt) does not exist"; exit 1; }
  ARGS+=("lm_checkpoint=${INIT_FROM}")
fi
read -r -a EXTRA <<< "${EXTRA_OVERRIDES:-}"   # one Hydra argument per word, applied last
ARGS+=("${EXTRA[@]+"${EXTRA[@]}"}")
if [[ -f "$CKPT" ]]; then
  echo "resuming from $CKPT"
  ARGS+=("+resume_from_checkpoint=${CKPT}")
fi
echo "command: .venv/bin/python $ENTRY ${ARGS[*]}"
.venv/bin/python "$ENTRY" "${ARGS[@]}"
echo "TRAIN DONE $EXPERIMENT"
