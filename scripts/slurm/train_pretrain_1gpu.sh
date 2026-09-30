#!/usr/bin/env bash
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # outputs/, HF_HOME and the Inductor cache live on /sc/scratch
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --job-name=train_1gpu
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # aisc-batch is preemptible: without this a requeue TRUNCATES the log
#SBATCH --requeue
#
# train_pretrain_1gpu.sh -- one run of scripts/train_pretrain.py (plan P5-B; port map §11.1, §11.2,
# §11.4). The screen array (screen_array.sh), the 4-GPU wrapper (train_pretrain_4gpu.sh) and the
# probes delegate to this body with `bash`; their own #SBATCH header is the one SLURM reads.
# Submit directly from the Mac:
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && MODEL_CONFIG=hybrid_legal_base SEED=42 sbatch scripts/slurm/train_pretrain_1gpu.sh"
#
# Levers (environment; unset ones leave the yaml's value):
#   MODEL_CONFIG SEED DATASET_CONFIG TRAINER_CFG DISTILL_CFG(=none: CE only) EXPERIMENT NUM_GPUS
#   SAVE_TOP_K CKPT_EVERY MAX_STEPS ACCUM VAL_EVERY BATCH_SIZE ROW_LEN COMPILE LR WARMUP GRAD_CLIP
#   GRAD_CKPT, and EXTRA_OVERRIDES (Hydra arguments, word-split, applied last: the last value wins).
#   ARM_EXPECT / ARM_FORBID ("|"-separated): tokens the ARCH line must / must not show (FL1).
# A requeued run resumes from $OUT/checkpoints/last.ckpt (defect 12: the reference restarted at 0).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# Environment block, port map §11.2
SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export SCRATCH_ROOT   # configs/config.yaml places output_dir under it
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # training jobs read the fetched cache offline
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${SCRATCH_ROOT}/.torchinductor/train_${SLURM_JOB_ID:-local}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
if [[ -f "$HOME/.hf_token" ]]; then
  HF_TOKEN="$(cat "$HOME/.hf_token")"
  export HF_TOKEN
fi
mkdir -p logs "$TORCHINDUCTOR_CACHE_DIR"

NUM_GPUS="${NUM_GPUS:-1}"
MODEL_CONFIG="${MODEL_CONFIG:-hybrid_legal_base}"
SEED="${SEED:-42}"
DATASET_CONFIG="${DATASET_CONFIG:-mixture_pretrain}"
TRAINER_CFG="${TRAINER_CFG:-h100_single_gpu}"
DISTILL_CFG="${DISTILL_CFG:-qwen3_1p7b}"   # decision 8's default; the run follows decisions.teacher
SAVE_TOP_K="${SAVE_TOP_K:-0}"             # R6: 0 on arms; pipeline runs pass 1 (+ last.ckpt)
CKPT_EVERY="${CKPT_EVERY:-1000}"
EXPERIMENT="${EXPERIMENT:-pretrain_${MODEL_CONFIG}_s${SEED}}"
OUT="${SCRATCH_ROOT}/outputs/${EXPERIMENT}"
CKPT="${OUT}/checkpoints/last.ckpt"

# R6: how full scratch is, before anything else
du -sh "${SCRATCH_ROOT}/outputs" 2>/dev/null || echo "du: ${SCRATCH_ROOT}/outputs does not exist yet"
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  array: ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}  restart: ${SLURM_RESTART_COUNT:-0}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
echo "experiment: $EXPERIMENT  model: $MODEL_CONFIG  seed: $SEED  trainer: $TRAINER_CFG  distill: $DISTILL_CFG  NUM_GPUS: $NUM_GPUS"
echo "output_dir: $OUT"
# A directive is not overridden by an environment variable (port map §11.1): the run needs the
# GPUs it is configured for, or it stops here.
.venv/bin/python -c "import torch, sys; n = torch.cuda.device_count(); print('GPUs:', n, [torch.cuda.get_device_name(i) for i in range(n)]); sys.exit(0 if n >= int(sys.argv[1]) else 1)" "$NUM_GPUS" \
  || { echo "FATAL: fewer than NUM_GPUS=$NUM_GPUS CUDA devices visible"; exit 1; }
nvidia-smi

ARGS=(
  "model=${MODEL_CONFIG}" "seed=${SEED}" "dataset=${DATASET_CONFIG}" "trainer=${TRAINER_CFG}"
  "trainer.devices=${NUM_GPUS}"
  "callbacks.checkpoint.every_n_train_steps=${CKPT_EVERY}" "callbacks.checkpoint.save_top_k=${SAVE_TOP_K}"
  "experiment_name=${EXPERIMENT}" "output_dir=${OUT}"
)
if [[ "$DISTILL_CFG" == none ]]; then ARGS+=("distill=null"); else ARGS+=("distill=${DISTILL_CFG}"); fi
opt() {  # hydra_key, value: passed only when the value is set
  if [[ -n "$2" ]]; then ARGS+=("$1=$2"); fi
}
opt trainer.max_steps "${MAX_STEPS:-}"
opt trainer.accumulate_grad_batches "${ACCUM:-}"
opt trainer.val_every_opt_steps "${VAL_EVERY:-}"
opt trainer.compile_model "${COMPILE:-}"
opt dataset.batch_size "${BATCH_SIZE:-}"
opt dataset.row_len "${ROW_LEN:-}"
opt model.learning_rate "${LR:-}"
opt model.warmup_steps "${WARMUP:-}"
opt model.gradient_clip_val "${GRAD_CLIP:-}"
opt model.use_gradient_checkpointing "${GRAD_CKPT:-}"
read -r -a EXTRA <<< "${EXTRA_OVERRIDES:-}"   # one Hydra argument per word, applied last
ARGS+=("${EXTRA[@]+"${EXTRA[@]}"}")

# FL1: the ARCH line of the model these exact overrides build, checked before step 0.
ARCH_LINE="$(.venv/bin/python scripts/train_pretrain.py "${ARGS[@]}" +arch_only=true | grep '^ARCH ' | tail -n 1)" \
  || { echo "FATAL: no ARCH line from train_pretrain.py +arch_only=true"; exit 1; }
echo "$ARCH_LINE"
IFS='|' read -r -a EXPECT <<< "${ARM_EXPECT:-}"
for tok in "${EXPECT[@]+"${EXPECT[@]}"}"; do
  if ! grep -qF -- "$tok" <<< "$ARCH_LINE"; then
    echo "FATAL: ARCH line lacks '$tok' (a lever did not reach the model; FL1)"; exit 1
  fi
done
IFS='|' read -r -a FORBID <<< "${ARM_FORBID:-}"
for tok in "${FORBID[@]+"${FORBID[@]}"}"; do
  if grep -qF -- "$tok" <<< "$ARCH_LINE"; then
    echo "FATAL: ARCH line shows forbidden '$tok' (FL1)"; exit 1
  fi
done
echo "ARCH CHECK OK (${#EXPECT[@]} expected, ${#FORBID[@]} forbidden tokens)"

if [[ -f "$CKPT" ]]; then
  echo "resuming from $CKPT"
  ARGS+=("+resume_from_checkpoint=${CKPT}")
fi
echo "command: .venv/bin/python scripts/train_pretrain.py ${ARGS[*]}"
.venv/bin/python scripts/train_pretrain.py "${ARGS[@]}"
echo "TRAIN DONE $EXPERIMENT"
