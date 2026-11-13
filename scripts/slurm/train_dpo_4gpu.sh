#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # outputs/ and HF_HOME live on /sc/scratch
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --job-name=train_dpo
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # pot-hpi-aisc-batch is preemptible: without this a requeue TRUNCATES the log
#SBATCH --requeue
#
# train_dpo_4gpu.sh -- preference optimisation (DPO or KTO) of an SFT checkpoint on the pairs of
# build_dpo_pairs.sh, 4 x H100 DDP on one node (scripts/train_dpo.py).
# Submit from your workstation, with --gpus=4 on the sbatch line too (a directive is not overridden by an
# environment variable):
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && SFT_CKPT=<sft run>/checkpoints/last.ckpt sbatch --gpus=4 scripts/slurm/train_dpo_4gpu.sh"
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && HF_BASELINE=1 SFT_CKPT=<baseline sft run>/checkpoints/last.ckpt LOSS=kto LR=1e-6 sbatch --gpus=4 scripts/slurm/train_dpo_4gpu.sh"
#
# Levers (environment):
#   SFT_CKPT (required: <sft run>/checkpoints/<file>) PAIRS VAL_PAIRS (default the pairs of that SFT run,
#   $SCRATCH_ROOT/data/dpo/<sft run>/{train,dev}.jsonl) LOSS (dpo|kto) LR BETA (always passed: the
#   experiment is named after them; the defaults are configs/dpo/default.yaml's) WARMUP BATCH_SIZE
#   MODEL_CONFIG (the SFT run's) HF_BASELINE(=1: dpo.hf_model) HF_MODEL SEED EXPERIMENT NUM_GPUS TRAINER_CFG
#   MAX_STEPS ACCUM VAL_EVERY CKPT_EVERY SAVE_TOP_K (best val/loss checkpoints besides last.ckpt), and
#   EXTRA_OVERRIDES (Hydra arguments, word-split, applied last).
# EXPERIMENT defaults to dpo_<sft run>_<loss>_lr<LR>_beta<BETA>_s<seed>, so sweep runs and SFT models never
# share a directory. One SLURM task; Lightning starts the rank processes itself. A requeued run resumes from
# $OUT/checkpoints/last.ckpt (train_dpo.py refuses one saved with other settings); the frozen reference is
# rebuilt from SFT_CKPT.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export SCRATCH_ROOT   # configs/config.yaml places output_dir under it
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
mkdir -p logs

: "${SFT_CKPT:?set SFT_CKPT to the SFT checkpoint the policy and the reference start from}"
SFT_RUN="$(basename "$(dirname "$(dirname "$SFT_CKPT")")")"
[[ -f "$SFT_CKPT" && "$(basename "$(dirname "$SFT_CKPT")")" == checkpoints && "$SFT_RUN" != . ]] \
  || { echo "FATAL: SFT_CKPT=$SFT_CKPT is not a file <sft run>/checkpoints/<file>"; exit 1; }
PAIRS="${PAIRS:-${SCRATCH_ROOT}/data/dpo/${SFT_RUN}/train.jsonl}"
VAL_PAIRS="${VAL_PAIRS:-${SCRATCH_ROOT}/data/dpo/${SFT_RUN}/dev.jsonl}"
for f in "$PAIRS" "$VAL_PAIRS"; do
  [[ -f "$f" ]] || { echo "FATAL: $f does not exist (build_dpo_pairs.sh writes it)"; exit 1; }
done
NUM_GPUS="${NUM_GPUS:-4}"
MODEL_CONFIG="${MODEL_CONFIG:-hybrid_legal_base}"
SEED="${SEED:-42}"
TRAINER_CFG="${TRAINER_CFG:-h100_multi_ddp}"
LOSS="${LOSS:-dpo}"
LR="${LR:-5e-7}"
BETA="${BETA:-0.1}"
MAX_STEPS="${MAX_STEPS:-1000}"
VAL_EVERY="${VAL_EVERY:-100}"
CKPT_EVERY="${CKPT_EVERY:-100}"
SAVE_TOP_K="${SAVE_TOP_K:-1}"   # the best val/loss checkpoint besides last.ckpt
EXPERIMENT="${EXPERIMENT:-dpo_${SFT_RUN}_${LOSS}_lr${LR}_beta${BETA}_s${SEED}}"
OUT="${SCRATCH_ROOT}/outputs/${EXPERIMENT}"
CKPT="${OUT}/checkpoints/last.ckpt"

du -sh "${SCRATCH_ROOT}/outputs" 2>/dev/null || echo "du: ${SCRATCH_ROOT}/outputs does not exist yet"
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  restart: ${SLURM_RESTART_COUNT:-0}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
echo "experiment: $EXPERIMENT  sft: $SFT_CKPT  pairs: $PAIRS  val: $VAL_PAIRS  hf: ${HF_BASELINE:-0}  NUM_GPUS: $NUM_GPUS"
echo "output_dir: $OUT"
.venv/bin/python -c "import torch, sys; n = torch.cuda.device_count(); print('GPUs:', n, [torch.cuda.get_device_name(i) for i in range(n)]); sys.exit(0 if n >= int(sys.argv[1]) else 1)" "$NUM_GPUS" \
  || { echo "FATAL: fewer than NUM_GPUS=$NUM_GPUS CUDA devices visible"; exit 1; }
nvidia-smi

ARGS=(
  "model=${MODEL_CONFIG}" "seed=${SEED}" "trainer=${TRAINER_CFG}" "trainer.devices=${NUM_GPUS}"
  "trainer.max_steps=${MAX_STEPS}" "trainer.val_every_opt_steps=${VAL_EVERY}"
  "callbacks.checkpoint.every_n_train_steps=${CKPT_EVERY}" "callbacks.checkpoint.save_top_k=${SAVE_TOP_K}"
  "+dpo=default" "dpo.sft_checkpoint=${SFT_CKPT}" "dpo.pairs=${PAIRS}" "dpo.val_pairs=${VAL_PAIRS}"
  "dpo.loss=${LOSS}" "dpo.learning_rate=${LR}" "dpo.beta=${BETA}"
  "experiment_name=${EXPERIMENT}" "output_dir=${OUT}"
)
if [[ "${HF_BASELINE:-0}" == 1 ]]; then ARGS+=("dpo.hf_model=${HF_MODEL:-Qwen/Qwen3-0.6B-Base}"); fi
opt() {  # hydra_key, value: passed only when the value is set
  if [[ -n "$2" ]]; then ARGS+=("$1=$2"); fi
}
opt dpo.warmup_steps "${WARMUP:-}"
opt dpo.batch_size "${BATCH_SIZE:-}"
opt trainer.accumulate_grad_batches "${ACCUM:-}"
read -r -a EXTRA <<< "${EXTRA_OVERRIDES:-}"   # one Hydra argument per word, applied last
ARGS+=("${EXTRA[@]+"${EXTRA[@]}"}")

if [[ -f "$CKPT" ]]; then
  echo "resuming from $CKPT"
  ARGS+=("+resume_from_checkpoint=${CKPT}")
fi
echo "command: .venv/bin/python scripts/train_dpo.py ${ARGS[*]}"
.venv/bin/python scripts/train_dpo.py "${ARGS[@]}"
echo "DPO DONE $EXPERIMENT"
