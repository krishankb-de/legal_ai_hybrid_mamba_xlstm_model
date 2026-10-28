#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # outputs/, HF_HOME and the Inductor cache live on /sc/scratch
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --job-name=screen
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
#SBATCH --open-mode=append   # pot-hpi-aisc-batch is preemptible: without this a requeue TRUNCATES the log
#SBATCH --requeue
#
# screen_array.sh -- the P5 screen, one array task per arm-seed (plan P5-B; port map §11.1, §11.5).
# Submit from the Mac (P5-C, P5-E):
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && ARMS='S1-s42 S1-s43' sbatch --array=0-1 scripts/slurm/screen_array.sh"
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && ARMS='S5-s42 S5-s43' sbatch --array=0-1 --time=24:00:00 scripts/slurm/screen_array.sh"
# ARM is resolved HERE, on the compute node, by `screen_arms.py env` (a pre-submit eval on the login
# node once exported nothing and a 12K arm ran as a 120K default, job 2513581). The screen settings
# and the arm's levers arrive as EXTRA_OVERRIDES; the ARCH line is checked against the arm's
# expected and forbidden tokens before step 0 (train_pretrain_1gpu.sh), so a lever that does not
# reach the model stops the job (FL1). The teacher is decisions.teacher, Qwen3-8B-Base (P4-U), with
# gradient checkpointing (train_pretrain_1gpu.sh's default for the 8B).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

ARMS="${ARMS:-S0-s42 S2-s42 S3-s42 S4-s42 S6-s42}"
read -r -a ARM_LIST <<< "${ARMS}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${IDX}" -ge "${#ARM_LIST[@]}" ]]; then
  echo "FATAL: array index ${IDX} but only ${#ARM_LIST[@]} arms in ARMS='${ARMS}'"; exit 1
fi
ARM="${ARM_LIST[$IDX]}"
unset EXPERIMENT   # a stray EXPERIMENT in the submitting shell would funnel every arm into one output dir

ARM_ENV="$(.venv/bin/python scripts/screen_arms.py env "$ARM")" || { echo "FATAL: cannot resolve arm $ARM"; exit 1; }
eval "$ARM_ENV"
export ARM MODEL_CONFIG SEED EXPERIMENT EXTRA_OVERRIDES ARM_EXPECT ARM_FORBID
export SAVE_TOP_K="${SAVE_TOP_K_SCREEN:-0}"   # R6: no top-k checkpoints on screen arms, last.ckpt only
echo "arm: $ARM  (task ${IDX} of '${ARMS}')  config: $MODEL_CONFIG  seed: $SEED  walltime wanted: ${ARM_WALLTIME:-?}"
bash scripts/slurm/train_pretrain_1gpu.sh
