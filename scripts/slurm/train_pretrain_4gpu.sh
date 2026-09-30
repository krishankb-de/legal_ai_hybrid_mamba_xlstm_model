#!/usr/bin/env bash
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # outputs/, HF_HOME and the Inductor cache live on /sc/scratch
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=3-00:00:00
#SBATCH --job-name=train_4gpu
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # aisc-batch is preemptible: without this a requeue TRUNCATES the log
#SBATCH --requeue
#
# train_pretrain_4gpu.sh -- 4 x H100 DDP pretraining on one node (plan P6-A; port map §11.1).
# Submit from the Mac, with --gpus=4 on the sbatch line too (a directive is not overridden by an
# environment variable, and --gres is never used):
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && NUM_GPUS=4 MODEL_CONFIG=hybrid_legal_base SEED=42 EXPERIMENT=p6_hybrid_s42_A sbatch --gpus=4 scripts/slurm/train_pretrain_4gpu.sh"
# One SLURM task; Lightning starts the rank processes itself (scripts/train_pretrain.py builds
# DDPStrategy with LightningEnvironment). The body is train_pretrain_1gpu.sh: `du` of the outputs
# first, the NUM_GPUS check against torch.cuda.device_count(), the ARCH check, and
# CKPT=$OUT/checkpoints/last.ckpt passed as +resume_from_checkpoint= when present, so a requeue
# resumes. Under DDP trainer.max_steps counts global optimizer steps.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

export NUM_GPUS="${NUM_GPUS:-4}"
export TRAINER_CFG="${TRAINER_CFG:-h100_multi_ddp}"
export SAVE_TOP_K="${SAVE_TOP_K:-1}"   # R6: pipeline inputs keep 1 + last.ckpt
bash scripts/slurm/train_pretrain_1gpu.sh
