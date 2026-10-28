#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # outputs/, HF_HOME and the Inductor cache live on /sc/scratch
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --job-name=diag_spike
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # pot-hpi-aisc-batch is preemptible: without this a requeue TRUNCATES the log
#SBATCH --requeue
#
# diag_spike.sh -- re-run one screen arm with the loss-spike diagnostics on (decisions.p5_divergence,
# 2026-10-11): S1-s42 collapsed after a gradient spike at steps 3,500-3,749 (job 2626202_0).
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/diag_spike.sh"
# The arm's screen settings and seed are unchanged (same data order and schedule: max_steps stays
# 12,000); what differs is observation only: SpikeDiagnosticsCallback (callbacks.diagnostics, every
# DIAG_EVERY steps; per-part gradient and update norms every step), train scalars logged every step,
# and a checkpoint every CKPT_EVERY (250) steps, all kept (save_top_k=-1, no monitor), so a window
# can be replayed. Training stops after STOP_AT (7,000) steps. It keeps screen attempt 1's recipe,
# AdamW beta2 0.999 (the yamls now say 0.95, decisions.p5_fix), so it observes the failure. The output
# goes to its own experiment, diag_<arm>_s<seed>, so nothing resumes from or overwrites a screen run.
# Read it with
#   PREFIX=diag_ DUMP='^diag/' STEPS=<a>-<b> sbatch scripts/slurm/screen_metrics.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

ARM="${ARM:-S1-s42}"
unset EXPERIMENT
ARM_ENV="$(.venv/bin/python scripts/screen_arms.py env "$ARM")" || { echo "FATAL: cannot resolve arm $ARM"; exit 1; }
eval "$ARM_ENV"
EXPERIMENT="diag_${ARM//-/_}"
SAVE_TOP_K=-1
CKPT_EVERY="${CKPT_EVERY:-250}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES} model.adam_beta2=0.999 callbacks.checkpoint.monitor=null trainer.log_every_n_steps=1"
EXTRA_OVERRIDES+=" callbacks.diagnostics.enabled=true callbacks.diagnostics.every_n_steps=${DIAG_EVERY:-10}"
EXTRA_OVERRIDES+=" callbacks.diagnostics.stop_at_step=${STOP_AT:-7000}"
export ARM MODEL_CONFIG SEED EXPERIMENT EXTRA_OVERRIDES ARM_EXPECT ARM_FORBID SAVE_TOP_K CKPT_EVERY
echo "diag arm: $ARM  config: $MODEL_CONFIG  seed: $SEED  experiment: $EXPERIMENT  stop at: ${STOP_AT:-7000}"
bash scripts/slurm/train_pretrain_1gpu.sh
