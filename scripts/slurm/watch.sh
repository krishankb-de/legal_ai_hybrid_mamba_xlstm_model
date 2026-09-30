#!/usr/bin/env bash
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # it lists the checkpoints on /sc/scratch
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --job-name=watch
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# watch.sh -- one status report on every run (plan P4-V; port map §11.7): the queue, the last three
# days of jobs, preemptions and requeues, the checkpoints of every run on scratch, and an error
# count per log. No python anywhere, and every section runs even when one command fails
# (`set -uo pipefail` without -e), so the script is also safe to `source` inside another job.
# Submit from the Mac, then read logs/watch_<id>.log:
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/watch.sh"
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && tail -100 logs/watch_<id>.log"
# (squeue/sacct/tail/grep/ls/du are also allowed on the login node as one-liners; this job is the
# report that runs them all, plus `du`, which only a job may run over scratch.)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}" || return 1 2>/dev/null || exit 1

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
OUTPUTS="${SCRATCH_ROOT}/outputs"
LOG_DIR="${WATCH_LOG_DIR:-logs}"
ERROR_PATTERN='Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError|DUE TO TIME LIMIT|PREEMPT|\bnan\b|Segmentation fault|FATAL'
# A requeued run prints "restart: N" (SLURM_RESTART_COUNT, N >= 1) in its header and, if it picked
# up its checkpoint, "resuming from ..." (train_pretrain_1gpu.sh): fewer resumes than restarts is
# a run that started again from step 0 (FL5).
REQUEUE_PATTERN='restart: [1-9]'
RESUME_PATTERN='^resuming from '

echo "== watch  $(date -u +%Y-%m-%dT%H:%M:%SZ)  host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"

echo; echo "== queue"
squeue --me --format="%.10i %.12P %.9N %.2t %.10M %.20j" 2>&1

echo; echo "== jobs of the last 3 days (steps omitted)"
sacct --starttime=now-3days --user="$USER" \
  --format="JobID%-16,JobName%-18,State%-14,ExitCode%-8,Elapsed%-10,Start%-20,NodeList%-10" 2>&1 \
  | grep -Ev '\.(batch|extern|[0-9]+) '

echo; echo "== preempted, requeued, failed, timed out or out of memory (last 3 days)"
sacct --starttime=now-3days --user="$USER" --format="JobID%-16,JobName%-18,State%-14,ExitCode%-8" -n 2>&1 \
  | grep -Ev '\.(batch|extern|[0-9]+) ' | grep -E 'PREEMPT|REQUEUE|FAIL|TIMEOUT|OUT_OF_ME|NODE_FAIL' \
  || echo "(none)"

echo; echo "== checkpoints per run under $OUTPUTS"
if [[ -d "$OUTPUTS" ]]; then
  for run in "$OUTPUTS"/*/; do
    [[ -d "$run" ]] || continue
    listing=""
    for ckpt in "${run}checkpoints"/*.ckpt; do
      [[ -e "$ckpt" ]] && listing+="$(basename "$ckpt") ($(du -h "$ckpt" 2>/dev/null | cut -f1)) "
    done
    echo "$(basename "$run"): ${listing:-(no checkpoint yet)}"
  done
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo; echo "== scratch usage (R6: <= 40 GB of checkpoints kept)"
    du -sh "$OUTPUTS" "$OUTPUTS"/*/ 2>/dev/null | sort -h | tail -n 15
    df -h "$SCRATCH_ROOT" 2>/dev/null | tail -n 1
  fi
else
  echo "(no outputs directory yet)"
fi

echo; echo "== logs: errors, requeues and resumes (newest 30 logs)"
printf '%-48s %7s %9s %8s  %s\n' "log" "errors" "requeues" "resumes" "last line"
# newest first; log names carry no spaces (logs/%x_%j.log, logs/%x_%A_%a.log)
# shellcheck disable=SC2012
ls -1t "$LOG_DIR"/*.log 2>/dev/null | head -n 30 | while IFS= read -r log; do
  errors="$(grep -cE "$ERROR_PATTERN" "$log" 2>/dev/null)"
  requeues="$(grep -cE "$REQUEUE_PATTERN" "$log" 2>/dev/null)"
  resumes="$(grep -cE "$RESUME_PATTERN" "$log" 2>/dev/null)"
  flag=""
  if [[ "${requeues:-0}" -gt "${resumes:-0}" ]]; then
    flag="  <- requeued ${requeues}x but resumed ${resumes}x: restarted from step 0? (FL5)"
  fi
  printf '%-48s %7s %9s %8s  %s%s\n' "$(basename "$log")" "${errors:-0}" "${requeues:-0}" "${resumes:-0}" \
    "$(tail -n 1 "$log" 2>/dev/null | cut -c1-60)" "$flag"
done
echo; echo "== watch done"
