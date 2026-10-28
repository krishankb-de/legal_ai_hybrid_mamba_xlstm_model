#!/usr/bin/env bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node (x86 .venv -> "Exec format error"); gx13v1: faulty GPU
#SBATCH --constraint=GLB_SCRATCH   # the paths live on /sc/scratch
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --time=0-02:00:00
#SBATCH --job-name=cleanup
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# cleanup.sh -- delete what the user approved, by an explicit list. The login node runs no rm, so
# deletion goes through a job (rule e).
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && DELETE='outputs/screen_S0_s42 .torchinductor repo:envs/scrub/.venv' sbatch scripts/slurm/cleanup.sh"
# DELETE holds paths relative to $SCRATCH_ROOT, or relative to the repository when they carry a
# "repo:" prefix. The whole list is checked before anything is deleted, and any refusal stops the job
# with nothing removed. Refused: an absolute path, "..", and anything outside the deletable roots below.
#   - scratch: outputs/screen_* outputs/screen2_* outputs/diag_* .torchinductor .uv-cache data/raw data/http_cache
#   - repo:    envs/scrub/.venv
# Never deletable: P6's outputs (outputs/p6_*), the training shards (data/shards), the scrubbed
# corpus (data/scrubbed), data/manifests, the HF cache (.hf) and the project .venv. A listed path
# that is already gone is reported and skipped. The log ends in "CLEANUP DONE".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
REPO="$PWD"
read -r -a LIST <<< "${DELETE:-}"
if [[ "${#LIST[@]}" -eq 0 ]]; then echo "FATAL: DELETE is empty"; exit 1; fi

resolve() {  # entry -> absolute path, or a refusal on stderr and exit status 1
  local entry="$1" base rel
  if [[ "$entry" == repo:* ]]; then base="$REPO"; rel="${entry#repo:}"; else base="$SCRATCH_ROOT"; rel="$entry"; fi
  rel="${rel%/}"
  if [[ -z "$rel" || "$rel" == /* || "$rel" == *..* ]]; then echo "refused (absolute or ..): $entry" >&2; return 1; fi
  if [[ "$base" == "$REPO" ]]; then
    case "$rel" in envs/scrub/.venv) ;; *) echo "refused (not a deletable repo path): $entry" >&2; return 1 ;; esac
  else
    case "$rel" in
      outputs/screen_?*|outputs/screen2_?*|outputs/diag_?*|.torchinductor|.torchinductor/*|.uv-cache|data/raw|data/raw/*|data/http_cache|data/http_cache/*) ;;
      *) echo "refused (not a deletable scratch path): $entry" >&2; return 1 ;;
    esac
  fi
  echo "$base/$rel"
}

TARGETS=()
for entry in "${LIST[@]}"; do
  target="$(resolve "$entry")" || { echo "FATAL: nothing deleted"; exit 1; }
  TARGETS+=("$target")
done

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
df -h "$SCRATCH_ROOT" "$HOME" 2>/dev/null || true
for target in "${TARGETS[@]}"; do
  if [[ ! -e "$target" ]]; then echo "already gone: $target"; continue; fi
  echo "delete: $target ($(du -sh "$target" 2>/dev/null | cut -f1))"
  rm -rf -- "$target"
done
df -h "$SCRATCH_ROOT" "$HOME" 2>/dev/null || true
echo "CLEANUP DONE"
