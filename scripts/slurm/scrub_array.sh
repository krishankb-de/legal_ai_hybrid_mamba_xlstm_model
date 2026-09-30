#!/usr/bin/env bash
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --constraint=GLB_SCRATCH   # data/raw, data/scrubbed and HF_HOME live on /sc/scratch
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --job-name=scrub
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
#SBATCH --open-mode=append   # aisc-batch is preemptible: a requeued task appends to its log
#SBATCH --requeue
#
# scrub_array.sh -- the LER scrub at corpus scale (plan P4-L, first step; R11): flair/ner-german-legal
# on one H100 per task. Task i scrubs part i % PARTS of source SOURCES[i / PARTS] (stride shares:
# lines k, k+PARTS, ...), so the array size is (number of sources) x PARTS. Submit from the Mac:
#   ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch --array=0-79 scripts/slurm/scrub_array.sh"   # 10 sources x 8 parts
# (measured on the Mac CPU: ~1,000 characters/s, P3-R; the GPU is what makes the corpus feasible).
# Writes $SCRATCH_ROOT/data/scrubbed/<src>.part-<k>-of-<PARTS>.jsonl and the scrub log
# $SCRATCH_ROOT/data/manifests/scrub_<src>.part-<k>-of-<PARTS>.jsonl, renamed from *.partial when
# the part completes; a part already written is skipped, so resubmitting the array redoes only what
# is missing. Every part hashes names with ONE key ($SCRATCH_ROOT/.secrets/scrub_hmac.key, created
# atomically by the first task; never printed). Needs the scrub environment (setup_env.sh builds it)
# and flair's weights in $HF_HOME (fetch_hf.sh). The log ends in "SCRUB <src> PART <k>/<n> DONE".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_HUB_OFFLINE=1   # the tagger's weights come from the fetched cache
export PYTHONUNBUFFERED=1

SOURCES="${SOURCES:-gii rii oldp eurlex ris fedlex bger dip fineweb2_de multilegalpile}"
PARTS="${PARTS:-8}"
read -r -a SOURCE_LIST <<< "${SOURCES}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
N_TASKS=$(( ${#SOURCE_LIST[@]} * PARTS ))
if [[ "${IDX}" -ge "${N_TASKS}" ]]; then
  echo "FATAL: array index ${IDX} but only ${N_TASKS} tasks (${#SOURCE_LIST[@]} sources x ${PARTS} parts)"; exit 1
fi
SRC="${SOURCE_LIST[$(( IDX / PARTS ))]}"
PART=$(( IDX % PARTS ))
RAW="${SCRATCH_ROOT}/data/raw/${SRC}/${SRC}.jsonl"
NAME="${SRC}.part-${PART}-of-${PARTS}.jsonl"
OUT="${SCRATCH_ROOT}/data/scrubbed/${NAME}"
LOG="${SCRATCH_ROOT}/data/manifests/scrub_${NAME}"
mkdir -p logs "$(dirname "$OUT")" "$(dirname "$LOG")"

echo "host: $(hostname)  job: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}}_${IDX}  restart: ${SLURM_RESTART_COUNT:-0}"
echo "sync_stamp: $(cat .sync_stamp 2>/dev/null || echo missing)"
echo "source: $SRC  part: $PART of $PARTS  in: $RAW  out: $OUT"
if [[ -f "$OUT" && -f "$LOG" ]]; then
  echo "SCRUB ${SRC} PART ${PART}/${PARTS} DONE (already scrubbed: $OUT)"; exit 0
fi
[[ -f "$RAW" ]] || { echo "FATAL: $RAW is missing (collect it first: build_corpus_array.sh, P4-K)"; exit 1; }
[[ -x envs/scrub/.venv/bin/python ]] || { echo "FATAL: no scrub environment: sbatch scripts/slurm/setup_env.sh"; exit 1; }

# One HMAC key for every part and every source: the first task links a complete key file into
# place (ln fails if it exists, so a racing task reads the winner's key instead).
KEY_FILE="${SCRATCH_ROOT}/.secrets/scrub_hmac.key"
if [[ ! -s "$KEY_FILE" ]]; then
  mkdir -p "$(dirname "$KEY_FILE")"
  chmod 700 "$(dirname "$KEY_FILE")"
  TMP_KEY="${KEY_FILE}.${SLURM_JOB_ID:-$$}.${IDX}"
  (umask 077; od -An -tx1 -N32 /dev/urandom | tr -d ' \n' > "$TMP_KEY")
  ln "$TMP_KEY" "$KEY_FILE" 2>/dev/null || true
  rm -f "$TMP_KEY"
fi
LEXHYBRID_SCRUB_KEY="$(cat "$KEY_FILE")"
export LEXHYBRID_SCRUB_KEY
[[ ${#LEXHYBRID_SCRUB_KEY} -eq 64 ]] || { echo "FATAL: the scrub key at $KEY_FILE is not 32 hex bytes"; exit 1; }

envs/scrub/.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable in the scrub env'; print('GPU:', torch.cuda.get_device_name(0))"
.venv/bin/python -m lexhybrid.data.corpus.scrub_ler --source "$SRC" --in "$RAW" --out "$OUT" --log "$LOG" \
  --part "$PART" --parts "$PARTS" --docs-per-batch "${DOCS_PER_BATCH:-64}" --batch-size "${NER_BATCH:-128}" \
  --cache-dir "${HF_HOME}/hub"
echo "scrubbed: $(wc -l < "$OUT" | tr -d ' ') documents, $(wc -l < "$LOG" | tr -d ' ') entities logged"
echo "SCRUB ${SRC} PART ${PART}/${PARTS} DONE"
