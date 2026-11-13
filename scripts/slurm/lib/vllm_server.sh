#!/usr/bin/env bash
# vllm_server.sh -- start, wait for and stop a vLLM OpenAI-compatible server from .venv-vllm.
# Sourced by the jobs that serve a model (vllm_smoke.sh, synth_qa.sh); not a job itself. Set
# VLLM_LOG first. The server listens on 127.0.0.1:$VLLM_PORT only: one port per job, from its job id
# (10000-29999, below Linux's ephemeral range), because several jobs can share a node.
#   vllm_start <model> [vllm serve options...]   # in the background; an EXIT trap stops it again;
#                                                 # fails if anything already answers on the port
#   vllm_wait                                     # until /health answers while the server we started
#                                                 # lives; fails if it exits or $VLLM_TIMEOUT seconds
#                                                 # pass (loading 32B takes minutes)
VLLM_PORT="${VLLM_PORT:-$(( 10000 + ${SLURM_JOB_ID:-0} % 20000 ))}"
VLLM_TIMEOUT="${VLLM_TIMEOUT:-1800}"
VLLM_URL="http://127.0.0.1:${VLLM_PORT}"

vllm_stop() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    echo "vllm stopped"
  fi
}

vllm_start() {
  [[ -x .venv-vllm/bin/vllm ]] || { echo "FATAL: no .venv-vllm: sbatch scripts/slurm/setup_vllm.sh"; exit 1; }
  # any HTTP answer counts: ours would die on bind and its /health would be another job's
  if curl -s -o /dev/null --max-time 5 "${VLLM_URL}/health"; then
    echo "FATAL: port ${VLLM_PORT} already answers: another server on this node"; exit 1
  fi
  # vLLM's compile cache outlives the job and points at inductor files by absolute path. Inductor's
  # default dir is the job's node-local $TMPDIR (/scratch/<job>), gone and not writable in the next
  # job, so both live on $SCRATCH_ROOT.
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${SCRATCH_ROOT}/.cache/vllm}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${SCRATCH_ROOT}/.torchinductor/vllm}"
  echo "vllm serve $* on ${VLLM_URL} (log: ${VLLM_LOG})"
  .venv-vllm/bin/vllm serve "$@" --host 127.0.0.1 --port "$VLLM_PORT" > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  trap vllm_stop EXIT
}

vllm_wait() {
  local waited=0
  # /health is ours only while our process lives: checked before and after asking
  while kill -0 "$VLLM_PID" 2>/dev/null; do
    if curl -sf --max-time 5 "${VLLM_URL}/health" > /dev/null && kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vllm healthy after ${waited}s"
      return 0
    fi
    if (( waited >= VLLM_TIMEOUT )); then
      echo "FATAL: no /health after ${VLLM_TIMEOUT}s"; tail -n 40 "$VLLM_LOG"; exit 1
    fi
    sleep 1
    waited=$(( waited + 1 ))
  done
  echo "FATAL: vllm exited before /health answered"; tail -n 40 "$VLLM_LOG"; exit 1
}
