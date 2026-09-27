#!/usr/bin/env bash
# sync_to_cluster.sh -- push this working tree to the cluster without git (plan P4-D, rule f).
#
#   bash scripts/sync_to_cluster.sh             # write .sync_stamp, mkdir the remote dirs, rsync
#   bash scripts/sync_to_cluster.sh --dry-run   # show what rsync would change; no stamp, no mkdir
#
# Reads CLUSTER_HOST, CLUSTER_REPO, SCRATCH_ROOT from scripts/slurm/cluster.env ($CLUSTER_ENV
# overrides the path). `.sync_stamp` = ISO date + tree hash (lexhybrid.utils.run_metadata.tree_hash),
# the provenance run_metadata.json records in place of a commit id. rsync runs with --delete, and
# .rsync-exclude protects what the cluster owns (.venv, logs/, outputs/, cluster.env).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

ENV_FILE="${CLUSTER_ENV:-$REPO_ROOT/scripts/slurm/cluster.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE: copy scripts/slurm/cluster.env.example and fill it (plan P4-A)" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE"
: "${CLUSTER_HOST:?CLUSTER_HOST not set in $ENV_FILE}"
: "${CLUSTER_REPO:?CLUSTER_REPO not set in $ENV_FILE}"
: "${SCRATCH_ROOT:?SCRATCH_ROOT not set in $ENV_FILE}"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "No venv interpreter at $PYTHON. Build it with: uv sync --locked" >&2
  exit 1
fi

if [[ $DRY_RUN -eq 0 ]]; then
  # Load run_metadata.py by path so the stamp is computed over this tree, not an installed copy.
  TREE_HASH="$("$PYTHON" - "$REPO_ROOT" <<'EOF'
import importlib.util, sys
from pathlib import Path
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("run_metadata", root / "lexhybrid/utils/run_metadata.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.tree_hash(root))
EOF
)"
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TREE_HASH" > .sync_stamp
  echo "[sync] .sync_stamp: $(cat .sync_stamp)"
  # shellcheck disable=SC2029  # expanding on the Mac is intended: the paths come from cluster.env
  ssh "$CLUSTER_HOST" "mkdir -p '$CLUSTER_REPO' '$SCRATCH_ROOT'"
fi

RSYNC_ARGS=(-az --delete "--exclude-from=$REPO_ROOT/.rsync-exclude")
if [[ $DRY_RUN -eq 1 ]]; then RSYNC_ARGS+=(--dry-run -v); fi
rsync "${RSYNC_ARGS[@]}" ./ "$CLUSTER_HOST:$CLUSTER_REPO/"
echo "[sync] $REPO_ROOT -> $CLUSTER_HOST:$CLUSTER_REPO/ (dry_run=$DRY_RUN)"
