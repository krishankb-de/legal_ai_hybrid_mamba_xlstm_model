#!/usr/bin/env bash
# sync_from_cluster.sh -- pull job logs or analysis files from the cluster (plan P4-D, rule f).
#
#   bash scripts/sync_from_cluster.sh logs       # $CLUSTER_REPO/logs/     -> cluster/logs/
#   bash scripts/sync_from_cluster.sh analysis   # $CLUSTER_REPO/analysis/ -> cluster/analysis/
#
# Only logs/ and analysis/ are allowed: never outputs/ (checkpoints stay on scratch), never
# --delete (a file removed on the cluster stays here as evidence).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -ne 1 ]]; then
  echo "usage: bash scripts/sync_from_cluster.sh logs|analysis" >&2
  exit 2
fi
SUBDIR="$1"
case "$SUBDIR" in
  logs|analysis) ;;
  *) echo "refusing to pull '$SUBDIR': only logs or analysis" >&2; exit 2 ;;
esac

ENV_FILE="${CLUSTER_ENV:-$REPO_ROOT/scripts/slurm/cluster.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE: copy scripts/slurm/cluster.env.example and fill it (plan P4-A)" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE"
: "${CLUSTER_HOST:?CLUSTER_HOST not set in $ENV_FILE}"
: "${CLUSTER_REPO:?CLUSTER_REPO not set in $ENV_FILE}"

mkdir -p "cluster/$SUBDIR"
rsync -az "$CLUSTER_HOST:$CLUSTER_REPO/$SUBDIR/" "cluster/$SUBDIR/"
echo "[sync] $CLUSTER_HOST:$CLUSTER_REPO/$SUBDIR/ -> cluster/$SUBDIR/"
