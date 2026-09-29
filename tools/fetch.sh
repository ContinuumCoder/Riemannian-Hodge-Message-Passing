#!/usr/bin/env bash
# Fetch the result files written on the remote host ($HOST:~/$REMOTE_DIR -> local checkout): bench/RESULTS_*.md,
# bench/*.json and runs/ (json / md / log / png / csv files only; no checkpoints).  Settings: tools/ssh_opts.sh.
#   HOST=mygpu tools/fetch.sh
set -euo pipefail
source "$(dirname "$0")/ssh_opts.sh"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
rsync -az -e "ssh $SSH_OPTS" --include 'RESULTS_*.md' --include '*.json' --exclude '*' "$HOST:~/$REMOTE_DIR/bench/" "$LOCAL/bench/"
mkdir -p "$LOCAL/runs"
rsync -az -e "ssh $SSH_OPTS" --prune-empty-dirs --include '*/' --include '*.json' --include '*.md' --include '*.log' \
  --include '*.png' --include '*.csv' --exclude '*' "$HOST:~/$REMOTE_DIR/runs/" "$LOCAL/runs/"
echo "fetched from $HOST -> $LOCAL"
