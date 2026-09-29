#!/usr/bin/env bash
# Push the local checkout to the remote host ($HOST:~/$REMOTE_DIR); additive, never deletes remote files.
#   HOST=mygpu tools/sync.sh
# Data (datasets/*.pkl, datasets/v2/*.pt), run outputs (runs/) and checkpoints are not pushed; result files written
# on the remote host (bench/RESULTS_*.md, bench/*.json) are not overwritten either: fetch them with tools/fetch.sh.
# Settings: tools/ssh_opts.sh.
set -euo pipefail
source "$(dirname "$0")/ssh_opts.sh"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
rsync -az -e "ssh $SSH_OPTS" --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'datasets/*.pkl' --exclude 'datasets/v2/*.pt' --exclude 'datasets/v2/*.pkl' \
  --exclude 'checkpoints_v1' --exclude 'logs' --exclude 'runs' \
  --exclude 'bench/RESULTS_*.md' --exclude 'bench/*.json' --exclude '.pytest_cache' --exclude '*.egg-info' \
  "$LOCAL/" "$HOST:~/$REMOTE_DIR/"
echo "synced -> $HOST:~/$REMOTE_DIR"
