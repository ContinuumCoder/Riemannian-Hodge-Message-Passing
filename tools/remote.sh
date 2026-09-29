#!/usr/bin/env bash
# Run a shell command on the remote host $HOST inside ~/$REMOTE_DIR (CUDA_VISIBLE_DEVICES=$GPU).  Settings:
# tools/ssh_opts.sh.
#   HOST=mygpu tools/remote.sh 'python3 -m pytest tests -x -q'
#   HOST=mygpu GPU=1 tools/remote.sh 'python3 bench/step_bench.py'
set -euo pipefail
source "$(dirname "$0")/ssh_opts.sh"
ssh $SSH_OPTS "$HOST" "cd ~/$REMOTE_DIR && export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 PY=$PY && { [ \"$PY\" = python3 ] || export PATH=\$(dirname $PY):\$PATH; } && $*"
