#!/usr/bin/env bash
# Launch a detached long-running job on the remote host $HOST and return immediately; its output is written to
# ~/$REMOTE_DIR/runs/logs/<NAME>.log (NAME defaults to a time stamp).  Settings: tools/ssh_opts.sh.
#   HOST=mygpu NAME=paper GPU=0 tools/run_bg.sh 'bash scripts/run_paper_tasks.sh'   # the command is run by bash -c
set -euo pipefail
source "$(dirname "$0")/ssh_opts.sh"
NAME="${NAME:-job_$(date +%Y%m%d_%H%M%S)}"
CMD="$*"
ssh $SSH_OPTS "$HOST" "cd ~/$REMOTE_DIR && mkdir -p runs/logs && ({ [ \"$PY\" = python3 ] || export PATH=\$(dirname $PY):\$PATH; }; CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 PY=$PY setsid nohup bash -c $(printf '%q' "$CMD") > runs/logs/$NAME.log 2>&1 < /dev/null &) ; sleep 1; echo \"started $NAME -> runs/logs/$NAME.log\""
