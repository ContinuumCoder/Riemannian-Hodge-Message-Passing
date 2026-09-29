# Shared settings of the remote-workflow scripts in tools/ (sourced, not executed).
#   HOST        ssh host of the GPU machine (required; e.g. an entry of ~/.ssh/config)
#   REMOTE_DIR  checkout on that host, relative to its home directory (default: rhmp)
#   PY          python executable on the host (default: python3)
#   GPU         CUDA_VISIBLE_DEVICES for remote commands (default: 0)
# One multiplexed ssh connection avoids sshd banner timeouts when many commands run concurrently.
SSH_OPTS="-o BatchMode=yes -o ControlMaster=auto -o ControlPath=/tmp/cm_rhmp_%C -o ControlPersist=1800 -o ConnectTimeout=20"
HOST="${HOST:?set HOST=<ssh host of the GPU machine>}"
REMOTE_DIR="${REMOTE_DIR:-rhmp}"
PY="${PY:-python3}"
GPU="${GPU:-0}"
