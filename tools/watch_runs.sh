#!/usr/bin/env bash
# Print one line per newly finished run on one or more remote hosts: the "exit" / "=== done" / "##### all done"
# lines that the scripts in scripts/ write into runs/logs/*.log.  Polls every POLL seconds (default 300).
#   HOSTS="gpu1 gpu2" tools/watch_runs.sh          (REMOTE_DIR as in tools/ssh_opts.sh; default rhmp)
set -uo pipefail
HOSTS="${HOSTS:-${HOST:?set HOST or HOSTS}}"
REMOTE_DIR="${REMOTE_DIR:-rhmp}"
POLL="${POLL:-300}"
SEEN="${SEEN:-/tmp/rhmp_seen_runs.txt}"; touch "$SEEN"
OPTS="-o BatchMode=yes -o ControlMaster=auto -o ControlPath=/tmp/cm_rhmp_%C -o ControlPersist=1800 -o ConnectTimeout=20"
while true; do
  for h in $HOSTS; do
    ssh $OPTS "$h" "cd ~/$REMOTE_DIR && grep -h 'exit\|=== done\|##### all done' runs/logs/*.log 2>/dev/null" 2>/dev/null \
      | sed "s/^/[$h] /"
  done | tr -s ' ' | cut -c1-230 | sort -u > "$SEEN.now"
  comm -13 "$SEEN" "$SEEN.now"
  cp "$SEEN.now" "$SEEN"
  sleep "$POLL"
done
