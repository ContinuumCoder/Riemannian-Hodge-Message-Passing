#!/usr/bin/env bash
# Run on REMOTE: watches scheduler PID, relaunches on death.
# Idempotent: refuses to start a second copy of itself.
# Usage:  nohup bash scripts/keepalive.sh > status/keepalive.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

KEEPALIVE_PID=status/keepalive.pid
SCHED_PID=status/scheduler.pid
INTERVAL=60

if [ -f "$KEEPALIVE_PID" ] && kill -0 "$(cat $KEEPALIVE_PID)" 2>/dev/null; then
    echo "keepalive already running (PID $(cat $KEEPALIVE_PID))"
    exit 1
fi
echo $$ > "$KEEPALIVE_PID"
trap "rm -f $KEEPALIVE_PID" EXIT

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Exit if no pending work remains
all_done() {
    python3 -c "
import json
d = json.load(open('status/jobs.json'))
pend = sum(1 for j in d if j['status'] in ('pending','running'))
print(pend)" 2>/dev/null
}

echo "[$(ts)] keepalive started (PID $$, interval ${INTERVAL}s)"
while true; do
    pend=$(all_done)
    if [ "$pend" = "0" ]; then
        echo "[$(ts)] all jobs done (pending+running=0). exiting keepalive."
        break
    fi

    if [ -f "$SCHED_PID" ] && kill -0 "$(cat $SCHED_PID)" 2>/dev/null; then
        :  # alive
    else
        # restore args from persisted launch_args.txt (preserves --only / --per-gpu)
        ARGS=""
        [ -f status/launch_args.txt ] && ARGS=$(cat status/launch_args.txt)
        echo "[$(ts)] scheduler down (pending=$pend), relaunching with: $ARGS"
        rm -f "$SCHED_PID"
        bash scripts/launch.sh $ARGS >> status/keepalive.out 2>&1 || echo "[$(ts)] relaunch failed"
    fi
    sleep $INTERVAL
done
