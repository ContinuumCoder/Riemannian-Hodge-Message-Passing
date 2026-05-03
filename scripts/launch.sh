#!/usr/bin/env bash
# Launch scheduler in background with nohup so it survives SSH disconnect.
# Usage:  bash scripts/launch.sh [--only T1,T6] [--gpus 0,1]

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

if [ ! -f status/jobs.json ]; then
    echo "jobs.json missing. Running build_jobs.py first..."
    python3 scripts/build_jobs.py
fi

# Refuse to start if one is already running
if [ -f status/scheduler.pid ]; then
    PID=$(cat status/scheduler.pid)
    if kill -0 "$PID" 2>/dev/null; then
        echo "scheduler already running (PID $PID). Tail status/scheduler.out or run monitor.py"
        exit 1
    else
        echo "stale pid file, removing"
        rm -f status/scheduler.pid
    fi
fi

mkdir -p status logs
LOG="status/scheduler.out"
echo "launching scheduler; stdout -> $LOG"
# default: 2 concurrent workers per GPU (override with --per-gpu N in args)
if ! printf '%s\n' "$@" | grep -q -- '--per-gpu'; then
    set -- --per-gpu 2 "$@"
fi
# persist args so keepalive can re-launch with the same filter on crash recovery
printf '%q ' "$@" > status/launch_args.txt
echo >> status/launch_args.txt
nohup python3 -u scripts/scheduler.py "$@" > "$LOG" 2>&1 &
PID=$!
disown
sleep 1

if kill -0 "$PID" 2>/dev/null; then
    echo "scheduler PID=$PID"
    echo "  live monitor:  python3 scripts/monitor.py --watch"
    echo "  tail log:      tail -f $LOG"
    echo "  stop:          kill $PID   (workers drain current jobs first)"
else
    echo "scheduler failed to start; see $LOG"
    tail -40 "$LOG"
    exit 1
fi
