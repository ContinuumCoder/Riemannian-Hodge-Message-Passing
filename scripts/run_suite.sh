#!/usr/bin/env bash
# Extension-suite runs (SURF / DYN), sequential on one GPU. Logs: runs/logs/suite_<task>_s<seed>.log, out: runs/suite/
#   NAME=suite GPU=0 tools/run_bg.sh 'bash scripts/run_suite.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-100}"; TASKS="${TASKS:-SURF SURF_heat DYN DYNfix}"; EXTRA="${EXTRA:-}"; OUT="${OUT:-runs/suite}"
mkdir -p runs/logs "$OUT"
for t in $TASKS; do
  name="${t}_s${SEED}"
  echo "=== $(date '+%F %T') $name"
  python3 -u -m rhmp.train --task "$t" --native --epochs "$EPOCHS" --seed "$SEED" --out "$OUT/$name" $EXTRA > "runs/logs/suite_${name}.log" 2>&1
  echo "    exit $? ; $(tail -1 runs/logs/suite_${name}.log)"
done
echo "=== done $(date '+%F %T')"
