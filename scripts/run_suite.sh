#!/usr/bin/env bash
# Trains the v2 model on the extension suite (by default SURF, SURF_heat, DYN and DYNfix; 100 epochs, seed 42), one
# task after the other on one GPU.  Needs the extension-suite data sets (SUITE=1 bash scripts/gen_datasets.sh) and a
# GPU; writes $OUT/<task>_s<seed>/ (default runs/extension_suite/) and the logs runs/logs/suite_<task>_s<seed>.log.
#   TASKS="DYNfix_cons DYN_cons" bash scripts/run_suite.sh                  # the conservative variants
#   HOST=<ssh host> NAME=suite GPU=0 tools/run_bg.sh 'bash scripts/run_suite.sh'    # detached on a remote host
set -uo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-100}"; TASKS="${TASKS:-SURF SURF_heat DYN DYNfix}"; EXTRA="${EXTRA:-}"; OUT="${OUT:-runs/extension_suite}"
mkdir -p runs/logs "$OUT"
for t in $TASKS; do
  name="${t}_s${SEED}"
  echo "=== $(date '+%F %T') $name"
  python3 -u -m rhmp.train --task "$t" --native --epochs "$EPOCHS" --seed "$SEED" --out "$OUT/$name" $EXTRA > "runs/logs/suite_${name}.log" 2>&1
  echo "    exit $? ; $(tail -1 runs/logs/suite_${name}.log)"
done
echo "=== done $(date '+%F %T')"
