#!/usr/bin/env bash
# Stage 2b (rerun with raw material reference): high-contrast HP with / without --metric-ref 1:0. 50 epochs, diag, poly, --log-range 3.
set -uo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-50}"; OUT="${OUT:-runs/stage2b}"
mkdir -p runs/logs "$OUT"
run() { local name="$1"; shift; local task="$1"; shift
  echo "=== $(date '+%F %T') $name ($*)"
  python3 -u -m rhmp.train --task "$task" --native --epochs "$EPOCHS" --seed "$SEED" --out "$OUT/$name" --log-range 3 --no-resume "$@" > "runs/logs/stage2b_${name}.log" 2>&1
  echo "    exit $? ; $(tail -1 runs/logs/stage2b_${name}.log)"; }
run HP_k1000_ref_s${SEED}           HP_k1000          --metric-ref 1:0
run HP_k10000_noref_s${SEED}        HP_k10000
run HP_k10000_ref_s${SEED}          HP_k10000         --metric-ref 1:0
run HP_k100_aniso100_ref_s${SEED}   HP_k100_aniso100  --metric-ref 1:0
run HP_k1000_ref_res_s${SEED}       HP_k1000          --metric-ref 1:0 --layers poly,poly,resolvent,poly
echo "=== done $(date '+%F %T')"
