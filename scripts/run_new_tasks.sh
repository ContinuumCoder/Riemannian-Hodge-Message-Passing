#!/usr/bin/env bash
# v2 on the new tasks (100 epochs, seed 42, sequential on one GPU): quad-grid T1, native face-target variants,
# hetero(-anisotropic) Poisson at three contrasts (+ zero-shot 4x resolution test, reported as fine_* in
# result.json), the edge-flux variant, and 3-D tetrahedral Poisson.  Requires the datasets of
# scripts/gen_datasets.sh.  Logs: runs/logs/new_<task>_s<seed>.log, outputs: runs/new/<task>_s<seed>/
#
#   NAME=new tools/run_bg.sh bash scripts/run_new_tasks.sh
#   TASKS="HP_k1000 TET" EXTRA="--amp" bash scripts/run_new_tasks.sh
set -uo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
TASKS="${TASKS:-T1q T6f T7f HP_k10 HP_k100 HP_k1000 HP_k100_aniso10 HP_k100_aniso100 HPflux_k100 HPfluxfem_k100 HPgrad_k100 HPfluxd_k100 TET_k100 TETflux_k100}"
EXTRA="${EXTRA:-}"
OUT="${OUT:-runs/new}"
mkdir -p runs/logs "$OUT"
# Bounded metric correction H/star in [e^-a, e^a] spans e^{2a}; scale a with the material contrast of the task
# (the edge input spans ln kappa, plus ln R for the tensor sets).  Override: LOG_RANGE=<a> (all runs) or
# SCALE_LOG_RANGE=0 (model default 2).
log_range_for() {
  if [[ -n "${LOG_RANGE:-}" ]]; then echo "$LOG_RANGE"; return; fi
  if [[ "${SCALE_LOG_RANGE:-1}" != "1" ]]; then echo ""; return; fi
  case "$1" in
    *aniso100*) echo 5 ;;
    *aniso10*)  echo 4.5 ;;
    *_k10000*)  echo 5.5 ;;
    *_k1000*)   echo 4.5 ;;
    *_k10)      echo 2 ;;
    HP*|TET*)   echo 3 ;;      # kappa = 100 (default contrast)
    *)          echo "" ;;
  esac
}
for t in $TASKS; do
  name="${t}_s${SEED}"
  echo "=== $(date '+%F %T') $name"
  lr="$(log_range_for "$t")"
  lrf=""; if [[ -n "$lr" ]]; then lrf="--log-range $lr"; fi
  python3 -u -m rhmp.train --task "$t" --native --epochs "$EPOCHS" --seed "$SEED" --out "$OUT/$name" $lrf $EXTRA \
    > "runs/logs/new_${name}.log" 2>&1
  echo "    exit $? ; $(tail -1 runs/logs/new_${name}.log)"
done
echo "=== done $(date '+%F %T')"
