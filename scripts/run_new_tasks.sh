#!/usr/bin/env bash
# Trains the v2 model on the new tasks (100 epochs, seed 42, one task after the other on one GPU): T1 on a quad grid
# (T1q), the native face-target variants T6f and T7f, heterogeneous Poisson at three contrasts (HP_k10, HP_k100,
# HP_k1000) and with anisotropic conductivity (HP_k100_aniso10, HP_k100_aniso100), each with a zero-shot test on 4x
# finer meshes (reported as fine_* in result.json), the edge-flux, FEM-flux, gradient and flux-density targets of
# HP_k100 (HPflux_k100, HPfluxfem_k100, HPgrad_k100, HPfluxd_k100) and 3-D tetrahedral Poisson (TET_k100,
# TETflux_k100).  Needs the v1 pickles (python3 datasets/download_v1.py), the data sets of scripts/gen_datasets.sh
# and a GPU; writes $OUT/<task>_s<seed>/ (default runs/new_tasks/) and the logs runs/logs/new_<task>_s<seed>.log.
#
#   TASKS="HP_k1000 TET" EXTRA="--amp" bash scripts/run_new_tasks.sh
#   HOST=<ssh host> NAME=new tools/run_bg.sh bash scripts/run_new_tasks.sh    # detached on a remote host
set -uo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
TASKS="${TASKS:-T1q T6f T7f HP_k10 HP_k100 HP_k1000 HP_k100_aniso10 HP_k100_aniso100 HPflux_k100 HPfluxfem_k100 HPgrad_k100 HPfluxd_k100 TET_k100 TETflux_k100}"
EXTRA="${EXTRA:-}"
OUT="${OUT:-runs/new_tasks}"
mkdir -p runs/logs "$OUT"
# The bounded metric correction H/star lies in [e^-a, e^a], a range of e^{2a}; a is scaled with the material contrast
# of the task (the edge input spans ln kappa, plus ln R on the tensor sets).  LOG_RANGE=<a> sets a for all runs,
# SCALE_LOG_RANGE=0 keeps the model default (2).
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
