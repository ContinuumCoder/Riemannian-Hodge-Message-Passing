#!/usr/bin/env bash
# v2 accuracy benchmark on the 7 paper tasks with the v1 protocol (100 epochs, seed 42), native and legacy I/O,
# run sequentially on one GPU (GPU 1 by default).  Each run resumes automatically if interrupted and is skipped when
# its result.json is complete.  Logs: runs/logs/paper_<task>_<mode>_s<seed>.log, outputs: runs/paper/<task>_<mode>_s<seed>/
#
#   NAME=paper tools/run_bg.sh bash scripts/run_paper_tasks.sh            # (or run it inside tmux/screen)
#   SEED=1 EPOCHS=100 TASKS="T6 T7" MODES="native" bash scripts/run_paper_tasks.sh
#   EXTRA="--amp" bash scripts/run_paper_tasks.sh                          # extra trainer flags for every run
#   V1_EVAL=1 bash scripts/run_paper_tasks.sh    # also evaluate the v1 checkpoints with the same script/metrics
set -uo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
TASKS="${TASKS:-T1 T2 T3 T5 T6 T7 T8}"
MODES="${MODES:-native legacy}"
EXTRA="${EXTRA:-}"
OUT="${OUT:-runs/paper}"
mkdir -p runs/logs "$OUT"
declare -A V1DIR=([T1]=T1_cns_vorticity [T2]=T2_torus_advection_diffusion [T3]=T3_ellipsoid_surface_flow
                  [T5]=T5_maxwell_poisson [T6]=T6_wilson_loop [T7]=T7_yang_mills_su2 [T8]=T8_airfoil_pressure)
for t in $TASKS; do
  for m in $MODES; do
    # T2 and T8 have identical native/legacy inputs: run them once (as native)
    if [[ "$m" == "legacy" && ( "$t" == "T2" || "$t" == "T8" ) ]]; then continue; fi
    name="${t}_${m}_s${SEED}"
    echo "=== $(date '+%F %T') $name"
    python3 -u -m rhmp.train --task "$t" --"$m" --epochs "$EPOCHS" --seed "$SEED" --out "$OUT/$name" $EXTRA \
      > "runs/logs/paper_${name}.log" 2>&1
    echo "    exit $? ; $(tail -1 runs/logs/paper_${name}.log)"
  done
  if [[ "${V1_EVAL:-0}" == "1" ]]; then
    ck="checkpoints_v1/${V1DIR[$t]}/ours/best_model.pt"
    if [[ "$SEED" != "42" ]]; then ck="checkpoints_v1/seed_${SEED}/${V1DIR[$t]}/ours/best_model.pt"; fi
    python3 -u -m rhmp.train --task "$t" --legacy --eval-v1 "$ck" --seed "$SEED" --out "$OUT/${t}_v1_s${SEED}" \
      > "runs/logs/paper_${t}_v1_s${SEED}.log" 2>&1
    echo "    v1: $(tail -1 runs/logs/paper_${t}_v1_s${SEED}.log)"
  fi
done
echo "=== done $(date '+%F %T')"
