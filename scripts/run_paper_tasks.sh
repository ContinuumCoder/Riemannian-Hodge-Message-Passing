#!/usr/bin/env bash
# Trains the v2 model on the seven paper tasks with the v1 protocol (100 epochs, seed 42), with native and legacy
# inputs and outputs, one run after the other on one GPU; a run resumes automatically if it was interrupted and is
# skipped when its result.json is complete.  Needs the v1 pickles (python3 datasets/download_v1.py), for native T6/T7
# the data sets of scripts/gen_datasets.sh, and a GPU; V1_EVAL=1 also re-evaluates the v1 checkpoints with the same
# split and metric code (download_v1.py --ckpt, or --seed-ckpt for seeds 1 and 2).  Writes
# $OUT/<task>_<mode>_s<seed>/ (default runs/paper_tasks/) and the logs runs/logs/paper_<task>_<mode>_s<seed>.log.
#
#   SEED=1 TASKS="T6 T7" MODES="native" EXTRA="--amp" bash scripts/run_paper_tasks.sh
#   V1_EVAL=1 bash scripts/run_paper_tasks.sh
#   HOST=<ssh host> NAME=paper tools/run_bg.sh bash scripts/run_paper_tasks.sh    # detached (or use tmux / screen)
set -uo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
TASKS="${TASKS:-T1 T2 T3 T5 T6 T7 T8}"
MODES="${MODES:-native legacy}"
EXTRA="${EXTRA:-}"
OUT="${OUT:-runs/paper_tasks}"
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
