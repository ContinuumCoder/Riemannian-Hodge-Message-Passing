#!/usr/bin/env bash
# Material identification: does the learned metric carry the material?  Trains solver-mode runs (the linear model is
# a FEEC solver with a learned or frozen metric; 30 epochs, the frozen FEM control 5) and general-stack runs (50
# epochs) on HP_k100 (inputs 0:f, 1:log sigma_e (even), 2:log sigma_f (even); seed 42, --log-range 3), and writes the
# metric-recovery analysis (scripts/metric_recovery.py) and the robustness table (scripts/eval_robustness.py) into the
# directory of every completed run.  Needs the HP_k100 data set and a GPU; writes $OUT/<task>_<run>/ (default
# runs/material_identification/) and the logs runs/logs/material_identification_<task>_<run>.log.
#   bash scripts/run_material_identification.sh
#   SEED=1 OUT=runs/material_identification_s1 bash scripts/run_material_identification.sh
# TASK (default HP_k100) and EXTRA (extra trainer flags for every run) can also be set.  Runs:
#   fem_frozen_tensor_faceref  exact P1 control: solver mode, frozen tensor metric with the face sigma as metric
#                              reference (test R2 close to 1)
#   dec_frozen_diag_edgeref    DEC two-point control: solver mode, frozen diagonal metric with the edge sigma as
#                              metric reference
#   solver_tensor_learn, solver_diag_learn
#                              solver mode with a learned tensor / diagonal metric; the material columns reach only
#                              the metric heads
#   material_res               material columns routed only to the metric heads, resolvent layer
#   material_ref_res           as material_res, with the edge sigma as metric reference
#   auxpde_res                 auxiliary PDE-residual loss (--aux-pde 1.0), resolvent layer
#   frozen_ref_res             frozen diagonal metric with the edge sigma as metric reference, resolvent layer
#   frozen_noref_res           frozen diagonal metric without reference, resolvent layer (pure DEC; sigma enters only
#                              through the features)
set -uo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-42}"; OUT="${OUT:-runs/material_identification}"; TASK="${TASK:-HP_k100}"; EXTRA="${EXTRA:-}"
mkdir -p runs/logs "$OUT"
run() { local name="$1"; shift; local ep="$1"; shift
  echo "=== $(date '+%F %T') $name ($*)"
  python3 -u -m rhmp.train --task "$TASK" --native --epochs "$ep" --seed "$SEED" --out "$OUT/${TASK}_$name" --log-range 3 --no-resume "$@" $EXTRA > "runs/logs/material_identification_${TASK}_${name}.log" 2>&1
  local code=$?
  echo "    exit $code ; $(tail -1 runs/logs/material_identification_${TASK}_${name}.log)"
  if [[ $code -eq 0 ]]; then
    python3 -u scripts/metric_recovery.py "$OUT/${TASK}_$name" --n 64 >> "runs/logs/material_identification_${TASK}_${name}.log" 2>&1
    python3 -u scripts/eval_robustness.py "$OUT/${TASK}_$name" --all --max-test 300 >> "runs/logs/material_identification_${TASK}_${name}.log" 2>&1
  fi; }
R="--layers poly,poly,resolvent,poly"
S="--solver-mode --solve-precond twolevel --solve-iters 128"
run fem_frozen_tensor_faceref  5 $S --no-learn-metric --metric-type tensor --metric-ref 2:0
run solver_tensor_learn        30 $S --metric-type tensor --material 1:1,2:1
run solver_diag_learn          30 $S --material 1:1,2:1
run material_res               50 --material 1:1,2:1 $R
run auxpde_res                 50 --aux-pde 1.0 $R
run frozen_ref_res             50 --no-learn-metric --metric-ref 1:0 $R
run frozen_noref_res           50 --no-learn-metric $R
run material_ref_res           50 --material 1:1,2:1 --metric-ref 1:0 $R
run dec_frozen_diag_edgeref    30 $S --no-learn-metric --metric-ref 1:0
echo "=== done $(date '+%F %T')"
