#!/usr/bin/env bash
# Stage 3: does the metric carry the material?  HP_k100 (inputs: 0:f, 1:log sigma_e (even), 2:log sigma_f (even)), seed 42, --log-range 3.
# Solver-mode runs (linear model = FEEC solver with a learned/frozen metric) use 30 epochs; the general stack uses 50.
#   S3e_fem   exact P1 control: solver mode, frozen tensor metric with the face sigma reference (should reach ~1)
#   S3e_dec   DEC two-point control: solver mode, frozen diag metric with the edge sigma reference
#   S3c_*     solver mode with a LEARNED metric, material columns routed only to the metric heads (tensor / diag)
#   S3a       material-only routing + resolvent            S3b  + raw edge reference
#   S3d       aux PDE-residual loss + resolvent             S3f  frozen diag + edge reference + resolvent
#   S3g       frozen diag, no reference, resolvent (pure DEC; sigma only through features)
set -uo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-42}"; OUT="${OUT:-runs/stage3}"; TASK="${TASK:-HP_k100}"; EXTRA="${EXTRA:-}"
mkdir -p runs/logs "$OUT"
run() { local name="$1"; shift; local ep="$1"; shift
  echo "=== $(date '+%F %T') $name ($*)"
  python3 -u -m rhmp.train --task "$TASK" --native --epochs "$ep" --seed "$SEED" --out "$OUT/${TASK}_$name" --log-range 3 --no-resume "$@" $EXTRA > "runs/logs/stage3_${TASK}_${name}.log" 2>&1
  local code=$?
  echo "    exit $code ; $(tail -1 runs/logs/stage3_${TASK}_${name}.log)"
  if [[ $code -eq 0 ]]; then
    python3 -u scripts/metric_recovery.py "$OUT/${TASK}_$name" --n 64 >> "runs/logs/stage3_${TASK}_${name}.log" 2>&1
    python3 -u scripts/eval_robustness.py "$OUT/${TASK}_$name" --all --max-test 300 >> "runs/logs/stage3_${TASK}_${name}.log" 2>&1
  fi; }
R="--layers poly,poly,resolvent,poly"
S="--solver-mode --solve-precond twolevel --solve-iters 128"
run S3e_fem_frozen_tensor_faceref  5 $S --no-learn-metric --metric-type tensor --metric-ref 2:0
run S3c_solver_tensor_learn        30 $S --metric-type tensor --material 1:1,2:1
run S3c_solver_diag_learn          30 $S --material 1:1,2:1
run S3a_material_res               50 --material 1:1,2:1 $R
run S3d_auxpde_res                 50 --aux-pde 1.0 $R
run S3f_frozen_ref_res             50 --no-learn-metric --metric-ref 1:0 $R
run S3g_frozen_noref_res           50 --no-learn-metric $R
run S3b_material_ref_res           50 --material 1:1,2:1 --metric-ref 1:0 $R
run S3e_dec_frozen_diag_edgeref    30 $S --no-learn-metric --metric-ref 1:0
echo "=== done $(date '+%F %T')"
