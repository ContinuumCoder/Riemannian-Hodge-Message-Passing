#!/usr/bin/env bash
# Trains the v2 model on high-contrast heterogeneous Poisson problems with the metric reference (the raw edge log
# conductivity, --metric-ref 1:0) on HP_k1000, HP_k10000 and HP_k100_aniso100, on HP_k10000 also without it, and on
# HP_k1000 also with the reference and a resolvent layer (50 epochs, seed 42, diagonal metric, polynomial layers,
# --log-range 3).  Needs the HP data sets of scripts/gen_datasets.sh (including HP_k10000) and a GPU; writes
# $OUT/<run>/ (default runs/high_contrast/) and the logs runs/logs/high_contrast_<run>.log.  The reference-free
# HP_k1000 run of results/high_contrast/ uses the same settings without --metric-ref.
#   bash scripts/run_high_contrast.sh
#   SEED=1 OUT=runs/high_contrast_s1 bash scripts/run_high_contrast.sh
set -uo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-50}"; OUT="${OUT:-runs/high_contrast}"
mkdir -p runs/logs "$OUT"
run() { local name="$1"; shift; local task="$1"; shift
  echo "=== $(date '+%F %T') $name ($*)"
  python3 -u -m rhmp.train --task "$task" --native --epochs "$EPOCHS" --seed "$SEED" --out "$OUT/$name" --log-range 3 --no-resume "$@" > "runs/logs/high_contrast_${name}.log" 2>&1
  echo "    exit $? ; $(tail -1 runs/logs/high_contrast_${name}.log)"; }
run HP_k1000_ref_s${SEED}           HP_k1000          --metric-ref 1:0
run HP_k10000_noref_s${SEED}        HP_k10000
run HP_k10000_ref_s${SEED}          HP_k10000         --metric-ref 1:0
run HP_k100_aniso100_ref_s${SEED}   HP_k100_aniso100  --metric-ref 1:0
run HP_k1000_ref_res_s${SEED}       HP_k1000          --metric-ref 1:0 --layers poly,poly,resolvent,poly
echo "=== done $(date '+%F %T')"
