#!/usr/bin/env bash
# Generates the v2 data sets in datasets/v2/ (git-ignored; see datasets/README.md): the core sets (T6/T7 native, HP_*,
# TET_k100), with SUITE=1 also the extension suite (SURF, DYN, HP_qual_*) and with ANISO=1 the anisotropy suite (AHP,
# ASURF, ACURL, ADARCY); SKIP_CORE=1 skips the core sets, SKIP_NATIVE=1 only the T6/T7 native sets.  The T6/T7 native
# sets are regenerated from the v1 pickles (python3 datasets/download_v1.py) with the CUDA random number generator and
# need a GPU to reproduce the v1 samples; all other sets are generated on the CPU with WORKERS processes (default 32).
# The core sets take about 6 minutes, the extension suite about 1 minute and the anisotropy suite (3-D Nedelec / RT0
# solves) several hours.
#   bash scripts/gen_datasets.sh                              # core sets
#   SUITE=1 ANISO=1 bash scripts/gen_datasets.sh              # core sets, extension suite and anisotropy suite
#   HOST=<ssh host> NAME=gen_datasets tools/run_bg.sh bash scripts/gen_datasets.sh    # detached on a remote host
set -euo pipefail
cd "$(dirname "$0")/.."
W="${WORKERS:-32}"
if [[ "${SKIP_CORE:-0}" != "1" ]]; then
  if [[ "${SKIP_NATIVE:-0}" != "1" ]]; then
    python3 -u datasets/generators/gen_T6_native.py
    python3 -u datasets/generators/gen_T7_native.py
  fi
  for k in 10 100 1000; do
    python3 -u datasets/generators/gen_HP.py --kappa "$k" --n 5000 --n-fine 500 --workers "$W"
  done
  python3 -u datasets/generators/gen_HP.py --kappa 100 --aniso --n 5000 --n-fine 500 --workers "$W"   # HP_k100_aniso
  for r in 10 100; do                                                                                 # HP_k100_aniso<r>
    python3 -u datasets/generators/gen_HP.py --kappa 100 --aniso-max "$r" --n 5000 --n-fine 500 --workers "$W"
  done
  # HP_k10000 for the contrast sweep: 1000 samples, the fields and meshes of the first 1000 samples of the HP_k* sets
  python3 -u datasets/generators/gen_HP.py --kappa 10000 --n 1000 --n-fine 0 --workers "$W"
  python3 -u datasets/generators/gen_TET.py --kappa 100 --n 3000 --n-fine 200 --workers "$W"
fi
if [[ "${SUITE:-0}" == "1" ]]; then          # extension suite (docs/TASK_SUITE_DETAILS.md); gen_qual needs HP_k100.pt
  python3 -u datasets/generators/gen_surf.py --workers "$W"
  python3 -u datasets/generators/gen_dyn.py --workers "$W"
  python3 -u datasets/generators/gen_qual.py --workers "$W"
fi
if [[ "${ANISO:-0}" == "1" ]]; then          # anisotropy suite (docs/ANISO_TASKS.md)
  python3 -u datasets/generators/gen_aniso.py --tasks AHP ASURF ACURL ADARCY --workers "$W"
  python3 -u datasets/generators/gen_aniso.py --analyse 12 --workers "$W"      # oracle representability report
fi
ls -la datasets/v2/
