#!/usr/bin/env bash
# Generate the v2 datasets (outputs in datasets/v2/, git-ignored; see datasets/README.md).
#   bash scripts/gen_datasets.sh                              # core: T6/T7 native, HP_*, TET_k100
#   SUITE=1 ANISO=1 bash scripts/gen_datasets.sh              # + SURF / DYN / HP_qual_* and AHP / ASURF / ACURL / ADARCY
#   SKIP_CORE=1 ANISO=1 bash scripts/gen_datasets.sh          # only the anisotropy suite
#   tools/remote.sh '(setsid nohup bash scripts/gen_datasets.sh > runs/logs/gen_datasets.log 2>&1 < /dev/null &)'
# T6/T7 native regeneration uses the CUDA RNG (must run on a GPU to reproduce the v1 samples) and reads the v1 pickles
# (python3 datasets/download_v1.py); everything else is CPU only (multiprocessing, WORKERS processes).  The core sets
# take about 6 minutes, the suite about 1 minute, the anisotropy suite (3-D Nedelec / RT0 solves) several hours.
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
  python3 -u datasets/generators/gen_HP.py --kappa 100 --aniso --n 5000 --n-fine 500 --workers "$W"        # legacy aniso
  for r in 10 100; do                                                                                 # tensor aniso
    python3 -u datasets/generators/gen_HP.py --kappa 100 --aniso-max "$r" --n 5000 --n-fine 500 --workers "$W"
  done
  # heterogeneity robustness sweep (test-size set; same fields/meshes as HP_k* for the first 1000 ids)
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
