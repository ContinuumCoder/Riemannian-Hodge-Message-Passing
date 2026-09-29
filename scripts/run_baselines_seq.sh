#!/usr/bin/env bash
# Runs scripts/run_baselines.sh on several tasks in turn (100 epochs, seed 42) with the baseline models listed in
# MODELS.  Needs the data sets of the tasks and a GPU (machines without pyvista need PYTHONPATH=<repo>/shims to
# unpickle the T3 data set); the outputs are those of run_baselines.sh: runs/baselines/<task>_<mode>/<model>_s<seed>/,
# the logs runs/logs/bl_*.log and the summary runs/baselines/summary.txt.
#   bash scripts/run_baselines_seq.sh "T3 HP_k100"
#   HOST=<ssh host> NAME=bl_seq GPU=0 tools/run_bg.sh 'bash scripts/run_baselines_seq.sh "T3 HP_k100"'   # detached
set -uo pipefail
cd "$(dirname "$0")/.."
TASKS="${1:?tasks}"
MODELS="${MODELS:-mgn egnn gat gcn schnet cw_net sccnn mpsn gauge_cnn gem_cnn clifford_smpn ours_v1 dec_fixed}"
for t in $TASKS; do
  echo "##### $(date '+%F %T') task $t"
  EPOCHS="${EPOCHS:-100}" bash scripts/run_baselines.sh "$t" $MODELS
done
echo "##### all done $(date '+%F %T')"
