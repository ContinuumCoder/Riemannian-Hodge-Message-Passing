#!/usr/bin/env bash
# Sequential baseline sweeps over several tasks (100 epochs, seed 42) through scripts/run_baselines.sh.
#   bash scripts/run_baselines_seq.sh "T3 HP_k100"                                   # locally / on the GPU machine
#   NAME=blA GPU=0 tools/run_bg.sh 'bash scripts/run_baselines_seq.sh "T3 HP_k100"'      # detached on a remote host
# (machines without pyvista need PYTHONPATH=<repo>/shims to unpickle the T3 dataset)
set -uo pipefail
cd "$(dirname "$0")/.."
TASKS="${1:?tasks}"
MODELS="${MODELS:-mgn egnn gat gcn schnet cw_net sccnn mpsn gauge_cnn gem_cnn clifford_smpn ours_v1 dec_fixed}"
for t in $TASKS; do
  echo "##### $(date '+%F %T') task $t"
  EPOCHS="${EPOCHS:-100}" bash scripts/run_baselines.sh "$t" $MODELS
done
echo "##### all done $(date '+%F %T')"
