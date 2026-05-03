#!/usr/bin/env bash
# Launch all ablation runs across 2 GPUs.
# 4 variants x 3 tasks = 12 runs. Split half/half: GPU0 does noH+relu, GPU1 does nocross+learnD.
set -e
cd "$(dirname "$0")/.."
EPOCHS=${EPOCHS:-50}
mkdir -p ablation/logs

run() {
    local gpu=$1 task=$2 variant=$3
    local logf="ablation/logs/${task}_${variant}.log"
    echo "[launch] GPU${gpu}  ${task}/${variant}  -> ${logf}"
    CUDA_VISIBLE_DEVICES=${gpu} python3 -u ablation/run_ablation.py \
        "${task}" "${variant}" --epochs ${EPOCHS} >"${logf}" 2>&1
}

# Serial per GPU, parallel across GPUs
(
    for t in T1 T6 T7; do
        run 0 "$t" noH
        run 0 "$t" relu
    done
) &
PID0=$!
(
    for t in T1 T6 T7; do
        run 1 "$t" nocross
        run 1 "$t" learnD
    done
) &
PID1=$!

wait $PID0 $PID1
echo "All ablation runs finished."
