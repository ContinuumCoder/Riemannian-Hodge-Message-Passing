#!/usr/bin/env bash
# Trains baseline models on one task through the common trainer (python -m rhmp.train --model), one model after the
# other on one GPU, with the v1 protocol by default (100 epochs, seed 42, Adam with learning rate 1e-3 and cosine
# decay, batches of 64 samples or 8 meshes, the checkpoint with the best validation R2).  Needs the data set of the
# task (datasets/README.md) and a GPU; a run resumes if it was interrupted, is skipped when its result.json is
# complete, and (model, task) pairs that are not applicable are skipped with the reason given by
# rhmp.baselines.registry.applicable.  Writes $OUT/<task>_<mode>/<model>_s<seed>/ (result.json, history.json,
# config.json), the logs runs/logs/bl_*.log and one summary line per run, appended to $OUT/summary.txt.
#
#   bash scripts/run_baselines.sh T6                                  # all registered models
#   EPOCHS=2 OUT=runs/baselines_2ep bash scripts/run_baselines.sh HP_k100 mgn egnn gat dec_fixed    # short 2-epoch runs
#   HOST=<ssh host> NAME=bl_t6 tools/run_bg.sh bash scripts/run_baselines.sh T6       # detached on a remote host
#
# Environment:
#   MODE    auto (default: legacy v1 inputs for node-input baselines on T1/T3/T5/T6/T7/T6_100K, native otherwise;
#           MeshGraphNet and the v2 family always native) | native | legacy
#   EPOCHS (100)  SEED (42)  OUT (runs/baselines)
#   BUDGET  parameter budget in millions (default: the parameter count of the v2 model)
#   OPTS    JSON builder options passed as --model-opts to every model (e.g. '{"n_layers": 8}')
#   EXTRA   extra trainer flags for every run (e.g. "--amp" or "--max-samples 300")
set -uo pipefail
cd "$(dirname "$0")/.."
TASK="${1:?usage: run_baselines.sh <task> [models...]}"
shift
MODELS="${*:-}"
MODE="${MODE:-auto}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-42}"
OUT="${OUT:-runs/baselines}"
EXTRA="${EXTRA:-}"
mkdir -p "$OUT" runs/logs
if [[ -z "$MODELS" ]]; then
  MODELS="$(python3 -c 'from rhmp.baselines.registry import model_names; print(" ".join(model_names()))')"
fi
CSV="$(echo $MODELS | tr ' ' ',')"
declare -A AUTO_MODE OK WHY
while read -r m md ok why; do
  AUTO_MODE[$m]="$md"; OK[$m]="$ok"; WHY[$m]="$why"
done < <(python3 -m rhmp.baselines.registry --applicable "$TASK" --models "$CSV" 2>/dev/null)
for m in $MODELS; do
  if [[ "${OK[$m]:-no}" != "yes" ]]; then
    echo "$(printf '%-14s' "$m") $TASK skipped: ${WHY[$m]:-applicability check failed}" | tee -a "$OUT/summary.txt"
    continue
  fi
  mode="$MODE"
  [[ "$mode" == "auto" ]] && mode="${AUTO_MODE[$m]}"
  name="${m}_s${SEED}"
  dir="$OUT/${TASK}_${mode}/$name"
  log="runs/logs/bl_${TASK}_${mode}_${name}.log"
  args=(--task "$TASK" "--$mode" --model "$m" --epochs "$EPOCHS" --seed "$SEED" --out "$dir")
  [[ -n "${BUDGET:-}" ]] && args+=(--param-budget "$BUDGET")
  [[ -n "${OPTS:-}" ]] && args+=(--model-opts "$OPTS")
  echo "=== $(date '+%F %T') $TASK/$mode $m"
  python3 -u -m rhmp.train "${args[@]}" $EXTRA > "$log" 2>&1
  code=$?
  python3 - "$dir" "$m" "$TASK/$mode" "$code" <<'PY' | tee -a "$OUT/summary.txt"
import json, os, sys
out, model, tm, code = sys.argv[1:5]
try:
    r = json.load(open(os.path.join(out, "result.json")))
    h = json.load(open(os.path.join(out, "history.json")))
    fine = f" fine_R2={r['fine']['R2']:.4f}" if "fine" in r else ""
    info = r.get("model_info") or {}
    width = info.get("hidden", info.get("C", r.get("config", {}).get("C")))
    print(f"{model:14s} {tm:16s} exit={code} val_R2(ep{len(h)})={h[-1]['val_R2']:.4f} test_R2={r['test']['R2']:.4f}"
          f"{fine} s/epoch={r['s_per_epoch']:.1f} peak={r['peak_mem_GB']:.2f}GB params={r['params_M']:.3f}M "
          f"width={width}")
except Exception as e:  # noqa: BLE001
    print(f"{model:14s} {tm:16s} exit={code} FAILED ({e})")
PY
done
echo "=== done $(date '+%F %T')"
