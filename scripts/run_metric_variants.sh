#!/usr/bin/env bash
# Trains the metric and layer variants of docs/DESIGN.md (section 9) one after the other on one GPU: diagonal vs
# tensor metric and polynomial vs resolvent layers (the default diag-poly, tensor-poly, diag-resolvent and
# tensor-resolvent), EPOCHS epochs each (default 50, seed 42); after every completed run the robustness table
# (scripts/eval_robustness.py) and, for HP/TET, the metric-recovery analysis (scripts/metric_recovery.py) are written
# into the run directory.  Needs the data sets of scripts/gen_datasets.sh (the robustness evaluation of the HP runs
# also uses QUAL_TASKS, by default HP_qual_graded,HP_qual_sliver of SUITE=1) and a GPU.  Writes
# $OUT/<task>_<variant>_s<seed>/ (default runs/metric_variants/), the logs
# runs/logs/metric_variants_<task>_<variant>_s<seed>.log and the table $OUT/SUMMARY.md.
#
#   TASKS="HP_k100_aniso100" VARIANTS="diag-poly tensor-poly" EPOCHS=5 bash scripts/run_metric_variants.sh  # 5 epochs
#   VARIANTS="diag-poly diag-poly+ref tensor-resolvent+ref" bash scripts/run_metric_variants.sh
#   HOST=<ssh host> NAME=metric_variants tools/run_bg.sh bash scripts/run_metric_variants.sh   # detached, remote
#
# "+ref" adds the metric reference --metric-ref 1:0 (the edge log conductivity as a fixed log offset of H_1; HP/TET
# only), so that the bounded correction is relative to the known material.  RESOLVENT_LAYERS sets the layer types of
# the resolvent variants (default poly,poly,resolvent,poly), EXTRA adds trainer flags (e.g. "--amp") and POST=0 skips
# the evaluations.  --log-range is scaled with the material contrast of each task (log_range_for below).
set -uo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
TASKS="${TASKS:-HP_k100 HP_k100_aniso100 T6f TET_k100}"
VARIANTS="${VARIANTS:-diag-poly tensor-poly diag-resolvent tensor-resolvent}"
RESOLVENT_LAYERS="${RESOLVENT_LAYERS:-poly,poly,resolvent,poly}"
EXTRA="${EXTRA:-}"
POST="${POST:-1}"
OUT="${OUT:-runs/metric_variants}"
mkdir -p runs/logs "$OUT"
# The bounded metric correction H/star lies in [e^-a, e^a], a range of e^{2a}; a is scaled with the material contrast
# of the task (the edge input spans ln kappa, plus ln R on the tensor sets).  LOG_RANGE=<a> sets a for all runs,
# SCALE_LOG_RANGE=0 keeps the model default (2).
log_range_for() {
  if [[ -n "${LOG_RANGE:-}" ]]; then echo "$LOG_RANGE"; return; fi
  if [[ "${SCALE_LOG_RANGE:-1}" != "1" ]]; then echo ""; return; fi
  case "$1" in
    *aniso100*) echo 5 ;;
    *aniso10*)  echo 4.5 ;;
    *_k10000*)  echo 5.5 ;;
    *_k1000*)   echo 4.5 ;;
    *_k10)      echo 2 ;;
    HP*|TET*)   echo 3 ;;      # kappa = 100 (default contrast)
    *)          echo "" ;;
  esac
}
for t in $TASKS; do
  for v in $VARIANTS; do
    base="${v%+ref}"
    metric="${base%%-*}"; layer="${base##*-}"
    flags="--metric-type $metric"
    if [[ "$v" == *+ref ]]; then
      case "$t" in HP*|TET*) flags="$flags --metric-ref 1:0" ;; *) echo "    (+ref ignored for $t)" ;; esac
    fi
    if [[ "$layer" == "resolvent" ]]; then flags="$flags --layers $RESOLVENT_LAYERS"; fi
    lr="$(log_range_for "$t")"
    if [[ -n "$lr" ]]; then flags="$flags --log-range $lr"; fi
    name="${t}_${v}_s${SEED}"
    echo "=== $(date '+%F %T') $name ($flags)"
    python3 -u -m rhmp.train --task "$t" --native --epochs "$EPOCHS" --seed "$SEED" --out "$OUT/$name" $flags $EXTRA \
      > "runs/logs/metric_variants_${name}.log" 2>&1
    code=$?
    echo "    exit $code ; $(tail -1 runs/logs/metric_variants_${name}.log)"
    if [[ "$POST" == "1" && $code -eq 0 ]]; then
      also=""
      case "$t" in HP*) also="--also-tasks ${QUAL_TASKS:-HP_qual_graded,HP_qual_sliver}" ;; esac
      python3 -u scripts/eval_robustness.py "$OUT/$name" --all --max-test 300 $also \
        >> "runs/logs/metric_variants_${name}.log" 2>&1
      case "$t" in HP*|TET*)
        python3 -u scripts/metric_recovery.py "$OUT/$name" --n 64 >> "runs/logs/metric_variants_${name}.log" 2>&1 ;;
      esac
    fi
  done
done
python3 - "$OUT" <<'PY'
import glob, json, os, sys
rows = []
for d in sorted(glob.glob(os.path.join(sys.argv[1], "*"))):
    p = os.path.join(d, "result.json")
    if not os.path.exists(p):
        continue
    r = json.load(open(p))
    mr = glob.glob(os.path.join(d, "metric_recovery_*.json"))
    rec = sat = ""
    if mr:
        m = json.load(open(mr[0]))
        corr = [l.get("diag_pearson", l.get("tensor_logdet_pearson")) for l in m["layers"]]
        rec = " / ".join("-" if c is None or c != c else f"{c:+.2f}" for c in corr)
        sat = " / ".join(f"{l['sat']:.2f}" if "sat" in l else "-" for l in m["layers"])
    rb = glob.glob(os.path.join(d, "robustness_*.json"))
    sym = ""
    if rb:
        rr = json.load(open(rb[0]))["rows"]
        dev = [abs(x["dR2"]) for x in rr if x["condition"] in ("relabel", "flip-orient", "rotate", "reflect")
               or x["condition"].startswith("gauge")]
        sym = f"{max(dev):.1e}" if dev else ""
    fine = r.get("fine", {}).get("R2")
    lr = r.get("config", {}).get("log_range", "")
    r2d = "u" if str(r.get("r2_definition", "")).startswith("uncentred") else "c"
    rows.append(f"| {os.path.basename(d)} | {lr} | {r['test']['R2']:.4f} ({r2d}) | "
                f"{'' if fine is None else f'{fine:.4f}'} | {r['s_per_epoch']:.1f} | {r['peak_mem_GB']:.1f} | "
                f"{r['params_M']:.3f} | {rec} | {sat} | {sym} |")
hdr = ["Test R2: (c) centred (v1 definition), (u) uncentred 1 - SS_res / sum t^2 for orientation-odd cochain targets.",
       "Metric recovery r: per layer, signed Pearson correlation of the learned log(H_1/star_1) (diag) or of the "
       "learned log det sigma_f (tensor) with the true log conductivity; sat: per-layer fraction of edges whose bounded "
       "correction is clamped (|phi| > 0.95 log_range); sym: max |dR2| over relabel / flip-orient / rotate / reflect "
       "/ gauge noise (exact symmetries -> round-off).", "",
       "| run | log_range | test R2 | fine (4x) R2 | s/epoch | peak GB | params M | metric recovery r (per layer) "
       "| sat (per layer) | sym |",
       "|---|---:|---:|---:|---:|---:|---:|---|---|---:|"]
open(os.path.join(sys.argv[1], "SUMMARY.md"), "w").write("\n".join(hdr + rows) + "\n")
print("\n".join(hdr + rows))
PY
echo "=== done $(date '+%F %T')"
