#!/usr/bin/env bash
# Trains the anisotropy task suite (docs/ANISO_TASKS.md), one run after the other on one GPU: diagonal vs
# Whitney/Galerkin tensor metrics and the baselines, 50 epochs, seed 42.  After every completed run it writes the
# metric recovery (scripts/metric_recovery.py, JSON and PNG; v2 runs only) and the physics residuals of the predictions
# (rhmp.tasks.aniso.structure_metrics, test split and 4x finer split) into the run directory, and at the end the table
# $OUT/SUMMARY.md.  Needs the anisotropy data sets (ANISO=1 bash scripts/gen_datasets.sh) and a GPU; writes
# $OUT/<run>/ (default runs/anisotropy/) and the logs runs/logs/aniso_<run>.log.
#
#   bash scripts/run_aniso.sh                                                        # core preset (default)
#   PRESET=full bash scripts/run_aniso.sh                                            # every task and variant
#   HOST=<ssh host> GPU=1 NAME=aniso tools/run_bg.sh bash scripts/run_aniso.sh       # detached on a remote host
# TASKS, VARIANTS and MODELS override the preset (e.g. TASKS="AHP_r100" VARIANTS="diag-solver+ref tensor-solver+ref"
# MODELS="": the two solver-mode variants on AHP_r100, no baselines); EPOCHS=2 EXTRA="--max-train-batches 20"
# OUT=runs/aniso_2ep gives a short check.
#
# Variants (v2 model; "+ref" adds --metric-ref <degree:column>, the task's metric reference, see ref_for below):
#   diag            diagonal metrics (DEC reference star times a bounded learned correction)
#   tensor          Galerkin tensor metric, full SPD parameterisation b expm(sum s t t^T) (--tensor-param full)
#   cone            Galerkin tensor metric, cone parameterisation b I + sum a t t^T, a >= 0 (--tensor-param cone)
#   diagfixed       --no-learn-metric (learn_metric=False): fixed DEC star times the reference (diagonal physics prior)
#   tensorfixed     --no-learn-metric with the tensor metric: fixed Galerkin star (times the isotropic reference)
#   diag-solver     --solver-mode (AHP / ADARCYp only): one solve layer in physical units, linear lifting and readout
#   tensor-solver   --solver-mode, full tensor metric (AHP / ADARCYp only): exactly the target's FEM operator family
# Baselines (MODELS, parameter-matched to the v2 model): mgn (MeshGraphNet), egnn, dec_fixed (frozen DEC metric heads).
set -uo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
PRESET="${PRESET:-core}"
OUT="${OUT:-runs/anisotropy}"
EXTRA="${EXTRA:-}"
POST="${POST:-1}"
N3D="${N3D:-_n1500}"        # tet tasks: first 1500 samples (~68 GB host RAM with Whitney blocks); N3D="" = all 3000
if [[ "$PRESET" == "full" ]]; then
  DEF_TASKS="AHP_r10 AHP_r100 ASURF_r10 ASURF_r100 ASURF_heat_r10 ASURF_heat_r100 ACURL_r10$N3D ACURL_r100$N3D \
ACURLb_r10$N3D ACURLb_r100$N3D ADARCY_r10$N3D ADARCY_r100$N3D ADARCYp_r10$N3D ADARCYp_r100$N3D"
  DEF_VARIANTS="diag diag+ref tensor tensor+ref cone+ref diagfixed+ref tensorfixed+ref diag-solver+ref tensor-solver+ref"
  DEF_MODELS="mgn egnn dec_fixed"
else
  DEF_TASKS="AHP_r100 ASURF_r100 ACURLb_r100$N3D ADARCY_r100$N3D ADARCYp_r100$N3D AHP_r10"
  DEF_VARIANTS="diag+ref tensor+ref cone+ref tensorfixed+ref diag-solver+ref tensor-solver+ref"
  DEF_MODELS="mgn dec_fixed"
fi
TASKS="${TASKS:-$DEF_TASKS}"
VARIANTS="${VARIANTS:-$DEF_VARIANTS}"
MODELS="${MODELS-$DEF_MODELS}"
mkdir -p runs/logs "$OUT"

ref_for() {        # --metric-ref of the degree whose metric carries the material (rhmp/tasks/aniso.py SPECS)
  case "$1" in
    ACURL*|ADARCY_r*) echo "2:0" ;;       # 2-form metrics: log n^T nu n (ACURL), log n^T K^-1 n (ADARCY)
    *)                echo "1:0" ;;       # 1-form metric: log t^T Sigma t on the edges
  esac
}
log_range_for() {  # bounded learned log-range a: covers the full-tensor s-range of the task (docs/ANISO_TASKS.md §6:
  # median / p90 of the smallest max|s_j|: AHP r10 1.4 / 2.2, r100 2.8 / 4.5; ASURF r10 1.3 / 1.8, r100 2.8 / 3.7;
  # tets r10 1.7 / 3.0, r100 3.4 / 6.2)
  if [[ -n "${LOG_RANGE:-}" ]]; then echo "$LOG_RANGE"; return; fi
  case "$1" in ACURL*_r100*|ADARCY*_r100*) echo 6 ;; *_r100*) echo 5 ;; *) echo 3 ;; esac
}
solver_ok() { case "$1" in AHP_r*|ADARCYp_r*) return 0 ;; *) return 1 ;; esac; }

post() {           # $1 run dir, $2 task, $3 log
  [[ "$POST" == "1" ]] || return 0
  if [[ -f "$1/best.pt" && "$(python3 -c "import json;print(json.load(open('$1/config.json'))['args'].get('model','rhmp'))")" == "rhmp" ]]; then
    python3 -u scripts/metric_recovery.py "$1" --n 32 >> "$3" 2>&1
  fi
  python3 - "$1" "$2" >> "$3" 2>&1 <<'PY'
import json, os, sys, torch
from rhmp.tasks import load_task
from rhmp.tasks.aniso import structure_metrics
from rhmp.train import _renormalize
run, task = sys.argv[1:3]
conf = json.load(open(os.path.join(run, "config.json")))
a = conf["args"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = conf.get("model", {}).get("metric_type") == "tensor"
# only what is evaluated: 64 test samples (max_samples = 3 x 64 keeps the first 64 of the test split) and 16 fine ones
td = _renormalize(load_task(task, a.get("root"), device=dev, whitney=tensor, max_samples=192, fine_max=16,
                            abs_scale=bool(a.get("abs_scale"))), conf["data"])
ck = torch.load(os.path.join(run, "best.pt"), map_location=dev, weights_only=False)
if (a.get("model") or "rhmp") == "rhmp":
    from rhmp.model import RHMP
    model = RHMP.from_checkpoint(ck, map_location=dev)
else:
    from rhmp.baselines.registry import from_checkpoint, prepare_task
    td = prepare_task(a["model"], td, ck.get("build"))
    model = from_checkpoint(ck, td, map_location=dev)
out = {"test": structure_metrics(model, td, n=64, device=dev)}
if "fine" in td.extra_tests:
    out["fine"] = structure_metrics(model, td, source=td.extra_tests["fine"], n=16, device=dev)
json.dump(out, open(os.path.join(run, "structure_metrics.json"), "w"), indent=2)
print("structure", {k: {m: round(v["mean"], 5) for m, v in d.items() if isinstance(v, dict)} for k, d in out.items()})
PY
}

run_one() {        # $1 run name, $2 task, remaining: trainer flags
  local name="$1" t="$2"; shift 2
  local dir="$OUT/$name" log="runs/logs/aniso_${name}.log"
  echo "=== $(date '+%F %T') $name ($*)"
  python3 -u -m rhmp.train --task "$t" --native --epochs "$EPOCHS" --seed "$SEED" --out "$dir" "$@" $EXTRA > "$log" 2>&1
  local code=$?
  echo "    exit $code ; $(tail -1 "$log")"
  [[ $code -eq 0 ]] && post "$dir" "$t" "$log"
}

for t in $TASKS; do
  ref="$(ref_for "$t")"; lr="$(log_range_for "$t")"
  for v in $VARIANTS; do
    base="${v%+ref}"; flags=(--log-range "$lr")
    [[ "$v" == *+ref ]] && flags+=(--metric-ref "$ref")
    case "$base" in
      diag)          flags+=(--metric-type diag) ;;
      tensor)        flags+=(--metric-type tensor --tensor-param full) ;;
      cone)          flags+=(--metric-type tensor --tensor-param cone) ;;
      diagfixed)     flags+=(--metric-type diag --no-learn-metric) ;;
      tensorfixed)   flags+=(--metric-type tensor --no-learn-metric) ;;
      diag-solver)   solver_ok "$t" || continue; flags+=(--metric-type diag --solver-mode) ;;
      tensor-solver) solver_ok "$t" || continue; flags+=(--metric-type tensor --tensor-param full --solver-mode) ;;
      *) echo "unknown variant $v"; continue ;;
    esac
    run_one "${t}_${v}_s${SEED}" "$t" "${flags[@]}"
  done
  for m in $MODELS; do
    run_one "${t}_${m}_s${SEED}" "$t" --model "$m"
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
    fine = r.get("fine", {}).get("R2")
    rec = ""
    mr = glob.glob(os.path.join(d, "metric_recovery_*.json"))
    if mr:
        last = json.load(open(mr[0]))["layers"][-1]
        ops = [f"{k.split('_vs_')[0]} r {v['pearson']:+.3f}" for k, v in last.items()
               if k.endswith(("_vs_edge", "_vs_face")) and isinstance(v, dict) and v.get("pearson") is not None]
        ang = [f"{v:.0f}deg" for k, v in last.items() if k.endswith("angle_err_deg_median") and v is not None]
        rec = ", ".join(ops + ang)
    sm = ""
    sp = os.path.join(d, "structure_metrics.json")
    if os.path.exists(sp):
        t = json.load(open(sp))["test"]
        sm = ", ".join(f"{k} {v['mean']:.2e}" for k, v in t.items() if isinstance(v, dict) and not k.startswith("target_"))
    r2d = "u" if str(r.get("r2_definition", "")).startswith("uncentred") else "c"
    rows.append(f"| {os.path.basename(d)} | {r['test']['R2']:.4f} ({r2d}) | {'' if fine is None else f'{fine:.4f}'} | "
                f"{r.get('s_per_epoch') or 0:.1f} | {r.get('peak_mem_GB', 0):.1f} | {r.get('params_M', 0):.3f} | {sm} | {rec} |")
hdr = ["Test R2: (c) centred, (u) uncentred 1 - SS_res / sum t^2 (orientation-odd cochain targets).  structure: mean "
       "physics residuals of the predictions on the test split (rhmp.tasks.aniso.structure_metrics).  recovery (last "
       "layer): Pearson r of the learned metric with the true material (diagonal log H1/star1, tensor edge action and "
       "face log det) and, on tensor sets, the median principal-direction error (scripts/metric_recovery.py).", "",
       "| run | test R2 | fine (4x) R2 | s/epoch | peak GB | params M | structure (test) | recovery (last layer) |",
       "|---|---:|---:|---:|---:|---:|---|---|"]
open(os.path.join(sys.argv[1], "SUMMARY.md"), "w").write("\n".join(hdr + rows) + "\n")
print("\n".join(hdr + rows))
PY
echo "=== done $(date '+%F %T')"
