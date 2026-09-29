#!/usr/bin/env bash
# 2-epoch smoke training of every task/mode with the v2 model (full data), one line per run in
# runs/smoke/summary.txt (val R2 after 2 epochs, s/epoch, peak memory).
#   bash scripts/smoke_all.sh            TASKS="T6 T8" bash scripts/smoke_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."
EPOCHS="${EPOCHS:-2}"
OUT="${OUT:-runs/smoke}"
RUNS="${RUNS:-T1:native T1:legacy T1q:native T2:native T3:native T3:legacy T5:native T5:legacy T6:native T6:legacy T6f:native T7:native T7:legacy T7f:native T8:native HP_k100:native HP_k100_aniso:native HPflux_k100:native TET_k100:native TETflux_k100:native}"
EXTRA="${EXTRA:-}"
mkdir -p "$OUT" runs/logs
for r in $RUNS; do
  t="${r%%:*}"; m="${r##*:}"
  name="${t}_${m}"
  python3 -u -m rhmp.train --task "$t" --"$m" --epochs "$EPOCHS" --seed 42 --out "$OUT/$name" --force --no-resume \
    $EXTRA > "runs/logs/smoke_${name}.log" 2>&1
  code=$?
  python3 - "$OUT/$name" "$name" "$code" <<'PY' | tee -a "$OUT/summary.txt"
import json, os, sys
out, name, code = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    r = json.load(open(os.path.join(out, "result.json")))
    h = json.load(open(os.path.join(out, "history.json")))
    fine = f" fine_R2={r['fine']['R2']:.4f}" if "fine" in r else ""
    print(f"{name:22s} exit={code} val_R2(ep{len(h)})={h[-1]['val_R2']:.4f} test_R2={r['test']['R2']:.4f}{fine} "
          f"s/epoch={r['s_per_epoch']:.1f} peak={r['peak_mem_GB']:.2f}GB params={r['params_M']:.3f}M")
except Exception as e:  # noqa: BLE001
    print(f"{name:22s} exit={code} FAILED ({e})")
PY
done
