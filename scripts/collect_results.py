"""Regenerate ``results/RESULTS.md`` from the result files in ``results/``.

The script only reads JSON files (the per-run results in ``results/``, or in a ``runs/`` directory of your own) and
needs neither torch, nor a GPU, nor the data sets; it writes the Markdown tables to ``--out`` (default: standard
output).

    python3 scripts/collect_results.py results --out results/RESULTS.md      # the tables of REPORT.md
    python3 scripts/collect_results.py runs --index-only                      # index of runs/**/result.json

``results/`` holds the per-run summaries of the release experiments, one directory per experiment group and run
(``results/<group>/<run>/``: ``result.json``, ``result_v1.json``, ``robustness_*.json``, ``metric_recovery_*.json``,
``rollout_*.json``, ``eval_*.json``, ``transfer_*.json``, ...; the groups are listed in ``results/README.md``), and
``results/v1_paper/all_metrics.json`` (the paper's own v1 metrics, seed 42).  Every number of the tables below is read
from these files; a missing file gives ``-``.

R2 conventions (``rhmp.metrics``; recorded as ``r2_definition`` in each ``result.json``): centred ``1 - SS_res / SS_tot``
on normalised targets (v1 formula) for scalar and vector targets; *uncentred* ``1 - SS_res / sum t^2`` for
orientation-odd cochain targets (T6f, T7f, HPflux*, TETflux*), marked with a dagger.  ``test`` is the full test split,
``test100`` the first 100 test samples (the v1 paper tables), ``4x`` the zero-shot 4x-finer test set, ``geo`` / ``topo``
the SURF geometry / topology transfer sets.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Any

DAGGER = "†"


# ================================================================================================================
# helpers
# ================================================================================================================
class Results:
    def __init__(self, root: str) -> None:
        self.root = root
        self._cache: dict[str, Any] = {}

    def load(self, rel: str) -> dict | None:
        """JSON file ``root/rel`` (``None`` if missing or unreadable)."""
        if rel not in self._cache:
            p = os.path.join(self.root, rel)
            try:
                with open(p) as f:
                    self._cache[rel] = json.load(f)
            except (OSError, ValueError):
                self._cache[rel] = None
        return self._cache[rel]

    def run(self, rel_dir: str, name: str = "result.json") -> dict | None:
        return self.load(os.path.join(rel_dir, name))


def get(d: Any, *path: Any, default: Any = None) -> Any:
    for p in path:
        if isinstance(d, dict) and p in d:
            d = d[p]
        elif isinstance(d, list) and isinstance(p, int) and -len(d) <= p < len(d):
            d = d[p]
        else:
            return default
    return d


def r2(res: dict | None, key: str = "test") -> float | None:
    v = get(res, key)
    if isinstance(v, dict):
        return v.get("R2")
    if key == "test":
        return get(res, "test_R2")
    return None


def f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(x):
        return "nan"
    if abs(x) >= 1e4 or (0 < abs(x) < 0.5 * 10 ** (-nd)):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def fe(x: Any) -> str:
    """Signed scientific notation (symmetry deviations)."""
    if x is None:
        return "-"
    try:
        return f"{float(x):+.1e}"
    except (TypeError, ValueError):
        return str(x)


def fk(n: Any) -> str:
    """Parameter count as 1.2K / 90K / 0.43M."""
    if n is None:
        return "-"
    n = int(n)
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.0f}K" if n >= 10_000 else f"{n / 1000:.1f}K"
    return f"{n / 1e6:.2f}M"


def odd(res: dict | None) -> str:
    return DAGGER if res and "uncentred" in str(res.get("r2_definition", "")) else ""


def table(header: list[str], rows: list[list[Any]], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


# ================================================================================================================
# sections
# ================================================================================================================
PAPER = [  # task, v1 dir, native dir, legacy dir, [(variant label, dir)], note
    ("T1 vorticity", "paper_tasks/T1_v1_s42", "paper_tasks/T1_native_s42", "paper_tasks/T1_legacy_s42",
     [("+ latent 1:8", "paper_tasks/T1_native_latent8_s42"), ("T1q (quad CW complex)", "new_tasks/T1q_s42")],
     "legacy input = frame components (not representable by an invariant model)"),
    ("T2 torus transport", "paper_tasks/T2_v1_s42", "paper_tasks/T2_native_s42", None,
     [("+ latent 1:8", "paper_tasks/T2_native_latent8_s42"), ("+ latent 1:16", "paper_tasks/T2_native_latent16_s42"),
      ("+ latent 1:8 + resolvent", "paper_tasks/T2_native_latent8_res_s42")],
     "the data hide a fixed velocity field; latent edge columns are explicit per-cell parameters"),
    ("T3 ellipsoid flow", "paper_tasks/T3_v1_s42", "paper_tasks/T3_native_s42", "paper_tasks/T3_legacy_s42", [],
     "least-squares vector readout at its round-trip ceiling"),
    ("T5 electrostatics", "paper_tasks/T5_v1_s42", "paper_tasks/T5_native_s42", "paper_tasks/T5_legacy_s42",
     [("+ resolvent", "paper_tasks/T5_native_res_s42"), ("T5g (grad readout), poly", "paper_tasks/T5g_native_poly_s42"),
      ("T5g, resolvent", "paper_tasks/T5g_native_res_s42"),
      ("T5g solver mode, frozen tensor", "paper_tasks/T5g_solver_frozen_tensor_s42"),
      ("T5g solver mode, learned tensor", "paper_tasks/T5g_solver_learn_tensor_s42"),
      ("T5g solver mode, learned diagonal", "paper_tasks/T5g_solver_learn_diag_s42")],
     "target = v1 edge->node average of -d0 phi; generator has negative-weight edges (tensor metric only)"),
    ("T6 Wilson loop", "paper_tasks/T6_v1_s42", "paper_tasks/T6_native_s42", "paper_tasks/T6_legacy_s42",
     [("T6f (face target)" + DAGGER, "new_tasks/T6f_s42")], ""),
    ("T7 Yang-Mills SU(2)", "paper_tasks/T7_v1_s42", "paper_tasks/T7_native_s42", "paper_tasks/T7_legacy_s42",
     [("T7f (face target)" + DAGGER, "new_tasks/T7f_s42")], ""),
    ("T8 airfoil pressure", "paper_tasks/T8_v1_s42", "paper_tasks/T8_native_s42", None,
     [("T8v (+ inflow 1-form)", "paper_tasks/T8v_native_s42"), ("T8v + abs. scale", "paper_tasks/T8v_native_abs_s42"),
      ("T8v + resolvent", "paper_tasks/T8v_native_res_s42")],
     "v1 used absolute-coordinate features and selected on the test split; v2 is strictly E(n)-invariant"),
]


def sec_paper(R: Results) -> str:
    rows, det = [], []
    for task, v1d, nat, leg, variants, note in PAPER:
        v1 = R.run(v1d, "result_v1.json")
        n, l_ = R.run(nat), (R.run(leg) if leg else None)
        var = "; ".join(f"{lab} {f(r2(R.run(d), 'test100'), 3)}" for lab, d in variants if R.run(d))
        rows.append([task, f(r2(v1, "test100"), 3), f(r2(n, "test100"), 3),
                     f(r2(l_, "test100"), 3) if leg else "= native", var or "-", note or "-"])
        for lab, d in [("v1 (re-evaluated)", v1d)] + [("v2 native", nat)] + ([("v2 legacy", leg)] if leg else []) + \
                list(variants):
            res = R.run(d, "result_v1.json") if lab.startswith("v1") else R.run(d)
            if res is None:
                continue
            det.append([task.split()[0], lab, f(r2(res)), f(r2(res, "test100")), f(get(res, "test", "SSIM")),
                        f(get(res, "test", "Pearson")), f(get(res, "test", "NRMSE")), fk(res.get("params")),
                        str(res.get("best_epoch", "-")), f"`{d}`"])
    s = ["## 1. Paper tasks (v1 protocol: 100 epochs, seed 42, sequential 70/15/15 split, best by validation R2)\n",
         "test100 R2: the first 100 test samples, as in the tables of the v1 paper.  *v1 (re-evaluated)*: the v1 "
         "checkpoints, evaluated by `python -m rhmp.train --eval-v1` with the same split and metric code.\n",
         table(["task", "v1 (re-evaluated)", "v2 native", "v2 legacy", "v2 variants", "note"], rows, "lrrrll"),
         "\nAll runs of the table (full test split and test100):\n",
         table(["task", "run", "test R2", "test100 R2", "SSIM", "Pearson", "NRMSE", "params", "best ep.", "result"],
               det, "llrrrrrrrl")]
    return "\n".join(s)


NEW = [  # label, dir, note
    ("T1q (quad CW complex)", "new_tasks/T1q_s42", ""),
    ("T6f (edge connection -> face flux)", "new_tasks/T6f_s42", ""),
    ("T7f (SU(2) edge -> face)", "new_tasks/T7f_s42", ""),
    ("HP_k10", "new_tasks/HP_k10_s42", ""),
    ("HP_k100", "new_tasks/HP_k100_s42", ""),
    ("HP_k1000", "new_tasks/HP_k1000_s42", ""),
    ("HP_k100_aniso", "new_tasks/HP_k100_aniso_s42", ""),
    ("HPflux_k100 (edge flux)", "new_tasks/HPflux_k100_s42", "flux halves under refinement; see HPfluxd"),
    ("HPfluxd_k100 (flux density)", "new_tasks/HPfluxd_k100_s42", ""),
    ("TET_k100 (3-D tetrahedra)", "new_tasks/TET_k100_s42", ""),
    ("TETflux_k100 (face flux)", "new_tasks/TETflux_k100_s42", ""),
    ("SURF (variable closed surfaces)", "extension_suite/SURF_s42", ""),
    ("SURF_heat", "extension_suite/SURF_heat_s42", ""),
    ("DYN (one step, variable meshes)", "extension_suite/DYN_s42", "rollouts: section 8"),
    ("DYNfix (one step, fixed mesh)", "extension_suite/DYNfix_s42", "rollouts: section 8"),
    ("DYNfix_cons (exactly conservative map)", "extension_suite/DYNfix_cons_s42", "one-step du target"),
    ("DYN_cons (first convention, M du)", "extension_suite/DYN_cons_s42", "diverges in rollouts"),
]


def sec_new(R: Results) -> str:
    rows = []
    for lab, d, note in NEW:
        res = R.run(d)
        if res is None:
            rows.append([lab, "-", "-", "-", "-", "-", "-", note or "-", f"`{d}`"])
            continue
        extra = ", ".join(f"{k} {f(r2(res, k))}" for k in ("geo", "topo") if isinstance(res.get(k), dict))
        rows.append([lab + odd(res), f(r2(res)), f(r2(res, "test100")) if isinstance(res.get("test100"), dict) else "-",
                     f(r2(res, "fine")) if isinstance(res.get("fine"), dict) else "-", extra or "-",
                     fk(res.get("params")), str(res.get("epochs_run", "-")), note or "-", f"`{d}`"])
    return "\n".join(["## 2. New tasks and the extension suite (v2 defaults, 100 epochs, seed 42)\n",
                      f"{DAGGER} uncentred R2 (orientation-odd cochain target).  4x: zero-shot test on 4x finer "
                      "meshes.\n",
                      table(["task", "test R2", "test100 R2", "4x R2", "transfer sets", "params", "epochs", "note",
                             "result"], rows, "lrrrlrrll")])


VARIANTS = [  # label, dir
    ("diag + poly (default)", "metric_variants/HP_k100_diag-poly_s42"),
    ("tensor (cone) + poly", "metric_variants/HP_k100_tensor-poly_s42"),
    ("tensor (full SPD) + poly", "metric_variants_full_spd/HP_k100_tensor-poly_s42"),
    ("diag + resolvent", "metric_variants/HP_k100_diag-resolvent_s42"),
    ("tensor (cone) + resolvent", "metric_variants/HP_k100_tensor-resolvent_s42"),
    ("tensor (full SPD) + resolvent", "metric_variants_full_spd/HP_k100_tensor-resolvent_s42"),
    ("solver mode, frozen tensor metric + face sigma reference",
     "material_identification/HP_k100_fem_frozen_tensor_faceref"),
    ("solver mode, frozen diagonal metric + edge sigma reference",
     "material_identification/HP_k100_dec_frozen_diag_edgeref"),
    ("solver mode, learned tensor metric (material -> metric only)",
     "material_identification/HP_k100_solver_tensor_learn"),
    ("solver mode, learned diagonal metric (material -> metric only)",
     "material_identification/HP_k100_solver_diag_learn"),
    ("material-only routing + resolvent", "material_identification/HP_k100_material_res"),
    ("material-only routing + edge reference + resolvent", "material_identification/HP_k100_material_ref_res"),
    ("aux PDE-residual loss (w = 1) + resolvent", "material_identification/HP_k100_auxpde_res"),
    ("frozen diagonal + true edge reference + resolvent", "material_identification/HP_k100_frozen_ref_res"),
    ("pure DEC (frozen, no reference) + resolvent", "material_identification/HP_k100_frozen_noref_res"),
]
VARIANTS_OTHER = [
    ("HP_k1000 diag + poly", "high_contrast/HP_k1000_noref_s42"),
    ("HP_k1000 diag + poly + edge reference", "high_contrast/HP_k1000_ref_s42"),
    ("HP_k1000 diag + resolvent + edge reference", "high_contrast/HP_k1000_ref_res_s42"),
    ("HP_k10000 diag + poly", "high_contrast/HP_k10000_noref_s42"),
    ("HP_k10000 diag + poly + edge reference", "high_contrast/HP_k10000_ref_s42"),
    ("HP_k100_aniso100 diag + poly", "metric_variants/HP_k100_aniso100_diag-poly_s42"),
    ("HP_k100_aniso100 diag + resolvent", "metric_variants/HP_k100_aniso100_diag-resolvent_s42"),
    ("HP_k100_aniso100 diag + poly + edge reference", "high_contrast/HP_k100_aniso100_ref_s42"),
    ("HP_k100_aniso100 tensor (cone) + poly", "metric_variants/HP_k100_aniso100_tensor-poly_s42"),
    ("HP_k100_aniso100 tensor (full) + poly", "metric_variants_full_spd/HP_k100_aniso100_tensor-poly_s42"),
    ("HP_k100_aniso100 tensor (cone) + resolvent", "metric_variants/HP_k100_aniso100_tensor-resolvent_s42"),
    ("HP_k100_aniso100 tensor (full) + resolvent", "metric_variants_full_spd/HP_k100_aniso100_tensor-resolvent_s42"),
    ("TET_k100 diag + poly", "metric_variants/TET_k100_diag-poly_s42"),
    ("TET_k100 diag + resolvent", "metric_variants/TET_k100_diag-resolvent_s42"),
    ("TET_k100 (1500 samples) diag + poly", "metric_variants_tet1500/TET_k100_diag-poly_s42"),
    ("TET_k100 (1500 samples) diag + resolvent", "metric_variants_tet1500/TET_k100_diag-resolvent_s42"),
    ("TET_k100 (1500 samples) tensor + resolvent", "metric_variants_tet1500/TET_k100_tensor-resolvent_s42"),
    ("T6f diag + poly", "metric_variants/T6f_diag-poly_s42"),
    ("T6f diag + resolvent", "metric_variants/T6f_diag-resolvent_s42"),
    ("T6f tensor + poly", "metric_variants/T6f_tensor-poly_s42"),
    ("T6f tensor + resolvent", "metric_variants/T6f_tensor-resolvent_s42"),
]


def _variant_rows(R: Results, spec) -> list:
    rows = []
    for lab, d in spec:
        res = R.run(d)
        if res is None:
            rows.append([lab, "-", "-", "-", "-", f"`{d}` (missing)"])
            continue
        rows.append([lab + odd(res), f(r2(res)), f(r2(res, "fine")) if isinstance(res.get("fine"), dict) else "-",
                     fk(res.get("params")), str(res.get("epochs_run", "-")), f"`{d}`"])
    return rows


def sec_variants(R: Results) -> str:
    return "\n".join([
        "## 3. Metric and layer variants\n",
        "HP_k100 with the inputs f and log sigma (on edges and faces); `--log-range 3` for the material-identification "
        "runs.  General-stack runs are trained for 50 epochs, solver-mode runs for 30 epochs (the frozen FEM control "
        "for 5).\n",
        table(["variant (HP_k100)", "test R2", "4x R2", "params", "epochs", "result"], _variant_rows(R, VARIANTS),
              "lrrrrl"),
        "\nOther tasks (50 epochs):\n",
        table(["variant", "test R2", "4x R2", "params", "epochs", "result"], _variant_rows(R, VARIANTS_OTHER),
              "lrrrrl"),
        "\nThe tensor-metric TET runs on all 3000 samples and the 1500-sample tensor + poly run ran out of host "
        "memory (Whitney blocks) and are not listed; see the limitations in REPORT.md.\n"])


# ----------------------------------------------------------------------------------------------------------------
def _recovery_rows(R: Results, dirs: list[str]) -> list:
    rows = []
    for d in dirs:
        js = sorted(glob.glob(os.path.join(R.root, d, "metric_recovery_*.json")))
        for jp in js:
            try:
                res = json.load(open(jp))
            except ValueError:
                continue
            layers = res.get("layers", [])
            edge = [x.get("diag_logratio_vs_edge") or x.get("tensor_action_vs_edge") for x in layers]
            face = [x.get("tensor_halflogdet_vs_face") for x in layers]

            def best(cs):
                cs = [c for c in cs if c and c.get("pearson") is not None and math.isfinite(c["pearson"])]
                return max(cs, key=lambda c: abs(c["pearson"])) if cs else None
            be, bf = best(edge), best(face)
            refmat = ", ".join([f"ref {k}:{v}" for k, v in (res.get("metric_reference") or {}).items()] +
                               [f"mat {k}:{v}" for k, v in (res.get("material_dims") or {}).items()]) or "-"
            ratio = " / ".join(f"{x['tensor_anisotropy']['median']:.2f}" for x in layers if "tensor_anisotropy" in x)

            def fr(c):
                return "-" if not c or c.get("pearson") is None else f"{c['pearson']:+.3f}"

            def sl(c):
                return "-" if not c else f"{c.get('slope', float('nan')):.3f} / {c.get('intercept', float('nan')):+.3f}"
            rows.append([os.path.basename(os.path.normpath(d)), res.get("metric_type", "-"),
                         ",".join(res.get("layer_types", [])), refmat, f(res.get("test_R2")),
                         " / ".join(fr(c) for c in edge), sl(be),
                         " / ".join(fr(c) for c in face) if any(face) else "-", sl(bf), ratio or "-"])
    return rows


def sec_recovery(R: Results) -> str:
    dirs = sorted({os.path.dirname(os.path.relpath(p, R.root))
                   for p in glob.glob(os.path.join(R.root, "*", "*", "metric_recovery_*.json"))})
    order = ["material_identification", "metric_variants_full_spd", "metric_variants", "anisotropy"]
    # the material-identification runs in a fixed order (learned metrics before the frozen controls), the others
    # alphabetically
    runs = ["HP_k100_material_res", "HP_k100_material_ref_res", "HP_k100_solver_diag_learn",
            "HP_k100_solver_tensor_learn", "HP_k100_auxpde_res", "HP_k100_dec_frozen_diag_edgeref",
            "HP_k100_fem_frozen_tensor_faceref", "HP_k100_frozen_ref_res", "HP_k100_frozen_noref_res"]

    def rank(d: str) -> tuple:
        run = os.path.basename(d)
        return (next((i for i, o in enumerate(order) if f"/{o}/" in f"/{d}/"), 9),
                runs.index(run) if run in runs else len(runs), d)
    dirs.sort(key=rank)
    rows = _recovery_rows(R, dirs)
    return "\n".join([
        "## 4. Metric recovery: learned metric vs true material (first 32 test samples, physical units)\n",
        "r: pooled Pearson correlation per layer, for diagonal metrics between `log(H_1/star_1)` and the true edge "
        "`log sigma`, for tensor metrics between the edge action `log mean_f t_e^T sigma_f t_e` and the edge "
        "`log sigma` and between the face `log det(sigma_f)/2` and the face `log sigma` (tensor sets: half the log "
        "determinant of the true tensor).  Slope and intercept: least-squares fit `learned = slope * true + intercept` "
        "for the layer with the largest |r| (slope 1 and intercept 0: the metric equals the material in physical "
        "units).  Generated by `scripts/metric_recovery.py`.\n",
        table(["run", "metric", "layers", "ref / material", "test R2", "edge r per layer", "edge slope / icpt",
               "face r per layer", "face slope / icpt", "anisotropy ratio median"], rows, "llllrlllll")])


# ----------------------------------------------------------------------------------------------------------------
ANISO_SOLVER = [  # task, diag dir, tensor dir
    ("AHP_r10", "anisotropy/AHP_r10_diag-solver_s42", "anisotropy/AHP_r10_tensor-solver_s42"),
    ("AHP_r100", "anisotropy/AHP_r100_diag-solver_s42", "anisotropy/AHP_r100_tensor-solver_lr3e-4_s42"),
    ("AHP_r100 (tensor lr 1e-3, 30 ep.)", "anisotropy/AHP_r100_diag-solver_s42",
     "anisotropy/AHP_r100_tensor-solver_s42"),
    ("ASURF_r100 (surfaces)", "anisotropy_surfaces/ASURF_r100_diag-solver_s42",
     "anisotropy_surfaces/ASURF_r100_tensor-solver_s42"),
    ("ADARCYp_r100 (3-D, 1500 samples)", "anisotropy_3d/ADARCYp_r100_n1500_diag-solver_s42",
     "anisotropy_3d/ADARCYp_r100_n1500_tensor-solver_s42"),
]
ANISO_GENERAL = [
    ("AHP_r100 diag + ref", "anisotropy_general/AHP_r100_diag+ref_s42"),
    ("AHP_r100 tensor + ref", "anisotropy_general/AHP_r100_tensor+ref_s42"),
    ("AHP_r100 MeshGraphNet", "anisotropy_general/AHP_r100_mgn_s42"),
    ("ASURF_r100 diag + ref", "anisotropy_general/ASURF_r100_diag+ref_s42"),
    ("ASURF_r100 tensor + ref", "anisotropy_general/ASURF_r100_tensor+ref_s42"),
    ("ASURF_r100 dec_fixed (frozen DEC star)", "anisotropy_general/ASURF_r100_dec_fixed_s42"),
    ("ACURLb_r10 (3-D curl-curl, 1500) diag + ref", "anisotropy_general/ACURLb_r10_n1500_diag+ref_s42"),
    ("ACURLb_r10 (3-D curl-curl, 1500) tensor + ref", "anisotropy_general/ACURLb_r10_n1500_tensor+ref_s42"),
]


def sec_aniso(R: Results) -> str:
    out = ["## 5. Anisotropy family (AHP, ASURF, ACURL, ADARCY)\n"]
    rep = R.load("anisotropy_oracle/all.json")
    if rep:
        def m(task, key):
            return get(rep, task, "oracle_rel_err", key, "mean")
        rows = [["AHP r100 (u)", f(m("AHP_r100", "diag_ref"), 3), f(m("AHP_r100", "cone"), 3),
                 f(m("AHP_r100", "full_L5"), 3)],
                ["ACURL r100 (A / B)", f"{f(m('ACURL_r100', 'diag_ref'), 3)} / {f(m('ACURL_r100', 'diag_ref_B'), 3)}",
                 f"{f(m('ACURL_r100', 'cone'), 3)} / {f(m('ACURL_r100', 'cone_B'), 3)}",
                 f"{f(m('ACURL_r100', 'full_L5'), 3)} / {f(m('ACURL_r100', 'full_L5_B'), 3)}"],
                ["ADARCY r100 (face flux J)", f(m("ADARCY_r100", "J_diag_ref"), 3), f(m("ADARCY_r100", "J_cone"), 3),
                 f(m("ADARCY_r100", "J_full_L5"), 3)]]
        cone = [[t, f(get(rep, t, "dataset", "cone_frac", "mean"), 3)] for t in rep
                if get(rep, t, "dataset", "cone_frac", "mean") is not None]
        out += ["Oracle representability: relative L2 solution error when the true tensors are replaced by the best "
                "member of each metric family (12 test samples; `datasets/generators/gen_aniso.py --analyse 12`, "
                "`results/anisotropy_oracle/all.json`).\n",
                table(["task", "diagonal star (TPFA)", "cone tensor", "full tensor (s bounded by 5)"], rows, "lrrr"),
                "\nFraction of cells whose true tensor lies in the cone parameterisation:\n",
                table(["task", "cone fraction"], cone, "lr")]
    rows = []
    for task, dd, td in ANISO_SOLVER:
        a, b = R.run(dd), R.run(td)
        fam = ""
        if b and any(k.startswith("test_") and isinstance(b[k], dict) for k in b):
            fam = ", ".join(f"{k[5:]} {f(r2(b, k), 3)}" for k in sorted(b) if k.startswith("test_") and
                            isinstance(b[k], dict))
        rows.append([task, f(r2(a)), f(r2(a, "fine")), fk(get(a, "params")), f(r2(b)), f(r2(b, "fine")),
                     fk(get(b, "params")), fam or "-"])
    out += ["\nSolver mode (the material enters only through the metric; 30 epochs; AHP_r100 tensor: lr 3e-4, 60 "
            "epochs):\n",
            table(["task", "diagonal test R2", "diagonal 4x R2", "params", "tensor test R2", "tensor 4x R2", "params",
                   "tensor per-family test"], rows, "lrrrrrrl")]
    rows = []
    for lab, d in ANISO_GENERAL:
        res = R.run(d)
        fam = ", ".join(f"{k[5:]} {f(r2(res, k), 3)}" for k in sorted(res or {}) if k.startswith("test_") and
                        isinstance(res[k], dict)) if res else ""
        rows.append([lab, f(r2(res)), f(r2(res, "fine")), fk(get(res, "params")), str(get(res, "epochs_run", default="-")),
                     fam or "-"])
    out += ["\nGeneral stack (polynomial layers; `+ ref`: with the metric reference; 50 epochs) and baselines:\n",
            table(["run", "test R2", "4x R2", "params", "epochs", "per-family test"], rows, "lrrrrl")]
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------------
BL_MODELS = ["gem_cnn", "gauge_cnn", "schnet", "cw_net", "egnn", "gat", "mpsn", "sccnn", "clifford_smpn", "gcn",
             "mgn", "ours_v1", "dec_fixed"]
BL_NAMES = {"gem_cnn": "GEM-CNN", "gauge_cnn": "GaugeEquivCNN", "schnet": "SchNet", "cw_net": "CW Net", "egnn": "EGNN",
            "gat": "GAT", "mpsn": "MPSN", "sccnn": "SCCNN", "clifford_smpn": "Clifford-SMPN", "gcn": "GCN",
            "mgn": "MeshGraphNet", "ours_v1": "v1 model (ours_v1)", "dec_fixed": "dec_fixed (v2, frozen DEC star)"}


def _paper_v1(R: Results, task: str, model: str) -> float | None:
    pm = R.load("v1_paper/all_metrics.json") or {}
    return get(pm, task, model, "R2")


def sec_baselines(R: Results) -> str:
    out = ["## 6. Baselines (common trainer, v1 protocol, 100 epochs, seed 42)\n",
           "v2 budget: parameter-matched to the v2 model (about 90K on T6 / HP / SURF, 0.31M on T8); v1 budget: about "
           "0.43M (the parameter count of the v1 model).  MPSN / SCCNN / Clifford-SMPN use the *v1 edge protocol* on "
           "T3/T6 (edge-averaged targets, edge-space R2, as in the v1 paper).  paper: the numbers reported in the v1 "
           "paper (`results/v1_paper/all_metrics.json`).  All values are R2 on the full test split.\n"]
    rows = []
    for m in BL_MODELS:
        a = R.run(f"baselines/T6_legacy/{m}_s42")
        b = R.run(f"baselines_v1_budget/T6_legacy/{m}_s42")
        c = R.run(f"baselines/T3_legacy/{m}_s42")
        n = R.run(f"baselines/T6_native/{m}_s42")
        nv1 = R.run(f"baselines_v1_budget/T6_native/{m}_s42")
        t3n = R.run(f"baselines/T3_native/{m}_s42")
        if not any((a, b, c, n, nv1, t3n)):
            continue
        rows.append([BL_NAMES[m], f(r2(a), 3), f(r2(b), 3), f(_paper_v1(R, "T6", m), 3), f(r2(c), 3),
                     f(r2(n), 3), f(r2(nv1), 3), f(r2(t3n), 3)])
    v2 = [R.run("paper_tasks/T6_legacy_s42"), R.run("paper_tasks/T3_native_s42"), R.run("paper_tasks/T6_native_s42"),
          R.run("paper_tasks/T3_legacy_s42")]
    rows.insert(0, ["RHMP v2", f(r2(v2[0]), 3), "-", f(_paper_v1(R, "T6", "ours"), 3) + " (v1)",
                    f(r2(v2[3]), 3) + " (native " + f(r2(v2[1]), 3) + ")", f(r2(v2[2]), 3), "-", f(r2(v2[1]), 3)])
    out += ["### T6 and T3\n",
            table(["model", "T6 legacy (v2 budget)", "T6 legacy (v1 budget)", "T6 paper", "T3 legacy",
                   "T6 native (v2 budget)", "T6 native (v1 budget)", "T3 native"], rows, "lrrrrrrr")]
    # HP_k100
    rows = []
    v2 = [("RHMP v2 general stack, diag + poly (100 ep.)", "new_tasks/HP_k100_s42"),
          ("RHMP v2 general stack, diag + resolvent (50 ep.)", "metric_variants/HP_k100_diag-resolvent_s42"),
          ("RHMP v2 solver mode, learned diagonal (30 ep.)", "material_identification/HP_k100_solver_diag_learn"),
          ("RHMP v2 solver mode, learned tensor (30 ep.)", "material_identification/HP_k100_solver_tensor_learn")]
    for lab, d in v2:
        res = R.run(d)
        rows.append([lab, f(r2(res), 4), f(r2(res, "fine"), 4), fk(get(res, "params")), "-", "-", "-"])
    for m in BL_MODELS:
        a = R.run(f"baselines/HP_k100_native/{m}_s42")
        b = R.run(f"baselines_v1_budget/HP_k100_native/{m}_s42")
        if not (a or b):
            continue
        rows.append([BL_NAMES[m], f(r2(a), 3), f(r2(a, "fine"), 3), fk(get(a, "params")), f(r2(b), 3),
                     f(r2(b, "fine"), 3), fk(get(b, "params"))])
    out += ["\n### HP_k100 (variable meshes; 4x = zero-shot 4x resolution)\n",
            table(["model", "test R2 (v2 budget)", "4x R2", "params", "test R2 (v1 budget)", "4x R2", "params"], rows,
                  "lrrrrrr")]
    # SURF
    rows = []
    res = R.run("extension_suite/SURF_s42")
    rows.append(["RHMP v2", f(r2(res), 4), f(r2(res, "geo"), 4), f(r2(res, "topo"), 4), fk(get(res, "params"))])
    for m in BL_MODELS:
        a = R.run(f"baselines/SURF_native/{m}_s42")
        if a:
            rows.append([BL_NAMES[m], f(r2(a), 4), f(r2(a, "geo"), 4), f(r2(a, "topo"), 4), fk(get(a, "params"))])
    out += ["\n### SURF (closed surfaces; geometry transfer to superquadrics, topology transfer to genus 2)\n",
            table(["model", "test R2", "geo R2", "topo R2", "params"], rows, "lrrrr")]
    # T8
    rows = []
    for lab, d in [("RHMP v2 (T8 native)", "paper_tasks/T8_native_s42"), ("RHMP v2 (T8v)", "paper_tasks/T8v_native_s42")]:
        res = R.run(d)
        rows.append([lab, f(r2(res), 3), f(r2(res, "test100"), 3), "-", fk(get(res, "params"))])
    v1 = R.run("paper_tasks/T8_v1_s42", "result_v1.json")
    rows.append(["v1 model (re-evaluated checkpoint)", f(r2(v1), 3), f(r2(v1, "test100"), 3),
                 f(_paper_v1(R, "T8", "ours"), 3), fk(get(v1, "params"))])
    for m in BL_MODELS:
        a = R.run(f"baselines/T8_native/{m}_s42")
        if a:
            rows.append([BL_NAMES[m], f(r2(a), 3), f(r2(a, "test100"), 3), f(_paper_v1(R, "T8", m), 3),
                         fk(get(a, "params"))])
    out += ["\n### T8 (AirfRANS airfoil pressure; v2 budget about 0.31M)\n",
            table(["model", "test R2", "test100 R2", "paper", "params"], rows, "lrrrr")]
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------------
def sec_transfer(R: Results) -> str:
    out = ["## 7. Transfer, symmetry and robustness\n"]
    rows = []
    base = R.run("new_tasks/T6f_s42")
    rows.append(["training mesh (test split)", "1024", f(r2(base), 4)])
    for tag, lab in (("mesh7_n1024", "new random mesh (seed 7)"), ("mesh11_n1024", "new random mesh (seed 11)"),
                     ("mesh7_n4096", "4x finer new mesh"), ("mesh7_n256", "4x coarser new mesh")):
        res = R.load(f"mesh_transfer/T6f_s42/transfer_T6f_{tag}.json")
        rows.append([lab, str(get(res, "n_pts", default="-")), f(r2(res), 4)])
    out += ["### T6f zero-shot mesh / resolution transfer (`scripts/t6_mesh_transfer.py`; 200 test samples per mesh, "
            f"uncentred R2{DAGGER})\n", table(["evaluation mesh", "vertices", "R2"], rows, "lrr")]
    # robustness
    rob = [("T6f", "new_tasks/T6f_s42/robustness_T6f.json"), ("T3", "paper_tasks/T3_native_s42/robustness_T3.json"),
           ("HP_k100 solver tensor", "material_identification/HP_k100_solver_tensor_learn/robustness_HP_k100.json"),
           ("HP_k100 diag + resolvent", "metric_variants/HP_k100_diag-resolvent_s42/robustness_HP_k100.json"),
           ("T1q", "new_tasks/T1q_s42/robustness_T1q.json"), ("T7f", "new_tasks/T7f_s42/robustness_T7f.json")]
    conds = ["batch=1", "relabel", "flip-orient", "rotate", "reflect", "gauge=0.1", "gauge=1", "gauge=10",
             "noise=0.01", "noise=0.1"]
    rows = []
    for lab, p in rob:
        res = R.load(p)
        if not res:
            continue
        d = {x["condition"]: x for x in res.get("rows", [])}
        rows.append([lab, f(get(d, "base", "R2"), 6)] +
                    [fe(get(d, c, "dR2")) if c in d else "-" for c in conds])
    out += ["\n### Symmetry and robustness of trained models (300 test samples; dR2 relative to the base R2; "
            "`scripts/eval_robustness.py RUN --all`)\n",
            table(["model", "base R2"] + [f"dR2 {c}" for c in conds], rows, "l" + "r" * (1 + len(conds)))]
    # mesh-quality shift
    rows = []
    for lab, d in (("diag + poly (50 ep.)", "metric_variants/HP_k100_diag-poly_s42"),
                   ("tensor + poly (50 ep.)", "metric_variants/HP_k100_tensor-poly_s42")):
        for kind in ("sliver_ref", "graded_ref"):
            res = R.run(d, f"eval_HP_qual_{kind}.json")
            if not res:
                continue
            lv = res.get("levels", [])
            rows.append([lab, kind] + [f"{k} {f(get(res, k, 'R2'), 3)}" for k in lv])
    width = max((len(r) for r in rows), default=2)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out += ["\n### Mesh-quality shift (HP_k100 models on sliver / graded meshes, 100 instances per level, targets "
            "= 4x-finer reference; `scripts/eval_on.py --task HP_qual_*_ref`)\n",
            table(["model", "set"] + [f"level {i}" for i in range(width - 2)], rows, "ll" + "l" * (width - 2))]
    rows = []
    for lab, p in (("short 2-epoch run", "mesh_quality/eval_HP_qual_graded.json"),
                   ("diag + poly (50 ep.)", "metric_variants/HP_k100_diag-poly_s42/eval_HP_qual_graded.json"),
                   ("tensor + poly (50 ep.)", "metric_variants/HP_k100_tensor-poly_s42/eval_HP_qual_graded.json")):
        res = R.load(p)
        if res:
            rows.append([lab, f(get(res, "base", "R2"), 3), f(get(res, "corner_r64", "R2"), 3),
                         f(get(res, "base", "relL2M_pooled"), 3), f(get(res, "corner_r64", "relL2M_pooled"), 3)])
    out += ["\nCorner refinement (graded meshes): the node-uniform R2 drops because the refined nodes lie where u "
            "is about 0; the pooled mass-weighted relative L2 error measures the field error:\n",
            table(["model", "R2 base", "R2 corner_r64", "mass-weighted rel. L2 base", "corner_r64"], rows, "lrrrr")]
    return "\n".join(out)


def sec_rollout(R: Results) -> str:
    rows = []
    specs = [("DYNfix, mass projection", "extension_suite/DYNfix_s42/rollout_DYNfix_proj.json"),
             ("DYNfix, no projection", "extension_suite/DYNfix_s42/rollout_DYNfix.json"),
             ("DYN (variable meshes), no projection", "extension_suite/DYN_s42/rollout_DYN.json"),
             ("DYN, mass projection", "extension_suite/DYN_s42/rollout_DYN_proj.json"),
             ("DYNfix_cons (exactly conservative map)", "extension_suite/DYNfix_cons_s42/rollout_DYNfix_cons.json"),
             ("DYN_cons (first convention, M du)", "extension_suite/DYN_cons_s42/rollout_DYN_cons.json")]
    hs = (1, 5, 10, 25, 50, 100)
    pers = None
    for lab, p in specs:
        res = R.load(p)
        if not res:
            continue
        s = res.get("summary", {})
        rows.append([lab] + [f(s.get(f"R2@{h}"), 3) for h in hs] +
                    [f(s.get("mass_drift@100"), 2), str(res.get("first_nonfinite_step") or "-")])
        if pers is None and "DYNfix" in lab:
            pers = [f(s.get(f"persistence_R2@{h}"), 3) for h in hs]
    if pers:
        rows.append(["persistence (u_{t+k} = u_t), DYNfix"] + pers + ["0", "-"])
    return "\n".join(["## 8. Autoregressive rollouts (DYN / DYNfix, 100 steps, full test trajectories)\n",
                      "Written by `scripts/eval_on.py --run RUN --task DYN[fix] --rollout [--project-mass]`.\n",
                      table(["model"] + [f"R2@{h}" for h in hs] + ["mass drift @100", "first non-finite step"], rows,
                            "l" + "r" * (len(hs) + 2))])


def sec_index(R: Results, root_is_results: bool = True) -> str:
    rows = []
    for p in sorted(glob.glob(os.path.join(R.root, "**", "result.json"), recursive=True)):
        rel = os.path.relpath(os.path.dirname(p), R.root)
        if rel.startswith("checkpoints"):
            continue
        try:
            res = json.load(open(p))
        except ValueError:
            continue
        rows.append([f"`{rel}`", str(res.get("task", "")), str(res.get("model", "rhmp")), str(res.get("mode", "")),
                     f(r2(res)) + odd(res), f(r2(res, "test100")) if isinstance(res.get("test100"), dict) else "-",
                     f(r2(res, "fine")) if isinstance(res.get("fine"), dict) else "-", fk(res.get("params")),
                     str(res.get("epochs_run", "-")), f(res.get("s_per_epoch"), 1), f(res.get("peak_mem_GB"), 2)])
    return "\n".join(["## Appendix: index of all runs\n",
                      table(["run", "task", "model", "mode", "test R2", "test100 R2", "4x R2", "params", "epochs",
                             "s/epoch", "peak GB"], rows, "lllrrrrrrrr")])


# ================================================================================================================
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="results", help="results directory (or a runs/ directory)")
    ap.add_argument("--out", default=None, help="Markdown file to write (default: standard output)")
    ap.add_argument("--index-only", action="store_true", help="only the index of all result.json files")
    a = ap.parse_args(argv)
    R = Results(a.root)
    if a.index_only:
        text = "# Runs\n\n" + sec_index(R)
    else:
        parts = ["# RHMP v2 results\n",
                 "Generated by `python3 scripts/collect_results.py results --out results/RESULTS.md` from the result "
                 "files in `results/`; the conventions are described in the docstring of that script.  The results "
                 "are discussed in [REPORT.md](../REPORT.md); speed and memory measurements are in "
                 "[bench/RESULTS_step.md](../bench/RESULTS_step.md) and [bench/RESULTS_ops.md](../bench/RESULTS_ops.md).\n",
                 sec_paper(R), sec_new(R), sec_variants(R), sec_recovery(R), sec_aniso(R), sec_baselines(R),
                 sec_transfer(R), sec_rollout(R), sec_index(R)]
        text = "\n".join(parts)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            fh.write(text)
        print(f"wrote {a.out} ({len(text.splitlines())} lines)")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
