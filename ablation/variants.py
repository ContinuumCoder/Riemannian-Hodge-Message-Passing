"""
Ablation variants for Gauge-Structured Hodge MP.

Four variants target the four core design pillars:
  A1 noH       : H_k = I (identity metric, no learnable gauge metric)
                 -> tests the value of metric learning.
  A3 nocross   : alpha forced to 1 (no cross-dimensional transfer)
                 -> tests the value of cross-dimensional communication.
  A5 relu      : replace O(C)-equivariant norm-gated update with ReLU activation
                 -> tests the value of O(C)-equivariant nonlinearity.
  A4 learnD    : replace fixed CW coboundary d_k with learnable linear maps
                 -> tests the value of the d^2 = 0 hard constraint.

Each variant only flips one switch relative to the full model (A0).
"""

VARIANT_SPEC = {
    "full":    {"desc": "Full model (A0, baseline)",
                "identity_metric": False, "no_cross": False,
                "relu_gate": False, "learned_d": False},
    "noH":     {"desc": "H_k = I (no learnable metric)",
                "identity_metric": True,  "no_cross": False,
                "relu_gate": False, "learned_d": False},
    "nocross": {"desc": "alpha = 1 (no cross-dim transfer)",
                "identity_metric": False, "no_cross": True,
                "relu_gate": False, "learned_d": False},
    "relu":    {"desc": "ReLU instead of norm-gated update",
                "identity_metric": False, "no_cross": False,
                "relu_gate": True,  "learned_d": False},
    "learnD":  {"desc": "learnable d_k (breaks d^2 = 0)",
                "identity_metric": False, "no_cross": False,
                "relu_gate": False, "learned_d": True},
}

ABLATION_VARIANTS = ["noH", "nocross", "relu", "learnD"]
ALL_VARIANTS = ["full"] + ABLATION_VARIANTS

TASKS = ["T1", "T3", "T6"]  # regular grid / curved surface vector / U(1) gauge
