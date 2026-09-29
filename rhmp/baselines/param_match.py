"""Parameter matching with v1's rule (v1 ``formal_benchmark.py::find_hidden_match``) and a JSON cache.

v1 rule: scan ``hidden = 16, 20, 24, ...`` (< 800) and take the first width whose parameter count is ``>= target``;
it is accepted if it is ``<= (1 + tol) * target`` (``tol = 0.2``), otherwise it is "the best we can do" (flagged
``within_tol = False``).  Refinement (``refine=True``): when the step-4 crossing misses the tolerance (small budgets,
where one step changes the count by > 20 %), the widths between the last two scanned values are tried one by one and
the first one ``>= target`` is taken (widths a model rejects, e.g. GAT widths not divisible by its 4 heads, are
skipped).  Extension for models with a second size knob (``options``, e.g. FNO's Fourier modes): the
scan is repeated for each option in order and the first option that lands inside the tolerance wins; if none does,
the candidate with the smallest ``|log(params / target)|`` is returned.

Parameter counts are obtained by building the model on the ``meta`` device (no memory, no RNG draws; CPU fallback)
inside ``torch.random.fork_rng`` so that matching never changes the random initialisation of the final model.
Results are cached in ``runs/param_match.json`` as ``{task: {model: {...}}}``; an entry is reused only if its
``target`` and ``signature`` (input/output layout and structural overrides) match.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch
from torch import nn

__all__ = ["MatchResult", "count_params", "match_params", "DEFAULT_CACHE", "V1_TOL"]

DEFAULT_CACHE = os.path.join("runs", "param_match.json")
V1_TOL = 0.2


@dataclass
class MatchResult:
    """Outcome of :func:`match_params`.

    Attributes:
        hidden: selected width.
        params: trainable parameters of the selected model.
        target: target parameter count.
        ratio: ``params / target``.
        within_tol: ``target <= params <= (1 + tol) target``.
        option: the selected extra keyword arguments (``{}`` without ``options``).
        cached: whether the result came from the cache.
    """
    hidden: int
    params: int
    target: int
    ratio: float
    within_tol: bool
    option: dict = field(default_factory=dict)
    cached: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def count_params(model: nn.Module) -> int:
    """Trainable scalars of ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count(build_fn: Callable[..., nn.Module], hidden: int, option: dict) -> int | None:
    """Trainable parameters of ``build_fn(hidden, **option)`` (``None`` if the width is invalid for the model,
    e.g. not divisible by GAT's number of heads)."""
    try:
        with torch.device("meta"):
            m = build_fn(hidden, **option)
    except Exception:  # noqa: BLE001 - some constructors do not support the meta device
        try:
            m = build_fn(hidden, **option)
        except Exception:  # noqa: BLE001
            return None
    return count_params(m)


def _load(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _store(path: str, task: str, model: str, entry: dict) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    data = _load(path)
    data.setdefault(task, {})[model] = entry
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def match_params(build_fn: Callable[..., nn.Module], target_params: int, tol: float = V1_TOL, *,
                 start: int = 16, stop: int = 800, step: int = 4, options: list[dict] | None = None,
                 key: tuple[str, str] | None = None, signature: Any = None,
                 cache_path: str | None = DEFAULT_CACHE, refine: bool = True,
                 refine_step: int | tuple[int, ...] = 1) -> MatchResult:
    """Width (and option) whose parameter count matches ``target_params`` by the v1 rule.

    Args:
        build_fn: ``build_fn(hidden, **option) -> nn.Module``.
        target_params: target number of trainable parameters (e.g. the v2 model's).
        tol: relative tolerance (v1: 0.2, i.e. ``params in [target, 1.2 target]``).
        start, stop, step: width scan (v1: 16, 800, 4).
        options: optional list of extra keyword-argument dicts tried in order.
        key: ``(model, task)`` cache key (no caching if ``None`` or ``cache_path`` is ``None``).
        signature: JSON-serialisable description of everything else that changes the parameter count.
        cache_path: JSON cache file.
        refine: refinement around the crossing when the scan misses the tolerance.
        refine_step: spacing(s) of the refined widths, tried in order until the tolerance is met (default 1;
            MeshGraphNet uses ``(4, 1)``: GEMM-friendly widths first, exact budget parity if they cannot meet it).
    Returns:
        :class:`MatchResult`.
    """
    target = int(target_params)
    steps = (refine_step,) if isinstance(refine_step, int) else tuple(int(v) for v in refine_step)
    if target <= 0:
        raise ValueError(f"target_params must be positive, got {target_params}")
    sig = json.loads(json.dumps(signature, default=str)) if signature is not None else None
    if key is not None and cache_path:
        entry = _load(cache_path).get(key[1], {}).get(key[0])
        if entry and entry.get("target") == target and entry.get("signature") == sig and entry.get("tol") == tol \
                and entry.get("scan") == [start, stop, step] and entry.get("refine", False) == refine \
                and entry.get("refine_step", [1]) == list(steps):
            r = MatchResult(entry["hidden"], entry["params"], target, entry["ratio"], entry["within_tol"],
                            entry.get("option", {}), cached=True)
            return r
    best: tuple[float, MatchResult] | None = None
    chosen: MatchResult | None = None
    with torch.random.fork_rng(devices=[]):
        for opt in (options or [{}]):
            last, prev_h = None, None
            for h in range(start, stop, step):
                n = _count(build_fn, h, opt)
                if n is None:
                    continue
                if n >= target:
                    last = (h, n)
                    break
                last, prev_h = (h, n), h
            if last is None:
                continue
            h, n = last
            lo = prev_h if prev_h is not None else h            # refine strictly between the last two widths
            for rs in (steps if refine else ()):
                if n <= (1 + tol) * target or rs >= step:
                    break
                for h2 in range(lo + rs, h, rs):
                    n2 = _count(build_fn, h2, opt)
                    if n2 is not None and n2 >= target:
                        h, n = h2, n2
                        break
            r = MatchResult(h, n, target, n / target, target <= n <= (1 + tol) * target, dict(opt))
            score = abs(math.log(max(n, 1) / target))
            if best is None or score < best[0]:
                best = (score, r)
            if r.within_tol:
                chosen = r
                break
    if chosen is None:
        if best is None:
            raise RuntimeError("match_params: empty scan")
        chosen = best[1]
    if key is not None and cache_path:
        _store(cache_path, key[1], key[0], dict(chosen.to_dict(), signature=sig, tol=tol,
                                                 scan=[start, stop, step], refine=refine, refine_step=list(steps),
                                                 cached=False))
    return chosen
