"""Render ``docs/API.md`` from the docstrings of the ``rhmp`` package.

    python3 docs/gen_api.py                  # writes docs/API.md next to this script
    python3 docs/gen_api.py --out FILE       # another file
    python3 docs/gen_api.py --stdout         # print to stdout

The modules are imported (torch, numpy and scipy must be installed); for every module the names in ``__all__`` are
documented with their signature, a one-line summary, the full docstring (argument and return shapes) and, for
dataclasses with an ``Attributes:`` / ``Fields`` section, a table of fields with types and defaults.  The task
registry, the baseline registry and the trainer CLI (``python -m rhmp.train --help``) are rendered from the code, so
the reference cannot drift from the implementation.  Regenerate after changing public docstrings.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib
import inspect
import io
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (module, section title, short description); order = reading order
MODULES = [
    ("rhmp", "Package", "top-level exports (lazy imports)"),
    ("rhmp.complex", "Cochain complexes", "meshes -> exact oriented incidence, stars, geometry, batching"),
    ("rhmp.model", "Model", "configuration and the RHMP network"),
    ("rhmp.dec", "DEC toolkit", "Galerkin metrics, CG solvers, Hodge Laplacians/decomposition, Whitney maps"),
    ("rhmp.geometry", "Geometry", "E(n)-invariant descriptors, reference Hodge stars, Whitney blocks"),
    ("rhmp.ops", "Sparse and segment operations", "spmm on (n, B, C), Gershgorin bounds, segment reductions"),
    ("rhmp.data", "Data containers", "TaskData, normalisation, splits, minibatching, cochain alignment"),
    ("rhmp.metrics", "Metrics", "R2 / MSE / MAE / NRMSE / SSIM / Pearson (v1-compatible)"),
    ("rhmp.train", "Trainer", "training / evaluation loop and CLI"),
    ("rhmp.tasks", "Task registry", "load_task and the paper / synthetic tasks"),
    ("rhmp.tasks.suite", "Extension task suite", "SURF, DYN, QUAL loaders and rollouts"),
    ("rhmp.baselines", "Baselines", "lazy entry points"),
    ("rhmp.baselines.registry", "Baseline registry", "model specs, applicability, parameter matching"),
    ("rhmp.layers", "Building blocks: layers", "normalised metric Hodge operators, resolvent, RHMPLayer"),
    ("rhmp.metric", "Building blocks: metric heads", "bounded log-metric and material-tensor heads"),
    ("rhmp.lifting", "Building blocks: lifting", "raw inputs on any degree -> hidden cochains"),
    ("rhmp.readout", "Building blocks: readouts", "node / cochain / even / constraint / vector readouts"),
]

_ROLE = re.compile(r":(?:py:)?(?:class|func|meth|mod|attr|data|obj|exc):`~?([^`]+)`")
_DOUBLE_TICK = re.compile(r"``([^`]+)``")


# ---------------------------------------------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------------------------------------------
def md_inline(text: str) -> str:
    """RST inline markup -> Markdown (``x`` -> `x`, :func:`a.b` -> `a.b`)."""
    text = _ROLE.sub(lambda m: f"``{m.group(1)}``", text)
    return _DOUBLE_TICK.sub(lambda m: f"`{m.group(1)}`", text)


def split_doc(obj) -> tuple[str, str]:
    """``(summary, rest)`` of a docstring: first paragraph joined into one line, remaining text."""
    doc = inspect.getdoc(obj) or ""
    if not doc:
        return "", ""
    paras = doc.split("\n\n", 1)
    summary = " ".join(line.strip() for line in paras[0].splitlines())
    return md_inline(summary), (paras[1] if len(paras) > 1 else "")


def fence(text: str, lang: str = "text") -> str:
    ticks = "````" if "```" in text else "```"
    return f"{ticks}{lang}\n{text.rstrip()}\n{ticks}\n"


def details(summary: str, body: str) -> str:
    return f"<details><summary>{summary}</summary>\n\n{body}\n</details>\n\n"


def cell(text: str) -> str:
    """Markdown table cell: inline markup converted, pipes escaped, newlines removed."""
    return md_inline(" ".join(str(text).split())).replace("|", "\\|")


def ann(a) -> str:
    if a is inspect.Parameter.empty:
        return ""
    if isinstance(a, str):
        return a.strip("'\"")
    return inspect.formatannotation(a)


def short_repr(v, limit: int = 120) -> str:
    if isinstance(v, dict):
        keys = list(v)
        r = "{" + ", ".join(repr(k) for k in keys[:12]) + (", ..." if len(keys) > 12 else "") + "}"
        return f"dict with keys {r}"
    r = repr(v)
    return r if len(r) <= limit else r[:limit - 3] + "..."


# ---------------------------------------------------------------------------------------------------------------
# signatures
# ---------------------------------------------------------------------------------------------------------------
def _join(name: str, parts: list[str], ret: str) -> str:
    one = f"{name}({', '.join(parts)}){ret}"
    if len(one) <= 100:
        return one
    return f"{name}(\n" + "".join(f"    {p},\n" for p in parts) + f"){ret}"


def fmt_signature(name: str, obj, drop_first: bool = False) -> str:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return f"{name}(...)"
    params = list(sig.parameters.values())
    if drop_first and params and params[0].name in ("self", "cls"):
        params = params[1:]
    parts: list[str] = []
    star_done = any(p.kind == p.VAR_POSITIONAL for p in params)
    slash_pending = False
    for p in params:
        if p.kind == p.POSITIONAL_ONLY:
            slash_pending = True
        elif slash_pending:
            parts.append("/")
            slash_pending = False
        if p.kind == p.KEYWORD_ONLY and not star_done:
            parts.append("*")
            star_done = True
        s = {p.VAR_POSITIONAL: "*", p.VAR_KEYWORD: "**"}.get(p.kind, "") + p.name
        a = ann(p.annotation)
        if a:
            s += f": {a}"
        if p.default is not p.empty:
            s += (" = " if a else "=") + repr(p.default)
        parts.append(s)
    if slash_pending:
        parts.append("/")
    ret = ann(sig.return_annotation)
    return _join(name, parts, f" -> {ret}" if ret else "")


def dataclass_signature(cls) -> str:
    parts = []
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        s = f"{f.name}: {ann(f.type)}"
        if f.default is not dataclasses.MISSING:
            s += f" = {f.default!r}"
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            s += f" = {f.default_factory()!r}"  # type: ignore[misc]
        parts.append(s)
    return _join(cls.__name__, parts, "")


# ---------------------------------------------------------------------------------------------------------------
# dataclass field tables
# ---------------------------------------------------------------------------------------------------------------
def parse_fields_section(doc: str) -> dict[str, str]:
    """``{name: description}`` from a Google-style ``Attributes:`` or ``Fields ...:`` section."""
    lines = doc.splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(r"^\s*(Attributes|Fields)\b.*:\s*$", l)), None)
    if start is None:
        return {}
    out: dict[str, str] = {}
    base = None
    names: list[str] = []
    for line in lines[start + 1:]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if base is None:
            base = indent
        if indent < base:
            break
        m = re.match(r"^([A-Za-z_][\w]*(?:\s*[,/]\s*[A-Za-z_][\w]*)*):\s*(.*)$", line.strip())
        if indent == base and m:
            names = [n.strip() for n in re.split(r"[,/]", m.group(1))]
            for n in names:
                out[n] = m.group(2)
        elif names:
            for n in names:
                out[n] += " " + line.strip()
    return out


def field_table(cls) -> str:
    desc = parse_fields_section(inspect.getdoc(cls) or "")
    rows = ["| field | type | default | description |", "|---|---|---|---|"]
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            d = f"`{f.default!r}`"
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            d = f"`{f.default_factory()!r}`"  # type: ignore[misc]
        else:
            d = "required"
        rows.append(f"| `{f.name}` | `{cell(ann(f.type))}` | {d.replace('|', chr(92) + '|')} | "
                    f"{cell(desc.get(f.name, ''))} |")
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# objects
# ---------------------------------------------------------------------------------------------------------------
def doc_block(obj, what: str = "docstring") -> str:
    _, rest = split_doc(obj)
    return details(what, fence(rest)) if rest.strip() else ""


def render_function(qualname: str, fn, level: int = 3, drop_first: bool = False, kind: str = "") -> str:
    summary, _ = split_doc(fn)
    head = "#" * level + f" {kind}`{qualname}`\n\n"
    return head + fence(fmt_signature(qualname.split(".")[-1], fn, drop_first), "python") + \
        (summary + "\n\n" if summary else "") + doc_block(fn)


def class_members(cls) -> list[tuple[str, str, object]]:
    """Public members defined on the class itself: ``(name, kind, object)``."""
    out = []
    for name, raw in cls.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(raw, property):
            out.append((name, "property", raw.fget))
        elif isinstance(raw, staticmethod):
            out.append((name, "staticmethod", raw.__func__))
        elif isinstance(raw, classmethod):
            out.append((name, "classmethod", raw.__func__))
        elif inspect.isfunction(raw):
            out.append((name, "method", raw))
    return out


def render_class(qualname: str, cls) -> str:
    summary, _ = split_doc(cls)
    sig = dataclass_signature(cls) if dataclasses.is_dataclass(cls) else fmt_signature(cls.__name__, cls)
    bases = [b.__name__ for b in cls.__bases__ if b is not object]
    parts = [f"### class `{qualname}`\n\n", fence(sig, "python")]
    if bases:
        parts.append(f"Bases: {', '.join(f'`{b}`' for b in bases)}.\n\n")
    if summary:
        parts.append(summary + "\n\n")
    if dataclasses.is_dataclass(cls):
        parts.append(field_table(cls) + "\n")
    parts.append(doc_block(cls))
    for name, kind, obj in class_members(cls):
        q = f"{cls.__name__}.{name}"
        if kind == "property":
            s, _ = split_doc(obj)
            parts.append(f"#### `{q}` (property)\n\n" + (s + "\n\n" if s else "") + doc_block(obj))
        else:
            label = "" if kind == "method" else f"{kind} "
            parts.append(render_function(q, obj, level=4, drop_first=kind in ("method", "classmethod"),
                                         kind=label))
    return "".join(parts)


def render_module(modname: str, title: str, blurb: str) -> tuple[str, list[str]]:
    mod = importlib.import_module(modname)
    names = list(getattr(mod, "__all__", []))
    out = [f"## {title}: `{modname}`\n\n", f"{blurb[0].upper() + blurb[1:]}.\n\n"]
    doc = inspect.getdoc(mod) or ""
    if doc:
        out.append(details("module docstring", fence(doc)))
    toc = []
    consts = []
    for name in names:
        obj = getattr(mod, name)
        home = getattr(obj, "__module__", modname)
        if (inspect.isclass(obj) or inspect.isfunction(obj)) and home != modname:
            out.append(f"- `{modname}.{name}`: re-export of `{home}.{name}` (documented there).\n")
            continue
        if inspect.isclass(obj):
            out.append(render_class(name, obj))
            toc.append(f"class {name}")
        elif inspect.isfunction(obj):
            out.append(render_function(name, obj))
            toc.append(name)
        elif inspect.ismodule(obj):
            out.append(f"- `{modname}.{name}`: submodule.\n")
        else:
            consts.append(f"- `{name}`: {cell(short_repr(obj))}\n")
    if consts:
        out.append("### Constants\n\n" + "".join(consts) + "\n")
    return "".join(out) + "\n", toc


# ---------------------------------------------------------------------------------------------------------------
# registries and CLI
# ---------------------------------------------------------------------------------------------------------------
def render_registries() -> str:
    from rhmp import tasks
    from rhmp.baselines.registry import SPECS
    try:
        from rhmp.tasks.suite import SUITE_TASKS, suite_task_defaults
    except ImportError:  # the extension suite is optional
        SUITE_TASKS, suite_task_defaults = {}, None
    names = tasks.list_tasks()
    listed = [n for n in SUITE_TASKS if n in names]
    out = ["## Registries (generated from the code)\n\n", "### Tasks: `rhmp.tasks.load_task(name, root, ...)`\n\n",
           "Datasets are not part of the package: paper tasks read `datasets/*.pkl` (v1 pickles), the new tasks read "
           "`datasets/v2/*.pt` produced by the generators in `datasets/generators/` (see `docs/DATASETS.md`, "
           "`docs/TASK_SUITE.md`).  HP/TET names accept the suffixes `_k<kappa>` and `_aniso[<R>]` "
           "(module docstring of `rhmp.tasks`).  Every name below is accepted by `load_task` and by "
           "`python -m rhmp.train --task NAME`.\n\n",
           "| task | source | default C | default layers | default batch |\n|---|---|---:|---:|---:|\n"]

    def source(n: str) -> str:
        if n in SUITE_TASKS:
            return "extension suite (`rhmp.tasks.suite`)"
        if n.startswith(("HP", "TET")):
            return "synthetic (`rhmp.tasks.synthetic`)"
        return "paper (`rhmp.tasks.paper`)"
    for name in names:
        d = tasks.task_defaults(name)
        out.append(f"| `{name}` | {source(name)} | {d['C']} | {d['layers']} | {d['batch']} |\n")
    missing = [n for n in SUITE_TASKS if n not in listed]
    if missing:   # an older registry without the suite hook
        out.append("\nNot resolved by `load_task` at the time of generation (load with "
                   "`rhmp.tasks.suite.SUITE_TASKS[name](root, device=...)`):\n\n"
                   "| task | default C | default layers | default batch |\n|---|---:|---:|---:|\n")
        for name in missing:
            d = suite_task_defaults(name)
            out.append(f"| `{name}` | {d['C']} | {d['layers']} | {d['batch']} |\n")
    out.append("\n### Models: `python -m rhmp.train --model NAME` (`rhmp.baselines.registry.SPECS`)\n\n"
               "Details, fairness rules and applicability: `docs/BASELINES.md`.  Baselines that wrap the v1 code "
               "(`ours_v1`, the v1 graph / mesh / complex baselines) use its vendored copy in "
               "`rhmp.baselines.v1`.\n\n| model | family | description | param-matched | keeps task output map |\n"
               "|---|---|---|:-:|:-:|\n")
    for name, s in SPECS.items():
        out.append(f"| `{name}` | {s.family} | {cell(s.description)} | {'yes' if s.param_matched else 'no'} | "
                   f"{'yes' if s.uses_output_map else 'no'} |\n")
    return "".join(out) + "\n"


def render_cli() -> str:
    from rhmp.train import parse_args
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        parse_args(["--help"])
    help_text = re.sub(r"^usage: \S+", "usage: python -m rhmp.train", buf.getvalue())   # prog = this script
    return "## Trainer CLI: `python -m rhmp.train --help`\n\n" + fence(help_text) + "\n"


# ---------------------------------------------------------------------------------------------------------------
def package_version() -> str:
    try:
        from importlib.metadata import version
        return version("rhmp")
    except Exception:  # noqa: BLE001 - not installed: read pyproject.toml
        m = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(), flags=re.M)
        return m.group(1) if m else "unknown"


def render() -> str:
    import torch
    body, toc_lines = [], []
    for modname, title, blurb in MODULES:
        text, toc = render_module(modname, title, blurb)
        body.append(text)
        anchor = re.sub(r"[^a-z0-9 -]", "", f"{title}: {modname}".lower()).replace(" ", "-")
        entries = ", ".join(f"`{t}`" for t in toc[:14]) + (", ..." if len(toc) > 14 else "")
        toc_lines.append(f"- [{title} (`{modname}`)](#{anchor})" + (f": {entries}" if entries else ""))
    header = textwrap.dedent(f"""\
        # RHMP v2 API reference

        Generated by `python3 docs/gen_api.py` from the docstrings of `rhmp` {package_version()} (torch {torch.__version__}).
        Do not edit by hand; regenerate after changing public docstrings.  Conventions used throughout: cochain
        features have the layout `(n_k, B, C)` (cells of degree k, samples sharing the complex, channels); a
        block-diagonal batch of different meshes has `B = 1` and `K.batch[k]` holds the graph id of every k-cell.
        Each entry shows the signature, a one-line summary and, folded, the full docstring with argument and return
        shapes.  Start with `CochainComplex`, `RHMPConfig`, `RHMP`, `rhmp.dec`, `TaskData` and `rhmp.train.run`;
        the tutorial is `docs/TUTORIAL.md`, the mathematics `docs/MATH.md`.

        ## Contents

        """)
    toc = "\n".join(toc_lines + ["- [Registries](#registries-generated-from-the-code)",
                                 "- [Trainer CLI](#trainer-cli-python--m-rhmptrain---help)"]) + "\n\n"
    return header + toc + "".join(body) + render_registries() + render_cli()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "API.md"))
    ap.add_argument("--stdout", action="store_true")
    a = ap.parse_args()
    text = render()
    if a.stdout:
        sys.stdout.write(text)
    else:
        Path(a.out).write_text(text)
        print(f"wrote {a.out} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
