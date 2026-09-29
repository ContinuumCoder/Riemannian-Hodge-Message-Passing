"""Usability tests: every script in ``examples/`` and the README quickstart run end to end on the CPU.

Each example is run in a subprocess (fresh interpreter, CUDA hidden, 2 threads) from a temporary working directory;
the test checks the exit code and the final ``EXAMPLE NN OK`` marker the script prints.  Examples that write files
receive ``--out <tmp>``.  The quickstart test extracts the first ``python`` code block under "## Quickstart" in
``README.md`` and runs it the same way, so the documented code cannot silently go stale.  The API reference
generator is smoke-tested on its stdout mode.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))
TIMEOUT_S = 600  # the examples take < 2 min each on a busy CPU; the margin is for heavily loaded machines


def _env() -> dict:
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MPLBACKEND="Agg",
               PYTHONPATH=os.pathsep.join([str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])))
    return env


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=_env(), capture_output=True, text=True, timeout=TIMEOUT_S)


def _report(proc: subprocess.CompletedProcess) -> str:
    return f"exit {proc.returncode}\n--- stdout (tail)\n{proc.stdout[-4000:]}\n--- stderr (tail)\n{proc.stderr[-4000:]}"


def test_examples_are_present():
    assert [p.name[:2] for p in EXAMPLES] == ["01", "02", "03", "04", "05", "06"], [p.name for p in EXAMPLES]


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(path: Path, tmp_path: Path):
    cmd = [sys.executable, str(path)]
    if "--out" in path.read_text():
        cmd += ["--out", str(tmp_path / "out")]
    proc = _run(cmd, tmp_path)
    assert proc.returncode == 0, _report(proc)
    assert f"EXAMPLE {path.name[:2]} OK" in proc.stdout, _report(proc)
    if path.name.startswith("06"):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            assert "skipping the ellipse plot" in proc.stdout
        else:
            assert (tmp_path / "out" / "tensor_metric_ellipses.png").stat().st_size > 0


def _quickstart_code() -> str:
    text = (ROOT / "README.md").read_text()
    section = text.split("## Quickstart", 1)[1]
    match = re.search(r"```python\n(.*?)```", section, flags=re.S)
    assert match, "no python code block under '## Quickstart' in README.md"
    return match.group(1)


def test_readme_quickstart_runs(tmp_path: Path):
    script = tmp_path / "quickstart.py"
    script.write_text(_quickstart_code())
    proc = _run([sys.executable, str(script)], tmp_path)
    assert proc.returncode == 0, _report(proc)
    assert "CochainComplex(" in proc.stdout, _report(proc)


def test_tutorial_code_runs(tmp_path: Path):
    """The ``python`` blocks of docs/TUTORIAL.md form one script (run from the repository root)."""
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / "docs" / "TUTORIAL.md").read_text(), flags=re.S)
    assert len(blocks) >= 10, len(blocks)
    script = tmp_path / "tutorial.py"
    script.write_text("\n".join(blocks))
    proc = _run([sys.executable, str(script)], ROOT)
    assert proc.returncode == 0, _report(proc)


def test_api_reference_generator_runs(tmp_path: Path):
    out = tmp_path / "API.md"
    proc = _run([sys.executable, str(ROOT / "docs" / "gen_api.py"), "--out", str(out)], tmp_path)
    assert proc.returncode == 0, _report(proc)
    text = out.read_text()
    for needle in ("class `RHMPConfig`", "class `CochainComplex`", "`hodge_decompose", "`load_task", "--task"):
        assert needle in text, needle
