#!/usr/bin/env python3
# Adapted from `download.py` of the RHMP v1 code base, https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing
# (paper: "Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame Equivariance",
# Zheng & Allen-Blanchette, arXiv:2608.14556).  Same OSF archives; the v1 checkpoints are placed in `checkpoints_v1/`.
"""Download the v1 paper datasets (and optionally the v1 checkpoints) from OSF.

    python3 datasets/download_v1.py              # datasets/*.pkl                      (1.1 GB archive)
    python3 datasets/download_v1.py --ckpt       # + checkpoints_v1/ (seed 42)          (834 MB archive)
    python3 datasets/download_v1.py --seed-ckpt  # + checkpoints_v1/seed_{1,2}/          (315 MB archive)

Paths are relative to the repository root (the parent of this directory), whatever the working directory.  The
checkpoints are only needed to re-evaluate the v1 model with ``python -m rhmp.train --eval-v1`` (see README).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEW_ONLY = "0df302ed83154a8ba6bb1f8d8a1983ec"
OSF_PAGE = f"https://osf.io/35gcm/?view_only={VIEW_ONLY}"

FILES = {
    "datasets": {
        "url": f"https://osf.io/download/69dd1357d69645ec5e3e8eae/?view_only={VIEW_ONLY}",
        "filename": "gauge_hodge_mp_datasets.tar.gz",
        "member_root": "datasets",
        "target": "datasets",
        "suffix": ".pkl",
        "check_path": "datasets/T1_cns_vorticity.pkl",
    },
    "checkpoints": {
        "url": f"https://osf.io/download/69dd13f93dd88ea8473e8ec5/?view_only={VIEW_ONLY}",
        "filename": "gauge_hodge_mp_checkpoints.tar.gz",
        "member_root": "checkpoints",
        "target": "checkpoints_v1",
        "check_path": "checkpoints_v1/T1_cns_vorticity/ours/best_model.pt",
    },
    "seed_checkpoints": {
        "url": f"https://osf.io/download/69f7d3bf492480663e1e51e5/?view_only={VIEW_ONLY}",
        "filename": "gauge_hodge_mp_seed_checkpoints.tar.gz",
        "member_root": "checkpoints",
        "target": "checkpoints_v1",
        "check_path": "checkpoints_v1/seed_1/T1_cns_vorticity/ours/best_model.pt",
    },
}


def _progress(count: int, block_size: int, total_size: int) -> None:
    mb = count * block_size / 1e6
    if total_size > 0:
        sys.stdout.write(f"\r  {mb:.1f}/{total_size / 1e6:.1f} MB ({min(100.0, 100 * mb * 1e6 / total_size):.0f}%)")
    else:
        sys.stdout.write(f"\r  {mb:.1f} MB")
    sys.stdout.flush()


def _merge_tree(src: str, dst: str) -> None:
    """Move the contents of ``src`` into ``dst`` (existing files are kept)."""
    for dirpath, _, files in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        out = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(out, exist_ok=True)
        for f in files:
            target = os.path.join(out, f)
            if not os.path.exists(target):
                shutil.move(os.path.join(dirpath, f), target)


def download_and_extract(key: str, force: bool = False, keep_archive: bool = False) -> None:
    info = FILES[key]
    if os.path.exists(os.path.join(ROOT, info["check_path"])) and not force:
        print(f"[{key}] already present ({info['check_path']}), skipping")
        return
    archive = os.path.join(ROOT, info["filename"])
    if force or not os.path.exists(archive):
        print(f"[{key}] downloading {info['filename']} from OSF ({OSF_PAGE})")
        urllib.request.urlretrieve(info["url"], archive, reporthook=_progress)
        print()
    print(f"[{key}] extracting into {info['target']}/")
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        with tarfile.open(archive, "r:gz") as tar:
            def keep(m: tarfile.TarInfo) -> bool:
                name = m.name[2:] if m.name.startswith("./") else m.name
                return name.split("/")[0] == info["member_root"] and name.endswith(info.get("suffix", ""))
            members = [m for m in tar.getmembers() if keep(m)]
            try:
                tar.extractall(path=tmp, members=members, filter="data")
            except TypeError:  # Python < 3.12: no extraction filter argument
                tar.extractall(path=tmp, members=members)
        base = tmp if os.path.isdir(os.path.join(tmp, info["member_root"])) else os.path.join(tmp, ".")
        _merge_tree(os.path.join(base, info["member_root"]), os.path.join(ROOT, info["target"]))
    if not keep_archive:
        os.remove(archive)
    print(f"[{key}] done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="store_true", help="also download the seed-42 v1 checkpoints")
    ap.add_argument("--seed-ckpt", action="store_true", help="also download the seed 1 / 2 v1 checkpoints")
    ap.add_argument("--no-data", action="store_true", help="skip the datasets")
    ap.add_argument("--force", action="store_true", help="download and extract even if present")
    ap.add_argument("--keep-archive", action="store_true", help="keep the downloaded .tar.gz files")
    a = ap.parse_args()
    keys = ([] if a.no_data else ["datasets"]) + (["checkpoints"] if a.ckpt else []) + \
        (["seed_checkpoints"] if a.seed_ckpt else [])
    for key in keys:
        download_and_extract(key, force=a.force, keep_archive=a.keep_archive)


if __name__ == "__main__":
    main()
