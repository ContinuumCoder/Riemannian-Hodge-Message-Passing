#!/usr/bin/env python3
"""
Download datasets and pretrained checkpoints from OSF.

Usage:
    python3 download.py              # download everything (data + ckpt + seed-ckpt)
    python3 download.py --data       # datasets only
    python3 download.py --ckpt       # seed=42 checkpoints only
    python3 download.py --seed-ckpt  # seed=1,2 multi-seed checkpoints only
"""

import argparse
import os
import tarfile
import urllib.request
import sys

VIEW_ONLY = "0df302ed83154a8ba6bb1f8d8a1983ec"

FILES = {
    "datasets": {
        "url": f"https://osf.io/download/69dd1357d69645ec5e3e8eae/?view_only={VIEW_ONLY}",
        "filename": "gauge_hodge_mp_datasets.tar.gz",
        "extract_to": ".",
        "check_path": "datasets/T1_cns_vorticity.pkl",
    },
    "checkpoints": {
        "url": f"https://osf.io/download/69dd13f93dd88ea8473e8ec5/?view_only={VIEW_ONLY}",
        "filename": "gauge_hodge_mp_checkpoints.tar.gz",
        "extract_to": ".",
        "check_path": "checkpoints/T1_cns_vorticity/ours/best_model.pt",
    },
    "seed_checkpoints": {
        "url": f"https://osf.io/download/69f7d3bf492480663e1e51e5/?view_only={VIEW_ONLY}",
        "filename": "gauge_hodge_mp_seed_checkpoints.tar.gz",
        "extract_to": ".",
        "check_path": "checkpoints/seed_1/T1_cns_vorticity/ours/best_model.pt",
    },
}


def download_file(url, dest):
    print(f"Downloading {dest} ...")
    def _progress(count, block_size, total_size):
        if total_size > 0:
            mb = count * block_size / 1e6
            total_mb = total_size / 1e6
            pct = count * block_size * 100 / total_size
            sys.stdout.write(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)")
        else:
            mb = count * block_size / 1e6
            sys.stdout.write(f"\r  {mb:.1f} MB")
        sys.stdout.flush()
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def download_and_extract(key, force=False):
    info = FILES[key]
    dest = info["filename"]

    if os.path.exists(info["check_path"]) and not force:
        print(f"[{key}] Already present ({info['check_path']}), skipping.")
        return

    if not os.path.exists(dest) or force:
        download_file(info["url"], dest)

    print(f"  Extracting ...")
    with tarfile.open(dest, "r:gz") as tar:
        tar.extractall(path=info["extract_to"])

    os.remove(dest)
    print(f"  [{key}] Done.")


def main():
    parser = argparse.ArgumentParser(description="Download Gauge-Hodge MP data")
    parser.add_argument("--data", action="store_true", help="Download datasets only")
    parser.add_argument("--ckpt", action="store_true", help="Download seed=42 checkpoints only")
    parser.add_argument("--seed-ckpt", action="store_true",
                        help="Download multi-seed (seed=1,2) checkpoints only")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not args.data and not args.ckpt and not args.seed_ckpt:
        args.data = args.ckpt = args.seed_ckpt = True

    if args.data:
        download_and_extract("datasets", force=args.force)
    if args.ckpt:
        download_and_extract("checkpoints", force=args.force)
    if args.seed_ckpt:
        download_and_extract("seed_checkpoints", force=args.force)

    print("\nDone. To evaluate:")
    print("  python3 experiments/compute_all_metrics.py --n-eval 100")
    print("  # multi-seed aggregation (seed=42 + seed=1 + seed=2):")
    print("  python3 scripts/aggregate_seeds.py --seeds 42 1 2")


if __name__ == "__main__":
    main()
