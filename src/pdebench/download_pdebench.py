"""
PDEBench Dataset Download Script

Supported datasets:
  - Shallow Water 2D (SWE)
  - 2D Compressible Navier-Stokes (CNS)

Features:
  - Downloads HDF5 files from DaRUS (https://darus.uni-stuttgart.de)
  - Supports resumable downloads
  - Shows download progress
"""

import os
import sys
import argparse
import hashlib
import urllib.request
import time

# ---------------------------------------------------------------------------
# Dataset URL configuration
# ---------------------------------------------------------------------------
# PDEBench data hosted on DaRUS (Uni Stuttgart)
# SWE:  doi:10.18419/darus-2986
# CNS:  doi:10.18419/darus-2986 (different files under the same dataset)

DATASETS = {
    "swe": {
        "url": "https://darus.uni-stuttgart.de/api/access/datafile/133021",
        "filename": "2D_rdb_NA_NA.h5",
        "description": "Shallow Water Equations 2D (radial dam break)",
    },
    "cns_rand_m01_eta001": {
        "url": "https://darus.uni-stuttgart.de/api/access/datafile/164687",
        "filename": "2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5",
        "description": "2D Compressible NS (random init, M=0.1, Eta=0.01, 128)",
    },
    "cns_rand_m01_eta01": {
        "url": "https://darus.uni-stuttgart.de/api/access/datafile/164688",
        "filename": "2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5",
        "description": "2D Compressible NS (random init, M=0.1, Eta=0.1, 128)",
    },
    "cns_rand_m10_eta001": {
        "url": "https://darus.uni-stuttgart.de/api/access/datafile/164690",
        "filename": "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5",
        "description": "2D Compressible NS (random init, M=1.0, Eta=0.01, 128)",
    },
    "cns_turb_m01": {
        "url": "https://darus.uni-stuttgart.de/api/access/datafile/164685",
        "filename": "2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5",
        "description": "2D Compressible NS (turbulent, M=0.1, 512)",
    },
    "cns_turb_m10": {
        "url": "https://darus.uni-stuttgart.de/api/access/datafile/164686",
        "filename": "2D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5",
        "description": "2D Compressible NS (turbulent, M=1.0, 512)",
    },
}

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "pdebench"
)


# ---------------------------------------------------------------------------
# Download utilities
# ---------------------------------------------------------------------------

class DownloadProgressBar:
    """Callback for displaying download progress."""

    def __init__(self, filename: str):
        self.filename = filename
        self.start_time = time.time()
        self.last_print_time = 0.0
        self.downloaded = 0
        self.total = 0

    def __call__(self, block_num: int, block_size: int, total_size: int):
        self.total = total_size
        self.downloaded = block_num * block_size

        now = time.time()
        if now - self.last_print_time < 0.5 and self.downloaded < self.total:
            return
        self.last_print_time = now

        elapsed = now - self.start_time
        speed = self.downloaded / max(elapsed, 1e-6)

        if total_size > 0:
            pct = min(100.0, self.downloaded / total_size * 100)
            total_mb = total_size / (1024 * 1024)
            dl_mb = self.downloaded / (1024 * 1024)
            speed_mb = speed / (1024 * 1024)
            eta = (total_size - self.downloaded) / max(speed, 1e-6)
            sys.stdout.write(
                f"\r  [{self.filename}] "
                f"{dl_mb:8.1f}/{total_mb:.1f} MB "
                f"({pct:5.1f}%) "
                f"{speed_mb:6.2f} MB/s "
                f"ETA {eta:6.0f}s"
            )
        else:
            dl_mb = self.downloaded / (1024 * 1024)
            sys.stdout.write(
                f"\r  [{self.filename}] {dl_mb:8.1f} MB downloaded"
            )
        sys.stdout.flush()


def download_file(url: str, dest_path: str, resume: bool = True) -> str:
    """
    Download a file with optional resume support.

    Args:
        url: Download URL.
        dest_path: Destination file path.
        resume: Whether to enable resumable downloads.

    Returns:
        Final file path.
    """
    filename = os.path.basename(dest_path)
    partial_path = dest_path + ".partial"

    if os.path.exists(dest_path):
        print(f"  [SKIP] {filename} already exists ({os.path.getsize(dest_path) / 1e6:.1f} MB)")
        return dest_path

    existing_size = 0
    if resume and os.path.exists(partial_path):
        existing_size = os.path.getsize(partial_path)
        print(f"  [RESUME] Found partial download: {existing_size / 1e6:.1f} MB")

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if existing_size > 0:
        req = urllib.request.Request(url)
        req.add_header("Range", f"bytes={existing_size}-")
        try:
            response = urllib.request.urlopen(req, timeout=60)
            content_range = response.headers.get("Content-Range", "")
            if response.status == 206:
                total_size = int(content_range.split("/")[-1]) if "/" in content_range else 0
                print(f"  [RESUME] Continuing from byte {existing_size}, total {total_size / 1e6:.1f} MB")
                _download_with_progress(response, partial_path, filename,
                                        mode="ab", offset=existing_size, total=total_size)
            else:
                print(f"  [WARN] Server does not support resume. Restarting download.")
                response.close()
                _download_full(url, partial_path, filename)
        except urllib.error.HTTPError as e:
            if e.code == 416:
                print(f"  [INFO] Partial file may be complete. Verifying...")
            else:
                raise
    else:
        _download_full(url, partial_path, filename)

    os.rename(partial_path, dest_path)
    final_size = os.path.getsize(dest_path)
    print(f"\n  [DONE] {filename}: {final_size / 1e6:.1f} MB")
    return dest_path


def _download_full(url: str, dest_path: str, filename: str):
    """Download file from scratch."""
    progress = DownloadProgressBar(filename)
    urllib.request.urlretrieve(url, dest_path, reporthook=progress)


def _download_with_progress(response, dest_path: str, filename: str,
                            mode: str = "wb", offset: int = 0, total: int = 0):
    """Stream download with progress display."""
    chunk_size = 8192
    downloaded = offset
    start_time = time.time()
    last_print = 0.0

    with open(dest_path, mode) as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_print >= 0.5:
                last_print = now
                elapsed = now - start_time
                speed = (downloaded - offset) / max(elapsed, 1e-6)
                dl_mb = downloaded / (1024 * 1024)

                if total > 0:
                    pct = downloaded / total * 100
                    total_mb = total / (1024 * 1024)
                    speed_mb = speed / (1024 * 1024)
                    eta = (total - downloaded) / max(speed, 1e-6)
                    sys.stdout.write(
                        f"\r  [{filename}] "
                        f"{dl_mb:8.1f}/{total_mb:.1f} MB "
                        f"({pct:5.1f}%) "
                        f"{speed_mb:6.2f} MB/s "
                        f"ETA {eta:6.0f}s"
                    )
                else:
                    sys.stdout.write(f"\r  [{filename}] {dl_mb:8.1f} MB downloaded")
                sys.stdout.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download PDEBench HDF5 datasets for Gauge-Hodge MP experiments"
    )
    parser.add_argument(
        "--dataset", type=str, nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        default=["swe", "cns_rand_m01_eta001"],
        help="Which dataset(s) to download (default: swe cns_rand)"
    )
    parser.add_argument(
        "--data_dir", type=str, default=DEFAULT_DATA_DIR,
        help="Directory to save downloaded files"
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Disable resumable downloads (restart from scratch)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets and exit"
    )
    args = parser.parse_args()

    if args.list:
        print("Available PDEBench datasets:")
        print("-" * 60)
        for key, info in DATASETS.items():
            print(f"  {key:12s}  {info['filename']}")
            print(f"  {'':12s}  {info['description']}")
            print()
        return

    datasets_to_download = []
    if "all" in args.dataset:
        datasets_to_download = list(DATASETS.keys())
    else:
        datasets_to_download = args.dataset

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    resume = not args.no_resume

    print(f"PDEBench Download")
    print(f"  Target directory: {data_dir}")
    print(f"  Resumable: {resume}")
    print(f"  Datasets: {datasets_to_download}")
    print("=" * 60)

    for ds_key in datasets_to_download:
        info = DATASETS[ds_key]
        print(f"\n[{ds_key}] {info['description']}")
        print(f"  File: {info['filename']}")
        print(f"  URL:  {info['url'][:80]}...")

        dest = os.path.join(data_dir, info["filename"])
        try:
            download_file(info["url"], dest, resume=resume)
        except Exception as e:
            print(f"\n  [ERROR] Failed to download {ds_key}: {e}")
            print(f"  You can manually download from: {info['url']}")
            print(f"  Place the file at: {dest}")
            continue

    print("\n" + "=" * 60)
    print("Download complete. Files saved to:")
    print(f"  {data_dir}")
    print()

    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        h5_files = [f for f in files if f.endswith(('.h5', '.hdf5'))]
        if h5_files:
            print("Downloaded HDF5 files:")
            for f in sorted(h5_files):
                size = os.path.getsize(os.path.join(data_dir, f))
                print(f"  {f}: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
