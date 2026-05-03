"""
PDEBench Data Loader

Converts regular-grid PDE data (HDF5) to triangular-mesh CellComplex format
for Gauge-Hodge MP and baseline models.

Supported datasets:
  - SWE  (Shallow Water Equations 2D): height field h(x,y,t) + velocity u,v
  - CNS  (2D Compressible NS): density rho, velocity u,v, pressure p

Pipeline:
  1. Read grid data from HDF5
  2. Build regular grid node coordinates
  3. Delaunay triangulation -> faces
  4. CellComplex.from_triangulation(pos, faces)
  5. Return (node_features, cell_complex, target)
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import Tuple, Optional, Dict, List

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from gauge_hodge_mp.cell_complex import CellComplex

try:
    import h5py
except ImportError:
    raise ImportError("h5py is required: pip install h5py")

try:
    from scipy.spatial import Delaunay
except ImportError:
    raise ImportError("scipy is required: pip install scipy")


# ---------------------------------------------------------------------------
# Grid construction utilities
# ---------------------------------------------------------------------------

def build_regular_grid(nx: int, ny: int,
                       x_range: Tuple[float, float] = (0.0, 1.0),
                       y_range: Tuple[float, float] = (0.0, 1.0),
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a regular 2D grid and perform Delaunay triangulation.

    Args:
        nx, ny: Grid resolution (x-direction, y-direction).
        x_range, y_range: Coordinate ranges.

    Returns:
        pos: (nx*ny, 2) node coordinates.
        faces: (F, 3) triangle face indices.
    """
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')  # (nx, ny)
    pos = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (nx*ny, 2)

    tri = Delaunay(pos)
    faces = tri.simplices  # (F, 3)

    return pos, faces


def subsample_grid(data_grid: np.ndarray, target_size: int
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniformly subsample a regular grid.

    Args:
        data_grid: (nx, ny, ...) or (nx, ny) grid data.
        target_size: Target side length (total nodes = target_size^2).

    Returns:
        subsampled_data: Subsampled data.
        indices: Sampling indices (ix, iy) in the original grid.
    """
    nx, ny = data_grid.shape[:2]
    step_x = max(1, nx // target_size)
    step_y = max(1, ny // target_size)
    ix = np.arange(0, nx, step_x)[:target_size]
    iy = np.arange(0, ny, step_y)[:target_size]
    subsampled = data_grid[np.ix_(ix, iy)]
    return subsampled, (ix, iy)


# ---------------------------------------------------------------------------
# HDF5 data loading
# ---------------------------------------------------------------------------

def load_swe_hdf5(filepath: str) -> Dict:
    """
    Load a PDEBench SWE 2D HDF5 file.

    Expected HDF5 structure (PDEBench convention):
        /tensor: (N_samples, T, nx, ny, C) where C = {h, u, v} or similar
        or
        /0000/data: (T, nx, ny, C)  per-sample storage

    Returns:
        dict with keys: 'data' (N, T, nx, ny, C), 'dt', 'dx'
    """
    result = {}
    with h5py.File(filepath, 'r') as f:
        if 'tensor' in f:
            data = f['tensor'][:]
            result['data'] = data
        elif 't' in f and 'x' in f:
            sample_keys = sorted([k for k in f.keys() if k.isdigit()])
            if sample_keys:
                samples = []
                for k in sample_keys:
                    samples.append(f[k]['data'][:])
                result['data'] = np.stack(samples, axis=0)
            else:
                raise ValueError(f"Unrecognized SWE HDF5 structure. Keys: {list(f.keys())}")
        else:
            sample_keys = sorted([k for k in f.keys() if k.isdigit()])
            if sample_keys:
                samples = []
                for k in sample_keys:
                    if 'data' in f[k]:
                        samples.append(f[k]['data'][:])
                    else:
                        samples.append(f[k][:])
                result['data'] = np.stack(samples, axis=0)
            else:
                for key in f.keys():
                    ds = f[key]
                    if hasattr(ds, 'shape') and len(ds.shape) >= 4:
                        result['data'] = ds[:]
                        break
                if 'data' not in result:
                    raise ValueError(
                        f"Cannot parse SWE HDF5 file. Top-level keys: {list(f.keys())}"
                    )

        if 'dt' in f.attrs:
            result['dt'] = float(f.attrs['dt'])
        if 'dx' in f.attrs:
            result['dx'] = float(f.attrs['dx'])

    print(f"  Loaded SWE data: shape={result['data'].shape}, "
          f"dtype={result['data'].dtype}")
    return result


def load_cns_hdf5(filepath: str) -> Dict:
    """
    Load a PDEBench 2D Compressible NS HDF5 file.

    Expected structure:
        /Vx:       (N, T, nx, ny) -- x-velocity
        /Vy:       (N, T, nx, ny) -- y-velocity
        /density:  (N, T, nx, ny) -- density
        /pressure: (N, T, nx, ny) -- pressure
        or
        /tensor:   (N, T, nx, ny, 4)

    Returns:
        dict with 'data' (N, T, nx, ny, 4) ordered as [density, Vx, Vy, pressure]
    """
    result = {}
    with h5py.File(filepath, 'r') as f:
        if 'Vx' in f:
            Vx = f['Vx'][:]
            Vy = f['Vy'][:]
            density = f['density'][:]
            pressure = f['pressure'][:]
            data = np.stack([density, Vx, Vy, pressure], axis=-1)
            result['data'] = data
        elif 'tensor' in f:
            result['data'] = f['tensor'][:]
        else:
            sample_keys = sorted([k for k in f.keys() if k.isdigit()])
            if sample_keys:
                samples = []
                for k in sample_keys:
                    if 'data' in f[k]:
                        samples.append(f[k]['data'][:])
                    else:
                        samples.append(f[k][:])
                result['data'] = np.stack(samples, axis=0)
            else:
                raise ValueError(
                    f"Cannot parse CNS HDF5 file. Keys: {list(f.keys())}"
                )

        if 'dt' in f.attrs:
            result['dt'] = float(f.attrs['dt'])

    print(f"  Loaded CNS data: shape={result['data'].shape}, "
          f"dtype={result['data'].dtype}")
    return result


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PDEBenchDataset(Dataset):
    """
    PDEBench dataset (triangular mesh + CellComplex format).

    Converts regular-grid PDE data to node features on a triangular mesh
    paired with a CellComplex for Gauge-Hodge MP.

    Each sample: (node_features, cell_complex, target)
      - node_features: (n_nodes, f_in) input node features
      - cell_complex: shared CellComplex instance
      - target: (n_nodes, out_dim) prediction target
    """

    def __init__(
        self,
        data_dir: str,
        dataset: str = "swe",
        grid_size: int = 64,
        time_step_gap: int = 1,
        input_channels: Optional[List[int]] = None,
        target_channels: Optional[List[int]] = None,
        n_samples: Optional[int] = None,
        device: str = "cpu",
    ):
        """
        Args:
            data_dir: Directory containing HDF5 files.
            dataset: "swe" or "cns" (also accepts "cns_rand", "cns_turb").
            grid_size: Subsampled grid side length (nodes = grid_size^2).
            time_step_gap: Time step gap between input and target.
            input_channels: Input channel indices (None = auto).
            target_channels: Target channel indices (None = auto).
            n_samples: Max number of samples (None = all).
            device: Device for CellComplex construction.
        """
        super().__init__()
        self.dataset = dataset
        self.grid_size = grid_size
        self.time_step_gap = time_step_gap
        self.device = device

        # Load HDF5
        filepath = self._find_hdf5(data_dir, dataset)
        print(f"Loading PDEBench [{dataset}] from: {filepath}")

        if dataset.startswith("swe"):
            raw = load_swe_hdf5(filepath)
        else:
            raw = load_cns_hdf5(filepath)

        data = raw['data']  # (N, T, nx, ny, C)

        # Determine channels
        n_channels = data.shape[-1] if data.ndim == 5 else 1
        if data.ndim == 4:
            data = data[..., np.newaxis]  # (N, T, nx, ny, 1)

        if dataset.startswith("swe"):
            # SWE: h (height), u, v
            self.input_channels = input_channels or [0]
            self.target_channels = target_channels or [0]
            self.task_type = "scalar"
        else:
            # CNS: density, Vx, Vy, pressure
            self.input_channels = input_channels or list(range(n_channels))
            self.target_channels = target_channels or list(range(n_channels))
            self.task_type = "scalar"

        self.f_in = len(self.input_channels)
        self.out_dim = len(self.target_channels)

        # Subsample grid
        N, T, nx_orig, ny_orig, C = data.shape
        print(f"  Original grid: {nx_orig}x{ny_orig}, T={T}, C={C}, N={N}")

        if grid_size < min(nx_orig, ny_orig):
            data_sub = np.zeros((N, T, grid_size, grid_size, C), dtype=data.dtype)
            step_x = max(1, nx_orig // grid_size)
            step_y = max(1, ny_orig // grid_size)
            ix = np.arange(0, nx_orig, step_x)[:grid_size]
            iy = np.arange(0, ny_orig, step_y)[:grid_size]
            for n_idx in range(N):
                for t_idx in range(T):
                    data_sub[n_idx, t_idx] = data[n_idx, t_idx][np.ix_(ix, iy)]
            data = data_sub
            print(f"  Subsampled grid: {grid_size}x{grid_size}")
        else:
            grid_size = min(nx_orig, ny_orig)
            self.grid_size = grid_size
            print(f"  Using full grid: {grid_size}x{grid_size}")

        # Build triangular mesh (shared topology across all samples)
        pos_np, faces_np = build_regular_grid(
            grid_size, grid_size,
            x_range=(0.0, 1.0), y_range=(0.0, 1.0)
        )
        self.pos = torch.tensor(pos_np, dtype=torch.float32)
        self.faces = torch.tensor(faces_np, dtype=torch.long)
        self.n_nodes = self.pos.size(0)

        print(f"  Triangulation: {self.n_nodes} nodes, {self.faces.size(0)} faces")

        # Pre-build shared CellComplex
        self._cell_complex = CellComplex.from_triangulation(
            self.pos, self.faces, device="cpu"
        )
        print(f"  CellComplex: n0={self._cell_complex.n0}, "
              f"n1={self._cell_complex.n1}, n2={self._cell_complex.n2}")

        # Expand into (sample, time_pair) samples
        n_time_pairs = T - time_step_gap
        if n_time_pairs <= 0:
            raise ValueError(
                f"time_step_gap={time_step_gap} too large for T={T}"
            )

        if n_samples is not None:
            N = min(N, n_samples)
            data = data[:N]

        # Flatten grid data to node features
        self.inputs = []   # list of (n_nodes, f_in) tensors
        self.targets = []  # list of (n_nodes, out_dim) tensors

        for n_idx in range(N):
            for t_idx in range(n_time_pairs):
                # Input: selected channels at time t
                inp = data[n_idx, t_idx, :, :, :]  # (gs, gs, C)
                inp_flat = inp.reshape(-1, C)[:, self.input_channels]
                self.inputs.append(
                    torch.tensor(inp_flat, dtype=torch.float32)
                )

                # Target: selected channels at time t+dt
                tgt = data[n_idx, t_idx + time_step_gap, :, :, :]
                tgt_flat = tgt.reshape(-1, C)[:, self.target_channels]
                self.targets.append(
                    torch.tensor(tgt_flat, dtype=torch.float32)
                )

        print(f"  Total samples: {len(self.inputs)} "
              f"({N} trajectories x {n_time_pairs} time pairs)")
        print(f"  Input dim: {self.f_in}, Output dim: {self.out_dim}")

    def _find_hdf5(self, data_dir: str, dataset: str) -> str:
        """Find the corresponding HDF5 file."""
        from download_pdebench import DATASETS
        candidates = []

        # Try to get filename from DATASETS config
        for key, info in DATASETS.items():
            if key.startswith(dataset.split("_")[0]):
                candidates.append(info['filename'])

        # Search directory
        if os.path.isdir(data_dir):
            files = os.listdir(data_dir)
            h5_files = [f for f in files if f.endswith(('.h5', '.hdf5'))]

            for c in candidates:
                if c in h5_files:
                    return os.path.join(data_dir, c)

            if dataset.startswith("swe"):
                for f in h5_files:
                    if "rdb" in f.lower() or "swe" in f.lower() or "shallow" in f.lower():
                        return os.path.join(data_dir, f)
            elif dataset.startswith("cns") or dataset.startswith("ns"):
                for f in h5_files:
                    if "cfd" in f.lower() or "cns" in f.lower() or "ns" in f.lower():
                        if "rand" in dataset and "rand" in f.lower():
                            return os.path.join(data_dir, f)
                        if "turb" in dataset and "turb" in f.lower():
                            return os.path.join(data_dir, f)
                # Fallback: return first CFD file
                for f in h5_files:
                    if "cfd" in f.lower() or "cns" in f.lower():
                        return os.path.join(data_dir, f)

            if h5_files:
                return os.path.join(data_dir, h5_files[0])

        raise FileNotFoundError(
            f"No HDF5 file found for dataset '{dataset}' in {data_dir}. "
            f"Run download_pdebench.py first."
        )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, CellComplex, torch.Tensor]:
        """
        Returns:
            node_features: (n_nodes, f_in)
            cell_complex: shared CellComplex instance (CPU)
            target: (n_nodes, out_dim)
        """
        return self.inputs[idx], self._cell_complex, self.targets[idx]

    def get_cell_complex(self) -> CellComplex:
        """Return the shared CellComplex (for model initialization)."""
        return self._cell_complex


# ---------------------------------------------------------------------------
# Train / Val / Test split
# ---------------------------------------------------------------------------

def split_dataset(
    dataset: PDEBenchDataset,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    data_fraction: float = 1.0,
) -> Tuple[Subset, Subset, Subset]:
    """
    Split dataset into train/val/test subsets.

    Args:
        dataset: PDEBenchDataset instance.
        train_frac, val_frac, test_frac: Split ratios (should sum to 1).
        seed: Random seed.
        data_fraction: Fraction of training data to use (for data efficiency experiments).

    Returns:
        train_set, val_set, test_set: Subset instances.
    """
    n = len(dataset)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    if data_fraction < 1.0:
        n_use = max(1, int(len(train_idx) * data_fraction))
        train_idx = train_idx[:n_use]
        print(f"  Data efficiency: using {n_use}/{n_train} training samples "
              f"({data_fraction*100:.0f}%)")

    train_set = Subset(dataset, train_idx.tolist())
    val_set = Subset(dataset, val_idx.tolist())
    test_set = Subset(dataset, test_idx.tolist())

    print(f"  Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# Collate function (for DataLoader)
# ---------------------------------------------------------------------------

def pdebench_collate_fn(
    batch: List[Tuple[torch.Tensor, CellComplex, torch.Tensor]]
) -> Tuple[torch.Tensor, CellComplex, torch.Tensor]:
    """
    Collate function: all samples share the same CellComplex, so features
    can be directly stacked.

    Returns:
        features: (B, n_nodes, f_in)
        cell_complex: shared CellComplex instance
        targets: (B, n_nodes, out_dim)
    """
    features, complexes, targets = zip(*batch)
    features = torch.stack(features, dim=0)
    targets = torch.stack(targets, dim=0)
    return features, complexes[0], targets


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test PDEBench dataset loading")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(_PROJECT_ROOT, "data", "pdebench"))
    parser.add_argument("--dataset", type=str, default="swe",
                        choices=["swe", "cns_rand", "cns_turb"])
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    ds = PDEBenchDataset(
        data_dir=args.data_dir,
        dataset=args.dataset,
        grid_size=args.grid_size,
        n_samples=args.n_samples,
    )

    print(f"\nDataset length: {len(ds)}")
    feat, K, tgt = ds[0]
    print(f"  feat:   {feat.shape} {feat.dtype}")
    print(f"  target: {tgt.shape} {tgt.dtype}")
    print(f"  K.n0={K.n0}, K.n1={K.n1}, K.n2={K.n2}")
    print(f"  K.pos: {K.pos.shape}")
    print(f"  K.d0:  {K.d0.shape}")
    print(f"  K.d1:  {K.d1.shape}")

    # Verify d1 @ d0 = 0
    check = torch.sparse.mm(K.d1, K.d0).to_dense()
    print(f"  d1@d0 max abs: {check.abs().max().item():.2e}")

    # Test DataLoader
    from torch.utils.data import DataLoader
    train, val, test = split_dataset(ds, data_fraction=0.5)
    loader = DataLoader(train, batch_size=4, collate_fn=pdebench_collate_fn)
    for batch_feat, batch_K, batch_tgt in loader:
        print(f"\n  Batch: feat={batch_feat.shape}, tgt={batch_tgt.shape}")
        break
