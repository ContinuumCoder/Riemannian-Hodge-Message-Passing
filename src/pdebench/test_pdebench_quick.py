"""
Quick validation of the PDEBench SWE data loading + model training pipeline.
Uses a small grid (32x32) and few samples to verify the end-to-end flow.
"""
import sys
import os
import torch
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from dataset_pdebench import PDEBenchDataset
from gauge_hodge_mp.network import GaugeHodgeNetwork

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = os.path.join(PROJECT_ROOT, "data", "pdebench")

    print("=" * 60)
    print("PDEBench SWE Quick Test")
    print("=" * 60)

    # 1. Load data (small grid, few samples)
    t0 = time.time()
    dataset = PDEBenchDataset(
        data_dir=data_dir,
        dataset="swe",
        grid_size=32,
        time_step_gap=10,
        n_samples=50,
    )
    print(f"  Data loaded in {time.time() - t0:.1f}s")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  f_in={dataset.f_in}, out_dim={dataset.out_dim}")

    K = dataset.get_cell_complex()
    print(f"  CellComplex: n0={K.n0}, n1={K.n1}, n2={K.n2}")

    # 2. Build model
    model = GaugeHodgeNetwork(
        f_in=dataset.f_in, C=32, n_layers=2,
        n0=K.n0, n1=K.n1, n2=K.n2,
        task="scalar", out_dim=dataset.out_dim,
        mp_hidden=32, metric_type="diagonal",
    ).to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Move CellComplex to device
    K_dev = K.to(device) if hasattr(K, 'to') else K
    if K_dev.d0.device != torch.device(device):
        K_dev.d0 = K_dev.d0.to(device)
        K_dev.d1 = K_dev.d1.to(device)
        K_dev.pos = K_dev.pos.to(device)
        K_dev.edges = K_dev.edges.to(device)

    # 4. Train for a few epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    print("\nTraining (20 epochs, first 100 samples)...")
    model.train()
    n_train = min(100, len(dataset))
    for epoch in range(20):
        total_loss = 0
        for i in range(n_train):
            f0, _, target = dataset[i]
            f0 = f0.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(f0, K_dev)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_train
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.6f}")

    # 5. Save checkpoint
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "pdebench_swe_quick")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": 20,
        "loss": avg_loss,
        "grid_size": 32,
        "n_samples": 50,
    }, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")
    print("Quick test PASSED!")

if __name__ == "__main__":
    main()
