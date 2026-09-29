# Vendored v1 code.  Original: module `baselines_operator` of the RHMP v1 code base,
#   https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing
# which accompanies "Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame Equivariance"
# (Zheng & Allen-Blanchette, arXiv:2608.14556).  Kept verbatim (only this header was added).
# Used by rhmp.baselines (the v1 baselines and the v1 model `ours_v1`) and by `python -m rhmp.train --eval-v1`.
"""
Neural Operator Baselines: FNO + DeepONet
For regular-grid PDE tasks (CNS, SWE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# FNO (Fourier Neural Operator) — 2D
# ============================================================

class SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT -> learnable frequency weights -> iFFT"""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(torch.view_as_real(
            torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat) * scale))
        self.weights2 = nn.Parameter(torch.view_as_real(
            torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat) * scale))

    def compl_mul2d(self, input, weights):
        """Complex multiplication: (B,I,X,Y) x (I,O,X,Y) -> (B,O,X,Y)"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(B, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        w1 = torch.view_as_complex(self.weights1.contiguous())
        w2 = torch.view_as_complex(self.weights2.contiguous())
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], w1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], w2)

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    """
    Fourier Neural Operator for 2D PDE.
    Input:  (B, nx, ny, f_in) -> Output: (B, nx, ny, out_dim)
    """

    def __init__(self, f_in, out_dim, width=64, modes=12, n_layers=4):
        super().__init__()
        self.width = width
        self.n_layers = n_layers

        self.fc0 = nn.Linear(f_in, width)

        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SpectralConv2d(width, width, modes, modes))
            self.ws.append(nn.Conv2d(width, width, 1))
            self.norms.append(nn.InstanceNorm2d(width))

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x, grid_size=None):
        """
        x: (B, n_nodes, f_in) -- flattened grid features
        grid_size: (nx, ny) tuple
        Returns: (B, n_nodes, out_dim)
        """
        B, N, f_in_actual = x.shape
        if grid_size is None:
            gs = int(math.sqrt(N))
            grid_size = (gs, gs)
        nx, ny = grid_size

        x = x.reshape(B, nx, ny, -1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (B, width, nx, ny)

        for conv, w, norm in zip(self.convs, self.ws, self.norms):
            x1 = conv(x)
            x2 = w(x)
            x = norm(x1 + x2)
            x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)  # (B, nx, ny, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.reshape(B, N, -1)


# ============================================================
# DeepONet
# ============================================================

class DeepONet(nn.Module):
    """
    DeepONet: Branch (input function) + Trunk (query positions) -> output.
    Simplified variant with shared grid.
    """

    def __init__(self, f_in, out_dim, branch_width=128, trunk_width=128, n_basis=64, pos_dim=2):
        super().__init__()
        self.n_basis = n_basis
        self.out_dim = out_dim

        # Branch net: input function -> basis coefficients
        self.branch = nn.Sequential(
            nn.Linear(f_in, branch_width), nn.GELU(),
            nn.Linear(branch_width, branch_width), nn.GELU(),
            nn.Linear(branch_width, n_basis * out_dim)
        )

        # Trunk net: query positions -> basis functions
        self.trunk = nn.Sequential(
            nn.Linear(pos_dim, trunk_width), nn.GELU(),
            nn.Linear(trunk_width, trunk_width), nn.GELU(),
            nn.Linear(trunk_width, n_basis)
        )

        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, pos=None, grid_size=None):
        """
        x: (B, n_nodes, f_in) -- input function values
        pos: (n_nodes, 2) -- node positions (optional; generates uniform grid if None)
        Returns: (B, n_nodes, out_dim)
        """
        B, N, _ = x.shape

        if pos is None:
            if grid_size is None:
                gs = int(math.sqrt(N))
                grid_size = (gs, gs)
            nx, ny = grid_size
            gx = torch.linspace(0, 1, nx, device=x.device)
            gy = torch.linspace(0, 1, ny, device=x.device)
            gxx, gyy = torch.meshgrid(gx, gy, indexing='ij')
            pos = torch.stack([gxx.reshape(-1), gyy.reshape(-1)], dim=-1)

        # Branch: global pooling -> coefficients
        x_global = x.mean(dim=1)  # (B, f_in)
        branch_out = self.branch(x_global)  # (B, n_basis * out_dim)
        branch_out = branch_out.reshape(B, self.n_basis, self.out_dim)

        # Trunk: position -> basis functions
        trunk_out = self.trunk(pos[:, :self.trunk[0].in_features])  # (N, n_basis)

        # out = trunk @ branch: (B, N, out_dim)
        out = torch.einsum("nk,bko->bno", trunk_out, branch_out) + self.bias
        return out


# ============================================================
# Interface adapters: make FNO/DeepONet compatible with mesh-based training scripts
# ============================================================

class FNOWrapper(nn.Module):
    """Wrap FNO to accept (x, K) interface like GNN baselines."""

    def __init__(self, f_in, out_dim, width=64, modes=12, n_layers=4):
        super().__init__()
        self.fno = FNO2d(f_in, out_dim, width, modes, n_layers)
        self.grid_size = None

    def forward(self, x, K=None):
        """x: (n_nodes, f_in) -> (n_nodes, out_dim)"""
        return self.fno(x.unsqueeze(0), self.grid_size).squeeze(0)

    def forward_batch(self, x_batch, K=None):
        """x_batch: (B, n_nodes, f_in) -> (B, n_nodes, out_dim)"""
        return self.fno(x_batch, self.grid_size)


class DeepONetWrapper(nn.Module):
    """Wrap DeepONet to accept (x, K) interface. Works on any mesh via K.pos."""

    def __init__(self, f_in, out_dim, branch_width=128, trunk_width=128, n_basis=64):
        super().__init__()
        self.net = DeepONet(f_in, out_dim, branch_width, trunk_width, n_basis)

    def forward(self, x, K=None):
        pos = K.pos[:, :2] if K is not None else None
        return self.net(x.unsqueeze(0), pos=pos).squeeze(0)

    def forward_batch(self, x_batch, K=None):
        pos = K.pos[:, :2] if K is not None else None
        return self.net(x_batch, pos=pos)
