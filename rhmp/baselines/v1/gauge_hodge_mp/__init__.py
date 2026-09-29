# Vendored v1 code.  Original: module `gauge_hodge_mp` of the RHMP v1 code base,
#   https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing
# which accompanies "Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame Equivariance"
# (Zheng & Allen-Blanchette, arXiv:2608.14556).  Kept verbatim (only this header was added).
# Used by rhmp.baselines (the v1 baselines and the v1 model `ours_v1`) and by `python -m rhmp.train --eval-v1`.
from .cell_complex import CellComplex
from .lifting import InvariantLifting
from .hodge_mp import HodgeMPLayer
from .reconstruction import EquivariantReconstruction
from .network import GaugeHodgeNetwork

__all__ = [
    "CellComplex",
    "InvariantLifting",
    "HodgeMPLayer",
    "EquivariantReconstruction",
    "GaugeHodgeNetwork",
]
