"""Vendored v1 code (the RHMP v1 model and baselines), used only for comparisons.

Origin: the RHMP v1 code base, https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing, which accompanies
"Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame Equivariance" (Zheng & Allen-Blanchette,
arXiv:2608.14556).  The modules are kept verbatim except for package-relative imports:

* ``gauge_hodge_mp``: the v1 model (``GaugeHodgeNetwork``, ``CellComplex``); rebuilt by ``rhmp.train --eval-v1`` and
  by the ``ours_v1`` baseline;
* ``baselines_graph`` (GCN, GAT, SchNet, EGNN), ``baselines_topo`` (MPSN, SCCNN), ``baselines_advanced``
  (GaugeEquivCNN, GEM-CNN, CW Net, Clifford-SMPN), ``baselines_operator`` (FNO, DeepONet): the v1 baselines, whose
  parameters and initialisation ``rhmp.baselines.v1_wrappers`` reuse inside vectorised ``(n, B, C)`` cores;
* ``metrics_v1``: the v1 metric functions (R2 / MSE / MAE, NRMSE, SSIM / Pearson) that ``rhmp.metrics`` reproduces
  (``tests/test_metrics.py``).

Nothing here is imported by ``import rhmp``; ``rhmp.baselines.v1_wrappers.import_v1()`` loads the modules on demand.
"""
