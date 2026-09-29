# Baselines for the RHMP v2 task suite

The baselines are in the package `rhmp/baselines/`. Every baseline trains through the **same trainer** as the v2 model:

```
python3 -u -m rhmp.train --task T6 --legacy --model gat --epochs 100 --seed 42 --out runs/baselines/T6_legacy/gat_s42
bash scripts/run_baselines.sh T6                      # every registered model, sequentially, v1 protocol
bash scripts/run_baselines.sh HP_k100 mgn egnn gat dec_fixed
python3 -m rhmp.baselines.registry --budgets T6 HP_k100  # matched widths / parameter counts
python3 -m rhmp.baselines.registry --bench T6 --models rhmp,mgn,mgn_fast --batch 64   # step time, peak memory
```

The data, split, normalisation, loss, metrics, checkpoints, resume logic and logging are the ones of `rhmp.train`.
Only the model changes (`--model`, default `rhmp`). Trainer options for baselines:
- `--param-budget M`: parameter budget in millions. Default: the trainable-parameter count of the v2 model that
  `rhmp.train` would build for the same task, inputs, `--C` and `--layers`.
- `--model-opts JSON`: builder options, e.g. `'{"n_layers": 8}'`, `'{"hidden": 64}'`, `'{"invariant_inputs": false}'`.

`result.json` records `model` and `model_info`: width, layers, budget, match ratio, input encoding and head.

## 1. Models

| name | family | what it is | width chosen by |
|---|---|---|---|
| `rhmp` | v2 | RHMP v2: learned bounded metric around the reference DEC (cotan) Hodge star | task default `C` |
| `dec_fixed` | v2 control | same stack, **H = star frozen** (metric heads frozen at their zero initialisation): fixed DEC Hodge operators, no metric learning | same as `rhmp` |
| `unit_star` | v2 control | same stack, **star = 1** (combinatorial Laplacians as the prior) + learned metric (= the `--star unit` ablation) | same as `rhmp` |
| `unit_fixed` | v2 control | star = 1 and H = 1 frozen: combinatorial Hodge Laplacians (SCCNN-like operators) inside the v2 stack | same as `rhmp` |
| `ours_v1` | v1 model | v1 `GaugeHodgeNetwork`, paper configuration (diagonal rank-8 metric bases, `mp_hidden=16`, 4 layers) | v1 `C` (not matched) |
| `mgn` | mesh GNN | MeshGraphNet (Pfaff et al., ICLR 2021), **15 processor steps** (standard) | parameter-matched |
| `mgn_fast` | mesh GNN | MeshGraphNet, **8 processor steps** (wider at the same budget) | parameter-matched |
| `gcn` | graph | GCN (Kipf & Welling 2017): normalised adjacency, topology only | parameter-matched |
| `gat` | graph | GAT (Veličković et al. 2018): 4 heads, LayerNorm, residuals | parameter-matched |
| `schnet` | graph | SchNet (Schütt et al. 2017): RBF continuous filters on distances, E(n)-invariant | parameter-matched |
| `egnn` | graph | EGNN (Satorras et al. 2021) without coordinate updates; invariant inputs (v1 protocol) | parameter-matched |
| `gauge_cnn` | mesh | gauge-equivariant mesh CNN (Cohen et al. 2019, as simplified in v1): channel pairs rotated by the edge angle | parameter-matched |
| `gem_cnn` | mesh | GEM-CNN (de Haan et al. 2021, as simplified in v1): Fourier-angle kernels + pair transport | parameter-matched |
| `mpsn` | cell complex | MPSN (Bodnar et al. 2021, as simplified in v1): per-degree adjacency message passing | parameter-matched |
| `sccnn` | cell complex | SCCNN (Yang et al. 2022): scalar polynomial filters of the Hodge Laplacians per degree | parameter-matched |
| `cw_net` | cell complex | CW Network (Bodnar et al. 2021): boundary / coboundary messages through `d0`, `d1` | parameter-matched |
| `clifford_smpn` | cell complex | Clifford simplicial MP (Liu et al. 2024, as simplified in v1): Cl(2,0) multivectors on nodes/edges | parameter-matched |
| `fno` | operator | FNO (Li et al. 2021) on the 32×32 grid; Fourier modes reduced when the budget requires it | parameter-matched (width, modes) |
| `deeponet` | operator | DeepONet (Lu et al. 2021): branch on the per-mesh mean of the inputs, trunk on node positions | parameter-matched |

## 2. Protocol and fairness rules

**Budgets (v1 rule).** `rhmp.baselines.param_match.match_params` reproduces
`formal_benchmark.py::find_hidden_match` of v1. It scans widths 16, 20, 24, ... and takes the first width
whose trainable-parameter count is ≥ target. The width is accepted if the count is ≤ 1.2 × target.
- When a step of 4 overshoots the tolerance (budgets of ~30K), the widths in between are tried one by one.
- Widths a model rejects are skipped (GAT needs widths divisible by its 4 heads).
- Parameter counts come from models built on the `meta` device inside `torch.random.fork_rng`. Matching therefore
  never changes the random initialisation of the final model (tested).
- Results are cached in `runs/param_match.json` (the release caches: `results/param_match.json`,
  `results/param_match_baselines.json`).

The target is the parameter count of the v2 model on the same task and inputs. That budget is small (the v2 model is
parameter-efficient: scalar filter coefficients, no per-cell parameters), so `--param-budget` lets any run use another
budget, e.g. `--param-budget 0.43` for the v1 T6 budget.

**Inputs.** Node-input baselines (all v1 baselines, `ours_v1`, `fno`, `deeponet`) run on the **legacy (v1) inputs**
where these differ from the native ones (T1, T3, T5, T6, T7, T6_100K). This is exactly the v1 protocol and
`scripts/run_baselines.sh` selects it (`MODE=auto`). On all other tasks they read the native inputs through the
fixed, parameter-free `NodeInputEncoder` (`rhmp/baselines/adapters.py`):
- node inputs pass through;
- even inputs of degree k ≥ 1 become the mean over the k-cells incident to the node;
- odd edge inputs become the directional encoding of v1, `(avg, avg·dx, avg·dy[, avg·dz])`, exactly
  `datasets/gen_T6_wilson_loop.py::encode_edges_to_nodes`. An edge value `v` refers to the canonical orientation
  `src → dst` (`src < dst`), and the edge contributes the same `+v` (and `+v·t`, with `t` the unit vector
  `src → dst`) to both endpoints.
- `v·t` does not depend on the orientation convention; the `avg` column changes sign with it, as in v1.

Checks of the encoder:
- encoding the native T6 edge connection reproduces the stored v1 `X_data` to 1e-5 relative
  (`test_node_encoder_reproduces_T6_legacy_inputs`);
- it matches a literal port of the v1 Python loop (`test_node_encoder_matches_v1_loop`);
- encoded columns are standardised with fixed statistics of the training samples (buffers, never trained).

MeshGraphNet and the v2 family always read the native inputs.

The node features of MeshGraphNet are the node encoding above without its orientation-convention-dependent `avg`
column: node inputs, incident means of even inputs, and the one-ring vector field `mean_e v_e t_e` of odd edge
inputs. MeshGraphNet therefore stays exactly equivariant under vertex relabelling (tested).

Its edge features per directed edge are:
- the odd edge inputs `v` (+v forward, −v reverse);
- `v·t`, with `t` the unit vector of the directed edge (the same in both directions);
- the even edge inputs, the relative position `(x_recv − x_send)/s` and `|x_recv − x_send|/s`, with `s` the
  mesh's mean edge length.

These inputs matter on T6 (native, `mgn`, val R2 by epoch):

| MeshGraphNet inputs | ep 2 | ep 5 | ep 10 |
|---|---:|---:|---:|
| raw edge values `±v` only | −0.001 | −0.001 | 0.005 |
| + `v·t` edge features | −0.001 | −0.001 | 0.565 |
| + node vector field (default) | 0.423 | 0.729 | 0.828 |
| legacy node encoding (v1 inputs), geometric edges | 0.039 | 0.473 | 0.826 |

Without these inputs, narrow ReLU MLPs have to assemble the product of the 1-form with its direction, and the
one-ring aggregate, by themselves.

EGNN follows the v1 rule: it only sees the invariant `avg` columns. The v1 code intends this but selects columns 0, 3, 6
of the T7 inputs `[avg(3), avg·dx(3), avg·dy(3)]`; here EGNN gets the three `avg` columns.
`--model-opts '{"invariant_inputs": false}'` gives EGNN the directional columns.

**Targets.** Every model predicts the task target, on its cells, in normalised units. The loss is MSE, as for the v2 model.
- The v2 family keeps the orientation output map of the task. The v2 model is exactly orientation-equivariant and
  reflection-invariant, so it needs the map to output the pseudo-scalar targets of native T1/T6/T7 and the `n × ∇ψ`
  target of T3.
- The other models are not equivariant and predict those targets directly. The trainer drops `task.output_map`
  for them and records this as `model_info.dropped_output_map`.
- Heads (`adapters.py`):
  - node targets use the model's own v1 node head;
  - edge targets on node-feature models use `g([x_src, x_dst, ℓ_e])`, antisymmetrised `g(a,b) − g(b,a)` for
    orientation-odd cochain targets (exactly odd under edge reversal) and symmetrised for even ones;
  - face targets on node-feature models use `σ_f · MLP(mean_{v∈f} x_v)` for odd targets on planar meshes
    (`σ_f` = sign of the signed area), otherwise the plain mean head;
  - cell-complex models read the target degree's own features, concatenated with incidence means of their other
    degrees (edge heads also get `x_src + x_dst`, `x_dst − x_src`, `ℓ_e`). This is the symmetric **node head**
    (`head='node'`);
  - **v1 edge protocol** for `mpsn`, `sccnn` and `clifford_smpn` on the v1 paper tasks T1, T2, T3, T5, T6 and T7
    (the default there, `head='auto'`). v1 never scored these three models on node targets: the v1 training script
    `formal_benchmark.py` trained them with a 1-channel edge readout on the edge-averaged target `0.5·(y_src + y_dst)`,
    and the v1 evaluation script `compute_all_metrics.py` scored them in **edge space**, keeping only the first target
    component (T3, T7). `rhmp.baselines.registry.prepare_task` applies exactly this transform, and the trainer does so
    automatically:
    - the target becomes `even:1` on the edges, normalised with the node statistics as in v1;
    - the v1 readout reads `x_1`;
    - all metrics (R2, SSIM, Pearson, NRMSE, test100) are edge-space metrics.

    Their paper-task numbers are therefore comparable with the v1 paper tables, **not** with the node-target R2 of the
    other models. `--model-opts '{"head": "node"}'` gives the node-target version (the default on all other tasks),
    and `'{"v1_components": "all"}'` keeps every target component.
  - MeshGraphNet decodes node latents, or `dec(e_ab) − dec(e_ba)` of the two directed edge latents for odd edge
    targets.

**Execution.** All v1 baselines are re-executed in the v2 layout `(n, B, C)` by vectorised cores that hold the
original v1 module (vendored in `rhmp/baselines/v1/`; same parameters and initialisation; `rhmp/baselines/v1_wrappers.py`):
- one pass per batch; gathers `index_select`, scatters `index_add_`, sparse products `rhmp.ops.spmm` on CSR
  operators cached once per complex;
- block-diagonal batches of variable meshes;
- per-mesh constants (length scale, Laplacian normalisation, global means) are computed per graph, so predictions
  never depend on the other samples of a batch (tested, 1e-5);
- with `v1_exact=True`, raw geometry and un-normalised SCCNN, every core reproduces the v1 forward pass to float
  round-off (`test_core_fidelity`, 10 models), and `ours_v1` on the adapter reproduces the v1 `forward_batch` on the v1
  `CellComplex` (`test_ours_v1_wrapper_matches_v1`).

In v1, `BaselineBase.forward_batch` loops over the samples, and the v1 GCN/SCCNN densify `n × n` matrices in every
forward pass. The re-executed cores train one T6 epoch in 0.9-2.1 s (cw_net, egnn, gat, sccnn; measured in 2-epoch
runs). The v1 T6 runs of the same baselines took 5.7-30.3 s per epoch (`checkpoints_v1/T6_wilson_loop/*/history.json`,
at the larger v1 widths). A like-for-like timing of the unmodified v1 modules is available as
`registry --bench T6 --v1-original`.

**Deliberate deviations from v1.** Defaults are the stronger or fairer variant; all are switchable.

| model | v1 | here (default) | switch |
|---|---|---|---|
| SchNet, EGNN | raw coordinates of unit-square meshes: SchNet's fixed RBF grid on [0, 5] puts every edge length (~0.03) into its first bin; EGNN sees `d² ~ 1e-3` | distances divided by the mesh's mean edge length | `geometry='raw'` |
| GaugeEquivCNN | the message computed from `x_src` is added to both endpoints (the source receives its own transported feature) | `dst` receives `T(x_src)`, `src` receives `T'(x_dst)` (reverse direction) | `v1_exact=True` |
| GEM-CNN | aggregates only along canonical edges `src → dst` (`src < dst`: information flows from lower to higher vertex ids only); no nonlinearity (a linear model) | both directions (reverse angle θ+π), SiLU between convolutions as in the original GEM-CNN | `v1_exact=True` |
| SCCNN | raw combinatorial Laplacians with cubic polynomials (activations grow ~1e2 per layer: 1e7 after 4 layers on T6) | **v1 protocol (paper tasks): raw, as v1.** Elsewhere: each mesh block divided by its Gershgorin bound **and** filters initialised as identity (`w_0 += 1`) | `normalize`, `filter_init` |
| MPSN, SCCNN, Clifford | edge readout on edge-averaged targets, edge-space metrics, first component | **paper tasks: identical (v1 edge protocol).** Other tasks: node head on the task target | `head='node'` / `'v1_edge'` |
| DeepONet | global mean over all nodes of the batch | per-mesh mean in block-diagonal batches | — |

**SCCNN initialisation.** Normalised Laplacians combined with the `N(0, 0.1)` filter coefficients of v1 make SCCNN
collapse, which is why the default above also initialises the filters as identity.
- With these coefficients the identity term of the filter shrinks the signal ~10× per layer. After four layers only
  the biases remain: prediction spread 1e-6, gradients on the early layers 1e-10.
- The model then predicts a constant (T6 and T3 100-epoch runs: val R2 −0.0004).
- `test_sccnn_normalised_needs_identity_init` and `test_trainable_on_learnable_tiny_task` guard against this.
- All SCCNN results in REPORT.md use the defaults above.
- 3-epoch check on T6 legacy, val R2 at epoch 3:

  | variant | val R2 (epoch 3) |
  |---|---:|
  | v1 edge protocol, raw | 0.023 |
  | v1 edge protocol, normalised + identity | 0.022 |
  | node head, normalised + identity | 0.025 |
  | normalised, without identity initialisation | −0.0005 |

  MPSN is at 0.027 at the same point.

## 3. What each baseline can represent

Task families (see `rhmp/tasks/__init__.py`):
- **node** targets with node inputs: T2, T8, legacy T1/T3/T5/T6/T7, HP (legacy), TET (legacy);
- **edge-input** tasks: native T6/T7 (connection on edges), native T1 (velocity 1-form), HP/TET (even edge/face/tet
  conductivities);
- **face** targets: T6f (plaquette flux `d1 θ`) and T7f (`F = dA + [A, A]`);
- **edge** targets: HPflux / HPfluxd / TETflux (flux 1-cochains);
- mesh kinds: planar Delaunay (T5–T8, HP), 3-D surfaces (T2 torus, T3 ellipsoid), quads (T1q), tetrahedra (TET),
  grids (T1, T1q), variable meshes (T8, HP, TET).

| model | shared meshes | variable meshes | quads | tets | edge inputs | edge targets | face targets | notes |
|---|---|---|---|---|---|---|---|---|
| v2 family | ✓ | ✓ | ✓ | ✓ | native cochains, exact connection mode | odd `cochain:1` readout | `cochain:2` readout, exact `d1` structure available | the only models with exact `d²=0` / gauge structure |
| `ours_v1` | triangles only | ✗ per-cell metric parameters | ✗ | ✗ | node-encoded | v1 `edge_scalar` | ✗ no face readout | batch-coupled metric in training (v1) |
| `mgn` / `mgn_fast` | ✓ | ✓ | ✓ | ✓ | **native on edges** (`±v`, `v·t` per directed edge) + node vector field | exactly odd directed-edge decoder | mean of vertex latents × σ_f | translation-invariant, not rotation-invariant |
| node graph models (`gcn`, `gat`, `schnet`, `egnn`) | ✓ | ✓ | ✓ | ✓ | node-encoded (lossy) | exactly odd pair head | σ_f × mean-vertex head | no access to edge values themselves; `d1` (curl) must be learned from the encoding |
| `gauge_cnn`, `gem_cnn` | ✓ | ✓ | ✓ | ✗ 2-manifold methods | node-encoded | pair head | σ_f × mean head | edge angles from the **xy projection** (v1 simplification): not intrinsic on 3-D surfaces (T2, T3) |
| `mpsn`, `sccnn`, `cw_net` | ✓ | ✓ | ✓ (polygon faces) | ✓ 2-skeleton | node-encoded, lifted to edges/faces | edge features (no definite parity) + node difference | face features | v1 lift uses face means (`|d1|/3` for triangles); `mpsn`/`sccnn` on paper tasks: v1 edge protocol |
| `clifford_smpn` | ✓ | ✓ | ✓ | ✓ | node-encoded | edge multivectors | vertex/edge means | no face features; on paper tasks: v1 edge protocol |
| `fno` | 32×32 grid only (T1, T1q) | ✗ | T1q ✓ | ✗ | node-encoded | ✗ | ✗ | vertex `i·ny+j` ↔ grid cell `(i, j)` |
| `deeponet` | ✓ | ✓ | ✓ | ✓ | node-encoded | ✗ | ✗ | uses absolute positions; branch sees only the mesh mean of the inputs (no local operator) |

Consequences worth keeping in mind when reading the tables:
- **Pseudo-scalar targets** (T1 vorticity, T6/T7 node flux) are representable by the non-equivariant baselines
  because their inputs carry the frame: directional encodings, xy angles, absolute or relative positions.
- EGNN and SchNet with invariant inputs cannot tell the orientation of the plane: on T6/T7 they only see the
  orientation-convention-dependent `avg` columns.
- **Face targets** (T6f/T7f) need a circulation around each face. Node-feature baselines only see a per-node
  average of the edge values, so they can only approximate `d1 θ`. MeshGraphNet sees the edge values but has no
  face cells (it decodes face means of node latents). The v2 family has `d1` exactly.
- **Edge-flux targets** (HPflux) are exactly orientation-odd for the pair head of the node-feature models and for
  MeshGraphNet. The internal edge features of the cell-complex models have no parity (their nonlinearities break it);
  these models get the node difference `x_dst − x_src` in the head.
- The complete list of inapplicable (model, task) pairs is in §6.

## 4. Parameter budgets

Matched widths (and MeshGraphNet steps) per task, with the parameter count in parentheses.
- Target = the parameter count of the v2 model on the same task. Node-input models are matched to the v2 model on the legacy
  inputs where those are used (T1, T3, T5, T6, T7), all other models to the native one.
- `!` = outside [1, 1.2] × target after refinement (the "best we can do" result of the v1 rule).
- Regenerate with `python3 -m rhmp.baselines.registry --budgets <tasks> --device cpu`.

| model | T1 | T1q | T2 | T3 | T5 | T6 | T6f | T7 | T7f | T8 | HP_k100 | HPflux_k100 | TET_k100 | TETflux_k100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *target, node-input mode* | 89.9K | 73.4K | 31.4K | 27.3K | 131.1K | 89.8K | 73.0K | 132.5K | 106.3K | 304.6K | 89.9K | 73.4K | 116.7K | 100.2K |
| *target, native* | 73.4K | 73.4K | 31.4K | 29.6K | 105.1K | 73.0K | 73.0K | 106.3K | 106.3K | 304.6K | 89.9K | 73.4K | 116.7K | 100.2K |
| rhmp | 128 (73K) | 128 (73K) | 64 (31K) | 64 (30K) | 160 (105K) | 128 (73K) | 128 (73K) | 160 (106K) | 160 (106K) | 256 (305K) | 128 (90K) | 128 (73K) | 128 (117K) | 128 (100K) |
| dec_fixed | 128 (69K) | 128 (69K) | 64 (27K) | 64 (25K) | 160 (101K) | 128 (69K) | 128 (69K) | 160 (102K) | 160 (102K) | 256 (300K) | 128 (85K) | 128 (69K) | 128 (111K) | 128 (94K) |
| unit_star | 128 (73K) | 128 (73K) | 64 (31K) | 64 (30K) | 160 (105K) | 128 (73K) | 128 (73K) | 160 (106K) | 160 (106K) | 256 (305K) | 128 (90K) | 128 (73K) | 128 (117K) | 128 (100K) |
| unit_fixed | 128 (69K) | 128 (69K) | 64 (27K) | 64 (25K) | 160 (101K) | 128 (69K) | 128 (69K) | 160 (102K) | 160 (102K) | 256 (300K) | 128 (85K) | 128 (69K) | 128 (111K) | 128 (94K) |
| ours_v1 | 128 (424K) | n/a | 64 (532K) | 64 (372K) | 160 (509K) | 128 (433K) | n/a | 160 (510K) | n/a | n/a | n/a | n/a | n/a | n/a |
| mgn | 24x15 (85K) | 24x15 (85K) | 15x15 (34K) | 14x15 (30K) | 28x15 (115K) | 24x15 (85K) | 24x15 (85K) | 28x15 (116K) | 28x15 (116K) | 48x15 (333K) | 25x15 (92K) | 24x15 (85K) | 29x15 (124K) | 28x15 (115K) |
| mgn_fast | 32x8 (83K) | 32x8 (83K) | 20x8 (33K) | 20x8 (33K) | 37x8 (110K) | 32x8 (83K) | 32x8 (83K) | 37x8 (111K) | 37x8 (111K) | 64x8 (326K) | 36x8 (105K) | 32x8 (83K) | 40x8 (129K) | 36x8 (105K) |
| gcn | 172 (90K) | 156 (75K) | 104 (33K) | 96 (28K) | 208 (131K) | 172 (90K) | 136 (75K) | 208 (133K) | 164 (110K) | 320 (310K) | 172 (90K) | 124 (78K) | 196 (117K) | 144 (105K) |
| gat | 172 (92K) | 156 (76K) | 100 (32K) | 96 (30K) | 208 (134K) | 172 (92K) | 136 (77K) | 208 (136K) | 160 (107K) | 316 (306K) | 172 (92K) | 120 (75K) | 196 (119K) | 140 (101K) |
| schnet | 104 (96K) | 92 (76K) | 60 (34K) | 56 (30K) | 124 (134K) | 104 (96K) | 88 (78K) | 124 (135K) | 104 (108K) | 192 (312K) | 104 (96K) | 84 (78K) | 116 (118K) | 96 (101K) |
| egnn | 64 (100K) | 56 (77K) | 36 (32K) | 36 (32K) | 76 (141K) | 64 (100K) | 56 (80K) | 76 (141K) | 68 (118K) | 112 (305K) | 64 (100K) | 56 (83K) | 72 (127K) | 64 (109K) |
| gauge_cnn | 100 (92K) | 92 (78K) | 60 (33K) | 56 (29K) | 120 (131K) | 100 (91K) | 92 (77K) | 124 (141K) | 108 (107K) | 184 (307K) | 100 (91K) | 88 (79K) | n/a | n/a |
| gem_cnn | 48 (93K) | 44 (78K) | 28 (32K) | 28 (32K) | 60 (144K) | 48 (93K) | 44 (80K) | 60 (145K) | 52 (112K) | 88 (311K) | 48 (93K) | 44 (82K) | n/a | n/a |
| mpsn | 72 (97K) | 60 (75K) | 44 (37K) | 40 (31K) | 88 (143K) | 72 (97K) | 60 (75K) | 88 (144K) | 72 (108K) | 124 (313K) | 68 (96K) | 60 (78K) | 76 (119K) | 68 (100K) |
| sccnn | 72 (95K) | 60 (74K) | 44 (36K) | 40 (30K) | 88 (141K) | 72 (95K) | 60 (73K) | 88 (142K) | 76 (118K) | 124 (310K) | 68 (94K) | 60 (77K) | 76 (117K) | 72 (111K) |
| cw_net | 56 (99K) | 52 (86K) | 32 (33K) | 30 (29K) | 68 (146K) | 56 (99K) | 48 (73K) | 68 (146K) | 60 (114K) | 100 (313K) | 56 (99K) | 48 (75K) | 64 (129K) | 56 (102K) |
| clifford_smpn | 72 (91K) | 64 (76K) | 44 (34K) | 40 (29K) | 88 (135K) | 72 (91K) | 64 (76K) | 88 (136K) | 76 (107K) | 132 (319K) | 72 (96K) | 64 (80K) | 80 (118K) | 72 (101K) |
| fno | 20/m4 (107K) | 17/m4 (78K) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| deeponet | 184 (93K) | 160 (74K) | 96 (32K) | 68 (27K) | 212 (132K) | 184 (93K) | n/a | 200 (134K) | n/a | 360 (309K) | 184 (93K) | n/a | 212 (119K) | n/a |

Reading the table:
- `mgn` = width × 15 steps, `mgn_fast` = width × 8 steps; `fno` = width / Fourier modes.
- `mpsn`, `sccnn` and `clifford_smpn` on T1, T2, T3, T5, T6 and T7 carry the v1 edge readout (v1 edge protocol);
  everywhere else they carry the node head.
- The v2 family is not matched: it is the same architecture, and the frozen metric heads of `dec_fixed` /
  `unit_fixed` do not count as trainable.
- `ours_v1` is the v1 paper configuration, 4-5× the v2 budget; its per-cell metric bases account for most of it.
- Every matched entry is within [1, 1.2] × target.
- MeshGraphNet prefers widths divisible by 4. Such widths are ~1.8× faster than e.g. 30 because of kernel
  alignment; single steps are used only when no multiple of 4 meets the tolerance.

## 5. Speed: train step, inference, peak memory

`python3 -m rhmp.baselines.registry --bench <task>`:
- one real training batch: T6, B = 64 samples of the shared mesh; HP_k100, one block-diagonal batch of 8 meshes
  (≈ 12K nodes);
- 3 warm-up steps, then the best of two passes of 20 timed steps (forward + MSE backward + Adam);
- inference under `no_grad`; peak = `torch.cuda.max_memory_allocated`;
- TF32 matmuls on, as in the trainer;
- every model waits for an exclusive GPU (RTX PRO 6000 Blackwell); `exclusive=False` marks rows during which
  another process appeared.

The v2 rows are measured in the same process under the same conditions. They agree with `bench/RESULTS_step.md`:
T6 v2 native 60.7 ms / 17.9 ms / 4.52 GB there; HP_k100 block8 ≈ 38 ms per 8-mesh step (210 samples/s).

**T6, B = 64 (shared mesh), exclusive GPU, fp32 with TF32 matmuls** (`results/benchmarks/baselines_T6.json`):

| model | inputs | params | width × L | train ms | infer ms | peak GB |
|---|---|---:|---|---:|---:|---:|
| `rhmp` (v2) | native | 73.0K | 128 × 4 | 60.5 | 18.2 | 4.51 |
| `dec_fixed` | native | 68.7K | 128 × 4 | 60.0 | 17.9 | 4.51 |
| **`mgn`** | native | 85.2K | 24 × 15 | **29.8** | 10.9 | 3.49 |
| `mgn_fast` (H = 30, a width that is not a multiple of 4) | native | 73.1K | 30 × 8 | 33.9 | 20.1 | 2.48 |
| `ours_v1` (v1 paper model; per-sample evaluation) | legacy | 433.5K | 128 | 104.9 | 937.4 | 7.48 |
| `gcn` | legacy | 90.1K | 172 × 4 | 1.7 | 0.5 | 0.76 |
| `gat` | legacy | 92.2K | 172 × 4 | 15.8 | 4.8 | 3.56 |
| `schnet` | legacy | 95.8K | 104 × 4 | 8.5 | 2.5 | 2.05 |
| `egnn` | legacy | 100.3K | 64 × 4 | 9.6 | 4.2 | 1.90 |
| `gauge_cnn` | legacy | 91.4K | 100 × 4 | 17.7 | 6.5 | 2.37 |
| `gem_cnn` | legacy | 92.6K | 48 × 4 | 19.4 | 6.9 | 3.85 |
| `mpsn` | legacy | 95.7K | 68 × 4 | 13.6 | 4.2 | 2.05 |
| `sccnn` | legacy | 94.1K | 68 × 4 | 16.9 | 5.3 | 3.24 |
| `cw_net` | legacy | 99.1K | 56 × 4 | 6.8 | 2.4 | 1.71 |
| `clifford_smpn` | legacy | 96.0K | 72 × 4 | 28.7 | 7.7 | 1.66 |
| `deeponet` | legacy | 93.0K | 184 | 0.9 | 0.2 | 0.34 |

**T6, B = 64, bf16 AMP** (`--amp`, `results/benchmarks/baselines_T6_amp.json`):

| model | width × L | train ms | infer ms | peak GB |
|---|---|---:|---:|---:|
| `rhmp` | 128 × 4 | 59.6 | 18.1 | 4.91 |
| `mgn` | 24 × 15 | 27.9 | 9.1 | 2.61 |
| `mgn_fast` | 32 × 8 | 19.4 | 5.6 | 2.01 |

**HP_k100, one block-diagonal batch of 8 meshes** (≈ 12K nodes, 35K edges):

| model | width × L | train ms | infer ms | peak GB (step only) | exclusive GPU |
|---|---|---:|---:|---:|---|
| `rhmp` (v2) | 128 × 4 | 33.2 | 10.9 | 0.94 | yes |
| `dec_fixed` | 128 × 4 | 33.4 | 15.7 | 0.94 | no |
| `mgn` | 25 × 15 | 56.8 | 30.1 | 0.74 | no |

About these measurements:
- MeshGraphNet on T6: 15 steps train 2.0× faster than the v2 model at the same budget, and 8 steps with AMP 3.1× faster.
  On shared meshes the `(n, B, C)` layout gives it large, GEMM-friendly tensors.
- On block-diagonal batches of small meshes (B = 1, ~70K directed edges × 25 channels), the 15 MeshGraphNet blocks
  (~70 kernels each per training step) are launch/dispatch-bound and slower than the 4 layers of the v2 model. During
  this measurement the 8 CPUs of the benchmark machine were at load ≈ 15, which slows eager dispatch. Inductor would
  fuse these kernels, but `torch.compile` with Inductor was unavailable on that machine (no `Python.h`; see
  `bench/RESULTS_step.md`). MeshGraphNet compiles cleanly with the `aot_eager` backend (no graph break in the
  processor; tested).
- The MeshGraphNet rows were measured with an earlier input encoding, without the node vector field and the `v·t`
  edge features. The default encoding changes only the input width of the encoders (T6: node 1 → 2, edge 4 → 6
  columns), so the timings are representative.
- Rows marked "no" in the exclusive column were measured while another process used the GPU.  Further
  measurements can be taken with `python3 -m rhmp.baselines.registry --bench HP_k100 --batch 8` (all models,
  `--amp` for bf16) and `--bench T6 --v1-original` (the unmodified v1 modules at the same widths).

## 6. Inapplicable (model, task family) pairs

`rhmp.baselines.registry.applicable(name, task)` implements these rules. `run_baselines.sh` skips such pairs with
the reason, and `rhmp.train --model` refuses them.

| model | inapplicable task families | reason |
|---|---|---|
| `ours_v1` | variable meshes (T8, HP\*, TET\*) | per-cell parameters: metric bases `n_k × 8` sized for one mesh |
| `ours_v1` | quads (T1q) | v1 `CellComplex` / lifting need triangle faces (areas, angles) |
| `ours_v1` | face targets (T6f, T7f) | v1 readouts: `scalar` (nodes), `edge_scalar`, `vector` only |
| `fno` | every task except T1, T1q | needs a regular grid numbered `i·ny + j` (FFT on the 32×32 grid) |
| `fno` | non-node targets | predicts grid (node) values only |
| `deeponet` | face / edge targets (T6f, T7f, HPflux\*, TETflux\*) | the trunk is evaluated at the nodes |
| `gauge_cnn`, `gem_cnn` | tetrahedral volumes (TET\*) | 2-manifold methods (tangent-plane angles); v1 used xy-projected angles |

Applicable, with a caveat:
- `gauge_cnn` / `gem_cnn` on the 3-D surfaces T2 and T3 use the xy projection of the edge directions from v1.
- The cell-complex baselines run on the 2-skeleton of tetrahedral meshes (tet inputs reach them as node means).
- EGNN / SchNet cannot represent reflection-odd (pseudo-scalar) targets from invariant inputs.

## 7. Results

The 100-epoch results of the baselines (T6 and T3 with legacy and native inputs, HP_k100 with zero-shot 4x resolution,
SURF with geometry and topology transfer, T8; v2 and v1 parameter budgets) are in [REPORT.md](../REPORT.md) §6.5 and
[results/RESULTS.md](../results/RESULTS.md) §6; the run files are in `results/baselines/`,
`results/baselines_v1_budget/` and `results/baseline_diagnostics/`.  Two observations from the runs:

- **The GAT budget.** Budget-matched GAT (width 172, 92K parameters) reaches test R2 0.513 on T6 legacy after 100
  epochs; the v1 table reports 0.703.  There is no code difference (fidelity-tested).  v1 matched GAT to its own 0.433M
  model: width 296, 445K parameters counted, 269K of them used (the other 176K are in the unused edge MLP).  At the v1
  width (`--model-opts '{"hidden": 296}'`, `results/baseline_diagnostics/T6_gat_v1width_e100`) GAT learns faster, and with
  the v1 budget (`--param-budget 0.43`) it reaches 0.636.  The gap is a budget effect, not a protocol change.
- **T6 is slow to start for every non-structural baseline**, while the physics-prior controls are fast: after 2
  epochs `rhmp` 0.9975, `unit_star` 0.988, `dec_fixed` 0.984, `unit_fixed` 0.965 (validation R2), i.e. both the DEC
  prior and metric learning add to the exact `d² = 0` structure.  Loading HP_k100 builds 5,500 complexes on the CPU;
  use one thread (`OMP_NUM_THREADS=1`) when the CPU is shared with other jobs.
