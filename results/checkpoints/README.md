# Model zoo

Fifteen trained models from the experiments of [REPORT.md](../../REPORT.md) (4.0 MB in total).  Each directory is a
trainer run directory reduced to `best.pt` (the model with the best validation R2), `config.json` (trainer arguments,
model configuration and the data normalisation) and `result.json` (test metrics).  The full result files of every run
are under `results/cab75/<source>` (`results/cab16/<source>` for the two MeshGraphNet baselines, trained on the second
machine).  The last three rows are the comparison runs of the qualitative field figures (REPORT.md section 6.8,
`scripts/make_field_figures.py`).

| checkpoint | task | model | params | test R2 | 4x / transfer R2 | size | source |
|---|---|---|---:|---:|---|---:|---|
| `HP_k100_S3c_solver_tensor_learn` | HP_k100 | solver mode, learned full-SPD tensor metric; material only in the metric | 2355 | 0.9999 | 1.0000 | 20 KB | `stage3/HP_k100_S3c_solver_tensor_learn` |
| `HP_k100_S3c_solver_diag_learn` | HP_k100 | solver mode, learned diagonal metric; material only in the metric | 1201 | 0.9996 | 0.9996 | 14 KB | `stage3/HP_k100_S3c_solver_diag_learn` |
| `HP_k100_S3e_fem_frozen_tensor_faceref` | HP_k100 | solver mode, frozen tensor metric = true face conductivity (the exact P1 solver; only the linear gain is trained) | 46 | 1.0000 | 1.0000 | 6 KB | `stage3/HP_k100_S3e_fem_frozen_tensor_faceref` |
| `AHP_r100_tensor-solver_lr3e-4_s42` | AHP_r100 | solver mode, learned tensor metric, misaligned anisotropy ratio 100 (lr 3e-4, 60 epochs) | 2451 | 0.9639 | 0.9579 | 21 KB | `aniso/AHP_r100_tensor-solver_lr3e-4_s42` |
| `AHP_r100_diag-solver_s42` | AHP_r100 | solver mode, learned diagonal metric (the M-matrix-limited comparison) | 1233 | 0.8318 | 0.6989 | 14 KB | `aniso/AHP_r100_diag-solver_s42` |
| `ASURF_r100_tensor-solver_s42` | ASURF_r100 | solver mode, learned tensor metric, fibre diffusion on closed surfaces | 2451 | 0.9609 | 0.9562 | 21 KB | `aniso_solver/ASURF_r100_tensor-solver_s42` |
| `T5g_solver_learn_tensor_s42` | T5g | solver mode (Neumann), learned tensor metric; the paper's T5 electrostatics target | 2195 | 1.0000 | test100 1.0000 | 20 KB | `paper/T5g_solver_learn_tensor_s42` |
| `T6f_s42` | T6f | general stack; U(1) edge connection to face flux (transfers to new meshes) | 73044 | 1.0000† | test100 1.0000 | 334 KB | `new/T6f_s42` |
| `T6_native_s42` | T6 | general stack, native inputs + orientation map; the paper's T6 target | 73044 | 1.0000 | test100 1.0000 | 334 KB | `paper/T6_native_s42` |
| `T3_native_s42` | T3 | general stack, least-squares vector readout; the paper's T3 target | 29621 | 0.9958 | test100 0.9961 | 165 KB | `paper/T3_native_s42` |
| `SURF_s42` | SURF | general stack; screened Poisson on variable closed surfaces | 89557 | 0.9967 | geo 0.9965 / topo 0.9964 | 399 KB | `suite/SURF_s42` |
| `DYNfix_s42` | DYNfix | general stack; advection-diffusion step (1-form velocity input) for rollouts | 89749 | 0.9997 | - | 400 KB | `suite/DYNfix_s42` |
| `HP_k100_general_s42` | HP_k100 | general stack (diagonal metric, polynomial layers, 100 epochs); the material is an ordinary input | 89941 | 0.8255 | 0.6992 | 401 KB | `new/HP_k100_s42` |
| `HP_k100_mgn_s42` | HP_k100 | MeshGraphNet baseline (15 message-passing steps, parameter-matched to the v2 model) | 92376 | 0.9488 | 0.2581 | 463 KB | cab16 `baselines/HP_k100_native/mgn_s42` |
| `AHP_r100_mgn_s42` | AHP_r100 | MeshGraphNet baseline (15 message-passing steps, parameter-matched to the v2 model) | 92401 | 0.6649 | 0.3998 | 463 KB | cab16 `aniso/AHP_r100_mgn_s42` |

† uncentred R2 (odd cochain target).  4x: zero-shot test on 4x finer meshes.

## Loading

```python
from rhmp import RHMP

model = RHMP.from_checkpoint("results/checkpoints/HP_k100_S3c_solver_tensor_learn/best.pt", map_location="cpu")
print(model.cfg.layer_types, model.cfg.metric_type, model.num_parameters())     # ['solve'] tensor 2355
```

`RHMP.from_checkpoint` accepts the trainer's `best.pt` (and `last.pt`, and files written by `model.to_checkpoint()`).
Inputs are `{degree: (n_k, B, F_k)}` tensors in the normalisation of the run (`config.json['data']`); columns listed in
`config.json['raw_input_columns']` (material / metric-reference columns) are raw physical log values.  The two
MeshGraphNet checkpoints are baselines, not `RHMP` models: rebuild them with
`rhmp.baselines.registry.from_checkpoint(torch.load(path, weights_only=False), task)` (the task fixes their input
encoders), or evaluate them with `--eval-ckpt` as below, which handles both kinds.

[load_example.py](load_example.py) applies the HP_k100 tensor solver to a new random mesh with a synthetic
conductivity field (no data set needed; CPU, a few seconds) and compares its learned per-face tensors with the input
field.  Output on the release machine:

```
loaded HP_k100_S3c_solver_tensor_learn: layers ['solve'], metric tensor (full), material columns {1: 1, 2: 1}, 2355 parameters
new mesh: 1320 vertices, 2518 triangles; output u (1320, 1, 1), rms 0.028 (normalised units)
learned log det(sigma_f)/2 vs input log sigma_f: r = 0.9918, slope 1.004, intercept +0.028; median anisotropy ratio 1.36
```

## Evaluating on the task data

With the data sets in place ([datasets/README.md](../../datasets/README.md)), a zoo directory works as a run directory:

```bash
python -m rhmp.train --task HP_k100 --native --eval-ckpt results/checkpoints/HP_k100_S3c_solver_tensor_learn --out runs/zoo_hp
python3 scripts/eval_on.py --run results/checkpoints/DYNfix_s42 --task DYNfix --rollout --steps 100 --project-mass --out runs/zoo_dyn
python3 scripts/metric_recovery.py results/checkpoints/HP_k100_S3c_solver_tensor_learn --out runs/zoo_recovery
python3 scripts/t6_mesh_transfer.py --run results/checkpoints/T6f_s42 --mesh-seed 7 --n-pts 4096 --n 500
```

The DYNfix rollout with mass projection reproduces `results/cab75/suite/DYNfix_s42/rollout_DYNfix_proj.json`
exactly, and the metric recovery of the HP_k100 tensor solver reproduces the stored statistics (face r = 0.996,
slope 1.039, intercept +0.005).
