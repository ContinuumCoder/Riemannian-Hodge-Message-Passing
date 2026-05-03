"""
After seed runs complete, aggregate per-seed metrics into mean±std across seeds.

Reads checkpoints from:
  - seed=42 (baseline): checkpoints/{task}/{model}/best_model.pt
  - seed=1, 2, ...:     checkpoints/seed_<N>/{task}/{model}/best_model.pt

For each seed it invokes experiments/compute_all_metrics.py with --ckpt-root,
caching the per-seed output as status/all_metrics_seed<N>.json. Then collapses
across seeds.

Output: status/seed_metrics.json with structure
  {task: {model: {metric: {mean, std, n_seeds, per_seed: {seed_id: val}}}}}

Usage:
    python3 scripts/aggregate_seeds.py
    python3 scripts/aggregate_seeds.py --seeds 42 1 2
"""
import os, sys, json, subprocess, argparse
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATUS_DIR = os.path.join(ROOT, 'status')


def ckpt_root_for(seed):
    if seed == 42:
        return os.path.join(ROOT, 'checkpoints')
    return os.path.join(ROOT, 'checkpoints', f'seed_{seed}')


def run_metrics_for_seed(seed):
    """Invoke compute_all_metrics.py with --ckpt-root pointed at the right tree."""
    ckpt_root = ckpt_root_for(seed)
    if not os.path.isdir(ckpt_root):
        print(f'  no checkpoint root {ckpt_root}, skipping seed {seed}')
        return None

    if seed == 42:
        cached = os.path.join(ckpt_root, 'all_metrics.json')
        if os.path.exists(cached):
            data = json.load(open(cached))
            if any(data.get(t) for t in data):
                return data

    os.makedirs(STATUS_DIR, exist_ok=True)
    out_path = os.path.join(STATUS_DIR, f'all_metrics_seed{seed}.json')
    if os.path.exists(out_path):
        return json.load(open(out_path))

    cmd = [
        sys.executable, '-u',
        os.path.join(ROOT, 'experiments/compute_all_metrics.py'),
        '--ckpt-root', ckpt_root,
        '--out', out_path,
    ]
    print(f'  running eval for seed={seed}: {ckpt_root}', flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        print(f'  eval failed for seed={seed}, rc={rc}')
        return None
    if not os.path.exists(out_path):
        return None
    return json.load(open(out_path))


def aggregate(seeds):
    per_seed = {}
    for s in seeds:
        d = run_metrics_for_seed(s)
        if d is not None:
            per_seed[s] = d
    agg = {}
    tasks = set()
    for d in per_seed.values():
        tasks.update(d.keys())
    for t in sorted(tasks):
        agg[t] = {}
        models = set()
        for d in per_seed.values():
            if t in d:
                models.update(d[t].keys())
        for m in sorted(models):
            vals = {}
            for s, d in per_seed.items():
                if t in d and m in d[t]:
                    for metric, v in d[t][m].items():
                        vals.setdefault(metric, {})[s] = v
            agg[t][m] = {}
            for metric, by_seed in vals.items():
                arr = np.array(list(by_seed.values()), dtype=np.float64)
                agg[t][m][metric] = {
                    'mean': float(arr.mean()),
                    'std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    'n_seeds': int(len(arr)),
                    'per_seed': {str(k): float(v) for k, v in by_seed.items()},
                }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 1, 2])
    ap.add_argument('--out', default=os.path.join(STATUS_DIR, 'seed_metrics.json'))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    agg = aggregate(args.seeds)
    with open(args.out, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f'Wrote: {args.out}')
    for t in sorted(agg):
        if 'ours' in agg[t] and 'SSIM' in agg[t]['ours']:
            s = agg[t]['ours']['SSIM']
            print(f"  {t:<35} ours SSIM = {s['mean']:.3f} ± {s['std']:.3f}  (n={s['n_seeds']})")


if __name__ == '__main__':
    main()
