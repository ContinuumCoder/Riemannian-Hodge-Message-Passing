"""
Build jobs.json from (tasks × models × seeds), estimating cost from the seed=42
training history in checkpoints/<task>/<model>/history.json.
"""
import os, json, glob, sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Task -> ckpt dir name mapping (from formal_benchmark.py TASKS dict)
TASK_TO_DIRNAME = {
    'T1': 'T1_cns_vorticity',
    'T2': 'T2_torus_advection_diffusion',
    'T3': 'T3_ellipsoid_surface_flow',
    'T5': 'T5_maxwell_poisson',
    'T6': 'T6_wilson_loop',
    'T7': 'T7_yang_mills_su2',
    'T8': 'T8_airfoil_pressure',
}

# Models run per task (from all_metrics.json coverage)
MODELS_PER_TASK = {
    'T1': ['gcn', 'gat', 'schnet', 'egnn', 'mpsn', 'sccnn',
           'gauge_cnn', 'gem_cnn', 'cw_net', 'clifford_smpn', 'fno', 'deeponet', 'ours'],
    'T2': ['gcn', 'gat', 'schnet', 'egnn', 'mpsn', 'sccnn',
           'gauge_cnn', 'gem_cnn', 'cw_net', 'clifford_smpn', 'deeponet', 'ours'],
    'T3': ['gcn', 'gat', 'schnet', 'egnn', 'mpsn', 'sccnn',
           'gauge_cnn', 'gem_cnn', 'cw_net', 'clifford_smpn', 'deeponet', 'ours'],
    'T5': ['gcn', 'gat', 'schnet', 'egnn', 'mpsn', 'sccnn',
           'gauge_cnn', 'gem_cnn', 'cw_net', 'clifford_smpn', 'deeponet', 'ours'],
    'T6': ['gcn', 'gat', 'schnet', 'egnn', 'mpsn', 'sccnn',
           'gauge_cnn', 'gem_cnn', 'cw_net', 'clifford_smpn', 'deeponet', 'ours'],
    'T7': ['gcn', 'gat', 'schnet', 'egnn', 'mpsn', 'sccnn',
           'gauge_cnn', 'gem_cnn', 'cw_net', 'clifford_smpn', 'deeponet', 'ours'],
    'T8': ['gcn', 'gat', 'schnet', 'egnn', 'ours'],
}

DEFAULT_SEEDS = [1, 2]  # augment seed=42 baseline


def estimate_cost_sec(task, model):
    """Read formal_v1 history.json to estimate epoch time × 100 epochs."""
    dname = TASK_TO_DIRNAME[task]
    hp = os.path.join(ROOT, 'checkpoints', dname, model, 'history.json')
    if not os.path.exists(hp):
        return 600  # 10 min fallback
    try:
        h = json.load(open(hp))
        if not h:
            return 600
        total = sum(e.get('time', 0) for e in h)
        return float(total)
    except Exception:
        return 600


def build(seeds=None, epochs=100):
    seeds = seeds or DEFAULT_SEEDS
    jobs = []
    job_num = 0
    for task in sorted(MODELS_PER_TASK):
        for model in MODELS_PER_TASK[task]:
            for seed in seeds:
                est = estimate_cost_sec(task, model)
                # Check if already done in seed_dir
                ckpt = os.path.join(ROOT, 'checkpoints', f'seed_{seed}',
                                    TASK_TO_DIRNAME[task], model, 'best_model.pt')
                result = os.path.join(ROOT, 'checkpoints', f'seed_{seed}',
                                      TASK_TO_DIRNAME[task], model, 'result.json')
                done = os.path.exists(result)
                job_id = f'{task}__{model}__seed{seed}'
                jobs.append({
                    'id': job_id,
                    'task': task,
                    'model': model,
                    'seed': seed,
                    'epochs': epochs,
                    'est_sec': est,
                    'est_min': round(est / 60, 1),
                    'status': 'completed' if done else 'pending',
                    'gpu': None,
                    'pid': None,
                    'start': None,
                    'end': None,
                    'error': None,
                    'log': f'logs/{job_id}.log',
                })
                job_num += 1
    # Longest first (LPT scheduling: optimal-within-ε for makespan on identical machines)
    jobs.sort(key=lambda j: -j['est_sec'])
    return jobs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--out', default=os.path.join(ROOT, 'status/jobs.json'))
    args = ap.parse_args()

    jobs = build(args.seeds, args.epochs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(jobs, f, indent=2)

    total_sec = sum(j['est_sec'] for j in jobs if j['status'] == 'pending')
    per_gpu = total_sec / 2
    print(f'Wrote {len(jobs)} jobs ({sum(1 for j in jobs if j["status"]=="pending")} pending, '
          f'{sum(1 for j in jobs if j["status"]=="completed")} already done).')
    print(f'Total pending work: {total_sec/3600:.1f} GPU-hours')
    print(f'Perfect 2-GPU parallel: {per_gpu/3600:.1f} wall-clock hours')
    print(f'Top 10 longest:')
    for j in jobs[:10]:
        print(f'  {j["id"]:<40} {j["est_min"]:>6.1f} min')


if __name__ == '__main__':
    main()
