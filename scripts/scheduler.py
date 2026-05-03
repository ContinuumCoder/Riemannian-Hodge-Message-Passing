"""
Concurrent 2-GPU scheduler for multi-seed training.

- Reads jobs.json (status/jobs.json), pulls pending jobs longest-first.
- Spawns one worker thread per GPU.
- Each worker serially runs subprocesses of formal_benchmark.py with the
  right CUDA_VISIBLE_DEVICES env.
- Atomic status writes via fcntl lock; safe under crash/resume.
- On start, any 'running' jobs from a prior instance are requeued as 'pending'
  (their Python subprocess died when the parent did).
- The subprocess itself resumes from latest checkpoint_ep*.pt inside its own
  save_dir — so requeued jobs pick up where they left off.

Usage:
    python3 scripts/scheduler.py                  # foreground
    nohup python3 -u scripts/scheduler.py &       # background w/ nohup
    bash scripts/launch.sh                         # convenience wrapper

Options:
    --gpus 0,1            GPU IDs (default 0,1)
    --jobs-file path      jobs.json location
    --dry-run             print plan, don't execute
    --only task[,task]    filter to specific tasks (T1,T6,...)
"""
import os, sys, json, time, subprocess, threading, argparse, signal, fcntl
from contextlib import contextmanager
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_JOBS = os.path.join(ROOT, 'status/jobs.json')
LOG_DIR = os.path.join(ROOT, 'logs')


@contextmanager
def locked_json(path):
    """Open JSON file with exclusive lock, yield parsed data, write back on exit."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            data = json.load(f)
            yield data
            f.seek(0); f.truncate()
            json.dump(data, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def claim_next_job(jobs_path, gpu, task_filter=None):
    """Atomically pick highest-cost pending job not filtered out; mark it running."""
    with locked_json(jobs_path) as jobs:
        pending = [j for j in jobs if j['status'] == 'pending']
        if task_filter:
            pending = [j for j in pending if j['task'] in task_filter]
        if not pending:
            return None
        pending.sort(key=lambda j: -j['est_sec'])
        job = pending[0]
        job['status'] = 'running'
        job['gpu'] = int(gpu)
        job['start'] = now_str()
        return dict(job)  # return copy for worker


def finalize_job(jobs_path, job_id, status, error=None, pid=None):
    with locked_json(jobs_path) as jobs:
        for j in jobs:
            if j['id'] == job_id:
                j['status'] = status
                j['end'] = now_str()
                if error: j['error'] = error
                if pid:   j['pid'] = pid
                return


def reset_stale_running(jobs_path):
    """On scheduler startup, any 'running' jobs are leftovers from a dead parent."""
    count = 0
    with locked_json(jobs_path) as jobs:
        for j in jobs:
            if j['status'] == 'running':
                j['status'] = 'pending'
                j['gpu'] = None
                j['pid'] = None
                count += 1
    return count


def run_job(job, gpu, epochs_override=None):
    """Fork subprocess for formal_benchmark.py. Returns (returncode, log_path)."""
    log_path = os.path.join(LOG_DIR, f"{job['id']}.log")
    os.makedirs(LOG_DIR, exist_ok=True)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['PYTHONUNBUFFERED'] = '1'

    n_epochs = epochs_override if epochs_override else job['epochs']
    cmd = [
        'python3', '-u', 'experiments/formal_benchmark.py',
        job['task'],
        '--model', job['model'],
        '--seed', str(job['seed']),
        '--epochs', str(n_epochs),
        '--multi-seed-dir',
    ]

    with open(log_path, 'a') as lf:
        lf.write(f"\n{'='*70}\n[{now_str()}] launching on GPU {gpu}: {' '.join(cmd)}\n{'='*70}\n")
        lf.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        rc = proc.wait()
        lf.write(f"\n[{now_str()}] exit code {rc}\n")
    return rc, log_path, proc.pid


def worker_loop(gpu, jobs_path, task_filter=None, epochs_override=None, stop_flag=None):
    print(f'[GPU {gpu}] worker started', flush=True)
    while stop_flag is None or not stop_flag.is_set():
        job = claim_next_job(jobs_path, gpu, task_filter)
        if job is None:
            print(f'[GPU {gpu}] queue empty, worker exiting', flush=True)
            return
        print(f"[GPU {gpu}] >>> {job['id']} (est {job['est_min']:.1f} min)", flush=True)
        t0 = time.time()
        try:
            rc, log_path, pid = run_job(job, gpu, epochs_override=epochs_override)
            status = 'completed' if rc == 0 else 'failed'
            error = None if rc == 0 else f'exit {rc}'
            finalize_job(jobs_path, job['id'], status, error=error, pid=pid)
            print(f"[GPU {gpu}] <<< {job['id']} {status} in {(time.time()-t0)/60:.1f} min", flush=True)
        except Exception as e:
            finalize_job(jobs_path, job['id'], 'failed', error=str(e))
            print(f"[GPU {gpu}] ERROR {job['id']}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default='0,1', help='comma-sep GPU ids')
    ap.add_argument('--per-gpu', type=int, default=1, help='concurrent workers per GPU')
    ap.add_argument('--jobs-file', default=DEFAULT_JOBS)
    ap.add_argument('--only', default=None, help='comma-sep task filter (T1,T6)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--epochs', type=int, default=None, help='override per-job epochs')
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(',') if g.strip()]
    task_filter = None
    if args.only:
        task_filter = {t.strip() for t in args.only.split(',') if t.strip()}

    if not os.path.exists(args.jobs_file):
        print(f'ERROR: jobs file not found: {args.jobs_file}')
        print('Run scripts/build_jobs.py first.')
        sys.exit(1)

    if args.dry_run:
        with open(args.jobs_file) as f:
            jobs = json.load(f)
        pend = [j for j in jobs if j['status'] == 'pending']
        if task_filter:
            pend = [j for j in pend if j['task'] in task_filter]
        print(f'Would run {len(pend)} jobs on GPUs {gpus}')
        print(f'Estimated wall-clock: {sum(j["est_sec"] for j in pend)/60/len(gpus):.0f} min')
        return

    stale = reset_stale_running(args.jobs_file)
    if stale:
        print(f'Reset {stale} stale "running" jobs from prior crash.')

    # Write PID file so monitor can find us
    pid_file = os.path.join(ROOT, 'status/scheduler.pid')
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))

    stop_flag = threading.Event()

    def handle_signal(sig, frame):
        print(f'\nreceived signal {sig}, letting workers drain current jobs...', flush=True)
        stop_flag.set()
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    threads = []
    for g in gpus:
        for k in range(args.per_gpu):
            t = threading.Thread(
                target=worker_loop,
                args=(g, args.jobs_file, task_filter, args.epochs, stop_flag),
                name=f'gpu-{g}-w{k}',
                daemon=False,
            )
            t.start()
            threads.append(t)

    for t in threads:
        t.join()
    print('[scheduler] all workers exited')

    if os.path.exists(pid_file):
        os.remove(pid_file)


if __name__ == '__main__':
    main()
