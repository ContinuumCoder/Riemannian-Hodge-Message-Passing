"""
Live monitor for seed-training scheduler.

Shows:
- Overall progress (pending/running/done/failed)
- Per-GPU current job + wallclock
- Recent completions
- GPU utilization (nvidia-smi)

Usage:
    python3 scripts/monitor.py          # snapshot
    python3 scripts/monitor.py --watch  # refresh every 5s
"""
import os, json, time, argparse, subprocess
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
JOBS = os.path.join(ROOT, 'status/jobs.json')
PID_FILE = os.path.join(ROOT, 'status/scheduler.pid')


def nvidia_smi():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'], text=True, timeout=3)
        rows = []
        for line in out.strip().splitlines():
            idx, util, mem_used, mem_total = [x.strip() for x in line.split(',')]
            rows.append({'idx': idx, 'util': util, 'mem_used': mem_used, 'mem_total': mem_total})
        return rows
    except Exception as e:
        return []


def scheduler_alive():
    if not os.path.exists(PID_FILE):
        return False, None
    pid = int(open(PID_FILE).read().strip())
    try:
        os.kill(pid, 0)
        return True, pid
    except ProcessLookupError:
        return False, pid


def fmt_elapsed(start_str):
    if not start_str:
        return '-'
    try:
        t0 = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
        secs = (datetime.now() - t0).total_seconds()
        return f'{int(secs//60):d}m{int(secs%60):02d}s'
    except Exception:
        return '?'


def render():
    if not os.path.exists(JOBS):
        print(f'no jobs.json at {JOBS}; run scripts/build_jobs.py')
        return
    jobs = json.load(open(JOBS))
    alive, pid = scheduler_alive()

    counts = {'pending': 0, 'running': 0, 'completed': 0, 'failed': 0}
    for j in jobs:
        counts[j['status']] = counts.get(j['status'], 0) + 1

    total_sec = sum(j['est_sec'] for j in jobs)
    done_sec = sum(j['est_sec'] for j in jobs if j['status'] == 'completed')

    clear = '\033[2J\033[H'
    print(clear, end='')
    print(f'== seed-training monitor ({datetime.now().strftime("%H:%M:%S")}) ==')
    sched_str = f'PID {pid} (alive)' if alive else 'NOT RUNNING'
    print(f'scheduler: {sched_str}')
    print(f'progress: done={counts["completed"]}/{len(jobs)}  '
          f'running={counts["running"]}  pending={counts["pending"]}  failed={counts["failed"]}')
    pct = 100 * done_sec / max(total_sec, 1)
    bar = int(pct / 2)
    print(f'cost-weighted: [{"#"*bar}{"."*(50-bar)}] {pct:.1f}%')

    # Running jobs
    print('\n-- running --')
    rs = [j for j in jobs if j['status'] == 'running']
    if not rs:
        print('  (none)')
    for j in sorted(rs, key=lambda x: x.get('gpu', 0) or 0):
        print(f"  GPU{j['gpu']}  {j['id']:<42} est={j['est_min']:>5.1f}m  elapsed={fmt_elapsed(j['start'])}")

    # Recent completions (last 8)
    print('\n-- recent completions --')
    done = [j for j in jobs if j['status'] in ('completed', 'failed') and j.get('end')]
    done.sort(key=lambda j: j['end'] or '', reverse=True)
    for j in done[:8]:
        marker = '✓' if j['status'] == 'completed' else '✗'
        print(f"  {marker} {j['id']:<42} gpu={j['gpu']} end={j['end'][-8:]}"
              f"{'' if not j.get('error') else ' err=' + j['error']}")

    # GPU util
    rows = nvidia_smi()
    if rows:
        print('\n-- nvidia-smi --')
        for r in rows:
            print(f"  GPU{r['idx']}  util={r['util']}%  mem={r['mem_used']}/{r['mem_total']} MiB")

    # Failed summary
    if counts['failed']:
        print('\n-- failed --')
        for j in jobs:
            if j['status'] == 'failed':
                print(f"  {j['id']:<42} err={j.get('error')}  log={j['log']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch', action='store_true', help='refresh every 5s')
    ap.add_argument('--interval', type=int, default=5)
    args = ap.parse_args()
    if args.watch:
        try:
            while True:
                render()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    else:
        render()


if __name__ == '__main__':
    main()
