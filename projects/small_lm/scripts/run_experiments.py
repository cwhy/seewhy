"""
Experiment runner for small_lm.

Runs experiments sequentially (default) or in parallel across GPUs.
Logs each experiment to projects/small_lm/logs/{name}.log and prints a
results table when done.

Usage:
    # Run sequentially (blocks until done)
    uv run python projects/small_lm/scripts/run_experiments.py exp1 exp2

    # Fire and forget — detach to background, return immediately
    uv run python projects/small_lm/scripts/run_experiments.py --bg exp_baseline

    # Run in parallel — one per GPU
    uv run python projects/small_lm/scripts/run_experiments.py --parallel exp1 exp2

    # Force a specific GPU for sequential runs
    uv run python projects/small_lm/scripts/run_experiments.py --gpu 1 exp_baseline
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
LOG_DIR     = PROJECT_DIR / "logs"
JSONL       = PROJECT_DIR / "results.jsonl"


# ── GPU detection ─────────────────────────────────────────────────────────────

def get_gpu_count() -> int:
    try:
        r = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True)
        return max(1, len(r.stdout.strip().splitlines()))
    except FileNotFoundError:
        return 1


# ── Results ───────────────────────────────────────────────────────────────────

def read_result(exp_name: str) -> dict | None:
    if not JSONL.exists():
        return None
    for line in JSONL.read_text().strip().splitlines():
        try:
            r = json.loads(line)
            if r.get("experiment") == exp_name:
                return r
        except json.JSONDecodeError:
            continue
    return None


def print_results_table(names: list[str]):
    rows = []
    for name in names:
        r = read_result(name)
        rows.append(r or {"experiment": name, "name": "(no result)", "time_s": float("nan")})

    header = f"{'exp':<8} {'time':>7}  name"
    print("\n" + header)
    print("-" * 60)
    for r in rows:
        t = r.get("time_s", float("nan"))
        t_str = f"{t:.0f}s" if t == t else "—"
        print(f"{r.get('experiment', '?'):<8} {t_str:>7}  {r.get('name', '')}")


# ── Running ───────────────────────────────────────────────────────────────────

def resolve_script(name: str) -> Path:
    """Accept 'exp1', '1', or a full path."""
    direct = PROJECT_DIR / f"{name}.py"
    if direct.exists():
        return direct
    num = name.lstrip("exp")
    script = PROJECT_DIR / f"experiments{num}.py"
    if script.exists():
        return script
    raise FileNotFoundError(f"Cannot find script for '{name}' (tried {direct} and {script})")


def run_one(name: str, script: Path, log_path: Path, gpu_id: int | None, idx: int, total: int,
            viz_script: Path | None = None) -> int:
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_str = f"GPU {gpu_id}"
    else:
        gpu_str = "default GPU"

    print(f"[{idx}/{total}] Starting {name} on {gpu_str} → {log_path}")
    t0 = time.perf_counter()

    with open(log_path, "w") as f:
        proc = subprocess.Popen([sys.executable, str(script)], stdout=f, stderr=f, env=env)
        while proc.poll() is None:
            elapsed = time.perf_counter() - t0
            print(f"  {name}  running… {elapsed:.0f}s", end="\r", flush=True)
            time.sleep(10)

    elapsed = time.perf_counter() - t0
    rc = proc.returncode
    status = "done" if rc == 0 else f"FAILED (exit {rc})"
    print(f"  {name}  {status} in {elapsed:.0f}s" + " " * 10)

    if rc == 0 and viz_script is not None:
        print(f"  {name}  running viz → {viz_script.name} {name}")
        subprocess.run([sys.executable, str(viz_script), name], check=False)

    return rc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run {PROJECT_NAME} experiments")
    parser.add_argument("experiments", nargs="+", help="Experiment names, e.g. exp1 exp2")
    parser.add_argument("--parallel", action="store_true",
                        help="Run in parallel, one per GPU (needs enough GPUs)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Pin all sequential runs to this GPU ID")
    parser.add_argument("--viz", type=str, default=None, metavar="SCRIPT",
                        help="Viz script to run after each successful experiment (receives exp_name as arg)")
    parser.add_argument("--bg", action="store_true",
                        help="Detach to background immediately; print PID and log path, then exit")
    args = parser.parse_args()

    if args.bg:
        cmd = [sys.executable] + [a for a in sys.argv if a != "--bg"]
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        runner_log = LOG_DIR / f"runner_{'_'.join(args.experiments)}.log"
        with open(runner_log, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=lf,
                                    start_new_session=True)
        print(f"Detached PID {proc.pid} → {runner_log}")
        sys.exit(0)

    viz_script = None
    if args.viz:
        p = Path(args.viz)
        if not p.is_absolute():
            if (Path(__file__).parent / p).exists():
                p = Path(__file__).parent / p
            else:
                p = PROJECT_DIR / p
        if not p.exists():
            print(f"ERROR: viz script not found: {args.viz}", file=sys.stderr)
            sys.exit(1)
        viz_script = p

    exps = []
    for name in args.experiments:
        try:
            script = resolve_script(name)
            exps.append((name, script))
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    n_gpus = get_gpu_count()
    total = len(exps)
    print(f"Found {n_gpus} GPU(s). Running {total} experiment(s) "
          f"{'in parallel' if args.parallel else 'sequentially'}. Logs → {LOG_DIR}")

    if args.parallel:
        if total > n_gpus:
            print(f"WARNING: {total} experiments but only {n_gpus} GPU(s). "
                  f"Some will share a GPU.")
        assignments = [(name, script, i % n_gpus) for i, (name, script) in enumerate(exps)]

        def _run(item):
            name, script, gpu_id = item
            log = LOG_DIR / f"{name}.log"
            return name, run_one(name, script, log, gpu_id, idx=assignments.index(item) + 1,
                                 total=total, viz_script=viz_script)

        with ThreadPoolExecutor(max_workers=total) as pool:
            futures = {pool.submit(_run, item): item[0] for item in assignments}
            for future in as_completed(futures):
                name, rc = future.result()
                if rc != 0:
                    print(f"WARNING: {name} exited with code {rc}, check {LOG_DIR / f'{name}.log'}")

    else:
        for idx, (name, script) in enumerate(exps, 1):
            log = LOG_DIR / f"{name}.log"
            rc = run_one(name, script, log, args.gpu, idx, total, viz_script=viz_script)
            if rc != 0:
                print(f"Stopping: {name} failed. Check {log}")
                sys.exit(rc)

    print_results_table([name for name, _ in exps])


if __name__ == "__main__":
    main()
