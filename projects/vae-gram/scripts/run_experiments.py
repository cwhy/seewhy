"""
Experiment runner for vae-gram.

Runs experiments sequentially (default) or in parallel across GPUs.
Logs each experiment to /tmp/{name}.log and prints a results table when done.

Usage:
    # Run sequentially on GPU 0
    uv run python projects/vae-gram/scripts/run_experiments.py exp24 exp25

    # Run in parallel — one per GPU (requires enough GPUs)
    uv run python projects/vae-gram/scripts/run_experiments.py --parallel exp24 exp25

    # Force a specific GPU for sequential runs
    uv run python projects/vae-gram/scripts/run_experiments.py --gpu 1 exp24 exp25
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
JSONL = PROJECT_DIR / "results.jsonl"


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
        if r:
            rows.append(r)
        else:
            rows.append({"experiment": name, "final_bin_acc": float("nan"),
                         "final_recon_nll": float("nan"), "final_kl": float("nan"),
                         "time_s": float("nan"), "n_params": 0, "name": "(no result)"})

    header = f"{'exp':<8} {'acc':>8} {'recon':>8} {'kl':>7} {'time':>7} {'params':>10}  name"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        acc = r.get("final_bin_acc", float("nan"))
        acc_str = f"{acc:.4f}" if acc == acc else "  —   "
        t = r.get("time_s", float("nan"))
        t_str = f"{t:.0f}s" if t == t else "—"
        print(
            f"{r.get('experiment', '?'):<8}"
            f" {acc_str:>8}"
            f" {r.get('final_recon_nll', 0):>8.4f}"
            f" {r.get('final_kl', 0):>7.4f}"
            f" {t_str:>7}"
            f" {r.get('n_params', 0):>10,}"
            f"  {r.get('name', '')}"
        )


# ── Running ───────────────────────────────────────────────────────────────────

def resolve_script(name: str) -> Path:
    """Accept 'exp24', '24', or a full path."""
    # Try as-is first (e.g. full path or bare name)
    direct = PROJECT_DIR / f"{name}.py"
    if direct.exists():
        return direct
    # Strip 'exp' prefix if given as a number
    num = name.lstrip("exp")
    script = PROJECT_DIR / f"experiments{num}.py"
    if script.exists():
        return script
    raise FileNotFoundError(f"Cannot find script for '{name}' (tried {direct} and {script})")


def run_one(name: str, script: Path, log_path: Path, gpu_id: int | None, idx: int, total: int) -> int:
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
        # Stream a heartbeat to stdout while waiting
        while proc.poll() is None:
            elapsed = time.perf_counter() - t0
            print(f"  {name}  running… {elapsed:.0f}s", end="\r", flush=True)
            time.sleep(10)

    elapsed = time.perf_counter() - t0
    rc = proc.returncode
    status = "done" if rc == 0 else f"FAILED (exit {rc})"
    print(f"  {name}  {status} in {elapsed:.0f}s" + " " * 10)
    return rc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run vae-gram experiments")
    parser.add_argument("experiments", nargs="+", help="Experiment names, e.g. exp24 exp25")
    parser.add_argument("--parallel", action="store_true",
                        help="Run in parallel, one per GPU (needs enough GPUs)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Pin all sequential runs to this GPU ID")
    args = parser.parse_args()

    # Resolve scripts
    exps = []
    for name in args.experiments:
        try:
            script = resolve_script(name)
            exps.append((name, script))
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    n_gpus = get_gpu_count()
    total = len(exps)
    print(f"Found {n_gpus} GPU(s). Running {total} experiment(s) "
          f"{'in parallel' if args.parallel else 'sequentially'}.")

    if args.parallel:
        if total > n_gpus:
            print(f"WARNING: {total} experiments but only {n_gpus} GPU(s). "
                  f"Some will share a GPU.")
        # Assign GPUs round-robin
        assignments = [(name, script, i % n_gpus) for i, (name, script) in enumerate(exps)]

        def _run(item):
            name, script, gpu_id = item
            log = Path(f"/tmp/{name}.log")
            return name, run_one(name, script, log, gpu_id, idx=assignments.index(item) + 1, total=total)

        with ThreadPoolExecutor(max_workers=total) as pool:
            futures = {pool.submit(_run, item): item[0] for item in assignments}
            for future in as_completed(futures):
                name, rc = future.result()
                if rc != 0:
                    print(f"WARNING: {name} exited with code {rc}, check /tmp/{name}.log")

    else:
        # Sequential — wait for XLA compilation of one before starting the next
        for idx, (name, script) in enumerate(exps, 1):
            log = Path(f"/tmp/{name}.log")
            rc = run_one(name, script, log, args.gpu, idx, total)
            if rc != 0:
                print(f"Stopping: {name} failed. Check {log}")
                sys.exit(rc)

    print_results_table([name for name, _ in exps])


if __name__ == "__main__":
    main()
