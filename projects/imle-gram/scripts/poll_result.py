"""
Wait for an experiment result to appear in results.jsonl, then print it.

Usage:
    uv run python projects/imle-gram/scripts/poll_result.py exp13
    uv run python projects/imle-gram/scripts/poll_result.py exp13 --interval 5
"""

import sys, json, time, argparse
from pathlib import Path

JSONL = Path(__file__).parent.parent / "results.jsonl"


def read_results():
    if not JSONL.exists():
        return []
    lines = JSONL.read_text().strip().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def wait_for(exp_name: str, interval: int = 10):
    print(f"Polling for {exp_name} in {JSONL.name} every {interval}s …")
    while True:
        results = read_results()
        for r in results:
            if r.get("experiment") == exp_name:
                acc   = r.get("final_bin_acc", "?")
                recon = r.get("final_recon_nll", "?")
                kl    = r.get("final_kl", "?")
                t     = r.get("time_s", "?")
                n     = r.get("n_params", "?")
                print(
                    f"{exp_name}: acc={acc:.4f}  recon={recon:.4f}"
                    f"  kl={kl:.4f}  time={t}s  params={n:,}"
                )
                return r
        time.sleep(interval)


def show_all():
    results = read_results()
    if not results:
        print("No results yet.")
        return
    header = f"{'exp':<8} {'acc':>8} {'recon':>8} {'kl':>7} {'time':>7} {'params':>10}  name"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.get('experiment','?'):<8}"
            f" {r.get('final_bin_acc', 0):>8.4f}"
            f" {r.get('final_recon_nll', 0):>8.4f}"
            f" {r.get('final_kl', 0):>7.4f}"
            f" {r.get('time_s', 0):>7.0f}s"
            f" {r.get('n_params', 0):>10,}"
            f"  {r.get('name','')}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?", help="experiment name to wait for (e.g. exp13)")
    parser.add_argument("--interval", type=int, default=10, help="poll interval in seconds")
    parser.add_argument("--all", action="store_true", help="print all results table and exit")
    args = parser.parse_args()

    if args.all or args.experiment is None:
        show_all()
    else:
        wait_for(args.experiment, args.interval)
