"""
Wait for an experiment result to appear in results.jsonl, then print it.

Usage:
    uv run python projects/small_lm/scripts/poll_result.py exp1
    uv run python projects/small_lm/scripts/poll_result.py --all
    uv run python projects/small_lm/scripts/poll_result.py exp1 --interval 5
"""

import sys, json, time, argparse
from pathlib import Path

JSONL = Path(__file__).parent.parent / "results.jsonl"


def read_results() -> list[dict]:
    if not JSONL.exists():
        return []
    lines = JSONL.read_text().strip().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def wait_for(exp_name: str, interval: int = 10):
    print(f"Polling for {exp_name} in {JSONL.name} every {interval}s …")
    while True:
        for r in read_results():
            if r.get("experiment") == exp_name:
                print(json.dumps(r, indent=2))
                return r
        time.sleep(interval)


def show_all():
    results = read_results()
    if not results:
        print("No results yet.")
        return
    header = f"{'exp':<8} {'time':>7}  name"
    print(header)
    print("-" * 60)
    for r in results:
        t = r.get("time_s", float("nan"))
        t_str = f"{t:.0f}s" if t == t else "—"
        print(f"{r.get('experiment','?'):<8} {t_str:>7}  {r.get('name','')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?", help="experiment name to wait for (e.g. exp1)")
    parser.add_argument("--interval", type=int, default=10, help="poll interval in seconds")
    parser.add_argument("--all", action="store_true", help="print all results and exit")
    args = parser.parse_args()

    if args.all or args.experiment is None:
        show_all()
    else:
        wait_for(args.experiment, args.interval)
