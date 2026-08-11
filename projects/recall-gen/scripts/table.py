"""Print the results table straight from results.jsonl. No GPU, no model.

Columns are normalised MSE (model MSE / mean-image MSE) on the four conditions,
so 1.000 means "no better than predicting the dataset mean" and 0 is perfect.
`best D` is the minimum over the training curve — the absent-target conditions
get WORSE with training in every run so far, so the final value understates what
the run reached.

Usage:
    uv run python projects/recall-gen/scripts/table.py [--curves]
"""

import argparse
import json
from pathlib import Path

JSONL = Path(__file__).parent.parent / "results.jsonl"
CONDS = ["A_seen_present", "B_novel_present", "C_seen_absent", "D_novel_absent"]


def conds_of(r):
    """Runs with a digit split carry two extra conditions; keep the row's own order."""
    return list(r.get("final", r.get("baselines", {})).keys())


def gain(final):
    """How much having the answer in the context is worth, on NOVEL images.

    nMSE(D) - nMSE(B): both conditions use context images the model has never
    seen, and differ only in whether the query's true image is among them. This
    is the one number that says whether a model retrieves, and unlike
    identification accuracy it cannot be inflated by a model whose completions
    happen to be good enough to pick the right neighbour.
    """
    return final["D_novel_absent"]["nmse"] - final["B_novel_present"]["nmse"]


def rows():
    seen, out = set(), []
    for line in JSONL.read_text().strip().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        k = r.get("experiment", "")
        if k in seen or k.startswith("smoke_"):
            continue
        seen.add(k)
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", action="store_true")
    a = ap.parse_args()

    rs = rows()
    base = {r["experiment"]: r for r in rs if r["experiment"].startswith("baselines")}
    exps = [r for r in rs if not r["experiment"].startswith("baselines")]

    for name, b in base.items():
        print(f"\n=== {name} (normalised MSE; 1.0 = predict the dataset mean) ===")
        print(f"{'condition':<17} {'ridge':>7} {'nn1':>7} {'knn':>7}  knn tau")
        for c in conds_of(b):
            v = b["baselines"][c]
            print(f"{c:<17} {v['n_ridge']:>7.3f} {v['n_nn1']:>7.3f} {v['n_knn']:>7.3f}"
                  f"  {v['knn_tau']}")

    print(f"\n=== runs ===")
    allc = sorted({c for r in exps for c in conds_of(r)})
    hdr = (f"{'exp':<7} {'mode':<7} {'M':>4} {'sd':>3} "
           + " ".join(f"{c.split('_')[0]:>6}" for c in allc)
           + f" {'gain':>6} {'idB':>5} {'bestD':>6}  name")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(exps, key=lambda r: (r.get("M", 0), r.get("train_mode", ""),
                                         r.get("seed", 0))):
        f = r["final"]
        bd = min(r["history"]["nmse"]["D_novel_absent"])
        print(f"{r['experiment']:<7} {r.get('train_mode',''):<7} {r.get('M',0):>4} "
              f"{r.get('seed',0):>3} "
              + " ".join((f"{f[c]['nmse']:>6.3f}" if c in f else f"{'—':>6}")
                          for c in allc)
              + f" {gain(f):>6.3f}"
                f" {f['B_novel_present']['id_acc']:>5.2f}"
                f" {bd:>6.3f}  {r.get('name','')}")

    print("\n=== what the absent-target output actually resembles (final) ===")
    print(f"{'exp':<7} {'cond':<17} {'d(out,truth)':>13} {'d(out,lookup)':>14} "
          f"{'d(out,mean-img)':>16} {'points-at-lookup':>17}")
    for r in sorted(exps, key=lambda r: (r.get("M", 0), r.get("train_mode", ""))):
        for c in [c for c in conds_of(r) if "absent" in c]:
            f = r["final"][c]
            print(f"{r['experiment']:<7} {c:<17} {f['mse']:>13.4f} {f['mse_to_nn']:>14.4f} "
                  f"{f['mse_to_meanimg']:>16.4f} {f['nn_agree']:>17.3f}")

    if a.curves:
        for r in exps:
            print(f"\n--- {r['experiment']} nmse curve ---")
            h = r["history"]
            for i, s in enumerate(h["step"]):
                print(f"  {s:>6} " + "  ".join(
                    f"{c.split('_')[0]}:{h['nmse'][c][i]:.3f}" for c in conds_of(r)))


if __name__ == "__main__":
    main()
