"""
Regenerate the report's figures from results.jsonl.

Writes each figure's data to `report/assets/<name>.json` and its gribouille
spec to `report/figures/<name>.typ`. Files whose content is unchanged are left
alone, so this is idempotent and a partial regeneration only touches what moved.

    uv run python projects/omniglot-ar/scripts/gen_report.py            # charts
    uv run python projects/omniglot-ar/scripts/gen_report.py --grid     # + episode PNG

The episode grid is behind a flag because it is the only figure that needs the
dataset (and therefore JAX) rather than just results.jsonl — charts can be
regenerated anywhere, including a machine with no GPU stack.
"""

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lib.figures import (
    episode_grid, excess_over_chance, floor_comparison, learning_curves,
    load_results, loss_curve,
)
from shared_lib.typst_plot import write_figures

REPORT = PROJECT / "report"


def build_figures(rows: dict) -> list:
    figs = []
    if "exp1" in rows:
        r = rows["exp1"]
        figs += [learning_curves(r), loss_curve(r)]
    if "exp2" in rows:
        figs.append(learning_curves(rows["exp2"], name="learning_curves_exp2"))
    if "exp3" in rows:
        figs.append(learning_curves(rows["exp3"], name="learning_curves_exp3"))
    if "exp5" in rows:
        figs.append(learning_curves(rows["exp5"], name="learning_curves_exp5"))
    if len(rows) > 1:
        figs += [floor_comparison(rows), excess_over_chance(rows)]
    return figs


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate omniglot-ar report figures")
    ap.add_argument("--grid", action="store_true",
                    help="also rasterise an example episode (needs the dataset)")
    ap.add_argument("--into", default="report", choices=("report", "paper"),
                    help="which report tree to write figures into")
    args = ap.parse_args()

    target = PROJECT / args.into
    rows = load_results()
    if not rows:
        print("results.jsonl is empty — run an experiment first", file=sys.stderr)

    figs = build_figures(rows)
    written = write_figures(target, figs)
    print(f"{len(figs)} figures from {sorted(rows)}")
    for p in written:
        print(f"  wrote {p.relative_to(target)}")
    if not written and figs:
        print("  (all figures already up to date)")

    if args.grid:
        import numpy as np

        from lib.tasks import Spec, class_index
        from shared_lib.datasets import load_omniglot

        spec = Spec()
        if "exp1" in rows:
            r = rows["exp1"]
            spec = Spec(**{k[5:]: v for k, v in r.items() if k.startswith("spec_")})
        data = load_omniglot(size=spec.img_size, invert=True)
        X = np.asarray(data.X_bg).reshape(len(data.X_bg), -1).astype(np.uint8)
        idx = class_index(np.asarray(data.y_bg), data.n_char_bg)
        path = episode_grid(target, X, idx, spec, seed=3)
        print(f"  wrote {path.lstrip('/')}")


if __name__ == "__main__":
    main()
