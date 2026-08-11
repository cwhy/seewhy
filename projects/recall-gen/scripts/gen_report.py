"""
Regenerate the paper's figures from results.jsonl.

Writes each figure's data to `paper/assets/<name>.json` and its gribouille spec
to `paper/figures/<name>.typ`. Files whose content is unchanged are left alone,
so this is idempotent and a partial regeneration only touches what moved.

    uv run --no-sync python projects/recall-gen/scripts/gen_report.py
    uv run --no-sync python projects/recall-gen/scripts/gen_report.py --into report

Run this before publishing — `python -m shared_lib.publish` compiles what is on
disk and does not regenerate anything.

Figures live in `lib/figures.py` as `Figure` objects: data plus a declarative
spec, not drawing code. Keep them there rather than inline here, so the same
figure can also be dropped into a markdown report with
`shared_lib.typst_report.save_figure()`.
"""

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parents[1]))          # repo root, for shared_lib

from lib.figures import build_figures                          # noqa: E402
from shared_lib.results import load_results                    # noqa: E402
from shared_lib.typst_plot import write_figures                # noqa: E402

RESULTS = PROJECT / "results.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Regenerate recall-gen figures from results.jsonl")
    ap.add_argument("--into", default="paper",
                    help="which tree to write figures into (default: paper)")
    args = ap.parse_args()

    target = PROJECT / args.into
    if not target.is_dir():
        print(f"no such tree: {target}", file=sys.stderr)
        raise SystemExit(2)

    rows = load_results(RESULTS)
    if not rows:
        print("results.jsonl is empty — run an experiment first", file=sys.stderr)
        raise SystemExit(1)

    figures = build_figures(rows)
    written = write_figures(target, figures)

    print(f"{len(figures)} figures from {len(rows)} runs → {target.name}/")
    for path in written:
        print(f"  wrote {path.relative_to(target)}")
    if figures and not written:
        print("  (all figures already up to date)")


if __name__ == "__main__":
    main()
