"""
Regenerate the report's figures from results.jsonl.

Writes each figure's data to `report/assets/<name>.json` and its gribouille spec to
`report/figures/<name>.typ`. Files whose content is unchanged are left alone, so this is
idempotent and a partial regeneration only touches what moved.

    uv run --no-sync python projects/sparse-attn-emergence/scripts/gen_report.py
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lib.report_figures import (crossover, emergence_spread, induction,   # noqa: E402
                                kofm_candidates, kofm_difficulty, load_results,
                                pool_curve)
from shared_lib.typst_plot import write_figures                          # noqa: E402

REPORT = PROJECT / "report"


def main() -> None:
    rows = load_results()
    if not rows:
        print("results.jsonl is empty — run an experiment first", file=sys.stderr)
        raise SystemExit(1)

    figs = [crossover(rows), induction(rows), emergence_spread(rows), *pool_curve(rows),
            kofm_difficulty(rows), kofm_candidates(rows)]
    written = write_figures(REPORT, figs)
    print(f"{len(figs)} figures; {len(written)} files written/updated")
    for p in written:
        print(f"  {p.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
