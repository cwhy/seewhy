"""Regenerate the slide deck's figures from results.jsonl.

The deck is hand-written Typst with cetz drawings; only the depth curve comes
from measured data, and it comes from the same `results.jsonl` the paper reads,
so the two cannot disagree.

    uv run --no-sync python projects/local-learning/scripts/gen_slides.py
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))

from lib.figures import pc_depth                              # noqa: E402
from shared_lib.results import load_results                   # noqa: E402
from shared_lib.typst_plot import write_figures               # noqa: E402

rows = load_results(PROJECT / "results.jsonl")
written = write_figures(PROJECT / "slides", [pc_depth(rows)])
print(f"{len(written)} files written into slides/")
for path in written:
    print(f"  {path.relative_to(PROJECT / 'slides')}")
