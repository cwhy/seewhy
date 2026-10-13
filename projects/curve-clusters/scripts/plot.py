"""Render the probe's two figures and upload them. Usage: python .../plot.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.append(str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from lib import figures
from shared_lib.media import save_matplotlib_figure

rows = [json.loads(l) for l in (HERE / "results.jsonl").read_text().splitlines() if l.strip()]
print(f"{len(rows)} rows")

for name, fig in [
    ("curve_clusters_order_sweep_p1", figures.order_sweep(rows, n_passes=1)),
    ("curve_clusters_order_sweep_p3", figures.order_sweep(rows, n_passes=3)),
    ("curve_clusters_fitted", figures.fitted_curves()),
    ("curve_clusters_k_and_n", figures.k_and_n(rows)),
    ("curve_clusters_does_bend_help", figures.does_bend_help(rows)),
    ("curve_clusters_intrinsic_dim", figures.intrinsic_dimension(rows)),
    ("curve_clusters_dimension_flip", figures.dimension_flip(rows)),
    ("curve_clusters_low_order", figures.low_order_zoom(rows)),
    ("curve_clusters_rate_distortion", figures.rate_distortion(rows)),
    ("curve_clusters_comparison", figures.comparison_grid()),
    ("curve_clusters_bend_vs_budget", figures.bend_vs_budget(rows, orders=(2, 3))),
    ("curve_clusters_stick_vs_budget", figures.bend_vs_budget(rows, orders=(1, 2))),
]:
    print(name, save_matplotlib_figure(f"{name}.png", fig, dpi=150), flush=True)
