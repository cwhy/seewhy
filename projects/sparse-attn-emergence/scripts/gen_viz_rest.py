"""
Figures for exp3 (heads), exp5 (cellular automata) and exp6/exp7 (architectures), all read
from results.jsonl — no re-training. Skips any experiment with no rows yet.

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/gen_viz_rest.py
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from lib.viz import save_arch_panel, save_ca_panel, save_heads_panel      # noqa: E402

rows, seen = [], set()
for line in (PROJECT / "results.jsonl").read_text().splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    e = str(r.get("experiment", ""))
    if e and e not in seen and not e.startswith("smoke_"):
        seen.add(e)
        rows.append(r)


def exact_rate(r):
    if r.get("exact_rate") is not None:
        return r["exact_rate"]
    fl = r.get("final_loss2") or []
    return sum(1 for v in fl if v < 0.01) / len(fl) if fl else float("nan")


# ── exp3: heads vs head dim ───────────────────────────────────────────────────
h3 = [r for r in rows if r["experiment"].startswith("exp3_")]
if h3:
    heads = sorted([(r["n_heads"], r["solve_rate"], r.get("median_t_star"), exact_rate(r))
                    for r in h3 if r["leg"] == "heads"])
    hdims = sorted([(r["d_head"], r["solve_rate"], r.get("median_t_star"), exact_rate(r))
                    for r in h3 if r["leg"] == "headdim"])
    print(f"exp3: {len(heads)} head-count configs, {len(hdims)} head-dim configs")
    for x, s, t, e in heads:
        print(f"  H={x:>3}  solve {s:.2f}  exact {e:.2f}  median t* {t or float('nan'):.0f}")
    for x, s, t, e in hdims:
        print(f"  dh={x:>3} solve {s:.2f}  exact {e:.2f}  median t* {t or float('nan'):.0f}")
    print("  →", save_heads_panel("sparse_attn_emergence_exp3_heads", heads, hdims,
                                  h3[0]["n_seeds"]))

# ── exp6/exp7: architectures ──────────────────────────────────────────────────
arch = [r for r in rows if r["experiment"].startswith("exp7_")]
if arch:
    cells = {}
    for r in arch:
        curve = None
        if r.get("curve_loss2"):
            curve = (r["curve_step"], np.median(np.array(r["curve_loss2"]), axis=0))
        key = (r["s"], r["arch"])
        # Best LR per (s, arm) = highest solve rate, then lowest median loss. Ranking by
        # median loss alone is wrong: with fewer than half the seeds solved the median sits
        # at ln 2 regardless, so a 5/16 run ties a 0/16 run and the tie-break picked the
        # worse one.
        med = float(np.median(r["final_loss2"]))
        rank = (r["solve_rate"], -med)
        if key not in cells or rank > cells[key]["rank"]:
            cells[key] = {"solve": r["solve_rate"], "loss": med, "lr": r["lr"], "rank": rank,
                          "iou": float(np.mean(r["support_iou"])), "curve": curve,
                          "t": r.get("median_t_star")}
    print(f"\nexp7: {len(arch)} runs, {len(cells)} (s, arm) cells — best LR per cell")
    for (s, arm), d in sorted(cells.items()):
        print(f"  s={s:>2} {arm:<12} lr={d['lr']:.0e}  solve {d['solve']:.2f}  "
              f"loss {d['loss']:.4f}  support_iou {d['iou']:.2f}  "
              f"median t* {d['t'] or float('nan'):.0f}")
    print("  →", save_arch_panel("sparse_attn_emergence_exp7_arch", cells, float(np.log(2))))

# ── exp5: cellular automata ───────────────────────────────────────────────────
ca = [r for r in rows if r["experiment"].startswith("exp5_")]
if ca:
    print(f"\nexp5: {len(ca)} depths")
    for r in sorted(ca, key=lambda r: r["k"]):
        ps = np.array(r["per_state_loss"]).mean(0)
        print(f"  k={r['k']} span {r['span']}  solve {r['solve_rate']:.2f}  "
              f"loss_last med {np.median(r['final_loss_last']):.4f}  "
              f"per-state first→last {ps[0]:.3f}→{ps[-1]:.3f}")
    print("  →", save_ca_panel("sparse_attn_emergence_exp5_ca", ca, float(np.log(4))))
