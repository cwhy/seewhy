"""
Generate and upload the DiVeQ-PQ overview report with plots and results table.

Usage:
    uv run python projects/diveq/scripts/gen_report_diveq.py
"""

import sys, json, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared_lib.media import save_media
from shared_lib.html import save_html

PROJECT = Path(__file__).parent.parent
JSONL   = PROJECT / "results.jsonl"


# ── Load results ──────────────────────────────────────────────────────────────

def load_results():
    rows = []
    for line in JSONL.read_text().strip().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows

def upload_fig(fig, name, dpi=140):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    url = save_media(f"{name}.png", buf, "image/png")
    plt.close(fig)
    return url


rows = load_results()
by_exp = {r["experiment"]: r for r in rows}

valid = {e: r for e, r in by_exp.items()
         if not np.isnan(r.get("final_recon_mse", float("nan")))}

AE16 = valid.get("exp20", {}).get("final_recon_mse", 0.006779)
AE32 = valid.get("exp26", {}).get("final_recon_mse", 0.003618)
AE64 = valid.get("exp43", {}).get("final_recon_mse", 0.001810)

print(f"Loaded {len(valid)} valid experiments")

# ── Plot 1: Pareto front ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))

ax.axhline(AE16, color="steelblue", ls="--", lw=1.3, alpha=0.8,
           label=f"AE ceiling L=16 ({AE16:.6f})")
ax.axhline(AE32, color="darkorange", ls="--", lw=1.3, alpha=0.8,
           label=f"AE ceiling L=32 ({AE32:.6f})")
ax.axhline(AE64, color="tomato", ls="--", lw=1.3, alpha=0.8,
           label=f"AE ceiling L=64 ({AE64:.6f})")

groups = [
    # LATENT=16 — no BN
    ("DiVeQ L=16 D_g=2",     ["exp14","exp17","exp18","exp21"], "steelblue",     "o"),
    ("DiVeQ L=16 D_g≥4",     ["exp13","exp15","exp16"],         "cornflowerblue","D"),
    ("STE L=16",              ["exp24","exp25"],                 "crimson",       "s"),
    # LATENT=16 — with BN
    ("DiVeQ L=16 BN",         ["exp41"],                        "mediumpurple",  "P"),
    # LATENT=32 — no BN (contiguous / interleaved / projection)
    ("DiVeQ L=32 (no BN)",    ["exp27","exp28","exp29","exp30",
                                "exp31","exp32","exp33"],        "goldenrod",     "v"),
    # LATENT=32 — running-stats BN
    ("DiVeQ L=32 BN D_g=2",   ["exp34","exp35","exp38","exp39","exp40",
                                "exp44","exp49","exp50","exp51","exp52"], "darkorange", "^"),
    ("DiVeQ L=16 D_g=2 500ep",["exp53"],                                "navy",        "o"),
    ("DiVeQ L=32 BN D_g≥4",   ["exp36","exp37"],               "sandybrown",    "D"),
    ("STE L=32 BN",            ["exp42"],                       "darkred",       "s"),
    # LATENT=64 — BN
    ("DiVeQ L=64 BN D_g=2",   ["exp45","exp46"],               "tomato",        "^"),
    ("DiVeQ L=64 BN D_g≥4",   ["exp47"],                       "lightsalmon",   "D"),
    # Single-VQ baseline
    ("Single-VQ (no PQ)",      ["exp9","exp11"],                "gray",          "x"),
]

for label, exps, color, marker in groups:
    xs, ys, names = [], [], []
    for e in exps:
        if e in valid:
            r    = valid[e]
            bits = r.get("total_bits")
            mse  = r.get("final_recon_mse")
            if bits and mse and not np.isnan(mse):
                xs.append(bits); ys.append(mse); names.append(e)
    if xs:
        ax.scatter(xs, ys, color=color, marker=marker, s=80, label=label, zorder=5)
        for x, y, n in zip(xs, ys, names):
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7.5)

ax.set_xlabel("Total bits  (G × log₂K)", fontsize=11)
ax.set_ylabel("Test reconstruction MSE (log scale)", fontsize=11)
ax.set_title("DiVeQ-PQ on MNIST — Pareto front: bits vs MSE  (all experiments)", fontsize=13)
ax.legend(fontsize=8.5, ncol=2)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
plt.tight_layout()
pareto_url = upload_fig(fig, "diveq_pareto")
print(f"Pareto → {pareto_url}")

# ── Plot 2: STE vs DiVeQ (two regimes) ───────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: LATENT=16, no BN
ax = axes[0]
pairs16 = [("exp17","exp24","200 ep\nno BN"), ("exp21","exp25","500 ep\nno BN")]
x16 = np.arange(len(pairs16))
w = 0.32
d16 = [valid[a]["final_recon_mse"] for a,b,_ in pairs16 if a in valid and b in valid]
s16 = [valid[b]["final_recon_mse"] for a,b,_ in pairs16 if a in valid and b in valid]
l16 = [lbl for a,b,lbl in pairs16 if a in valid and b in valid]
b1 = ax.bar(x16 - w/2, d16, w, label="DiVeQ",       color="steelblue", alpha=0.85)
b2 = ax.bar(x16 + w/2, s16, w, label="STE β=0.25",  color="crimson",   alpha=0.85)
ax.axhline(AE16, color="gray", ls="--", lw=1.2, label="AE ceiling")
for bar, v in zip(list(b1)+list(b2), d16+s16):
    ax.text(bar.get_x()+bar.get_width()/2, v+2e-5, f"{v:.5f}", ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(x16); ax.set_xticklabels(l16, fontsize=10)
ax.set_ylabel("Test MSE", fontsize=11)
ax.set_title("LATENT=16, G=8, K=512, D_g=2, 72 bits", fontsize=10)
ax.legend(fontsize=9)
yvals = d16 + s16
ax.set_ylim(min(yvals)*0.993, max(yvals)*1.004)
ax.grid(True, alpha=0.3, axis="y")

# Right: LATENT=32, BN
ax = axes[1]
pairs32 = [("exp34","exp42","200 ep\n+BN"), ("exp35","exp42","500 ep\n+BN")]
# exp42 is 500ep; for 200ep BN we use exp34 vs same exp42 (500ep) — note differently
pairs32 = [("exp34", None, "200 ep+BN\n(DiVeQ only)"), ("exp35","exp42","500 ep+BN")]
x32 = np.arange(2)
d32, s32, l32 = [], [], []
# 200ep: only DiVeQ (exp34), placeholder
d32.append(valid["exp34"]["final_recon_mse"] if "exp34" in valid else None)
s32.append(None)
l32.append("200 ep\n+BN")
# 500ep: DiVeQ exp35 vs STE exp42
d32.append(valid["exp35"]["final_recon_mse"] if "exp35" in valid else None)
s32.append(valid["exp42"]["final_recon_mse"] if "exp42" in valid else None)
l32.append("500 ep\n+BN")

for i, (dv, sv) in enumerate(zip(d32, s32)):
    if dv is not None:
        bar = ax.bar(i - w/2, dv, w, color="darkorange", alpha=0.85, label="DiVeQ+BN" if i==0 else "_")
        ax.text(i-w/2, dv+1e-5, f"{dv:.5f}", ha="center", fontsize=8.5, fontweight="bold")
    if sv is not None:
        bar = ax.bar(i + w/2, sv, w, color="darkred",    alpha=0.85, label="STE+BN β=0.25" if i==1 else "_")
        ax.text(i+w/2, sv+1e-5, f"{sv:.5f}", ha="center", fontsize=8.5, fontweight="bold")

ax.axhline(AE32, color="gray", ls="--", lw=1.2, label="AE ceiling L=32")
ax.set_xticks(x32); ax.set_xticklabels(l32, fontsize=10)
ax.set_ylabel("Test MSE", fontsize=11)
ax.set_title("LATENT=32, G=16, K=512, D_g=2, 144 bits", fontsize=10)
ax.legend(fontsize=9)
all_vals = [v for v in d32+s32 if v is not None]
ax.set_ylim(min(all_vals)*0.993, max(all_vals)*1.006)
ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("DiVeQ vs STE: effect of training duration and BN", fontsize=12, fontweight="bold")
plt.tight_layout()
ste_url = upload_fig(fig, "diveq_ste_comparison")
print(f"STE comparison → {ste_url}")

# ── Plot 3: Per-group codebook utilization (LATENT=32) ────────────────────────

l32_exps = [(e, valid[e].get("final_codebook_per_group", []))
            for e in ["exp28","exp30","exp34","exp35","exp45","exp46"]
            if e in valid and valid[e].get("final_codebook_per_group")]

if l32_exps:
    n = len(l32_exps)
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4), sharey=True)
    if n == 1: axes = [axes]
    for ax, (exp_id, per_group) in zip(axes, l32_exps):
        K = valid[exp_id].get("codebook_size", 512)
        colors = ["crimson" if v < K*0.2 else "steelblue" for v in per_group]
        ax.bar(range(len(per_group)), per_group, color=colors)
        ax.axhline(K,     color="gray",   ls="--", lw=1)
        ax.axhline(K*0.2, color="crimson", ls=":",  lw=1)
        mse  = valid[exp_id]["final_recon_mse"]
        name = valid[exp_id].get("name", exp_id)
        ax.set_title(f"{exp_id}  MSE={mse:.5f}", fontsize=9)
        ax.set_xlabel("Group index")
        if ax is axes[0]: ax.set_ylabel("Codewords used")
    fig.suptitle("LATENT=32 — per-group codebook utilization (red = lazy < 20%)", fontsize=11)
    plt.tight_layout()
    cb_url = upload_fig(fig, "diveq_codebook_utilization")
    print(f"Codebook utilization → {cb_url}")
else:
    cb_url = None

# ── Results table HTML ────────────────────────────────────────────────────────

def result_row(r):
    exp   = r["experiment"]
    mse   = r.get("final_recon_mse", float("nan"))
    bits  = r.get("total_bits", "—")
    cb    = r.get("final_codebook_use", "—")
    K     = r.get("codebook_size", "—")
    G     = r.get("n_groups", "—")
    D_g   = r.get("group_dim", "—")
    ep    = r.get("epochs", "—")
    t     = r.get("time_s", "—")
    name  = r.get("name", "")
    if np.isnan(mse):
        mse_str = '<span style="color:red">NaN</span>'
    else:
        mse_str = f"{mse:.6f}"
    return (
        f"<tr><td>{exp}</td><td>{mse_str}</td><td>{bits}</td>"
        f"<td>{G}</td><td>{D_g}</td><td>{K}</td>"
        f"<td>{cb}</td><td>{ep}</td><td>{t}s</td><td style='font-size:0.85em'>{name}</td></tr>"
    )

table_rows = "".join(result_row(r) for r in sorted(rows, key=lambda r: r["experiment"]))

# ── Build combined HTML report ────────────────────────────────────────────────

cb_section = f"""
<h2>Per-group Codebook Utilization (LATENT=32)</h2>
<img src="{cb_url}" style="max-width:100%;border:1px solid #ddd;border-radius:4px;margin:1em 0">
<p>Red bars = lazy groups (&lt;20% utilization). exp33 (batch-norm) achieves near-perfect
utilization across all groups — the key difference vs contiguous/interleaved/projection approaches.</p>
""" if cb_url else ""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DiVeQ-PQ MNIST — Report</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1.5em; color: #222; }}
  h1 {{ font-size: 1.6em; font-weight: 700; border-bottom: 2px solid #555; padding-bottom:.3em; margin-bottom:1em; }}
  h2 {{ font-size: 1.2em; font-weight: 600; color: #333; margin-top: 2em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: .88em; }}
  th, td {{ border: 1px solid #ccc; padding: .35em .65em; text-align: left; }}
  th {{ background: #f0f0f0; font-weight: 600; }}
  tr:hover {{ background: #f7f7f7; }}
  .note {{ background: #e8f4fd; border-left: 4px solid #3a9; padding: .6em 1em; margin: 1em 0; border-radius: 0 4px 4px 0; }}
  .best {{ background: #fffbe6; font-weight: bold; }}
</style>
</head>
<body>
<h1>DiVeQ-PQ on MNIST — Experiment Report</h1>
<p><strong>Date:</strong> 2026-04-27 &nbsp;|&nbsp;
<strong>Experiments completed:</strong> {len(rows)} &nbsp;|&nbsp;
<strong>Best result:</strong> exp21 — MSE=0.006913 (2% above AE ceiling at 72 bits)</p>

<div class="note">
<strong>DiVeQ gradient path:</strong> zq = z + ‖c*−z‖₂ · sg[normalize(v + d̃)].
Gradient flows only through scalar magnitude — no STE, no commitment loss, single reconstruction loss.
</div>

<h2>Pareto Front: bits vs MSE</h2>
<img src="{pareto_url}" style="max-width:100%;border:1px solid #ddd;border-radius:4px;margin:1em 0">

<h2>DiVeQ vs STE at 200 and 500 Epochs</h2>
<img src="{ste_url}" style="max-width:100%;border:1px solid #ddd;border-radius:4px;margin:1em 0">
<div class="note">
<strong>Key finding:</strong> DiVeQ ≈ STE at 200 epochs. At 500 epochs, DiVeQ improves (0.006913)
while STE degrades (0.007577) — commitment loss conflicts with reconstruction loss in late training.
</div>

{cb_section}

<h2>All Experiments</h2>
<table>
<tr><th>Exp</th><th>MSE</th><th>bits</th><th>G</th><th>D_g</th><th>K</th><th>CB used</th><th>Ep</th><th>Time</th><th>Name</th></tr>
{table_rows}
</table>

<h2>Key Findings</h2>
<table>
<tr><th>Finding</th><th>Evidence</th></tr>
<tr><td>Product quantization is essential</td><td>Single-VQ best ≈0.031; PQ best 0.006913 (4.5×)</td></tr>
<tr><td>D_g=2 is optimal group dimension</td><td>D_g=2: 0.007041; D_g=4: 0.013170; D_g=8: 0.012318</td></tr>
<tr><td>K=512 slightly better than K=256</td><td>exp17 (K=512, 72b): 0.007041 vs exp14 (K=256, 64b): 0.007343</td></tr>
<tr><td>DiVeQ beats STE at 500 epochs</td><td>DiVeQ: 0.006913 vs STE: 0.007577 (exp21 vs exp25)</td></tr>
<tr><td>D_g=1 is fundamentally unstable</td><td>exp19/22/23: NaN or codebook collapse</td></tr>
<tr><td>LATENT=32 suffers lazy groups</td><td>6/16 groups unused regardless of grouping strategy</td></tr>
<tr><td>BN fixes lazy groups but causes train-eval mismatch</td><td>exp33: 8172/8192 CB used but MSE=0.007206</td></tr>
<tr class="best"><td>Best result</td><td>exp21: MSE=0.006913 (2% above AE ceiling), G=8 K=512 D_g=2 72 bits 500ep LATENT=16</td></tr>
</table>
</body>
</html>"""

url = save_html("diveq_report1", html)
print(f"Report → {url}")
