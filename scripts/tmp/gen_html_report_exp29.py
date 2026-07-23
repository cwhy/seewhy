"""
Generate a self-contained HTML report for exp29 (CIFAR-10, 200 epochs).

Usage:
    uv run python scripts/tmp/gen_html_report_exp29.py
"""

import sys, pickle, json, base64, io
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared_lib.media import save_media

EXP      = "exp29"
PROJECT  = ROOT / "projects" / "variations"
PLOTS    = PROJECT / "plots"
OUT_HTML = PROJECT / f"report_{EXP}.html"

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

with open(PROJECT / f"history_{EXP}.pkl", "rb") as f:
    history = pickle.load(f)

result = None
for line in (PROJECT / "results.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r.get("experiment") == EXP:
        result = r
        break

n_params_val = result["n_params"]
time_s       = result["time_s"]
final_mse    = result["final_recon_mse"]
batch_size   = result["batch_size"]
n_var        = result["n_var_samples"]
k_nn         = result["k_nn"]
latent_dim   = result["latent_dim"]
n_cat        = result["n_cat"]
vocab_size   = result["vocab_size"]
dc_c1        = result["dc_c1"]
dc_c2        = result["dc_c2"]
dc_c3        = result["dc_c3"]
dc_stem      = result["dc_stem"]
enc_c1       = result["enc_c1"]
enc_c2       = result["enc_c2"]
enc_c3       = result["enc_c3"]
enc_fc1      = result["enc_fc1"]
alpha        = result.get("alpha", 1.0)
epochs_val   = result.get("epochs", 200)

def to_data_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

epochs   = list(range(len(history["loss"])))
ev_ep,  ev_mse  = zip(*history["eval_mse"])
rec_ep, rec_mse = zip(*history["eval_recall_mse"])
rank_data = history["eval_recall_by_rank"]

fig, axes = plt.subplots(1, 3, figsize=(18, 4), facecolor="#fafafa")
for ax in axes: ax.set_facecolor("#fafafa")

ax = axes[0]
ax.plot(epochs, history["loss"],    label="total loss",     lw=1.8, color="#2563eb")
ax.plot(epochs, history["L_recon"], label="recon loss",     lw=1.5, ls="--", color="#16a34a")
ax.plot(epochs, history["L_var"],   label="variation loss", lw=1.5, ls=":",  color="#dc2626")
ax.set_xlabel("epoch", fontsize=11); ax.set_ylabel("MSE", fontsize=11)
ax.set_title("Training losses", fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.25)

ax = axes[1]
ax.plot(ev_ep,  ev_mse,  "o-",  lw=1.8, label="reconstruction MSE (test)", color="#2563eb", ms=4)
ax.plot(rec_ep, rec_mse, "s--", lw=1.8, label="variation recall MSE",      color="#dc2626", ms=4)
ax.set_xlabel("epoch", fontsize=11); ax.set_ylabel("MSE", fontsize=11)
ax.set_title("Eval metrics over training", fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.25)

ax = axes[2]
n_lines = len(rank_data)
colors  = plt.cm.coolwarm(np.linspace(0, 1, n_lines))
ranks   = np.arange(1, k_nn + 1)
for i, (ep, curve) in enumerate(rank_data):
    ax.plot(ranks, curve, color=colors[i], lw=1.2, alpha=0.85,
            label=f"ep {ep}" if i in (0, n_lines//2, n_lines-1) else None)
ax.set_xlabel("neighbor rank", fontsize=11); ax.set_ylabel("MSE", fontsize=11)
ax.set_title("Recall quality by neighbor rank\n(blue = early  →  red = final)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.25)

fig.suptitle(f"Learning dynamics — {EXP} (CIFAR-10, {epochs_val} epochs)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
curves_buf = io.BytesIO()
fig.savefig(curves_buf, format="png", dpi=130, bbox_inches="tight")
curves_uri = "data:image/png;base64," + base64.b64encode(curves_buf.getvalue()).decode()
plt.close(fig)
print("curves rendered")

recon_uri  = to_data_uri(PLOTS / f"variations_{EXP}_recon.png")
class_uris = {c: to_data_uri(PLOTS / f"variations_{EXP}_var_class{c}.png") for c in range(10)}
print("plots loaded")

class_sections = ""
for c in range(10):
    class_sections += f"""
      <div class="class-block">
        <h3>Class {c} &mdash; {CIFAR_CLASSES[c].capitalize()}</h3>
        <img src="{class_uris[c]}" alt="class {c}" class="plot full-width">
      </div>
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Learning to Generate Image Variations — {EXP} (CIFAR-10, {epochs_val} epochs)</title>
  <style>
    :root {{--blue:#2563eb;--green:#16a34a;--red:#dc2626;--gray:#6b7280;--light:#f8fafc;--border:#e2e8f0;}}
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{font-family:"Segoe UI",system-ui,sans-serif;font-size:15px;line-height:1.7;color:#1e293b;background:#fff;max-width:960px;margin:0 auto;padding:2.5rem 1.5rem 5rem;}}
    h1{{font-size:2rem;font-weight:700;color:#0f172a;margin-bottom:0.3rem;line-height:1.2;}}
    h2{{font-size:1.25rem;font-weight:700;color:#0f172a;border-bottom:2px solid var(--border);padding-bottom:0.4rem;margin:2.8rem 0 1rem;}}
    h3{{font-size:1rem;font-weight:600;color:#374151;margin:1.6rem 0 0.4rem;}}
    p{{margin-bottom:0.9rem;}}
    a{{color:var(--blue);}}
    .subtitle{{color:var(--gray);font-size:0.93rem;margin:0.2rem 0 2rem;}}
    .summary-card{{background:var(--light);border:1px solid var(--border);border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:2.2rem;}}
    .metrics{{display:flex;gap:1rem;flex-wrap:wrap;margin:0.8rem 0 1rem;}}
    .metric{{background:#fff;border:1px solid var(--border);border-radius:8px;padding:0.55rem 1rem;min-width:130px;flex:1;}}
    .metric .label{{font-size:0.75rem;color:var(--gray);text-transform:uppercase;letter-spacing:0.05em;}}
    .metric .value{{font-size:1.3rem;font-weight:700;color:#0f172a;}}
    .callout{{border-left:4px solid var(--blue);background:#eff6ff;border-radius:0 8px 8px 0;padding:0.85rem 1.2rem;margin:1rem 0 1.2rem;}}
    .callout.green{{border-color:var(--green);background:#f0fdf4;}}
    table{{border-collapse:collapse;width:100%;margin:1rem 0 1.5rem;font-size:0.9rem;}}
    th{{background:var(--light);text-align:left;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--gray);padding:0.5rem 0.9rem;border-bottom:2px solid var(--border);}}
    td{{padding:0.45rem 0.9rem;border-bottom:1px solid var(--border);}}
    tr:hover td{{background:var(--light);}}
    td:first-child{{font-weight:500;}}
    .plot{{display:block;max-width:100%;border-radius:8px;border:1px solid var(--border);margin:0.8rem 0 1.6rem;}}
    .full-width{{width:100%;}}
    .caption{{font-size:0.85rem;color:var(--gray);margin-bottom:0.4rem;}}
    .class-block{{margin:2rem 0;}}
    footer{{margin-top:4rem;padding-top:1.2rem;border-top:1px solid var(--border);font-size:0.8rem;color:var(--gray);text-align:center;}}
  </style>
</head>
<body>

<h1>Learning to Generate Image Variations</h1>
<p class="subtitle">
  {EXP} &nbsp;·&nbsp; CIFAR-10 &nbsp;·&nbsp; CNN encoder + wide deconvnet, LATENT={latent_dim}, {epochs_val} epochs &nbsp;·&nbsp;
  {int(time_s // 60)} min training on RTX 4090 &nbsp;·&nbsp; {n_params_val:,} parameters
</p>

<div class="summary-card">
  <strong style="font-size:1rem;">At a glance</strong>
  <div class="metrics">
    <div class="metric"><div class="label">Recon MSE (test)</div><div class="value">{final_mse:.4f}</div></div>
    <div class="metric"><div class="label">vs exp23 baseline</div><div class="value">{epochs_val} epochs</div></div>
    <div class="metric"><div class="label">N_VAR / K_NN</div><div class="value">{n_var} / {k_nn}</div></div>
    <div class="metric"><div class="label">Training time</div><div class="value">{int(time_s//60)} m {int(time_s%60)} s</div></div>
  </div>
  <p>
    exp29 doubles training duration from 100 to {epochs_val} epochs.
    Hypothesis: exp23 training curves were still declining at ep99 (−1.6e-5/checkpoint).
    Cosine schedule auto-scales to {epochs_val} epochs (num_steps = {epochs_val} × num_batches).
  </p>
</div>

<h2 id="results">Results</h2>

<h3>Learning curves</h3>
<p class="caption">
  <strong>Left:</strong> training losses. &nbsp;
  <strong>Centre:</strong> test reconstruction MSE and variation recall MSE. &nbsp;
  <strong>Right:</strong> recall MSE by neighbor rank.
</p>
<img src="{curves_uri}" alt="learning curves" class="plot full-width">

<h3>Reconstruction quality</h3>
<p class="caption">Original (top) vs reconstruction (bottom) for held-out test images.</p>
<img src="{recon_uri}" alt="reconstructions" class="plot full-width">

<h2>Configuration</h2>
<table>
  <tr><th>Setting</th><th>Value</th></tr>
  <tr><td>Dataset</td><td>CIFAR-10 — 50,000 train / 10,000 test, 32×32 RGB</td></tr>
  <tr><td>Encoder</td><td>CNN: Conv({enc_c1}/{enc_c2}/{enc_c3}, stride-2) + FC({enc_fc1}→{latent_dim})</td></tr>
  <tr><td>Content vector (latent)</td><td>{latent_dim} dimensions</td></tr>
  <tr><td>Style variables</td><td>{n_cat} categorical, {vocab_size} values each → 4,096 combinations</td></tr>
  <tr><td>Nearest neighbors</td><td>{k_nn} per training image (pixel space, sliced from k64)</td></tr>
  <tr><td>Variation candidates / step</td><td>{n_var} random style codes</td></tr>
  <tr><td>ALPHA (variation loss weight)</td><td>{alpha}</td></tr>
  <tr><td>Batch size</td><td>{batch_size}</td></tr>
  <tr><td>Decoder channels C1/C2/C3</td><td>{dc_c1} / {dc_c2} / {dc_c3}</td></tr>
  <tr><td>Decoder stem width</td><td>{dc_stem}</td></tr>
  <tr><td>Epochs</td><td><strong>{epochs_val}</strong> (was 100 in exp23), cosine LR 3×10⁻⁴ → 1.5×10⁻⁵</td></tr>
  <tr><td>Total parameters</td><td>{n_params_val:,}</td></tr>
</table>

<h2 id="grids">Variation grids — per class</h2>
<ul>
  <li><strong>Top strip:</strong> training image + 10 nearest pixel-space neighbors.</li>
  <li><strong>Middle strip:</strong> closest decoded output per neighbor (256 style codes swept).</li>
  <li><strong>16×16 mosaic:</strong> systematic style sweep.</li>
</ul>

{class_sections}

<footer>
  Conditioned variation model &nbsp;·&nbsp; {EXP} &nbsp;·&nbsp;
  CIFAR-10 &nbsp;·&nbsp; LATENT={latent_dim} {epochs_val}-epoch wide-deconvnet &nbsp;·&nbsp; generated 2026-04-17
</footer>
</body>
</html>
"""

r2_url = save_media(
    f"variations_{EXP}_report.html",
    io.BytesIO(html.encode("utf-8")),
    "text/html"
)
print(f"R2 URL → {r2_url}")
