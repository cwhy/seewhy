"""
Generate a self-contained HTML report for exp15 (CIFAR-10).
All plots are embedded as base64 PNGs. Uploads to R2.

Usage:
    uv run python scripts/tmp/gen_html_report_exp15.py
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

EXP      = "exp15"
PROJECT  = ROOT / "projects" / "variations"
PLOTS    = PROJECT / "plots"
OUT_HTML = PROJECT / f"report_{EXP}.html"

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── Load artifacts ─────────────────────────────────────────────────────────────

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

# ── Helper: PNG file → base64 data URI ────────────────────────────────────────

def to_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

# ── Regenerate learning curves plot (3 panels) ────────────────────────────────

epochs   = list(range(len(history["loss"])))
ev_ep,  ev_mse  = zip(*history["eval_mse"])
rec_ep, rec_mse = zip(*history["eval_recall_mse"])
rank_data = history["eval_recall_by_rank"]

fig, axes = plt.subplots(1, 3, figsize=(18, 4), facecolor="#fafafa")
for ax in axes:
    ax.set_facecolor("#fafafa")

ax = axes[0]
ax.plot(epochs, history["loss"],    label="total loss",      lw=1.8, color="#2563eb")
ax.plot(epochs, history["L_recon"], label="recon loss",      lw=1.5, ls="--", color="#16a34a")
ax.plot(epochs, history["L_var"],   label="variation loss",  lw=1.5, ls=":",  color="#dc2626")
ax.set_xlabel("epoch", fontsize=11); ax.set_ylabel("MSE", fontsize=11)
ax.set_title("Training losses", fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.25)

ax = axes[1]
ax.plot(ev_ep,  ev_mse,  "o-",  lw=1.8, label="reconstruction MSE (test set)", color="#2563eb", ms=4)
ax.plot(rec_ep, rec_mse, "s--", lw=1.8, label="variation recall MSE",          color="#dc2626", ms=4)
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
ax.set_xlabel("neighbor rank (1 = closest neighbor)", fontsize=11)
ax.set_ylabel("MSE", fontsize=11)
ax.set_title("Recall quality by neighbor rank\n(blue = early epoch  →  red = final epoch)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.25)

fig.suptitle("Learning dynamics — exp15 (CIFAR-10)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
curves_buf = io.BytesIO()
fig.savefig(curves_buf, format="png", dpi=130, bbox_inches="tight")
curves_uri = "data:image/png;base64," + base64.b64encode(curves_buf.getvalue()).decode()
plt.close(fig)
print("curves plot rendered")

# ── Load existing PNG plots ────────────────────────────────────────────────────

recon_uri   = to_data_uri(PLOTS / f"variations_{EXP}_recon.png")
class_uris  = {c: to_data_uri(PLOTS / f"variations_{EXP}_var_class{c}.png") for c in range(10)}
print("class plots loaded")

# ── Build class sections ───────────────────────────────────────────────────────

class_sections = ""
for c in range(10):
    class_sections += f"""
      <div class="class-block">
        <h3>Class {c} &mdash; {CIFAR_CLASSES[c].capitalize()}</h3>
        <img src="{class_uris[c]}" alt="class {c} {CIFAR_CLASSES[c]} variations" class="plot full-width">
      </div>
"""

# ── HTML ──────────────────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Learning to Generate Image Variations — exp15 (CIFAR-10)</title>
  <style>
    :root {{
      --blue:   #2563eb;
      --green:  #16a34a;
      --red:    #dc2626;
      --gray:   #6b7280;
      --light:  #f8fafc;
      --border: #e2e8f0;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Segoe UI", system-ui, sans-serif;
      font-size: 15px; line-height: 1.7;
      color: #1e293b; background: #fff;
      max-width: 960px; margin: 0 auto;
      padding: 2.5rem 1.5rem 5rem;
    }}
    h1 {{ font-size: 2rem; font-weight: 700; color: #0f172a; margin-bottom: 0.3rem; line-height: 1.2; }}
    h2 {{
      font-size: 1.25rem; font-weight: 700; color: #0f172a;
      border-bottom: 2px solid var(--border); padding-bottom: 0.4rem;
      margin: 2.8rem 0 1rem;
    }}
    h3 {{ font-size: 1rem; font-weight: 600; color: #374151; margin: 1.6rem 0 0.4rem; }}
    p  {{ margin-bottom: 0.9rem; }}
    ul, ol {{ margin: 0.4rem 0 0.9rem 1.5rem; }}
    li {{ margin-bottom: 0.3rem; }}
    a  {{ color: var(--blue); }}
    code {{
      font-family: "Fira Code", "Cascadia Code", monospace; font-size: 0.87em;
      background: #f1f5f9; border-radius: 4px; padding: 1px 5px;
    }}

    .subtitle {{ color: var(--gray); font-size: 0.93rem; margin: 0.2rem 0 2rem; }}

    .summary-card {{
      background: var(--light); border: 1px solid var(--border);
      border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 2.2rem;
    }}
    .summary-card > p:last-child {{ margin-bottom: 0; }}
    .metrics {{
      display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.8rem 0 1rem;
    }}
    .metric {{
      background: #fff; border: 1px solid var(--border); border-radius: 8px;
      padding: 0.55rem 1rem; min-width: 130px; flex: 1;
    }}
    .metric .label {{ font-size: 0.75rem; color: var(--gray);
                       text-transform: uppercase; letter-spacing: 0.05em; }}
    .metric .value {{ font-size: 1.3rem; font-weight: 700; color: #0f172a; }}

    .callout {{
      border-left: 4px solid var(--blue); background: #eff6ff;
      border-radius: 0 8px 8px 0; padding: 0.85rem 1.2rem; margin: 1rem 0 1.2rem;
    }}
    .callout.green {{ border-color: var(--green); background: #f0fdf4; }}
    .callout.amber {{ border-color: #d97706;       background: #fffbeb; }}
    .callout strong {{ display: block; margin-bottom: 0.2rem; }}

    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 1.5rem; font-size: 0.9rem; }}
    th {{
      background: var(--light); text-align: left; font-size: 0.78rem;
      text-transform: uppercase; letter-spacing: 0.04em; color: var(--gray);
      padding: 0.5rem 0.9rem; border-bottom: 2px solid var(--border);
    }}
    td {{ padding: 0.45rem 0.9rem; border-bottom: 1px solid var(--border); }}
    tr:hover td {{ background: var(--light); }}
    td:first-child {{ font-weight: 500; }}

    .plot {{
      display: block; max-width: 100%; border-radius: 8px;
      border: 1px solid var(--border); margin: 0.8rem 0 1.6rem;
    }}
    .full-width {{ width: 100%; }}
    .caption {{ font-size: 0.85rem; color: var(--gray); margin-bottom: 0.4rem; }}

    .class-block {{ margin: 2rem 0; }}
    .class-block h3 {{ font-size: 1.05rem; color: #374151; }}

    .diagram {{
      font-family: "Fira Code", monospace; font-size: 0.82rem;
      background: #f8fafc; border: 1px solid var(--border);
      border-radius: 8px; padding: 1rem 1.4rem; margin: 1rem 0;
      line-height: 2; color: #334155;
      overflow-x: auto;
    }}

    footer {{
      margin-top: 4rem; padding-top: 1.2rem; border-top: 1px solid var(--border);
      font-size: 0.8rem; color: var(--gray); text-align: center;
    }}
  </style>
</head>
<body>

<h1>Learning to Generate Image Variations</h1>
<p class="subtitle">
  exp15 &nbsp;·&nbsp; CIFAR-10 &nbsp;·&nbsp; deconvnet decoder &nbsp;·&nbsp;
  {int(time_s // 60)} min training on RTX 4090 &nbsp;·&nbsp;
  {n_params_val:,} parameters
</p>

<div class="summary-card">
  <strong style="font-size:1rem;">At a glance</strong>
  <div class="metrics">
    <div class="metric">
      <div class="label">Recon MSE (test)</div>
      <div class="value">{final_mse:.4f}</div>
    </div>
    <div class="metric">
      <div class="label">Style combinations</div>
      <div class="value">4,096</div>
    </div>
    <div class="metric">
      <div class="label">Training time</div>
      <div class="value">{int(time_s//60)} m {int(time_s%60)} s</div>
    </div>
    <div class="metric">
      <div class="label">Dataset</div>
      <div class="value">CIFAR-10</div>
    </div>
  </div>
  <p>
    exp15 scales the deconvnet variation model from MNIST (28×28 grayscale) to
    CIFAR-10 (32×32 RGB, 10 natural image classes, 50k training images).
    The encoder is deepened (3072→1024→512→128) and the decoder gains a third
    ConvTranspose stage (4×4→8×8→16×16→32×32, 3 output channels). The same
    inverted-IMLE training objective is used without modification.
  </p>
  <p style="margin-bottom:0">
    <strong>Jump to:</strong>
    <a href="#idea">The idea</a> &nbsp;·&nbsp;
    <a href="#architecture">Architecture</a> &nbsp;·&nbsp;
    <a href="#training">How training works</a> &nbsp;·&nbsp;
    <a href="#results">Results</a> &nbsp;·&nbsp;
    <a href="#grids">Variation grids</a>
  </p>
</div>

<h2 id="idea">The idea</h2>

<p>
  A "cat" photo can be taken in thousands of ways — close-up or distant, bright or shadowed,
  sitting or running. This model learns to separate <strong>what</strong> an image shows from
  <strong>how</strong> it appears, by splitting the decoder input into two parts:
</p>

<div class="callout">
  <strong>Content vector</strong>
  A {latent_dim}-dimensional continuous code from the encoder. Captures object identity,
  rough shape, and composition.
</div>
<div class="callout green">
  <strong>Style code</strong>
  Four independent categorical variables, each with {vocab_size} values — giving
  {vocab_size}<sup>{n_cat}</sup>&nbsp;=&nbsp;4,096 distinct combinations. Not derived from
  the image; freely controllable at generation time.
</div>

<div class="diagram">
  image (32×32 RGB) ──► <b>Encoder</b> (3072→1024→512→128) ──► content vector (128-d) ──────────────┐
                                                                                                       ▼
  style code (4 categorical) ── embed ── (16-d) ────────────────────────────────────────► <b>Decoder</b> ──► 32×32 RGB
</div>

<h2 id="architecture">Decoder architecture — 3-stage deconvnet</h2>

<p>
  The decoder concatenates content (128-d) and style (16-d) → <code>IN_DIM = 144</code>,
  then applies an MLP stem followed by three stride-2 ConvTranspose layers:
</p>

<div class="diagram">
  zc (B, 144)
    │
    ▼  Linear(144 → {dc_stem}) + GELU         &nbsp;&nbsp;&nbsp;  MLP stem
    ▼  Linear({dc_stem} → {dc_c1}×4×4) + GELU
    ▼  reshape → (B, {dc_c1}, 4, 4)
    │
    ▼  ConvTranspose(stride=2) + GELU  →  (B, {dc_c2}, 8, 8)
    ▼  ConvTranspose(stride=2) + GELU  →  (B, {dc_c3}, 16, 16)
    ▼  ConvTranspose(stride=2)         →  (B, 3, 32, 32)
    ▼  sigmoid → flatten → (B, 3072)
</div>

<p>
  Starting from a 4×4 spatial grid rather than 7×7 (MNIST) keeps the stem compact while
  allowing three doubling stages to reach the full 32×32 resolution.
</p>

<h2 id="training">How training works</h2>

<p>
  For every image in the 50,000-image CIFAR-10 training set we precompute its
  <strong>{k_nn} nearest neighbors</strong> in pixel space. These form our variation targets.
</p>

<div class="callout green">
  <strong>Each training step</strong>
  <ol style="margin-top: 0.5rem;">
    <li>Sample a batch of {batch_size} images. Encode each → content vector.</li>
    <li><strong>Reconstruction path:</strong> decode with the fixed "recon" style code. Minimize MSE to original.</li>
    <li><strong>Variation path:</strong> sample {n_var} random style codes. Decode each → {n_var} candidates.</li>
    <li>For each of the {k_nn} precomputed neighbors, assign the closest candidate (argmin, not differentiable).</li>
    <li>Minimize MSE between each neighbor and its assigned candidate.</li>
    <li>Sum both losses (equal weight), update with Adam.</li>
  </ol>
</div>

<p>
  Total decode calls per step: {batch_size}&nbsp;×&nbsp;{n_var}&nbsp;+&nbsp;{batch_size}&nbsp;=&nbsp;{batch_size*n_var + batch_size}.
  100 epochs, cosine LR 3×10⁻⁴ → 1.5×10⁻⁵.
</p>

<div class="callout amber">
  <strong>Coverage pressure</strong>
  Style codes that produce identical outputs share assignment targets — leaving other neighbors
  uncovered and increasing loss. The model is forced to diversify style codes to cover the full
  spread of real image variation.
</div>

<h2 id="results">Results</h2>

<h3>Learning curves</h3>
<p class="caption">
  <strong>Left:</strong> training losses over 100 epochs.<br>
  <strong>Centre:</strong> reconstruction MSE on the held-out test set and variation recall MSE.<br>
  <strong>Right:</strong> recall MSE by neighbor rank (blue = early, red = final epoch).
</p>
<img src="{curves_uri}" alt="learning curves" class="plot full-width">

<p>
  Test reconstruction MSE reaches <code>{final_mse:.4f}</code>.
  Recall MSE steadily decreases, indicating the variation space progressively covers more of
  the real image distribution. Per-rank curves show nearest neighbors are always best covered.
</p>

<h3>Reconstruction quality</h3>
<p class="caption">
  Each column: original test image (top) paired with reconstruction (bottom) using the fixed
  reconstruction style code. Images are 32×32 RGB; the model has not seen these during training.
</p>
<img src="{recon_uri}" alt="reconstructions" class="plot full-width">

<h2>Configuration</h2>
<table>
  <tr><th>Setting</th><th>Value</th></tr>
  <tr><td>Dataset</td><td>CIFAR-10 — 50,000 train / 10,000 test, 32×32 RGB</td></tr>
  <tr><td>Encoder</td><td>3072 → 1024 → 512 → {latent_dim} (GELU activations)</td></tr>
  <tr><td>Content vector (latent)</td><td>{latent_dim} dimensions</td></tr>
  <tr><td>Style variables</td><td>{n_cat} categorical, {vocab_size} values each → 4,096 combinations</td></tr>
  <tr><td>Nearest neighbors used</td><td>{k_nn} per training image (precomputed, pixel distance)</td></tr>
  <tr><td>Variation candidates per step</td><td>{n_var} random style codes</td></tr>
  <tr><td>Batch size</td><td>{batch_size}</td></tr>
  <tr><td>Decoder MLP stem width</td><td>{dc_stem}</td></tr>
  <tr><td>Decoder channels C1 / C2 / C3</td><td>{dc_c1} / {dc_c2} / {dc_c3}</td></tr>
  <tr><td>Decoder type</td><td>deconvnet — stem → reshape 4×4 → ConvTranspose ×3 → 32×32 RGB</td></tr>
  <tr><td>Epochs</td><td>100</td></tr>
  <tr><td>Optimizer</td><td>Adam, cosine LR 3×10⁻⁴ → 1.5×10⁻⁵</td></tr>
  <tr><td>Loss weighting</td><td>recon + 1.0 × variation</td></tr>
  <tr><td>Total parameters</td><td>{n_params_val:,}</td></tr>
</table>

<h2 id="grids">Variation grids — per class</h2>

<p>Each figure has three parts:</p>
<ul>
  <li><strong>Top strip:</strong> a training image (leftmost) and its 10 nearest pixel-space neighbors.</li>
  <li><strong>Middle strip:</strong> for each neighbor, the closest output found by sweeping 256 random style codes.</li>
  <li>
    <strong>16×16 mosaic:</strong> systematic sweep of the four style knobs, arranged as four 8×8
    sub-grids. Each quadrant fixes knob 3 and knob 4; rows vary knob 1 (0→7), columns vary knob 2 (0→7).
    Red lines divide quadrants.
  </li>
</ul>

{class_sections}

<footer>
  Conditioned variation model &nbsp;·&nbsp;
  exp15 &nbsp;·&nbsp;
  CIFAR-10 dataset &nbsp;·&nbsp;
  deconvnet decoder &nbsp;·&nbsp;
  generated 2026-04-15
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
