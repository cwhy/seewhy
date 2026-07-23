"""
Generate a self-contained HTML report for exp14.
All plots are embedded as base64 PNGs. Uploads to R2.

Usage:
    uv run python scripts/tmp/gen_html_report_exp14.py
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

EXP      = "exp14"
PROJECT  = ROOT / "projects" / "variations"
PLOTS    = PROJECT / "plots"
OUT_HTML = PROJECT / f"report_{EXP}.html"

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
dc_stem      = result["dc_stem"]

# ── Helper: PNG file → base64 data URI ────────────────────────────────────────

def to_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

# ── Regenerate learning curves plot (3 panels) ────────────────────────────────

epochs   = list(range(len(history["loss"])))
ev_ep,  ev_mse  = zip(*history["eval_mse"])
rec_ep, rec_mse = zip(*history["eval_recall_mse"])
rank_data = history["eval_recall_by_rank"]   # [(epoch, [64 floats]), ...]

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

fig.suptitle("Learning dynamics — exp14", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
curves_buf = io.BytesIO()
fig.savefig(curves_buf, format="png", dpi=130, bbox_inches="tight")
curves_uri = "data:image/png;base64," + base64.b64encode(curves_buf.getvalue()).decode()
plt.close(fig)
print("curves plot rendered")

# ── Load existing PNG plots ────────────────────────────────────────────────────

recon_uri  = to_data_uri(PLOTS / f"variations_{EXP}_recon.png")
digit_uris = {d: to_data_uri(PLOTS / f"variations_{EXP}_var_digit{d}.png") for d in range(10)}
print("digit plots loaded")

# ── Build digit sections ───────────────────────────────────────────────────────

DIGIT_NAMES = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]

digit_sections = ""
for d in range(10):
    digit_sections += f"""
      <div class="digit-block">
        <h3>Digit {d} &mdash; {DIGIT_NAMES[d]}</h3>
        <img src="{digit_uris[d]}" alt="digit {d} variations" class="plot full-width">
      </div>
"""

# ── HTML ──────────────────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Learning to Generate Handwriting Variations — exp14</title>
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

    /* ── summary card ── */
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

    /* ── callout boxes ── */
    .callout {{
      border-left: 4px solid var(--blue); background: #eff6ff;
      border-radius: 0 8px 8px 0; padding: 0.85rem 1.2rem; margin: 1rem 0 1.2rem;
    }}
    .callout.green {{ border-color: var(--green); background: #f0fdf4; }}
    .callout.amber {{ border-color: #d97706;       background: #fffbeb; }}
    .callout strong {{ display: block; margin-bottom: 0.2rem; }}

    /* ── table ── */
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 1.5rem; font-size: 0.9rem; }}
    th {{
      background: var(--light); text-align: left; font-size: 0.78rem;
      text-transform: uppercase; letter-spacing: 0.04em; color: var(--gray);
      padding: 0.5rem 0.9rem; border-bottom: 2px solid var(--border);
    }}
    td {{ padding: 0.45rem 0.9rem; border-bottom: 1px solid var(--border); }}
    tr:hover td {{ background: var(--light); }}
    td:first-child {{ font-weight: 500; }}

    /* ── images ── */
    .plot {{
      display: block; max-width: 100%; border-radius: 8px;
      border: 1px solid var(--border); margin: 0.8rem 0 1.6rem;
    }}
    .full-width {{ width: 100%; }}
    .caption {{ font-size: 0.85rem; color: var(--gray); margin-bottom: 0.4rem; }}

    .digit-block {{ margin: 2rem 0; }}
    .digit-block h3 {{ font-size: 1.05rem; color: #374151; }}

    /* ── diagram ── */
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

<!-- ═══════════════════════════ HEADER ═══════════════════════════════════════ -->

<h1>Learning to Generate Handwriting Variations</h1>
<p class="subtitle">
  exp14 &nbsp;·&nbsp; MNIST &nbsp;·&nbsp; deconvnet decoder &nbsp;·&nbsp;
  {int(time_s // 60)} min training on RTX 4090 &nbsp;·&nbsp;
  {n_params_val:,} parameters
</p>

<!-- ═══════════════════════════ SUMMARY ══════════════════════════════════════ -->

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
      <div class="label">Speedup vs exp11</div>
      <div class="value">~1.5×</div>
    </div>
  </div>
  <p>
    exp14 replaces the Gram-matrix decoder from prior experiments with a transposed-convolution
    (deconvnet) decoder. The architecture generates sharper reconstructions and trains ~33% faster
    than the equivalent Gram-matrix experiment (exp13: ~31 min vs exp14: ~21 min), as cuDNN
    handles ConvTranspose far more efficiently than the Gram decoder's large intermediate tensors.
    Best reconstruction MSE so far across all variation experiments.
  </p>
  <p style="margin-bottom:0">
    <strong>Jump to:</strong>
    <a href="#idea">The idea</a> &nbsp;·&nbsp;
    <a href="#architecture">Architecture</a> &nbsp;·&nbsp;
    <a href="#training">How training works</a> &nbsp;·&nbsp;
    <a href="#intuition">Why it works</a> &nbsp;·&nbsp;
    <a href="#results">Results</a> &nbsp;·&nbsp;
    <a href="#grids">Variation grids</a>
  </p>
</div>

<!-- ═══════════════════════════ IDEA ═════════════════════════════════════════ -->

<h2 id="idea">The idea</h2>

<p>
  The digit "3" can be written in thousands of ways — tall or squat, slanted left or right,
  with a sharp or rounded middle bend. A standard image reconstruction model encodes the exact
  appearance and spits it back, but gives you no handle to say "give me that 3, but rounder."
</p>

<p>
  This model learns to separate <strong>what</strong> a digit is from <strong>how</strong> it
  is drawn. It does this by splitting the input to the decoder into two parts:
</p>

<div class="callout">
  <strong>Content vector</strong>
  A {latent_dim}-dimensional continuous code produced by an encoder network from the input image.
  It captures digit identity, gross shape, and proportions.
</div>
<div class="callout green">
  <strong>Style code</strong>
  Four independent categorical variables, each with {vocab_size} possible values — giving
  {vocab_size}<sup>{n_cat}</sup>&nbsp;=&nbsp;4,096 distinct combinations. This code is
  <em>not</em> derived from the image; it is a discrete "knob" the user can set freely at
  generation time.
</div>

<p>
  At generation time you feed in any image to get its content vector, then sweep the four style
  knobs to produce a grid of variations that all look like the same digit in different handwriting.
</p>

<div class="diagram">
  image ──► <b>Encoder</b> (784→512→64) ──► content vector (64-d) ──────────────────────────────────┐
                                                                                                      ▼
  style code (4 categorical) ── embed ── (16-d) ─────────────────────────────────────────► <b>Decoder</b> ──► 28×28 image
</div>

<!-- ═══════════════════════════ ARCHITECTURE ═════════════════════════════════ -->

<h2 id="architecture">Decoder architecture — deconvnet</h2>

<p>
  The key change in exp14 is replacing the Gram-matrix decoder with a
  <strong>transposed-convolution (deconvnet) decoder</strong>. The decoder receives a
  concatenated vector of content (64-d) and style (16-d) = <code>IN_DIM = 80</code>.
</p>

<div class="diagram">
  zc (B, 80)
    │
    ▼  Linear(80 → {dc_stem}) + GELU          &nbsp;&nbsp;&nbsp;  MLP stem
    ▼  Linear({dc_stem} → {dc_c1}×7×7) + GELU
    ▼  reshape → (B, {dc_c1}, 7, 7)
    │
    ▼  ConvTranspose(stride=2, 4×4 kernel) + GELU  →  (B, {dc_c2}, 14, 14)
    ▼  ConvTranspose(stride=2, 4×4 kernel)         →  (B, 1, 28, 28)
    ▼  sigmoid → flatten → (B, 784)
</div>

<p>
  ConvTranspose with stride 2 doubles the spatial resolution at each stage: 7×7 → 14×14 → 28×28.
  cuDNN implements transposed convolution as a regular convolution on the backward pass, which
  avoids the large intermediate tensors that made the Gram decoder slow. Result: the same
  batch size (128) and variation budget (N=16) now completes in ~21 min instead of ~31 min.
</p>

<!-- ═══════════════════════════ TRAINING ═════════════════════════════════════ -->

<h2 id="training">How training works</h2>

<p>
  The central challenge is supervision: we have no labelled pairs of "same digit, different
  handwriting." The solution is to manufacture those pairs from the training data itself using
  nearest-neighbor search.
</p>

<p>
  Before training begins, for every image in the 60,000-image MNIST training set we precompute
  its <strong>{k_nn} nearest neighbors</strong> in pixel space — the {k_nn} images that look
  most similar. These neighbors are real samples: same digit, slightly different pen strokes.
  They form our variation targets.
</p>

<h3>Each training step</h3>

<div class="callout green">
  <strong>Step-by-step</strong>
  <ol style="margin-top: 0.5rem;">
    <li>
      Sample a batch of {batch_size} images. Encode each → content vector.
    </li>
    <li>
      <strong>Reconstruction path:</strong> decode with a fixed "reconstruction" style code
      (all zeros). Minimize pixel MSE to the original image. This anchors the content vector
      to faithfully represent the input.
    </li>
    <li>
      <strong>Variation path:</strong> sample {n_var} random style codes. For each image in
      the batch, decode once per style code → {n_var} candidate output images.
    </li>
    <li>
      Look up the {k_nn} precomputed neighbors of each image. For every neighbor, find the
      <em>closest candidate</em> (by pixel distance) among the {n_var} options and assign it.
      This assignment is not differentiable and acts purely as a target-selection step.
    </li>
    <li>
      Minimize pixel MSE between each neighbor and the candidate assigned to it. Gradients
      flow only through the decoder and encoder, not through the assignment.
    </li>
    <li>
      Add both losses (equal weight) and update all parameters with Adam.
    </li>
  </ol>
</div>

<p>
  Total decode calls per step: {batch_size}&nbsp;×&nbsp;{n_var}&nbsp;=&nbsp;{batch_size*n_var}
  (variation path) plus {batch_size} (reconstruction) = {batch_size*n_var + batch_size}.
  Training runs for 100 epochs with a cosine learning-rate schedule from 3×10⁻⁴ down to
  ~1.5×10⁻⁵.
</p>

<!-- ═══════════════════════════ INTUITION ════════════════════════════════════ -->

<h2 id="intuition">Why it works</h2>

<p>
  The logic behind the variation loss is a <strong>coverage pressure</strong>.
</p>

<div class="callout amber">
  <strong>The key argument</strong>
  If two different style codes produce nearly identical images, they will both get assigned
  to the <em>same</em> neighbor — and the other neighbors remain uncovered, contributing to
  a large loss. To minimize loss the model must spread its {n_var} candidates to cover
  <em>different</em> neighbors. This is only possible if different style codes produce
  genuinely different images.
</div>

<p>
  Over many steps, each style code specializes for a region of the handwriting-variation space.
  One code might drift toward slanted strokes; another toward thick upright ones. Because this
  emerges from matching real neighbor images, the resulting variations are grounded in actual
  handwriting diversity rather than arbitrary pixel noise.
</p>

<p>
  The reconstruction path prevents the content vector from collapsing: if the decoder could
  invent content freely, it might ignore the encoder altogether and simply memorize each style
  code as a prototype image. The reconstruction loss keeps the content vector informative and
  the style code in charge of variation only.
</p>

<!-- ═══════════════════════════ RESULTS ══════════════════════════════════════ -->

<h2 id="results">Results</h2>

<h3>Learning curves</h3>
<p class="caption">
  <strong>Left:</strong> training losses (total · reconstruction · variation) over 100 epochs.<br>
  <strong>Centre:</strong> reconstruction MSE on the held-out test set and variation recall MSE,
  evaluated every 5 epochs.<br>
  <strong>Right:</strong> recall quality broken down by neighbor rank (rank 1 = nearest neighbor).
  Blue lines = early epochs; red = final epoch. Closer neighbors are always better covered.
</p>
<img src="{curves_uri}" alt="learning curves" class="plot full-width">

<p>
  Both losses decrease steadily without divergence. The test reconstruction MSE reaches
  <code>{final_mse:.4f}</code> — the best result across all variation experiments to date.
  The per-rank curve confirms that the model covers nearest neighbors best and degrades
  gracefully for more distant ones — a sensible prior.
</p>

<h3>Reconstruction quality</h3>
<p class="caption">
  Each column shows an original test image (top row) paired with its reconstruction (bottom row),
  obtained by decoding with the fixed reconstruction style code. The model has not seen these
  images during training.
</p>
<img src="{recon_uri}" alt="reconstructions" class="plot full-width">

<!-- ═══════════════════════════ HYPERPARAMS ══════════════════════════════════ -->

<h2>Configuration</h2>
<table>
  <tr><th>Setting</th><th>Value</th></tr>
  <tr><td>Encoder output (content vector)</td><td>{latent_dim} dimensions</td></tr>
  <tr><td>Style variables</td><td>{n_cat} categorical, {vocab_size} values each → 4,096 combinations</td></tr>
  <tr><td>Nearest neighbors used</td><td>{k_nn} per training image (precomputed, pixel distance)</td></tr>
  <tr><td>Variation candidates per step</td><td>{n_var} random style codes</td></tr>
  <tr><td>Batch size</td><td>{batch_size}</td></tr>
  <tr><td>Decoder MLP stem width</td><td>{dc_stem}</td></tr>
  <tr><td>Decoder channels C1 / C2</td><td>{dc_c1} / {dc_c2}</td></tr>
  <tr><td>Decoder type</td><td>deconvnet — MLP stem → reshape 7×7 → ConvTranspose ×2 → 28×28</td></tr>
  <tr><td>Epochs</td><td>100</td></tr>
  <tr><td>Optimizer</td><td>Adam, cosine LR 3×10⁻⁴ → 1.5×10⁻⁵</td></tr>
  <tr><td>Loss weighting</td><td>recon + 1.0 × variation</td></tr>
  <tr><td>Total parameters</td><td>{n_params_val:,}</td></tr>
</table>

<!-- ═══════════════════════════ GRIDS ════════════════════════════════════════ -->

<h2 id="grids">Variation grids — per digit</h2>

<p>
  Each figure below has three parts:
</p>
<ul>
  <li>
    <strong>Top strip:</strong> a training image (leftmost) and its 10 nearest neighbors from the
    training set, ordered left to right.
  </li>
  <li>
    <strong>Middle strip:</strong> for each neighbor, the closest output found by sweeping 256 random
    style codes through the decoder. This is a proxy for how well the learned variation space can
    actually "recall" real handwriting samples.
  </li>
  <li>
    <strong>16×16 mosaic:</strong> a systematic sweep across the four style knobs, arranged as four
    8×8 sub-grids (quadrants). Each quadrant fixes the values of knob&nbsp;3 and knob&nbsp;4; within
    each quadrant, rows vary knob&nbsp;1 (0→7) and columns vary knob&nbsp;2 (0→7). Red lines
    divide the quadrants. Smooth transitions across the grid indicate the style space has been
    organized into a coherent, structured manifold.
  </li>
</ul>

{digit_sections}

<!-- ═══════════════════════════ FOOTER ═══════════════════════════════════════ -->

<footer>
  Conditioned variation model &nbsp;·&nbsp;
  exp14 &nbsp;·&nbsp;
  MNIST dataset &nbsp;·&nbsp;
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
