"""
Visualization utilities for ema-viz.

All functions are pure: they take data arrays and return matplotlib figures or
HTML strings. No side effects, no file I/O, no logging.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CMAP = plt.cm.get_cmap("tab10", 10)

# tab10 hex colours for use in SVG/HTML (matches matplotlib CMAP)
_TAB10_HEX = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ── Analysis ──────────────────────────────────────────────────────────────────

def cluster_quality(E, Y):
    """Intra/inter-class L2 distance ratio.

    Returns (intra, inter, ratio) where ratio = inter / intra.
    Higher ratio = better separated clusters.
    """
    classes = np.unique(Y)
    centroids = np.stack([E[Y == c].mean(0) for c in classes])
    intra = np.mean([
        np.linalg.norm(E[Y == c] - centroids[c], axis=1).mean()
        for c in classes
    ])
    diffs = centroids[:, None, :] - centroids[None, :, :]
    dist  = np.linalg.norm(diffs, axis=-1)
    inter = dist[np.triu_indices(len(classes), k=1)].mean()
    return float(intra), float(inter), float(inter / (intra + 1e-8))


def knn_retrieval(E, Y, X_imgs, n_queries=10, k=5):
    """For n_queries (one per class in order), return top-k nearest neighbours.

    Returns list of (query_img, nn_imgs, nn_labels) tuples.
    """
    results = []
    for c in range(n_queries):
        idx = np.where(Y == c)[0]
        q_i = idx[0]
        q_e = E[q_i]
        dists = np.linalg.norm(E - q_e, axis=1)
        dists[q_i] = np.inf
        nn_idx = np.argsort(dists)[:k]
        results.append((X_imgs[q_i], X_imgs[nn_idx], Y[nn_idx]))
    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_tsne(E_2d, Y, title="t-SNE of embeddings"):
    """Scatter plot of 2D t-SNE embeddings coloured by class label."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for c in range(len(np.unique(Y))):
        mask = Y == c
        ax.scatter(E_2d[mask, 0], E_2d[mask, 1], s=2, alpha=0.4,
                   color=CMAP(c), label=str(c))
    ax.legend(loc="upper right", markerscale=4, fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_retrieval(results, k=5):
    """Grid of query images (left) and their k nearest neighbours (right).

    results: list of (query_img, nn_imgs, nn_labels) from knn_retrieval().
    """
    n = len(results)
    fig, axes = plt.subplots(n, k + 1, figsize=(2 * (k + 1), 2 * n))
    for row, (q_img, nn_imgs, nn_labels) in enumerate(results):
        axes[row, 0].imshow(q_img.reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title("query", fontsize=7)
        axes[row, 0].axis("off")
        for col in range(k):
            axes[row, col + 1].imshow(nn_imgs[col].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
            axes[row, col + 1].set_title(f"nn{col+1}={nn_labels[col]}", fontsize=7)
            axes[row, col + 1].axis("off")
    fig.suptitle("Nearest-neighbour retrieval (L2)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_gram_components(params, dec_rank, d_rc):
    """Attribution maps for each gram basis vector.

    For component k, the map shows f[:,k]² — how strongly each pixel position
    activates that component (independent of pixel intensity).

    params: encoder param dict with enc_row_emb, enc_col_emb, W_pos_enc, W_enc.
    dec_rank: DEC_RANK hyperparameter (number of components).
    d_rc: D_RC hyperparameter (D_R * D_C).
    """
    rows_idx = np.arange(784) // 28
    cols_idx = np.arange(784) % 28
    re       = np.array(params["enc_row_emb"])[rows_idx]          # (784, D_R)
    ce       = np.array(params["enc_col_emb"])[cols_idx]          # (784, D_C)
    pos_rc   = np.einsum("pr,pc->prc", re, ce).reshape(784, d_rc) # (784, D_RC)
    pos_feat = pos_rc @ np.array(params["W_pos_enc"])              # (784, D)
    f        = pos_feat @ np.array(params["W_enc"])                # (784, dec_rank)
    attr     = f ** 2
    attr     = attr / (attr.max(0, keepdims=True) + 1e-8)

    cols = 8
    n_rows = dec_rank // cols
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 1.5, n_rows * 1.5))
    for k, ax in enumerate(axes.flat):
        ax.imshow(attr[:, k].reshape(28, 28), cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"k={k}", fontsize=6)
        ax.axis("off")
    fig.suptitle("Gram component attribution maps (W_enc columns, squared)", fontsize=9)
    fig.tight_layout()
    return fig


# ── t-SNE animation ───────────────────────────────────────────────────────────

def tsne_animation_html(snapshots, Y, title="t-SNE optimisation drift", fps=2.0):
    """Build a self-contained HTML file that animates t-SNE frame-by-frame.

    snapshots : list of (N, 2) float arrays — positions at each captured step.
    Y         : (N,) int array — class labels (0-9).
    title     : displayed above the SVG.
    fps       : default playback speed in frames/second.

    Returns a complete HTML string (inline SVG + JS, no external deps).

    Layout
    ------
    - 620×620 SVG; points normalised globally across all frames to [10, 610].
    - Each point is an SVG <circle>; positions are driven by JS interpolation
      between keyframes stored as a JSON array in a <script> block.
    - Controls: play/pause, scrubber, speed slider, frame label.
    - Smooth linear interpolation between consecutive frames.
    """
    SVG_SIZE  = 620
    PAD       = 20           # pixels of padding inside the SVG viewport
    INNER     = SVG_SIZE - 2 * PAD

    # ── normalise all frames together so scale is stable across time ──────────
    all_pts = np.concatenate(snapshots, axis=0)
    lo = all_pts.min(0)
    hi = all_pts.max(0)
    span = (hi - lo).clip(min=1e-8)

    def to_svg(pts):
        # (N,2) in data space → (N,2) in SVG px, rounded to 1 decimal
        scaled = (pts - lo) / span * INNER + PAD
        scaled[:, 1] = SVG_SIZE - scaled[:, 1]   # flip Y (SVG y-down)
        return np.round(scaled, 1).tolist()

    frames_svg = [to_svg(s) for s in snapshots]
    n_frames   = len(frames_svg)

    # ── per-point colour (hex string, indexed by class label) ─────────────────
    colors = [_TAB10_HEX[int(y) % len(_TAB10_HEX)] for y in Y]

    # ── label text: phase annotation per frame ────────────────────────────────
    # (caller can override by passing a list; here we just number them)
    frame_labels = [f"frame {i+1} / {n_frames}" for i in range(n_frames)]

    # ── serialise to compact JSON ─────────────────────────────────────────────
    frames_json = json.dumps(frames_svg, separators=(",", ":"))
    colors_json = json.dumps(colors,     separators=(",", ":"))
    labels_json = json.dumps(frame_labels, separators=(",", ":"))

    # ── build legend SVG fragment ─────────────────────────────────────────────
    n_classes = len(_TAB10_HEX)
    legend_items = "".join(
        f'<circle cx="16" cy="{14 + i*20}" r="5" fill="{_TAB10_HEX[i]}"/>'
        f'<text x="26" y="{18 + i*20}" fill="#8b949e" font-size="11" font-family="monospace">{i}</text>'
        for i in range(n_classes)
    )
    legend_svg = (
        f'<svg width="50" height="{14 + n_classes*20}" '
        f'style="position:absolute;top:10px;right:10px;">'
        f'{legend_items}</svg>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1117; color: #e6edf3;
    font-family: ui-monospace, monospace;
    display: flex; flex-direction: column; align-items: center;
    padding: 20px 0 32px;
  }}
  h2 {{ font-size: 13px; color: #8b949e; margin-bottom: 12px; font-weight: normal; }}
  #wrap {{ position: relative; display: inline-block; }}
  svg#viz {{ display: block; background: #0d1117; }}
  #controls {{
    display: flex; align-items: center; gap: 12px;
    margin-top: 10px; padding: 0 4px;
  }}
  button {{
    background: #21262d; border: 1px solid #30363d; color: #e6edf3;
    padding: 3px 10px; cursor: pointer; border-radius: 4px;
    font-family: inherit; font-size: 14px; line-height: 1.4;
  }}
  button:hover {{ background: #30363d; }}
  input[type=range] {{ accent-color: #58a6ff; cursor: pointer; }}
  #scrubber {{ width: 320px; }}
  #speed-range {{ width: 80px; }}
  .lbl {{ font-size: 11px; color: #8b949e; white-space: nowrap; }}
  #phase-bar {{
    width: {SVG_SIZE}px; height: 3px; background: #21262d;
    margin-top: 2px; border-radius: 2px; overflow: hidden;
  }}
  #phase-fill {{ height: 100%; background: #58a6ff; width: 0%; transition: width 0.1s; }}
</style>
</head>
<body>
<h2>{title}</h2>
<div id="wrap">
  <svg id="viz" width="{SVG_SIZE}" height="{SVG_SIZE}"
       viewBox="0 0 {SVG_SIZE} {SVG_SIZE}">
    <rect width="{SVG_SIZE}" height="{SVG_SIZE}" fill="#0d1117"/>
  </svg>
  {legend_svg}
</div>
<div id="phase-bar"><div id="phase-fill"></div></div>
<div id="controls">
  <button id="btn">⏸</button>
  <input type="range" id="scrubber" min="0" max="{n_frames - 1}" step="0.01" value="0">
  <span class="lbl" id="frame-lbl">frame 1 / {n_frames}</span>
  <span class="lbl">speed</span>
  <input type="range" id="speed-range" min="0.25" max="8" step="0.25" value="{fps}">
  <span class="lbl" id="speed-lbl">{fps}×</span>
</div>
<script>
const FRAMES  = {frames_json};
const COLORS  = {colors_json};
const LABELS  = {labels_json};
const N       = FRAMES[0].length;
const NF      = FRAMES.length;
const svg     = document.getElementById('viz');
const NS      = 'http://www.w3.org/2000/svg';

// Build circles once
const circles = new Array(N);
for (let i = 0; i < N; i++) {{
  const c = document.createElementNS(NS, 'circle');
  c.setAttribute('r', '2.2');
  c.setAttribute('fill', COLORS[i]);
  c.setAttribute('fill-opacity', '0.65');
  svg.appendChild(c);
  circles[i] = c;
}}

// State
let playing = true;
let t = 0;          // continuous fractional frame index
let speed = {fps};
let last = null;

function lerp(a, b, alpha) {{ return a + (b - a) * alpha; }}

function render(ft) {{
  const f0    = Math.floor(ft) % NF;
  const f1    = (f0 + 1) % NF;
  const alpha = ft - Math.floor(ft);
  const pts0  = FRAMES[f0];
  const pts1  = FRAMES[f1];
  for (let i = 0; i < N; i++) {{
    circles[i].setAttribute('cx', lerp(pts0[i][0], pts1[i][0], alpha).toFixed(1));
    circles[i].setAttribute('cy', lerp(pts0[i][1], pts1[i][1], alpha).toFixed(1));
  }}
  const fi = Math.min(Math.round(ft), NF - 1);
  document.getElementById('frame-lbl').textContent = LABELS[fi];
  document.getElementById('scrubber').value        = ft % NF;
  document.getElementById('phase-fill').style.width = ((ft % NF) / (NF - 1) * 100).toFixed(1) + '%';
}}

function tick(ts) {{
  if (playing) {{
    if (last !== null) t += (ts - last) / 1000 * speed;
    last = ts;
    if (t >= NF) t = 0;   // loop
    render(t);
  }}
  requestAnimationFrame(tick);
}}
requestAnimationFrame(tick);
render(0);

document.getElementById('btn').addEventListener('click', () => {{
  playing = !playing;
  document.getElementById('btn').textContent = playing ? '⏸' : '▶';
  if (playing) last = null;
}});
document.getElementById('scrubber').addEventListener('input', e => {{
  t = parseFloat(e.target.value);
  render(t);
}});
document.getElementById('speed-range').addEventListener('input', e => {{
  speed = parseFloat(e.target.value);
  document.getElementById('speed-lbl').textContent = speed + '×';
}});
</script>
</body>
</html>"""

    return html
