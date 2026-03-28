"""
Generate an interactive HTML t-SNE visualization for exp5 cluster centers.

Controls: scale slider, per-ring opacity, per-digit color pickers, zoom/pan.

Usage:
    uv run python projects/online-kmeans/scripts/gen_viz_tsne_interactive.py [exp_name]
"""
import base64, io, json, pickle, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.datasets import load_supervised_image
from shared_lib.r2 import upload_media
from sklearn.manifold import TSNE

EXP_NAME = sys.argv[1] if len(sys.argv) > 1 else "exp5"
BASE     = Path(__file__).parent.parent

# ── Load data ──────────────────────────────────────────────────────────────────

with open(BASE / f"params_{EXP_NAME}.pkl", "rb") as f:
    params = pickle.load(f)
centroids = np.array(params["centroids"])
N_CODES   = len(centroids)

label_probs    = np.load(BASE / f"label_probs_{EXP_NAME}.npy")
majority_label = label_probs.argmax(axis=1).astype(int)

# Encode each centroid as a base64 PNG (28×28, per-centroid min-max normalised)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def centroid_to_b64(vec):
    img = vec.copy()
    lo, hi = img.min(), img.max()
    img = (img - lo) / (hi - lo + 1e-9)
    img = img.reshape(28, 28)
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(1.12, 1.12), dpi=50)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

print("Encoding centroid images…")
centroid_imgs = [centroid_to_b64(centroids[c]) for c in range(N_CODES)]
print("Done.")

data  = load_supervised_image("mnist")
X_raw = data.X.reshape(60000, 784).astype(np.float32)
norms = np.linalg.norm(X_raw, axis=1, keepdims=True)
X_all = X_raw / np.where(norms > 0, norms, 1.0)
Y_all = np.array(data.y).astype(np.int32)

dots_all   = X_all @ centroids.T
assign_all = np.argmax(dots_all, axis=1)

# Stratified sample for t-SNE
N_SAMPLE  = 8000
rng       = np.random.default_rng(42)
idx_list  = [rng.choice(np.where(Y_all == d)[0], size=N_SAMPLE // 10, replace=False) for d in range(10)]
sample_idx  = np.concatenate(idx_list)
X_sample    = X_all[sample_idx]
Y_sample    = Y_all[sample_idx]
assign_samp = assign_all[sample_idx]

X_combined = np.vstack([X_sample, centroids])
print(f"Running t-SNE on {len(X_combined):,} points …")
tsne    = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=2000, metric="cosine")
pos_all = tsne.fit_transform(X_combined)
pos_data = pos_all[:N_SAMPLE]
pos_cent = pos_all[N_SAMPLE:]
print("t-SNE done.")

# ── Radii (2D distances from assigned points to centroid) ─────────────────────

radii_2d = np.zeros((N_CODES, 4))
for c in range(N_CODES):
    mask = assign_samp == c
    if mask.sum() > 0:
        dists = np.sqrt(((pos_data[mask] - pos_cent[c]) ** 2).sum(1))
        for i, p in enumerate([25, 50, 75, 100]):
            radii_2d[c, i] = np.percentile(dists, p)

# Global scale: p75 circles don't touch nearest neighbour
diff_cc  = pos_cent[:, None, :] - pos_cent[None, :, :]
dists_cc = np.sqrt((diff_cc ** 2).sum(axis=2))
np.fill_diagonal(dists_cc, np.inf)
nn_dists = dists_cc.min(axis=1)
scale    = float(np.min(nn_dists * 0.5 / (radii_2d[:, 2] + 1e-9)))

# ── Serialise ─────────────────────────────────────────────────────────────────

clusters = []
for c in range(N_CODES):
    clusters.append({
        "id":    c,
        "x":     round(float(pos_cent[c, 0]), 3),
        "y":     round(float(pos_cent[c, 1]), 3),
        "label": int(majority_label[c]),
        "radii": [round(float(radii_2d[c, i]), 3) for i in range(4)],
        "count": int((assign_samp == c).sum()),
        "purity": round(float(label_probs[c].max()), 3),
        "img": centroid_imgs[c],
        "dist": [round(float(label_probs[c, d] * 100), 1) for d in range(10)],
    })

points = [{"x": round(float(pos_data[i, 0]), 2),
           "y": round(float(pos_data[i, 1]), 2),
           "l": int(Y_sample[i])}
          for i in range(N_SAMPLE)]

# Tab10 colours
TAB10 = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
         "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

data_json     = json.dumps(points)
clusters_json = json.dumps(clusters)
colors_json   = json.dumps(TAB10)

# ── HTML ──────────────────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{EXP_NAME} — t-SNE clusters</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ display: flex; height: 100vh; font-family: monospace; font-size: 12px; background: #1a1a1a; color: #ccc; }}
#controls {{ width: 240px; flex-shrink: 0; padding: 12px; overflow-y: auto; background: #222; border-right: 1px solid #333; }}
#controls h2 {{ font-size: 13px; margin-bottom: 10px; color: #eee; }}
.section {{ margin-bottom: 14px; }}
.section label {{ display: block; margin-bottom: 3px; color: #aaa; }}
.section input[type=range] {{ width: 100%; }}
.color-row {{ display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }}
.color-row span {{ flex: 1; }}
.color-row input[type=color] {{ width: 36px; height: 22px; border: none; background: none; cursor: pointer; padding: 0; }}
#tooltip {{ position: fixed; background: rgba(20,20,20,0.95); color: #fff; padding: 8px 10px;
            border-radius: 5px; font-size: 11px; pointer-events: none; display: none; z-index: 99;
            min-width: 160px; border: 1px solid #444; }}
#tooltip img {{ display: block; width: 84px; height: 84px; image-rendering: pixelated;
                margin: 5px auto 6px; border: 1px solid #555; }}
#tooltip .bar-row {{ display: flex; align-items: center; gap: 4px; margin: 1px 0; }}
#tooltip .bar-row .lbl {{ width: 14px; text-align: right; color: #aaa; }}
#tooltip .bar-row .bar {{ height: 7px; background: #4a9eff; border-radius: 2px; min-width: 1px; }}
#tooltip .bar-row .pct {{ color: #ccc; font-size: 10px; }}
#viz {{ flex: 1; overflow: hidden; cursor: grab; }}
#viz:active {{ cursor: grabbing; }}
svg {{ width: 100%; height: 100%; }}
.val {{ color: #eee; float: right; }}
</style>
</head>
<body>

<div id="controls">
  <h2>{EXP_NAME} — t-SNE clusters</h2>

  <div class="section">
    <label>Scale <span class="val" id="scaleVal">1.0×</span></label>
    <input type="range" id="scaleSlider" min="0.1" max="5" step="0.05" value="1.0">
  </div>

  <div class="section">
    <label>p25 opacity <span class="val" id="a0Val">0.75</span></label>
    <input type="range" id="a0" min="0" max="1" step="0.01" value="0.75">
    <label>p50 opacity <span class="val" id="a1Val">0.54</span></label>
    <input type="range" id="a1" min="0" max="1" step="0.01" value="0.54">
    <label>p75 opacity <span class="val" id="a2Val">0.09</span></label>
    <input type="range" id="a2" min="0" max="1" step="0.01" value="0.09">
    <label>p100 opacity <span class="val" id="a3Val">0.09</span></label>
    <input type="range" id="a3" min="0" max="1" step="0.01" value="0.09">
  </div>

  <div class="section">
    <label>Point opacity <span class="val" id="ptVal">0.25</span></label>
    <input type="range" id="ptAlpha" min="0" max="1" step="0.01" value="0.25">
    <label style="margin-top:6px">
      <input type="checkbox" id="showPoints" checked> Show data points
    </label>
  </div>

  <div class="section">
    <label>Digit colours</label>
    <div id="colorPickers"></div>
  </div>

  <div class="section" style="color:#666; font-size:11px; line-height:1.5">
    Scroll to zoom · drag to pan<br>
    Hover cluster for info
  </div>
</div>

<div id="tooltip"></div>
<div id="viz"><svg id="svg"><g id="root"></g></svg></div>

<script>
const RAW_CLUSTERS = {clusters_json};
const RAW_POINTS   = {data_json};
const BASE_COLORS  = {colors_json};
const BASE_SCALE   = {scale:.4f};

// ── State ──────────────────────────────────────────────────────────────────────
let colors  = [...BASE_COLORS];
let alphas  = [0.75, 0.54, 0.09, 0.09];
let scaleMul = 1.0;
let ptAlpha  = 0.25;
let showPts  = true;

// ── Colour pickers ─────────────────────────────────────────────────────────────
const cpDiv = document.getElementById("colorPickers");
colors.forEach((c, i) => {{
  const row = document.createElement("div");
  row.className = "color-row";
  row.innerHTML = `<span>digit ${{i}}</span><input type="color" value="${{c}}" data-i="${{i}}">`;
  row.querySelector("input").addEventListener("input", e => {{
    colors[+e.target.dataset.i] = e.target.value;
    redraw();
  }});
  cpDiv.appendChild(row);
}});

// ── Sliders ────────────────────────────────────────────────────────────────────
function bindSlider(id, valId, setter, fmt) {{
  const el = document.getElementById(id);
  el.addEventListener("input", () => {{
    const v = parseFloat(el.value);
    document.getElementById(valId).textContent = fmt(v);
    setter(v);
    redraw();
  }});
}}
bindSlider("scaleSlider", "scaleVal", v => scaleMul = v, v => v.toFixed(2)+"×");
[0,1,2,3].forEach(i => bindSlider(`a${{i}}`, `a${{i}}Val`, v => alphas[i] = v, v => v.toFixed(2)));
bindSlider("ptAlpha", "ptVal", v => ptAlpha = v, v => v.toFixed(2));
document.getElementById("showPoints").addEventListener("change", e => {{
  showPts = e.target.checked;
  redraw();
}});

// ── Pan / zoom ─────────────────────────────────────────────────────────────────
const svg  = document.getElementById("svg");
const root = document.getElementById("root");
let tx=0, ty=0, zoom=1;
let dragging=false, startX, startY, startTx, startTy;

function applyTransform() {{
  root.setAttribute("transform", `translate(${{tx}},${{ty}}) scale(${{zoom}})`);
}}

svg.addEventListener("wheel", e => {{
  e.preventDefault();
  const rect = svg.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
  tx = mx - (mx - tx) * factor;
  ty = my - (my - ty) * factor;
  zoom *= factor;
  applyTransform();
}}, {{passive: false}});

svg.addEventListener("mousedown", e => {{
  dragging=true; startX=e.clientX; startY=e.clientY; startTx=tx; startTy=ty;
}});
window.addEventListener("mousemove", e => {{
  if (!dragging) return;
  tx = startTx + (e.clientX - startX);
  ty = startTy + (e.clientY - startY);
  applyTransform();
}});
window.addEventListener("mouseup", () => dragging=false);

// ── Tooltip ────────────────────────────────────────────────────────────────────
const tooltip = document.getElementById("tooltip");
function showTip(e, html) {{
  tooltip.innerHTML = html;
  tooltip.style.display = "block";
  tooltip.style.left = (e.clientX + 14) + "px";
  tooltip.style.top  = (e.clientY - 10) + "px";
}}
svg.addEventListener("mouseleave", () => tooltip.style.display = "none");

// ── Draw ───────────────────────────────────────────────────────────────────────
function hexToRgb(hex) {{
  const r = parseInt(hex.slice(1,3),16), g=parseInt(hex.slice(3,5),16), b=parseInt(hex.slice(5,7),16);
  return `${{r}},${{g}},${{b}}`;
}}

function redraw() {{
  const W = svg.clientWidth, H = svg.clientHeight;

  // Compute screen coords on first draw
  if (!redraw._init) {{
    const xs = RAW_CLUSTERS.map(c=>c.x).concat(RAW_POINTS.map(p=>p.x));
    const ys = RAW_CLUSTERS.map(c=>c.y).concat(RAW_POINTS.map(p=>p.y));
    const minX=Math.min(...xs), maxX=Math.max(...xs), minY=Math.min(...ys), maxY=Math.max(...ys);
    const pad = 20;
    const scaleX = (W - 2*pad) / (maxX - minX), scaleY = (H - 2*pad) / (maxY - minY);
    redraw._s = Math.min(scaleX, scaleY);
    redraw._ox = pad + ((W - 2*pad) - (maxX-minX)*redraw._s)/2 - minX*redraw._s;
    redraw._oy = pad + ((H - 2*pad) - (maxY-minY)*redraw._s)/2 - minY*redraw._s;
    tx = 0; ty = 0; zoom = 1;
    redraw._init = true;
    applyTransform();
  }}
  const s = redraw._s, ox = redraw._ox, oy = redraw._oy;
  const sc = BASE_SCALE * scaleMul;

  let html = "";

  // Data points
  if (showPts) {{
    RAW_POINTS.forEach(p => {{
      const x = p.x*s+ox, y = p.y*s+oy;
      const rgb = hexToRgb(colors[p.l]);
      html += `<circle cx="${{x.toFixed(1)}}" cy="${{y.toFixed(1)}}" r="1.5" fill="rgba(${{rgb}},${{ptAlpha}})" />`;
    }});
  }}

  // Cluster rings (p100→p25, largest first)
  RAW_CLUSTERS.forEach(c => {{
    const cx = c.x*s+ox, cy = c.y*s+oy;
    const rgb = hexToRgb(colors[c.label]);
    for (let i=3; i>=0; i--) {{
      const r = c.radii[i] * s * sc;
      html += `<circle cx="${{cx.toFixed(1)}}" cy="${{cy.toFixed(1)}}" r="${{r.toFixed(2)}}" `
            + `fill="rgba(${{rgb}},${{alphas[i]}})" data-id="${{c.id}}" class="cring" />`;
    }}
    html += `<circle cx="${{cx.toFixed(1)}}" cy="${{cy.toFixed(1)}}" r="3" fill="rgba(${{rgb}},0.95)" `
          + `stroke="white" stroke-width="0.5" data-id="${{c.id}}" class="cdot" />`;
  }});

  root.innerHTML = html;

  // Hover events on cluster elements
  root.querySelectorAll(".cring, .cdot").forEach(el => {{
    el.addEventListener("mouseenter", e => {{
      const c = RAW_CLUSTERS[+e.target.dataset.id];
      const bars = c.dist.map((v,i) => {{
        const w = Math.round(v * 0.9);
        const bold = i===c.label ? "font-weight:bold;color:#fff" : "";
        return `<div class="bar-row"><span class="lbl" style="${{bold}}">${{i}}</span>`
             + `<div class="bar" style="width:${{w}}px"></div>`
             + `<span class="pct" style="${{bold}}">${{v.toFixed(1)}}%</span></div>`;
      }}).join("");
      showTip(e,
        `<div style="text-align:center;color:#aaa;font-size:10px">cluster ${{c.id}}</div>`
        + `<img src="${{c.img}}">`
        + `<div style="margin-bottom:4px">digit <b>${{c.label}}</b> · purity <b>${{(c.purity*100).toFixed(1)}}%</b> · n=${{c.count}}</div>`
        + bars
      );
    }});
    el.addEventListener("mouseleave", () => tooltip.style.display="none");
    el.addEventListener("mousemove", e => {{
      tooltip.style.left = (e.clientX+14)+"px";
      tooltip.style.top  = (e.clientY-10)+"px";
    }});
  }});
}}

window.addEventListener("resize", () => {{ redraw._init=false; redraw(); }});
redraw();
</script>
</body>
</html>
"""

from datetime import datetime
date_dir = datetime.now().strftime("%y-%m-%d")
out_path = BASE / f"{EXP_NAME}_tsne_interactive.html"
out_path.write_text(html)
r2_key = f"seewhy/{date_dir}/{EXP_NAME}_tsne_interactive.html"
result = upload_media(r2_key, str(out_path), content_type="text/html")
print(result.get("url", out_path))
