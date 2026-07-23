"""
Interactive HTML visualizations for component-analysis.

Builds self-contained HTML pages with embedded JSON data, pan/zoom,
and hover tooltips — same pattern as online-kmeans/gen_viz_tsne_interactive.py.

Functions
---------
save_projection_interactive(scores, labels, images, exp_name, tag, ...)
    2D scatter with pan/zoom, per-digit toggle, hover thumbnail.

save_components_interactive(components, exp_name, tag, ...)
    Grid of component images with hover magnification and metadata.
"""

import base64
import io
import json
import logging
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.html import save_html


# ── Helpers ───────────────────────────────────────────────────────────────────

_TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _img_to_b64(vec: np.ndarray, img_shape: tuple[int, int] = (28, 28)) -> str:
    """Encode a float/uint8 vector as a base64 data-URI PNG."""
    v = vec.copy().astype(np.float64)
    lo, hi = v.min(), v.max()
    v = (v - lo) / (hi - lo + 1e-9)
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(1.12, 1.12), dpi=50)
    ax.imshow(v.reshape(img_shape), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Interactive 2D scatter ────────────────────────────────────────────────────

def save_projection_interactive(
    scores: np.ndarray,
    labels: np.ndarray,
    exp_name: str,
    tag: str,
    images: np.ndarray | None = None,
    n_max: int = 8000,
    seed: int = 0,
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    title: str | None = None,
    component_images: np.ndarray | None = None,
    component_labels: list[str] | None = None,
) -> str:
    """
    Interactive 2D scatter with pan/zoom and hover thumbnails.

    Parameters
    ----------
    scores    : (N, ≥2) — projection scores; only columns 0 and 1 are used.
    labels    : (N,) int — digit labels 0-9.
    images    : (N, 784) optional — raw pixel values for hover thumbnails.
    n_max     : subsample to this many points if N > n_max.
    component_images : (K, 784) optional — component vectors to show in sidebar.
    component_labels : list of K strings — labels for each component image.

    Returns the URL of the uploaded HTML page.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=int)

    # Subsample
    if len(scores) > n_max:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(scores), n_max, replace=False)
        scores = scores[idx]
        labels = labels[idx]
        if images is not None:
            images = images[idx]

    # Encode images as base64 thumbnails (or null)
    logging.info(f"  Encoding {len(scores)} point thumbnails…")
    if images is not None:
        img_b64 = [_img_to_b64(images[i]) for i in range(len(scores))]
    else:
        img_b64 = [""] * len(scores)

    # Encode component images
    comp_b64 = []
    if component_images is not None:
        logging.info(f"  Encoding {len(component_images)} component images…")
        comp_b64 = [_img_to_b64(component_images[i]) for i in range(len(component_images))]

    # Build JSON data
    points_js = json.dumps([
        {"x": round(float(scores[i, 0]), 3),
         "y": round(float(scores[i, 1]), 3),
         "l": int(labels[i]),
         "img": img_b64[i]}
        for i in range(len(scores))
    ])
    colors_js = json.dumps(_TAB10)
    comp_js = json.dumps([
        {"img": comp_b64[i],
         "label": component_labels[i] if component_labels else f"Component {i+1}"}
        for i in range(len(comp_b64))
    ])
    title_str = title or f"{exp_name} — {tag}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title_str}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ display: flex; height: 100vh; font-family: monospace; font-size: 12px;
       background: #1a1a1a; color: #ccc; }}
#sidebar {{ width: 220px; flex-shrink: 0; padding: 12px; overflow-y: auto;
           background: #222; border-right: 1px solid #333; }}
#sidebar h2 {{ font-size: 13px; margin-bottom: 10px; color: #eee; }}
.section {{ margin-bottom: 14px; }}
.section label {{ display: block; margin-bottom: 3px; color: #aaa; }}
.digit-row {{ display: flex; align-items: center; gap: 6px; margin-bottom: 3px;
              cursor: pointer; user-select: none; }}
.digit-row .swatch {{ width: 14px; height: 14px; border-radius: 50%;
                      border: 2px solid transparent; }}
.digit-row.hidden .swatch {{ opacity: 0.2; }}
.digit-row.hidden span {{ color: #555; }}
.comp-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 4px; }}
.comp-item {{ text-align: center; }}
.comp-item img {{ width: 100%; image-rendering: pixelated;
                  border: 1px solid #444; cursor: pointer; }}
.comp-item img:hover {{ border-color: #888; }}
.comp-item span {{ font-size: 9px; color: #777; display: block; }}
#tooltip {{ position: fixed; background: rgba(20,20,20,0.95); color: #fff;
            padding: 8px 10px; border-radius: 5px; font-size: 11px;
            pointer-events: none; display: none; z-index: 99;
            min-width: 120px; border: 1px solid #444; }}
#tooltip img {{ display: block; width: 84px; height: 84px;
                image-rendering: pixelated; margin: 4px auto;
                border: 1px solid #555; }}
#viz {{ flex: 1; overflow: hidden; cursor: grab; }}
#viz:active {{ cursor: grabbing; }}
svg {{ width: 100%; height: 100%; }}
.axis-label {{ font-size: 11px; fill: #888; }}
</style>
</head>
<body>

<div id="sidebar">
  <h2>{title_str}</h2>

  <div class="section">
    <label style="margin-bottom:6px">Digits</label>
    <div id="digitToggles"></div>
    <div style="margin-top:6px;display:flex;gap:8px">
      <button onclick="setAll(true)"
        style="font-size:10px;background:#333;border:1px solid #555;color:#aaa;padding:2px 6px;cursor:pointer">all</button>
      <button onclick="setAll(false)"
        style="font-size:10px;background:#333;border:1px solid #555;color:#aaa;padding:2px 6px;cursor:pointer">none</button>
    </div>
  </div>

  <div class="section">
    <label>Point opacity <span id="ptVal">0.4</span></label>
    <input type="range" id="ptAlpha" min="0.05" max="1" step="0.05" value="0.4"
      style="width:100%" oninput="ptAlpha=+this.value;document.getElementById('ptVal').textContent=this.value;redraw()">
    <label style="margin-top:6px">Point size <span id="ptSzVal">2</span></label>
    <input type="range" id="ptSize" min="1" max="8" step="0.5" value="2"
      style="width:100%" oninput="ptSize=+this.value;document.getElementById('ptSzVal').textContent=this.value;redraw()">
  </div>

  <div id="compSection" class="section" style="display:none">
    <label>Components</label>
    <div class="comp-grid" id="compGrid"></div>
  </div>

  <div class="section" style="color:#555;font-size:10px;line-height:1.5">
    Scroll to zoom · drag to pan<br>Hover for thumbnail
  </div>
</div>

<div id="tooltip"></div>
<div id="viz"><svg id="svg"><g id="root"></g></svg></div>

<script>
const POINTS  = {points_js};
const COLORS  = {colors_js};
const COMPS   = {comp_js};

let ptAlpha = 0.4, ptSize = 2;
let visible = Array(10).fill(true);

// ── Digit toggles ──────────────────────────────────────────────────────────────
const togDiv = document.getElementById("digitToggles");
COLORS.forEach((c, i) => {{
  const row = document.createElement("div");
  row.className = "digit-row";
  row.dataset.d = i;
  row.innerHTML = `<div class="swatch" style="background:${{c}}"></div><span>digit ${{i}}</span>`;
  row.addEventListener("click", () => {{
    visible[i] = !visible[i];
    row.classList.toggle("hidden", !visible[i]);
    redraw();
  }});
  togDiv.appendChild(row);
}});

function setAll(v) {{
  visible.fill(v);
  document.querySelectorAll(".digit-row").forEach(r =>
    r.classList.toggle("hidden", !v));
  redraw();
}}

// ── Component images ──────────────────────────────────────────────────────────
if (COMPS.length > 0) {{
  document.getElementById("compSection").style.display = "block";
  const grid = document.getElementById("compGrid");
  COMPS.forEach((c, i) => {{
    const div = document.createElement("div");
    div.className = "comp-item";
    div.innerHTML = `<img src="${{c.img}}" title="${{c.label}}"><span>${{c.label}}</span>`;
    grid.appendChild(div);
  }});
}}

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
  const f = e.deltaY < 0 ? 1.15 : 1/1.15;
  tx = mx - (mx - tx)*f; ty = my - (my - ty)*f; zoom *= f;
  applyTransform();
}}, {{passive: false}});
svg.addEventListener("mousedown", e => {{
  dragging=true; startX=e.clientX; startY=e.clientY; startTx=tx; startTy=ty;
}});
window.addEventListener("mousemove", e => {{
  if (!dragging) return;
  tx = startTx+(e.clientX-startX); ty = startTy+(e.clientY-startY);
  applyTransform();
}});
window.addEventListener("mouseup", () => dragging=false);

// ── Tooltip ────────────────────────────────────────────────────────────────────
const tip = document.getElementById("tooltip");
svg.addEventListener("mouseleave", () => tip.style.display="none");

// ── Draw ───────────────────────────────────────────────────────────────────────
function hexToRgb(hex) {{
  const r=parseInt(hex.slice(1,3),16), g=parseInt(hex.slice(3,5),16), b=parseInt(hex.slice(5,7),16);
  return `${{r}},${{g}},${{b}}`;
}}

let _s, _ox, _oy, _init=false;

function redraw() {{
  const W = svg.clientWidth, H = svg.clientHeight;
  if (!_init) {{
    const xs=POINTS.map(p=>p.x), ys=POINTS.map(p=>p.y);
    const minX=Math.min(...xs), maxX=Math.max(...xs);
    const minY=Math.min(...ys), maxY=Math.max(...ys);
    const pad=40;
    const sx=(W-2*pad)/(maxX-minX), sy=(H-2*pad)/(maxY-minY);
    _s=Math.min(sx,sy);
    _ox=pad+((W-2*pad)-(maxX-minX)*_s)/2-minX*_s;
    _oy=pad+((H-2*pad)-(maxY-minY)*_s)/2-minY*_s;
    tx=0; ty=0; zoom=1; applyTransform();
    _init=true;
  }}

  // Build axis labels (static, outside root transform)
  const axId="axlabels";
  let axEl=document.getElementById(axId);
  if (!axEl) {{
    axEl=document.createElementNS("http://www.w3.org/2000/svg","g");
    axEl.id=axId; svg.appendChild(axEl);
  }}
  axEl.innerHTML=`
    <text x="${{W/2}}" y="${{H-6}}" text-anchor="middle" class="axis-label">{xlabel}</text>
    <text x="14" y="${{H/2}}" text-anchor="middle" class="axis-label"
          transform="rotate(-90,14,${{H/2}})">{ylabel}</text>`;

  let html="";
  POINTS.forEach((p,i) => {{
    if (!visible[p.l]) return;
    const x=p.x*_s+_ox, y=p.y*_s+_oy;
    const rgb=hexToRgb(COLORS[p.l]);
    html+=`<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="${{ptSize}}"
      fill="rgba(${{rgb}},${{ptAlpha}})" data-i="${{i}}" class="pt"/>`;
  }});
  root.innerHTML=html;

  root.querySelectorAll(".pt").forEach(el => {{
    el.addEventListener("mouseenter", e => {{
      const p=POINTS[+e.target.dataset.i];
      const imgHtml=p.img ? `<img src="${{p.img}}">` : "";
      tip.innerHTML=imgHtml+`<div>digit <b>${{p.l}}</b></div>`
        +`<div style="color:#666;font-size:10px">(${{{{'x':p.x.toFixed(2),'y':p.y.toFixed(2)}}}}.x, .y)</div>`;
      tip.style.display="block";
    }});
    el.addEventListener("mousemove", e => {{
      tip.style.left=(e.clientX+14)+"px"; tip.style.top=(e.clientY-10)+"px";
    }});
    el.addEventListener("mouseleave", () => tip.style.display="none");
  }});
}}

window.addEventListener("resize", () => {{ _init=false; redraw(); }});
redraw();
</script>
</body>
</html>
"""
    url = save_html(f"{exp_name}_{tag}_interactive", html)
    logging.info(f"  interactive scatter → {url}")
    return url


# ── Interactive component grid ─────────────────────────────────────────────────

def save_components_interactive(
    components: np.ndarray,
    exp_name: str,
    tag: str,
    labels: list[str] | None = None,
    img_shape: tuple[int, int] = (28, 28),
    title: str | None = None,
    metadata: list[dict] | None = None,
) -> str:
    """
    Interactive grid of component images with hover magnification.

    components : (K, D) component vectors (e.g. PCA eigenvectors or MCCA pixel directions).
    labels     : list of K strings (default: "PC 1", "PC 2", …).
    metadata   : list of K dicts with extra info shown in tooltip.

    Returns the URL of the uploaded HTML page.
    """
    K = len(components)
    if labels is None:
        labels = [f"Component {i+1}" for i in range(K)]

    logging.info(f"  Encoding {K} component images…")
    imgs_b64 = [_img_to_b64(components[i], img_shape) for i in range(K)]

    items_js = json.dumps([
        {"img": imgs_b64[i],
         "label": labels[i],
         "meta": metadata[i] if metadata else {}}
        for i in range(K)
    ])
    title_str = title or f"{exp_name} — {tag} components"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title_str}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: monospace; font-size: 12px; background: #1a1a1a; color: #ccc; padding: 20px; }}
h1 {{ font-size: 15px; margin-bottom: 16px; color: #eee; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(90px, 1fr)); gap: 8px; }}
.item {{ text-align: center; cursor: pointer; }}
.item img {{ width: 100%; image-rendering: pixelated; border: 1px solid #333;
             transition: transform 0.1s; }}
.item:hover img {{ transform: scale(1.4); border-color: #888; z-index: 10;
                   position: relative; }}
.item span {{ font-size: 9px; color: #666; display: block; margin-top: 2px; }}
#overlay {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,0.7);
            z-index:100; align-items:center; justify-content:center; }}
#overlay.active {{ display:flex; }}
#overlay-inner {{ background:#222; border:1px solid #555; border-radius:6px;
                  padding:20px; max-width:380px; text-align:center; }}
#overlay-inner img {{ width:280px; height:280px; image-rendering:pixelated;
                      border:1px solid #555; display:block; margin:0 auto 12px; }}
#overlay-inner h2 {{ font-size:14px; margin-bottom:8px; color:#eee; }}
#overlay-inner pre {{ font-size:10px; color:#888; text-align:left;
                      background:#111; padding:8px; border-radius:4px;
                      overflow-x:auto; }}
#overlay-close {{ margin-top:12px; font-size:11px; color:#4a9eff; cursor:pointer; }}
</style>
</head>
<body>
<h1>{title_str}</h1>
<div class="grid" id="grid"></div>

<div id="overlay">
  <div id="overlay-inner">
    <img id="ov-img" src="">
    <h2 id="ov-label"></h2>
    <pre id="ov-meta"></pre>
    <div id="overlay-close" onclick="closeOverlay()">✕ close</div>
  </div>
</div>

<script>
const ITEMS = {items_js};

const grid = document.getElementById("grid");
ITEMS.forEach((item, i) => {{
  const div = document.createElement("div");
  div.className = "item";
  div.innerHTML = `<img src="${{item.img}}" alt="${{item.label}}">
                   <span>${{item.label}}</span>`;
  div.addEventListener("click", () => openOverlay(item));
  grid.appendChild(div);
}});

function openOverlay(item) {{
  document.getElementById("ov-img").src = item.img;
  document.getElementById("ov-label").textContent = item.label;
  const meta = Object.keys(item.meta).length
    ? JSON.stringify(item.meta, null, 2)
    : "(no metadata)";
  document.getElementById("ov-meta").textContent = meta;
  document.getElementById("overlay").classList.add("active");
}}
function closeOverlay() {{
  document.getElementById("overlay").classList.remove("active");
}}
document.getElementById("overlay").addEventListener("click", e => {{
  if (e.target === document.getElementById("overlay")) closeOverlay();
}});
</script>
</body>
</html>
"""
    url = save_html(f"{exp_name}_{tag}_components_interactive", html)
    logging.info(f"  interactive components → {url}")
    return url
