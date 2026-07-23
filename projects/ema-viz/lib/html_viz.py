"""
html_viz.py — Reusable HTML visualization builders.

All functions are pure (no I/O, no logging) and return HTML strings.

Public API
----------
to_b64_png(arr)                    — grayscale [0,1] or uint8 → base64 PNG
to_b64_png_diverging(arr, vabs)    — float any range → base64 PNG, RdBu_r symmetric
img_tag(b64, size, title)          — <img> HTML tag
build_dim_recon_html(...)          — per-dimension reconstruction viewer
build_sample_decomp_html(...)      — per-sample additive decomposition viewer
"""

import base64
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Image helpers ──────────────────────────────────────────────────────────────

def to_b64_png(arr):
    """(H, W) float [0,1] or uint8 → base64 PNG string."""
    arr = np.array(arr, dtype=np.float32)
    if arr.max() > 1.1:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    buf = io.BytesIO()
    plt.imsave(buf, arr, cmap="gray", vmin=0.0, vmax=1.0, format="png")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


def to_b64_png_diverging(arr, vabs=None):
    """(H, W) float (any range) → base64 PNG with RdBu_r colormap, symmetric around 0."""
    arr = np.array(arr, dtype=np.float32)
    if vabs is None:
        vabs = max(float(np.abs(arr).max()), 1e-6)
    buf = io.BytesIO()
    plt.imsave(buf, arr, cmap="RdBu_r", vmin=-vabs, vmax=vabs, format="png")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


def to_b64_png_sequential(arr, vmax=None, vmin=0.0, cmap="hot"):
    """(H, W) float → base64 PNG with a sequential colormap.
    Default cmap='hot' (black→red→yellow→white). vmin/vmax set the shared scale.
    Pass the same vmax to multiple images to make them visually comparable/additive.
    """
    arr = np.array(arr, dtype=np.float32)
    if vmax is None:
        vmax = max(float(arr.max()), 1e-6)
    buf = io.BytesIO()
    plt.imsave(buf, arr, cmap=cmap, vmin=vmin, vmax=vmax, format="png")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


def img_tag(b64, size=84, title=""):
    return (f'<img src="data:image/png;base64,{b64}" '
            f'width="{size}" height="{size}" '
            f'style="image-rendering:pixelated;" '
            f'title="{title}">')


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0d1117; color: #e6edf3;
  font-family: ui-monospace, monospace; font-size: 12px;
  padding: 24px 32px;
}
h1 { font-size: 15px; color: #e6edf3; font-weight: normal; margin-bottom: 6px; }
.description { color: #8b949e; font-size: 11px; margin-bottom: 24px; line-height: 1.6; }
.section-title {
  color: #f0883e; font-size: 12px; font-weight: bold;
  margin-bottom: 10px; margin-top: 28px;
  border-bottom: 1px solid #21262d; padding-bottom: 4px;
}
.dim { margin-bottom: 28px; }
.dim-title {
  color: #58a6ff; font-size: 12px; margin-bottom: 4px;
  display: flex; align-items: center; gap: 12px;
}
.dim-title span { color: #8b949e; font-size: 11px; }
.dim-subtitle { color: #8b949e; font-size: 10px; margin-bottom: 6px; }
.row { display: flex; gap: 6px; align-items: flex-end; flex-wrap: wrap; }
.sample { display: flex; flex-direction: column; align-items: center; gap: 2px; }
.sample img { border: 1px solid #21262d; border-radius: 2px; }
.sample .lbl { color: #8b949e; font-size: 10px; }
.sep {
  width: 1px; height: 84px; background: #30363d;
  margin: 0 6px; align-self: center;
}
.basis-img img { border: 1px solid #58a6ff !important; }
.arrow { color: #8b949e; font-size: 10px; align-self: center; padding: 0 4px; }
"""


# ── Full-reconstruction section ───────────────────────────────────────────────

def _full_recon_section(samples):
    """samples: list of (orig_b64, recon_b64) pairs."""
    items = []
    for orig_b64, recon_b64 in samples:
        items.append(
            f'<div class="sample">'
            f'{img_tag(orig_b64, title="original")}'
            f'<div class="lbl">orig</div>'
            f'</div>'
            f'<div class="sample">'
            f'{img_tag(recon_b64, title="full reconstruction")}'
            f'<div class="lbl">recon</div>'
            f'</div>'
        )
    return (
        f'<div class="section-title">Full reconstruction — all dims active (sanity check)</div>'
        f'<div class="description">'
        f'Random test samples reconstructed using all {len(samples[0]) if samples else "all"} '
        f'embedding dimensions. Low MSE here means the decoder training succeeded before '
        f'inspecting individual dimensions below.'
        f'</div>'
        f'<div class="dim"><div class="row">{"".join(items)}</div></div>'
    )


# ── Per-dimension section ─────────────────────────────────────────────────────

def _dim_section(dim_idx, z_min, z_max, basis_b64, samples):
    """samples: list of (orig_b64, recon_b64, z_val), sorted low→high z."""
    items = []

    # Basis template (pure unit vector, blue border)
    items.append(
        f'<div class="sample basis-img">'
        f'{img_tag(basis_b64, title="sigmoid(1·exp(P[:,d]) + b) — unit activation template")}'
        f'<div class="lbl">template</div>'
        f'</div>'
        f'<div class="sep"></div>'
    )

    # Arrow label on left
    items.append(f'<div class="arrow">low z[{dim_idx}] →</div>')

    for orig_b64, recon_b64, z_val in samples:
        sign = "+" if z_val >= 0 else ""
        items.append(
            f'<div class="sample">'
            f'{img_tag(orig_b64, title="original input")}'
            f'<div class="lbl">orig</div>'
            f'</div>'
            f'<div class="sample">'
            f'{img_tag(recon_b64, title=f"z[{dim_idx}]={z_val:.2f} (dim {dim_idx} only active)")}'
            f'<div class="lbl">{sign}{z_val:.2f}</div>'
            f'</div>'
        )

    items.append(f'<div class="arrow">→ high</div>')

    return (
        f'<div class="dim">'
        f'<div class="dim-title">'
        f'dim {dim_idx}'
        f'<span>z range [{z_min:.2f}, {z_max:.2f}]</span>'
        f'</div>'
        f'<div class="dim-subtitle">'
        f'Template (blue): what this dimension encodes at unit activation (z=1, all others=0). '
        f'Pairs: original image | reconstruction using only dim {dim_idx} (all other dims zeroed). '
        f'Sorted left→right by z[{dim_idx}] value.'
        f'</div>'
        f'<div class="row">{"".join(items)}</div>'
        f'</div>'
    )


# ── Top-level builder ─────────────────────────────────────────────────────────

def build_dim_recon_html(title, description, full_recon_samples, sections):
    """
    Build a self-contained HTML page for per-dimension reconstruction analysis.

    Args:
        title:               Page/experiment title string.
        description:         One or two sentences describing the experiment.
        full_recon_samples:  list of (orig_b64, recon_b64) — random full reconstructions.
        sections:            list of (dim_idx, (z_min, z_max), basis_b64, samples)
                             where samples = list of (orig_b64, recon_b64, z_val)
                             sorted low→high by z_val.

    Returns:
        HTML string (self-contained, no external dependencies).
    """
    body_parts = [
        f'<h1>{title}</h1>',
        f'<div class="description">{description}</div>',
        _full_recon_section(full_recon_samples),
        f'<div class="section-title">Per-dimension reconstruction</div>',
        f'<div class="description">'
        f'Each section shows one embedding dimension in isolation. '
        f'The blue-bordered <em>template</em> is sigmoid(1·exp(P[:,d]) + b) — '
        f'what that dimension codes for at unit activation. '
        f'The paired images show the original input and its reconstruction using '
        f'<em>only that dimension</em> (all others zeroed), sorted by activation value left→right.'
        f'</div>',
    ]

    for dim_idx, (z_min, z_max), basis_b64, samples in sections:
        body_parts.append(_dim_section(dim_idx, z_min, z_max, basis_b64, samples))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>
{''.join(body_parts)}
</body>
</html>"""


# ── Sample decomposition builder ──────────────────────────────────────────────

_CSS_DECOMP = _CSS + """
.sample-block { margin-bottom: 36px; border-top: 1px solid #21262d; padding-top: 16px; }
.sample-header { color: #e6edf3; font-size: 12px; margin-bottom: 8px; }
.sample-header span { color: #8b949e; font-size: 11px; margin-left: 12px; }
.decomp-row { display: flex; gap: 4px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 4px; }
.contrib { display: flex; flex-direction: column; align-items: center; gap: 2px; }
.contrib img { border: 1px solid #21262d; border-radius: 2px; }
.contrib .lbl { color: #8b949e; font-size: 9px; }
.contrib.pos .lbl { color: #3fb950; }
.contrib.neg .lbl { color: #f85149; }
.contrib.special img { border-color: #58a6ff !important; }
.contrib.recon img { border-color: #f0883e !important; }
.legend { color: #8b949e; font-size: 10px; margin-bottom: 10px; line-height: 1.6; }
.colorbar { display: inline-block; width: 80px; height: 8px;
  background: linear-gradient(to right, #2166ac, #f7f7f7, #d6604d);
  border-radius: 2px; vertical-align: middle; margin: 0 4px; }
"""


def build_sample_decomp_html(title, samples, img_size=56):
    """
    Per-sample additive decomposition viewer.

    Each sample shows, left to right:
      original | bias | dim_0 | dim_1 | ... | dim_D-1 | logit sum | reconstruction

    All contributions are shown pre-sigmoid (logit space) with a shared diverging
    colormap (RdBu_r), so they literally add up to the logit sum.
    reconstruction = sigmoid(logit sum).

    Args:
        title:     Page title string.
        samples:   list of dicts with keys:
                     label      int or str — digit class
                     mse        float — per-sample MSE
                     orig_b64   grayscale base64 PNG
                     bias_b64   diverging base64 PNG  (bias contribution b)
                     contribs   list of (b64, z_val) — one per dim, pre-sigmoid, diverging
                     logit_b64  diverging base64 PNG  (sum of all contribs + bias)
                     recon_b64  grayscale base64 PNG  (sigmoid of logit)
                     vabs       float — symmetric colormap bound used for all diverging images
        img_size:  pixel size for displayed images (default 56).

    Returns:
        HTML string.
    """
    body_parts = [
        f'<h1>{title}</h1>',
        f'<div class="legend">'
        f'Each row = one test sample decomposed into additive contributions (pre-sigmoid / logit space).<br>'
        f'Contributions &amp; logit share the same <strong>[0, vmax]</strong> scale (hot colormap: black→red→white). '
        f'vmax shown in header — you can visually add the contribution images to get the logit.<br>'
        f'<strong>orig</strong> = input image &nbsp;|&nbsp; '
        f'<strong>dim N (z=…)</strong> = softplus(z[N]·s[N]+e[N]) · exp(P[:,N]+c[N]) &nbsp;|&nbsp; '
        f'<strong>logit</strong> = Σ dims &nbsp;|&nbsp; '
        f'<strong>recon</strong> = sigmoid(logit)'
        f'</div>',
    ]

    for s in samples:
        items = []

        # Original (grayscale)
        items.append(
            f'<div class="contrib special">'
            f'{img_tag(s["orig_b64"], size=img_size, title="original input")}'
            f'<div class="lbl">orig</div>'
            f'</div>'
        )

        # Per-dim contributions
        for d, (b64, z_val) in enumerate(s["contribs"]):
            sign = "+" if z_val >= 0 else ""
            cls = "pos" if z_val >= 0 else "neg"
            items.append(
                f'<div class="contrib {cls}">'
                f'{img_tag(b64, size=img_size, title=f"dim {d}: z={z_val:.2f} · exp(P[:,{d}])")}'
                f'<div class="lbl">d{d} {sign}{z_val:.2f}</div>'
                f'</div>'
            )

        # Logit sum
        items.append(
            f'<div class="contrib special">'
            f'{img_tag(s["logit_b64"], size=img_size, title="logit = bias + Σ dims (pre-sigmoid)")}'
            f'<div class="lbl">logit</div>'
            f'</div>'
        )

        # Reconstruction
        items.append(
            f'<div class="contrib recon">'
            f'{img_tag(s["recon_b64"], size=img_size, title="reconstruction = sigmoid(logit)")}'
            f'<div class="lbl">recon</div>'
            f'</div>'
        )

        body_parts.append(
            f'<div class="sample-block">'
            f'<div class="sample-header">'
            f'class {s["label"]}'
            f'<span>mse={s["mse"]:.4f}</span>'
            f'<span>vmax={s["vabs"]:.2f}</span>'
            f'</div>'
            f'<div class="decomp-row">{"".join(items)}</div>'
            f'</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{_CSS_DECOMP}</style>
</head>
<body>
{''.join(body_parts)}
</body>
</html>"""
