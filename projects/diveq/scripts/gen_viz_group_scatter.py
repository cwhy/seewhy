"""
Interactive per-group scatter visualization for DiVeQ-PQ experiments.

For each PQ group, scatter the 2D group latent vectors colored by MNIST class.
Hover shows the actual digit image. Requires D_g=2.

Usage:
    uv run python projects/diveq/scripts/gen_viz_group_scatter.py [--exp exp35] [--n-per-class 100]

Reusable API:
    from projects.diveq.scripts.gen_viz_group_scatter import plot_group_scatter_html
    html_str = plot_group_scatter_html(z_all, X_raw, y, n_groups=16, group_dim=2, exp_id="exp35")
    url = save_html("diveq_group_scatter_exp35", html_str)
"""

import sys, argparse, pickle, base64, io, json, struct, zlib
import numpy as np
import jax, jax.numpy as jnp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared_lib.datasets import load_supervised_image
from shared_lib.html import save_html


# ── PNG encoder (no PIL dependency) ──────────────────────────────────────────

def _png_chunk(chunk_type, data):
    c = chunk_type + data
    return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)


def array_to_png_bytes(img_uint8):
    """Encode a 2D uint8 grayscale array as PNG bytes (no PIL)."""
    img = np.asarray(img_uint8, dtype=np.uint8)
    h, w = img.shape
    sig  = b"\x89PNG\r\n\x1a\n"
    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
    raw  = b"".join(b"\x00" + bytes(row.tolist()) for row in img)
    idat = _png_chunk(b"IDAT", zlib.compress(raw, 1))
    iend = _png_chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def digits_to_base64(X_raw, scale=3):
    """
    Convert N×784 uint8 array to list of base64 PNG data URLs.
    `scale`: nearest-neighbour upscale factor (3 → 84×84 px).
    """
    N = len(X_raw)
    urls = []
    for i in range(N):
        img = X_raw[i].reshape(28, 28).astype(np.uint8)
        if scale > 1:
            img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
        png = array_to_png_bytes(img)
        urls.append("data:image/png;base64," + base64.b64encode(png).decode())
        if (i + 1) % 200 == 0:
            print(f"    encoded {i+1}/{N} images")
    return urls


# ── Encoder (mirrors experiments35.py encode_eval) ───────────────────────────

@jax.jit
def _encode_raw_jit(params, x):
    h = jax.nn.gelu(x @ params["enc_W1"] + params["enc_b1"])
    h = jax.nn.gelu(h @ params["enc_W2"] + params["enc_b2"])
    return              h @ params["enc_W3"] + params["enc_b3"]


def encode_eval(params, bn_state, x_norm):
    """Encode normalised input [0,1]; use running-stats BN if bn_state provided."""
    z = np.array(_encode_raw_jit(params, jnp.array(x_norm)))
    if bn_state is not None:
        mean  = np.array(bn_state["mean"])
        var   = np.array(bn_state["var"])
        gamma = np.array(params["bn_gamma"])
        beta  = np.array(params["bn_beta"])
        z = (z - mean) / (np.sqrt(var) + 1e-6)
        z = z * gamma + beta
    return z


# ── Main reusable function ────────────────────────────────────────────────────

CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#a9a9a9",
]
CLASS_NAMES = [str(c) for c in range(10)]


def plot_group_scatter_html(
    z_all,
    X_raw,
    y,
    n_groups,
    group_dim,
    exp_id,
    n_cols=4,
    img_scale=3,
):
    """
    Build a full-page Plotly HTML with one scatter subplot per PQ group.

    Parameters
    ----------
    z_all     : (N, LATENT_DIM) float — encoded latent vectors (after BN if applicable)
    X_raw     : (N, 784)        uint8 — raw pixel values [0, 255]
    y         : (N,)            int   — class labels 0–9
    n_groups  : int  — number of PQ groups (G)
    group_dim : int  — must be 2 (D_g)
    exp_id    : str  — experiment label for the page title
    n_cols    : int  — subplot grid columns
    img_scale : int  — upscale factor for hover thumbnails (3 → 84×84)

    Returns
    -------
    html_str : str — complete self-contained HTML page (uses Plotly CDN)
    """
    if group_dim != 2:
        raise ValueError(f"plot_group_scatter_html requires group_dim=2, got {group_dim}")

    N = len(y)
    n_rows = (n_groups + n_cols - 1) // n_cols

    print(f"  Encoding {N} hover images (scale={img_scale})...")
    hover_imgs = digits_to_base64(X_raw.astype(np.uint8), scale=img_scale)
    img_px = 28 * img_scale

    subplot_titles = [f"Group {g}  (dims {g*2}–{g*2+1})" for g in range(n_groups)]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    for g in range(n_groups):
        zg   = z_all[:, g * 2 : (g + 1) * 2]
        row  = g // n_cols + 1
        col  = g % n_cols  + 1

        for cls in range(10):
            mask = (y == cls)
            if not mask.any():
                continue
            idx     = np.where(mask)[0]
            custom  = [[hover_imgs[i], cls] for i in idx]
            fig.add_trace(
                go.Scatter(
                    x=zg[mask, 0].tolist(),
                    y=zg[mask, 1].tolist(),
                    mode="markers",
                    name=f"digit {cls}",
                    legendgroup=str(cls),
                    showlegend=(g == 0),
                    marker=dict(color=CLASS_COLORS[cls], size=5, opacity=0.72),
                    customdata=custom,
                    hovertemplate=(
                        f"<img src='%{{customdata[0]}}' width='{img_px}' height='{img_px}'>"
                        "<br>class: <b>%{customdata[1]}</b>"
                        "<br>z=(%{x:.3f}, %{y:.3f})"
                        "<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )

    fig.update_layout(
        title=dict(
            text=f"{exp_id} — DiVeQ-PQ group scatter (D_g=2, {n_groups} groups, {N} samples)",
            font=dict(size=15),
        ),
        height=290 * n_rows + 100,
        width=310 * n_cols,
        template="plotly_white",
        legend=dict(
            title="Digit class",
            itemsizing="constant",
            tracegroupgap=2,
        ),
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=True)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload DiVeQ group scatter viz")
    parser.add_argument("--exp",          default="exp35",
                        help="Experiment ID, e.g. exp35 or exp21")
    parser.add_argument("--n-per-class",  type=int, default=100,
                        help="Samples per digit class (default 100)")
    parser.add_argument("--n-cols",       type=int, default=4,
                        help="Subplot columns (default 4)")
    parser.add_argument("--img-scale",    type=int, default=3,
                        help="Hover image upscale factor (default 3 → 84px)")
    args = parser.parse_args()

    exp_id  = args.exp
    PROJECT = Path(__file__).parent.parent

    # Load params
    params_path = PROJECT / f"params_{exp_id}.pkl"
    if not params_path.exists():
        print(f"ERROR: {params_path} not found — run the experiment first")
        sys.exit(1)
    with open(params_path, "rb") as f:
        params_np = pickle.load(f)
    params = {k: jnp.array(v) for k, v in params_np.items()}
    print(f"Loaded params: {sum(v.size for v in params_np.values()):,} parameters")

    # Load bn_state (optional)
    bn_state = None
    bn_path  = PROJECT / f"bn_state_{exp_id}.pkl"
    if bn_path.exists():
        with open(bn_path, "rb") as f:
            bn_np = pickle.load(f)
        bn_state = {k: np.array(v) for k, v in bn_np.items()}
        print(f"Loaded bn_state from {bn_path.name}")

    # Read experiment config from results.jsonl
    exp_cfg   = {}
    jsonl_path = PROJECT / "results.jsonl"
    for line in jsonl_path.read_text().strip().splitlines():
        try:
            r = json.loads(line)
            if r.get("experiment") == exp_id:
                exp_cfg = r
                break
        except Exception:
            pass

    n_groups  = exp_cfg.get("n_groups",  8)
    group_dim = exp_cfg.get("group_dim", 2)
    use_bn    = exp_cfg.get("bottleneck") == "batch_norm_running_stats"

    if group_dim != 2:
        print(f"ERROR: {exp_id} has group_dim={group_dim} — scatter requires D_g=2")
        sys.exit(1)

    print(f"Config: n_groups={n_groups}  group_dim={group_dim}  use_bn={use_bn}")

    # Load MNIST test set, sample balanced subset
    print("Loading MNIST test set...")
    data  = load_supervised_image("mnist")
    X_all = data.X_test.reshape(-1, 784).astype(np.float32)
    y_all = np.array(data.y_test)

    rng = np.random.default_rng(0)
    idx = []
    for cls in range(10):
        cls_idx = np.where(y_all == cls)[0]
        chosen  = rng.choice(cls_idx, min(args.n_per_class, len(cls_idx)), replace=False)
        idx.extend(chosen.tolist())
    idx = np.array(idx)
    rng.shuffle(idx)

    X_sub = X_all[idx]
    y_sub = y_all[idx]
    print(f"Sampled {len(y_sub)} examples ({args.n_per_class} per class)")

    # Encode in batches to avoid OOM
    print("Encoding latent vectors...")
    z_parts = []
    batch = 512
    for start in range(0, len(X_sub), batch):
        x_batch = X_sub[start : start + batch] / 255.0
        z_parts.append(encode_eval(params, bn_state if use_bn else None, x_batch))
    z_all_np = np.concatenate(z_parts, axis=0)

    # Build and upload
    print("Building Plotly figure...")
    html_str = plot_group_scatter_html(
        z_all     = z_all_np,
        X_raw     = X_sub.astype(np.uint8),
        y         = y_sub,
        n_groups  = n_groups,
        group_dim = group_dim,
        exp_id    = exp_id,
        n_cols    = args.n_cols,
        img_scale = args.img_scale,
    )

    name = f"diveq_group_scatter_{exp_id}"
    url  = save_html(name, html_str)
    print(f"\nScatter viz → {url}")


if __name__ == "__main__":
    main()
