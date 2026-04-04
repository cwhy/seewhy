"""
eval_graphs.py — MSE / linear-probe / KNN-1 / KNN-5 graphs across all encoders.

For each metric, produces a line plot (latent_dim on x-axis) comparing all encoder types.
Also runs pixel-space baseline (784-dim, no encoding).

Encoders:
  gram-EMA       — Hebbian gram + Hadamard, EMA, with augmentation
  gram-EMA-noaug — same, no augmentation
  SIGReg         — same architecture, SIGReg collapse prevention
  conv-EMA       — 3-layer CNN + GAP + Linear, EMA
  conv2-EMA      — 3-layer CNN + GAP + FC + Linear, EMA
  random         — frozen random linear projection
  pixel          — raw pixels / 255 (784-dim, single point)

Usage:
    uv run python projects/ema-viz/scripts/tmp/eval_graphs.py
"""
import pickle, sys, io, base64
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np, optax, faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_lib.datasets import load_supervised_image
from shared_lib.media import save_media

SSL_DIR = Path(__file__).parent.parent.parent.parent / "ssl"

# ── Encoder registry ──────────────────────────────────────────────────────────
# (label, pkl_name, arch)  — arch: "gram" | "conv" | "conv2"

ENCODER_REGISTRY = [
    ("gram-EMA",        "params_exp_ema_512.pkl",          "gram"),
    ("gram-EMA",        "params_exp_ema_1024.pkl",         "gram"),
    ("gram-EMA",        "params_exp_ema_2048.pkl",         "gram"),
    ("gram-EMA",        "params_exp_ema.pkl",              "gram"),  # 4096
    ("gram-EMA-noaug",  "params_exp_ema_noaug_512.pkl",   "gram"),
    ("gram-EMA-noaug",  "params_exp_ema_noaug_1024.pkl",  "gram"),
    ("gram-EMA-noaug",  "params_exp_ema_noaug_2048.pkl",  "gram"),
    ("gram-EMA-noaug",  "params_exp_ema_noaug_4096.pkl",  "gram"),
    ("SIGReg",          "params_exp16b.pkl",               "gram"),  # 2048
    ("SIGReg",          "params_exp16a.pkl",               "gram"),  # 4096
    ("conv-EMA",        "params_exp_conv_ema_512.pkl",    "conv"),
    ("conv-EMA",        "params_exp_conv_ema_1024.pkl",   "conv"),
    ("conv-EMA",        "params_exp_conv_ema_2048.pkl",   "conv"),
    ("conv-EMA",        "params_exp_conv_ema_4096.pkl",   "conv"),
    ("conv2-EMA",       "params_exp_conv2_ema_512.pkl",   "conv2"),
    ("conv2-EMA",       "params_exp_conv2_ema_1024.pkl",  "conv2"),
    ("conv2-EMA",       "params_exp_conv2_ema_2048.pkl",  "conv2"),
    ("conv2-EMA",       "params_exp_conv2_ema_4096.pkl",  "conv2"),
]

RANDOM_DIMS = [512, 1024, 2048, 4096]
KS = [1, 5, 10, 20]

# ── Pixel positions for gram encoder ──────────────────────────────────────────

_ROWS = jnp.array(np.arange(784) // 28)
_COLS = jnp.array(np.arange(784) % 28)

# ── Conv helpers ──────────────────────────────────────────────────────────────

_DN = ("NHWC", "HWIO", "NHWC")

def _conv2d(x, W, b):
    return jax.lax.conv_general_dilated(x, W, (1,1), "SAME", dimension_numbers=_DN) + b

def _maxpool(x):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1,2,2,1), (1,2,2,1), "VALID")

# ── Encode functions ──────────────────────────────────────────────────────────

@jax.jit
def _encode_gram(p, x):
    x_norm   = x / 255.0
    re       = p["enc_row_emb"][_ROWS]
    ce       = p["enc_col_emb"][_COLS]
    D_R      = p["enc_row_emb"].shape[1]
    D_C      = p["enc_col_emb"].shape[1]
    pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_R * D_C))(re, ce)
    pos_feat = pos_rc @ p["W_pos_enc"]
    f        = pos_feat @ p["W_enc"]
    F        = x_norm[:, :, None] * f[None, :, :]
    DEC_RANK = p["W_enc"].shape[1]
    H        = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK ** 2)
    return (H @ p["W_enc_A"]) * (H @ p["W_enc_B"])

@jax.jit
def _encode_conv(p, x):
    h = (x / 255.0).reshape(-1, 28, 28, 1)
    h = jax.nn.gelu(_conv2d(h, p["W_conv1"], p["b_conv1"]))
    h = _maxpool(h)
    h = jax.nn.gelu(_conv2d(h, p["W_conv2"], p["b_conv2"]))
    h = _maxpool(h)
    h = jax.nn.gelu(_conv2d(h, p["W_conv3"], p["b_conv3"]))
    h = h.mean(axis=(1, 2))
    return h @ p["W_fc"] + p["b_fc"]

@jax.jit
def _encode_conv2(p, x):
    h = (x / 255.0).reshape(-1, 28, 28, 1)
    h = jax.nn.gelu(_conv2d(h, p["W_conv1"], p["b_conv1"]))
    h = _maxpool(h)
    h = jax.nn.gelu(_conv2d(h, p["W_conv2"], p["b_conv2"]))
    h = _maxpool(h)
    h = jax.nn.gelu(_conv2d(h, p["W_conv3"], p["b_conv3"]))
    h = h.mean(axis=(1, 2))
    h = jax.nn.gelu(h @ p["W_fc1"] + p["b_fc1"])
    return h @ p["W_fc2"] + p["b_fc2"]

_ENCODE_FNS = {"gram": _encode_gram, "conv": _encode_conv, "conv2": _encode_conv2}

def encode_all(p, arch, X, bs=256):
    fn = _ENCODE_FNS[arch]
    return np.concatenate([
        np.array(fn(p, jnp.array(X[i:i+bs].astype(np.float32))))
        for i in range(0, len(X), bs)
    ])

def encode_random(X, dim, seed=1):
    W = np.random.default_rng(seed).standard_normal((784, dim)).astype(np.float32) / np.sqrt(784)
    return (X / 255.0) @ W

def latent_dim_from_params(p, arch):
    if arch == "gram":
        return int(p["W_enc_A"].shape[1])
    elif arch == "conv":
        return int(p["W_fc"].shape[1])
    else:  # conv2
        return int(p["W_fc2"].shape[1])

# ── Decoder (MSE) ─────────────────────────────────────────────────────────────

def eval_mse(Z_tr, X_tr, Z_te, X_te, latent_dim, epochs=200, lr=3e-3, batch=512, seed=42):
    z_std = Z_tr.std(axis=0) + 1e-8
    Z_tr  = Z_tr / z_std;  Z_te = Z_te / z_std
    k1, _ = jax.random.split(jax.random.PRNGKey(seed))
    params = {
        "P": jax.random.normal(k1, (784, latent_dim)) * latent_dim ** -0.5,
        "c": jnp.full(latent_dim, -float(np.log(latent_dim))),
        "s": jnp.ones(latent_dim), "e": jnp.zeros(latent_dim), "b": jnp.zeros(()),
    }
    opt = optax.adam(lr);  state = opt.init(params)
    n = (len(Z_tr) // batch) * batch
    Zb = jnp.array(Z_tr[:n].reshape(-1, batch, latent_dim))
    Xb = jnp.array(X_tr[:n].reshape(-1, batch, 784) / 255.0)

    @jax.jit
    def step(params, state, z, x):
        def loss(p):
            z_pos = jax.nn.softplus(z * p["s"][None, :] + p["e"][None, :])
            P_pos = jnp.exp(jnp.clip(p["P"] + p["c"][None, :], -10.0, 10.0))
            xh = jax.nn.sigmoid(z_pos @ P_pos.T + p["b"])
            return -(x * jnp.log(xh + 1e-7) + (1-x) * jnp.log(1-xh + 1e-7)).mean()
        l, g = jax.value_and_grad(loss)(params)
        u, s = opt.update(g, state, params)
        return optax.apply_updates(params, u), s, l

    for _ in range(epochs):
        for bi in range(len(Zb)):
            params, state, _ = step(params, state, Zb[bi], Xb[bi])

    z_pos = jax.nn.softplus(jnp.array(Z_te) * params["s"][None, :] + params["e"][None, :])
    P_pos = jnp.exp(jnp.clip(params["P"] + params["c"][None, :], -10.0, 10.0))
    xh = np.array(jax.nn.sigmoid(z_pos @ P_pos.T + params["b"]))
    return float(np.mean((xh - X_te / 255.0) ** 2))

# ── Linear probe ──────────────────────────────────────────────────────────────

def eval_probe(Z_tr, Y_tr, Z_te, Y_te, epochs=200, lr=1e-3, batch=512, seed=0):
    dim = Z_tr.shape[1]
    Z_tr = Z_tr / (np.linalg.norm(Z_tr, axis=1, keepdims=True) + 1e-8)
    Z_te = Z_te / (np.linalg.norm(Z_te, axis=1, keepdims=True) + 1e-8)
    params = {"W": jax.random.normal(jax.random.PRNGKey(seed), (dim, 10)) * 0.01,
              "b": jnp.zeros(10)}
    opt = optax.adam(lr);  state = opt.init(params)
    n = (len(Z_tr) // batch) * batch
    idx = np.random.default_rng(seed).permutation(len(Z_tr))[:n]
    Zb = jnp.array(Z_tr[idx].reshape(-1, batch, dim))
    Yb = jnp.array(Y_tr[idx].reshape(-1, batch))

    @jax.jit
    def step(params, state, z, y):
        def loss(p):
            return optax.softmax_cross_entropy_with_integer_labels(z @ p["W"] + p["b"], y).mean()
        l, g = jax.value_and_grad(loss)(params)
        u, s = opt.update(g, state, params)
        return optax.apply_updates(params, u), s, l

    for _ in range(epochs):
        for bi in range(len(Zb)):
            params, state, _ = step(params, state, Zb[bi], Yb[bi])

    logits = jnp.array(Z_te) @ params["W"] + params["b"]
    preds  = np.array(jnp.argmax(logits, axis=1))
    return float((preds == Y_te).mean())

# ── KNN ───────────────────────────────────────────────────────────────────────

def eval_knn(Z_tr, Y_tr, Z_te, Y_te):
    Z_tr = (Z_tr / (np.linalg.norm(Z_tr, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    Z_te = (Z_te / (np.linalg.norm(Z_te, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    index = faiss.IndexFlatL2(Z_tr.shape[1])
    index.add(Z_tr)
    _, I = index.search(Z_te, max(KS))
    return {
        k: float((np.array([np.bincount(Y_tr[I[i, :k]], minlength=10).argmax()
                             for i in range(len(Z_te))]) == Y_te).mean())
        for k in KS
    }

# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "gram-EMA":       "#58a6ff",
    "gram-EMA-noaug": "#79c0ff",
    "SIGReg":         "#f0883e",
    "conv-EMA":       "#3fb950",
    "conv2-EMA":      "#56d364",
    "random":         "#8b949e",
    "pixel":          "#d2a8ff",
}
MARKERS = {
    "gram-EMA":       "o",
    "gram-EMA-noaug": "s",
    "SIGReg":         "D",
    "conv-EMA":       "^",
    "conv2-EMA":      "v",
    "random":         "x",
    "pixel":          "*",
}

def make_plot(title, ylabel, data, higher_is_better=True, pixel_val=None, pixel_dim=784):
    """data: dict[label] = list of (dim, value)"""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.title.set_color("#e6edf3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.6)

    for label, pts in sorted(data.items()):
        if not pts:
            continue
        pts = sorted(pts)
        xs, ys = zip(*pts)
        c = COLORS.get(label, "#ffffff")
        m = MARKERS.get(label, "o")
        ax.plot(xs, ys, color=c, marker=m, linewidth=1.5, markersize=6, label=label)

    # Pixel baseline as horizontal dashed line
    if pixel_val is not None:
        ax.axhline(pixel_val, color=COLORS["pixel"], linewidth=1.2, linestyle="--",
                   label=f"pixel ({pixel_dim}d)")

    ax.set_xlabel("latent dim", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#e6edf3", loc="best")
    ax.set_xscale("log", base=2)
    xt = sorted({d for pts in data.values() for d, _ in pts})
    ax.set_xticks(xt)
    ax.set_xticklabels([str(x) for x in xt])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

def build_html(plots):
    """plots: list of (title, b64_png)"""
    _CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d1117; color: #e6edf3; font-family: ui-monospace, monospace;
       font-size: 13px; padding: 32px 40px; }
h1 { font-size: 15px; font-weight: normal; margin-bottom: 24px; }
.plot { margin-bottom: 40px; }
.plot img { max-width: 100%; border: 1px solid #21262d; border-radius: 4px; }
"""
    imgs = "".join(
        f'<div class="plot"><img src="data:image/png;base64,{b64}" alt="{title}"></div>'
        for title, b64 in plots
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Encoder evaluation graphs</title>
<style>{_CSS}</style></head>
<body>
<h1>MNIST encoder evaluation: MSE / linear probe / KNN across latent dims</h1>
{imgs}
</body></html>"""

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_supervised_image("mnist")
    X_tr = data.X.reshape(data.n_samples, 784).astype(np.float32)
    Y_tr = np.array(data.y, dtype=np.int32)
    X_te = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)
    Y_te = np.array(data.y_test, dtype=np.int32)

    # Results: label → list of (dim, value)
    mse_data   = {lbl: [] for lbl, _, _ in ENCODER_REGISTRY}
    probe_data = {lbl: [] for lbl, _, _ in ENCODER_REGISTRY}
    knn1_data  = {lbl: [] for lbl, _, _ in ENCODER_REGISTRY}
    knn5_data  = {lbl: [] for lbl, _, _ in ENCODER_REGISTRY}
    for d in [mse_data, probe_data, knn1_data, knn5_data]:
        d["random"] = []

    # ── SSL encoders ──────────────────────────────────────────────────────────
    for label, pkl_name, arch in ENCODER_REGISTRY:
        path = SSL_DIR / pkl_name
        if not path.exists():
            print(f"  SKIP {pkl_name} (not found)")
            continue
        raw = pickle.load(open(path, "rb"))
        p   = {k: jnp.array(v) for k, v in raw.items()}
        dim = latent_dim_from_params(p, arch)
        print(f"\n{label}  dim={dim}  ({pkl_name})")

        Z_tr = encode_all(p, arch, X_tr)
        Z_te = encode_all(p, arch, X_te)

        mse   = eval_mse(Z_tr, X_tr, Z_te, X_te, dim)
        probe = eval_probe(Z_tr, Y_tr, Z_te, Y_te)
        knns  = eval_knn(Z_tr, Y_tr, Z_te, Y_te)
        print(f"  mse={mse:.4f}  probe={probe*100:.2f}%  knn1={knns[1]*100:.2f}%  knn5={knns[5]*100:.2f}%")

        mse_data[label].append((dim, mse))
        probe_data[label].append((dim, probe * 100))
        knn1_data[label].append((dim, knns[1] * 100))
        knn5_data[label].append((dim, knns[5] * 100))

    # ── Random baseline ───────────────────────────────────────────────────────
    for dim in RANDOM_DIMS:
        print(f"\nrandom  dim={dim}")
        Z_tr_r = encode_random(X_tr, dim)
        Z_te_r = encode_random(X_te, dim)
        mse   = eval_mse(Z_tr_r, X_tr, Z_te_r, X_te, dim)
        probe = eval_probe(Z_tr_r, Y_tr, Z_te_r, Y_te)
        knns  = eval_knn(Z_tr_r, Y_tr, Z_te_r, Y_te)
        print(f"  mse={mse:.4f}  probe={probe*100:.2f}%  knn1={knns[1]*100:.2f}%  knn5={knns[5]*100:.2f}%")
        mse_data["random"].append((dim, mse))
        probe_data["random"].append((dim, probe * 100))
        knn1_data["random"].append((dim, knns[1] * 100))
        knn5_data["random"].append((dim, knns[5] * 100))

    # ── Pixel space baseline ──────────────────────────────────────────────────
    print("\npixel space  dim=784")
    Z_px_tr = (X_tr / 255.0).astype(np.float32)
    Z_px_te = (X_te / 255.0).astype(np.float32)
    px_mse   = eval_mse(Z_px_tr, X_tr, Z_px_te, X_te, 784)
    px_probe = eval_probe(Z_px_tr, Y_tr, Z_px_te, Y_te) * 100
    px_knns  = eval_knn(Z_px_tr, Y_tr, Z_px_te, Y_te)
    print(f"  mse={px_mse:.4f}  probe={px_probe:.2f}%  knn1={px_knns[1]*100:.2f}%  knn5={px_knns[5]*100:.2f}%")

    # ── Build plots ───────────────────────────────────────────────────────────
    # Remove empty series
    mse_data   = {k: v for k, v in mse_data.items()   if v}
    probe_data = {k: v for k, v in probe_data.items() if v}
    knn1_data  = {k: v for k, v in knn1_data.items()  if v}
    knn5_data  = {k: v for k, v in knn5_data.items()  if v}

    plots = [
        ("Reconstruction MSE (linear autodecoder, lower is better)",
         make_plot("Reconstruction MSE — lower is better", "MSE",
                   mse_data, higher_is_better=False, pixel_val=px_mse)),
        ("Linear probe accuracy (%, higher is better)",
         make_plot("Linear probe accuracy — higher is better", "accuracy (%)",
                   probe_data, pixel_val=px_probe)),
        ("KNN-1 accuracy (%, higher is better)",
         make_plot("KNN k=1 accuracy — higher is better", "accuracy (%)",
                   knn1_data, pixel_val=px_knns[1]*100)),
        ("KNN-5 accuracy (%, higher is better)",
         make_plot("KNN k=5 accuracy — higher is better", "accuracy (%)",
                   knn5_data, pixel_val=px_knns[5]*100)),
    ]

    html = build_html([(t, b) for t, b in plots])
    url  = save_media("encoder_eval_graphs.html", html.encode(), content_type="text/html")
    print(f"\n→ {url}")
