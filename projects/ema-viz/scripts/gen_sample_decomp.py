"""
gen_sample_decomp.py — Per-sample additive decomposition of the linear autodecoder.

For each selected test image, shows every dimension's contribution in logit (pre-sigmoid)
space alongside the original, logit sum, and final reconstruction. Contributions add up:

    logit = b  +  Σ_d softplus(z[d]*s[d]+e[d]) · exp(P[:,d]+c[d])
    recon = sigmoid(logit)

Encoder can be the trained SSL encoder (default) or a random frozen linear projection
(--encoder random), useful as a baseline comparison.

Usage:
    uv run python projects/ema-viz/scripts/gen_sample_decomp.py
    uv run python projects/ema-viz/scripts/gen_sample_decomp.py --latent 32 --n-samples 10
    uv run python projects/ema-viz/scripts/gen_sample_decomp.py --encoder random --latent 32
"""

import argparse
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.datasets import load_supervised_image
from shared_lib.media import save_media

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.html_viz import build_sample_decomp_html, to_b64_png, to_b64_png_diverging, to_b64_png_sequential

SSL_DIR = Path(__file__).parent.parent.parent / "ssl"

_ROWS = jnp.array(np.arange(784) // 28)
_COLS = jnp.array(np.arange(784) % 28)

# ── Encoder / decoder ──────────────────────────────────────────────────────────

def load_encoder(latent_dim):
    with open(SSL_DIR / f"params_exp_ema_{latent_dim}.pkl", "rb") as f:
        raw = pickle.load(f)
    return {k: jnp.array(v) for k, v in raw.items()}

@jax.jit
def _encode(p, x):
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

def encode_all(enc_params, X, bs=256):
    return np.concatenate([
        np.array(_encode(enc_params, jnp.array(X[i:i+bs].astype(np.float32))))
        for i in range(0, len(X), bs)
    ])

def random_project(X, latent_dim, seed=0):
    """Frozen random linear projection: z = (X/255) @ W,  W ~ N(0, 1/784)."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((784, latent_dim)).astype(np.float32) / np.sqrt(784)
    return (X / 255.0) @ W

def train_decoder(Z_tr, X_tr, latent_dim, epochs=200, lr=3e-3, batch=512, seed=42):
    import optax
    # Normalize z to unit std per dim so softplus(z*s) starts in a sane range
    # regardless of latent_dim or encoder scale.
    z_std = Z_tr.std(axis=0) + 1e-8   # (D,)
    Z_tr  = Z_tr / z_std

    k1, _ = jax.random.split(jax.random.PRNGKey(seed))
    params = {
        "P": jax.random.normal(k1, (784, latent_dim)) * latent_dim ** -0.5,
        "c": jnp.full(latent_dim, -float(np.log(latent_dim))),  # init P_pos = exp(P)/D so Σ ≈ O(1)
        "s": jnp.ones(latent_dim),                            # per-dim z scale
        "e": jnp.zeros(latent_dim),                           # per-dim z bias
        "b": jnp.zeros(()),            # universal scalar bias
    }
    opt   = optax.adam(lr)
    state = opt.init(params)

    n  = (len(Z_tr) // batch) * batch
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
        u, s_opt = opt.update(g, state, params)
        return optax.apply_updates(params, u), s_opt, l

    for ep in range(epochs):
        for bi in range(len(Zb)):
            params, state, _ = step(params, state, Zb[bi], Xb[bi])

    P = np.array(params["P"]); c = np.array(params["c"])
    s = np.array(params["s"]); e = np.array(params["e"]); b = float(params["b"])
    P_pos = np.exp(np.clip(P + c[None, :], -10.0, 10.0))
    z_pos = jax.nn.softplus(jnp.array(Z_tr[:batch]) * jnp.array(s)[None, :] + jnp.array(e)[None, :])
    xh = np.array(jax.nn.sigmoid(z_pos @ jnp.array(P_pos).T + b))
    mse = float(np.mean((xh - X_tr[:batch] / 255.0) ** 2))
    print(f"  train MSE (sample): {mse:.4f}  b={b:+.3f}  s [{s.min():+.2f},{s.max():+.2f}]  e [{e.min():+.2f},{e.max():+.2f}]")
    return P, c, s, e, b, z_std

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent",    type=int, default=512, help="encoder LATENT_DIM")
    ap.add_argument("--encoder",   default="ssl",         help="ssl (default) or random")
    ap.add_argument("--n-samples", type=int, default=8,   help="number of test images to show")
    ap.add_argument("--img-size",  type=int, default=56,  help="display size per image in px")
    ap.add_argument("--out",       default=None,          help="output filename stem")
    ap.add_argument("--seed",      type=int, default=0,   help="RNG seed for sample selection")
    args = ap.parse_args()

    latent_dim   = args.latent
    encoder_type = args.encoder
    out_stem     = args.out or f"sample_decomp_{encoder_type}_latent{latent_dim}"

    print(f"Loading data ({encoder_type} encoder, LATENT={latent_dim})...")
    data  = load_supervised_image("mnist")
    X_te  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)
    Y_te  = np.array(data.y_test, dtype=np.int32)
    X_tr  = data.X.reshape(data.n_samples, 784).astype(np.float32)

    if encoder_type == "random":
        Z_te = random_project(X_te, latent_dim, seed=args.seed + 1)
        Z_tr = random_project(X_tr, latent_dim, seed=args.seed + 1)
    else:
        enc_params = load_encoder(latent_dim)
        Z_te = encode_all(enc_params, X_te)
        Z_tr = encode_all(enc_params, X_tr)

    print("Training decoder (~20s)...")
    P, c, s, e, b, z_std = train_decoder(Z_tr, X_tr, latent_dim)
    Z_te = Z_te / z_std   # same normalization as training
    P_pos = np.exp(np.clip(P + c[None, :], -10.0, 10.0))   # (784, D), always positive

    # Pick one sample per class (or random if n_samples > 10)
    rng = np.random.default_rng(args.seed)
    n = args.n_samples
    if n <= 10:
        # one per class, cycling through classes
        chosen = []
        for cls in range(10):
            idxs = np.where(Y_te == cls)[0]
            chosen.append(rng.choice(idxs))
            if len(chosen) == n:
                break
    else:
        chosen = rng.choice(len(X_te), size=n, replace=False)

    # Build sample dicts
    samples = []
    for idx in chosen:
        x   = X_te[idx]          # (784,) uint8
        z   = Z_te[idx]          # (D,) float
        lbl = int(Y_te[idx])

        # Per-dim contributions in logit space: softplus(z*s+e) * exp(P+c)
        z_pos = np.array(jax.nn.softplus(jnp.array(z * s + e)))  # (D,) always positive
        contribs_logit = z_pos[None, :] * P_pos                    # (784, D) always positive

        # Total logit and reconstruction (scalar bias b)
        logit = contribs_logit.sum(axis=1) + b    # (784,)
        recon = 1.0 / (1.0 + np.exp(-logit))      # sigmoid, (784,)
        mse   = float(np.mean((recon - x / 255.0) ** 2))

        # Shared sequential scale for contributions + logit so they visually add up.
        # vmax = max of the contribution sum (logit without bias).
        contrib_sum = contribs_logit.sum(axis=1)   # (784,)
        shared_vmax = float(max(contrib_sum.max(), 1e-6))

        contrib_list = [
            (to_b64_png_sequential(contribs_logit[:, d].reshape(28, 28), vmax=shared_vmax), float(z_pos[d]))
            for d in range(latent_dim)
        ]

        samples.append({
            "label":     lbl,
            "mse":       mse,
            "orig_b64":  to_b64_png(x.reshape(28, 28)),
            "contribs":  contrib_list,
            "logit_b64": to_b64_png_sequential(contrib_sum.reshape(28, 28), vmax=shared_vmax),
            "recon_b64": to_b64_png(recon.reshape(28, 28)),
            "vabs":      shared_vmax,
        })
        print(f"  sample idx={idx}  class={lbl}  mse={mse:.4f}  vmax={shared_vmax:.2f}")

    title = f"Linear autodecoder — per-sample additive decomposition  (LATENT={latent_dim}, encoder={encoder_type})"
    html  = build_sample_decomp_html(title, samples, img_size=args.img_size)

    out_name = out_stem if out_stem.endswith(".html") else f"{out_stem}.html"
    url = save_media(out_name, html.encode(), content_type="text/html")
    print(f"\n→ {url}")


if __name__ == "__main__":
    main()
