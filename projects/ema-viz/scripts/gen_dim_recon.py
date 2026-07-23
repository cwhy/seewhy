"""
gen_dim_recon.py — Per-dimension reconstruction viewer for the linear autodecoder.

Layout
------
1. Full reconstruction strip (all dims active, random test samples) — sanity check.
2. Per-dimension sections: for each embedding dimension d, shows what that dimension
   alone reconstructs across N_SHOW images spread across the range of z[d]
   (sorted left→right from most-negative to most-positive activation).
   Also shows the pure basis template: sigmoid(1·exp(P[:,d]) + b).

Decoder: x̂ = sigmoid(z @ exp(P).T + b)
  exp(P) keeps pixel patterns positive; z=0 for inactive dims → 0 contribution.

Usage:
    uv run python projects/ema-viz/scripts/gen_dim_recon.py
    uv run python projects/ema-viz/scripts/gen_dim_recon.py --latent 32 --n-show 10
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
from lib.html_viz import build_dim_recon_html, to_b64_png

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

def decode(P, c, s, e, b, z):
    """z: (B, D) → (B, 784) in [0,1].
    z_pos[d]   = softplus(z[d]*s[d] + e[d])  — per-dim scale+bias, always positive
    P_pos[:,d] = exp(P[:,d] + c[d])           — per-dim log-scale, always positive
    contrib[d] = z_pos[d] * P_pos[:,d]        — always positive
    b          = universal scalar bias
    """
    z_pos = jax.nn.softplus(jnp.array(z) * jnp.array(s)[None, :] + jnp.array(e)[None, :])
    P_pos = jnp.exp(jnp.clip(jnp.array(P) + jnp.array(c)[None, :], -10.0, 10.0))
    return np.array(jax.nn.sigmoid(z_pos @ P_pos.T + b))

# ── Retrain decoder (fast — no encoder gradients) ─────────────────────────────

def train_decoder(Z_tr, X_tr, latent_dim, epochs=200, lr=3e-3, batch=512, seed=42):
    import optax
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
        def loss(params):
            z_pos = jax.nn.softplus(z * params["s"][None, :] + params["e"][None, :])
            P_pos = jnp.exp(jnp.clip(params["P"] + params["c"][None, :], -10.0, 10.0))
            xh = jax.nn.sigmoid(z_pos @ P_pos.T + params["b"])
            return -(x * jnp.log(xh + 1e-7) + (1-x) * jnp.log(1-xh+1e-7)).mean()
        l, g = jax.value_and_grad(loss)(params)
        u, s = opt.update(g, state, params)
        return optax.apply_updates(params, u), s, l

    for ep in range(epochs):
        for bi in range(len(Zb)):
            params, state, _ = step(params, state, Zb[bi], Xb[bi])

    P = np.array(params["P"]); c = np.array(params["c"])
    s = np.array(params["s"]); e = np.array(params["e"]); b = float(params["b"])
    x_hat = decode(P, c, s, e, b, b, Z_tr[:batch])
    mse = float(np.mean((x_hat - X_tr[:batch]/255.0)**2))
    print(f"  train MSE (sample): {mse:.4f}  b={b:+.3f}  s [{s.min():+.2f},{s.max():+.2f}]  e [{e.min():+.2f},{e.max():+.2f}]")
    return P, c, s, e, b

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent",   type=int, default=512, help="encoder LATENT_DIM")
    ap.add_argument("--n-show",   type=int, default=8,   help="samples per dimension")
    ap.add_argument("--n-full",   type=int, default=16,  help="random samples for full-recon strip")
    ap.add_argument("--out",      default=None,          help="output filename stem")
    ap.add_argument("--seed",     type=int, default=0,   help="RNG seed for full-recon sample selection")
    args = ap.parse_args()

    latent_dim = args.latent
    n_show     = args.n_show
    n_full     = args.n_full
    out_stem   = args.out or f"dim_recon_latent{latent_dim}"

    print(f"Loading data and encoder (LATENT={latent_dim})...")
    data  = load_supervised_image("mnist")
    X_te  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)
    X_tr  = data.X.reshape(data.n_samples, 784).astype(np.float32)

    enc_params = load_encoder(latent_dim)
    Z_te = encode_all(enc_params, X_te)
    Z_tr = encode_all(enc_params, X_tr)

    print("Training decoder (~20s)...")
    P, c, s, e, b = train_decoder(Z_tr, X_tr, latent_dim)
    print(f"P shape: {P.shape}")

    # ── Full-reconstruction strip (sanity check) ───────────────────────────────
    rng = np.random.default_rng(args.seed)
    full_idx   = rng.choice(len(X_te), size=n_full, replace=False)
    full_recon = decode(P, c, s, e, b, Z_te[full_idx])   # all dims active

    # measure test MSE on this sample
    test_mse = float(np.mean((full_recon - X_te[full_idx] / 255.0) ** 2))
    print(f"  full-recon test MSE (sample): {test_mse:.4f}")

    full_recon_samples = [
        (to_b64_png(X_te[i].reshape(28, 28)),
         to_b64_png(full_recon[j].reshape(28, 28)))
        for j, i in enumerate(full_idx)
    ]

    # ── Per-dimension sections ─────────────────────────────────────────────────
    sections = []
    for d in range(latent_dim):
        z_vals  = Z_te[:, d]
        z_min, z_max = float(z_vals.min()), float(z_vals.max())

        # Basis template: dim d at z=1, inactive dims at -1e6 so softplus(z*s+e) ≈ 0
        basis_z       = np.full((1, latent_dim), -1e6, dtype=np.float32)
        basis_z[0, d] = 1.0
        basis_b64     = to_b64_png(decode(P, c, s, e, b, basis_z)[0].reshape(28, 28))

        # n_show images equally spaced across z[d] range (sorted low→high)
        sorted_idx = np.argsort(z_vals)
        pick_pos   = np.linspace(0, len(sorted_idx) - 1, n_show, dtype=int)
        chosen_idx = sorted_idx[pick_pos]

        # For masking: set inactive dims to -1e6 so softplus(z*s+e) ≈ 0
        z_masked = np.full((len(chosen_idx), latent_dim), -1e6, dtype=np.float32)
        z_masked[:, d] = Z_te[chosen_idx, d]
        recons = decode(P, c, s, e, b, z_masked)

        samples = [
            (to_b64_png(X_te[idx].reshape(28, 28)),
             to_b64_png(recons[i].reshape(28, 28)),
             float(Z_te[idx, d]))
            for i, idx in enumerate(chosen_idx)
        ]

        sections.append((d, (z_min, z_max), basis_b64, samples))
        print(f"  dim {d:2d}  range [{z_min:+.2f}, {z_max:+.2f}]")

    # ── Build HTML ─────────────────────────────────────────────────────────────
    title = f"Linear autodecoder — per-dimension reconstruction  (LATENT={latent_dim})"
    description = (
        f"Decoder: x&#x302; = sigmoid(z &#xD7; exp(P + c)&#x1D40; + b),  "
        f"P &#x2208; &#x211D;<sup>784&#xD7;{latent_dim}</sup>,  c &#x2208; &#x211D;<sup>{latent_dim}</sup> (per-dim log-scale),  "
        f"frozen MLP-shallow encoder.  "
        f"Full-recon test MSE (random sample): <strong>{test_mse:.4f}</strong>."
    )
    html = build_dim_recon_html(title, description, full_recon_samples, sections)

    out_name = out_stem if out_stem.endswith(".html") else f"{out_stem}.html"
    url = save_media(out_name, html.encode(), content_type="text/html")
    print(f"\n→ {url}")


if __name__ == "__main__":
    main()
