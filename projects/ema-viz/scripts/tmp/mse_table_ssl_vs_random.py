"""
Compute MSE table: gram-EMA / gram-EMA-noaug / SIGReg / random projection across latent dims.

All SSL encoders use the same Hebbian gram + Hadamard readout architecture.
Differ only in collapse prevention / augmentation:
  gram-EMA      — EMA target encoder, with augmentation   (exp_ema_{dim}.pkl)
  gram-EMA-noaug— EMA target encoder, no augmentation     (exp_ema_noaug_{dim}.pkl)
  SIGReg        — SIGReg collapse prevention, with aug    (exp16a.pkl, 4096 only)
  random        — frozen random linear projection

Uploads result as a simple HTML table.
"""
import pickle, sys
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np, optax

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_lib.datasets import load_supervised_image
from shared_lib.media import save_media

SSL_DIR = Path(__file__).parent.parent.parent.parent / "ssl"

LATENT_DIMS = [512, 1024, 2048, 4096]

# gram-EMA (with aug) available at:
GRAM_EMA_DIMS = {512, 1024, 2048}
# gram-EMA-noaug available at:
GRAM_EMA_NOAUG_DIMS = {512, 1024, 2048, 4096}
# SIGReg available at:
SIGREG_DIMS = {4096}

# ── Precomputed pixel positions ────────────────────────────────────────────────

_ROWS = np.arange(784) // 28
_COLS = np.arange(784) % 28

# ── Gram encoder (shared by EMA and SIGReg — same architecture) ───────────────

def load_gram_encoder(path):
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {k: jnp.array(v) for k, v in raw.items()}

def _make_encode_gram():
    rows_j = jnp.array(_ROWS)
    cols_j = jnp.array(_COLS)

    @jax.jit
    def _encode(p, x):
        x_norm   = x / 255.0
        re       = p["enc_row_emb"][rows_j]
        ce       = p["enc_col_emb"][cols_j]
        D_R      = p["enc_row_emb"].shape[1]
        D_C      = p["enc_col_emb"].shape[1]
        pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_R * D_C))(re, ce)
        pos_feat = pos_rc @ p["W_pos_enc"]
        f        = pos_feat @ p["W_enc"]
        F        = x_norm[:, :, None] * f[None, :, :]
        DEC_RANK = p["W_enc"].shape[1]
        H        = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK ** 2)
        return (H @ p["W_enc_A"]) * (H @ p["W_enc_B"])
    return _encode

_encode_gram_jit = _make_encode_gram()

def encode_gram(p, X, bs=256):
    return np.concatenate([
        np.array(_encode_gram_jit(p, jnp.array(X[i:i+bs].astype(np.float32))))
        for i in range(0, len(X), bs)
    ])

def encode_random(X, latent_dim, seed=1):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((784, latent_dim)).astype(np.float32) / np.sqrt(784)
    return (X / 255.0) @ W

# ── Decoder ───────────────────────────────────────────────────────────────────

def train_and_eval(Z_tr, X_tr, Z_te, X_te, latent_dim, epochs=200, lr=3e-3, batch=512, seed=42):
    z_std = Z_tr.std(axis=0) + 1e-8
    Z_tr  = Z_tr / z_std
    Z_te  = Z_te / z_std

    k1, _ = jax.random.split(jax.random.PRNGKey(seed))
    params = {
        "P": jax.random.normal(k1, (784, latent_dim)) * latent_dim ** -0.5,
        "c": jnp.full(latent_dim, -float(np.log(latent_dim))),
        "s": jnp.ones(latent_dim),
        "e": jnp.zeros(latent_dim),
        "b": jnp.zeros(()),
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
        u, s = opt.update(g, state, params)
        return optax.apply_updates(params, u), s, l

    for ep in range(epochs):
        for bi in range(len(Zb)):
            params, state, _ = step(params, state, Zb[bi], Xb[bi])

    z_pos = jax.nn.softplus(jnp.array(Z_te) * params["s"][None, :] + params["e"][None, :])
    P_pos = jnp.exp(jnp.clip(params["P"] + params["c"][None, :], -10.0, 10.0))
    xh    = np.array(jax.nn.sigmoid(z_pos @ P_pos.T + params["b"]))
    return float(np.mean((xh - X_te / 255.0) ** 2))

# ── HTML table ────────────────────────────────────────────────────────────────

def build_table_html(rows):
    """rows: list of (latent_dim, ema_mse|None, noaug_mse|None, sigreg_mse|None, rand_mse)"""
    _CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d1117; color: #e6edf3; font-family: ui-monospace, monospace;
       font-size: 13px; padding: 32px 40px; }
h1 { font-size: 15px; font-weight: normal; color: #e6edf3; margin-bottom: 8px; }
p  { color: #8b949e; font-size: 11px; margin-bottom: 24px; line-height: 1.7; }
table { border-collapse: collapse; }
th, td { padding: 8px 20px; text-align: right; border-bottom: 1px solid #21262d; }
th { color: #58a6ff; font-weight: normal; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.better { color: #3fb950; }
.worse  { color: #f85149; }
.na     { color: #484f58; }
"""
    cols = [
        ("gram-EMA (aug)",   "ema"),
        ("gram-EMA (noaug)", "noaug"),
        ("SIGReg",           "sigreg"),
        ("random",           "rand"),
    ]
    header = "<tr><th>latent dim</th>" + "".join(f"<th>{label}</th>" for label, _ in cols) + "</tr>"

    body = ""
    for dim, ema_mse, noaug_mse, sigreg_mse, rand_mse in rows:
        def cell(mse):
            if mse is None:
                return "<td class='na'>—</td>"
            cls = "better" if mse < rand_mse else "worse"
            return f"<td class='{cls}'>{mse:.4f}</td>"

        body += (f"<tr><td>{dim}</td>"
                 f"{cell(ema_mse)}{cell(noaug_mse)}{cell(sigreg_mse)}{cell(rand_mse)}</tr>")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>SSL vs Random — decoder MSE</title>
<style>{_CSS}</style></head>
<body>
<h1>Linear autodecoder MSE: Hebbian gram encoders vs random projection</h1>
<p>All SSL encoders share the same architecture: Hebbian gram + Hadamard readout.<br>
Columns differ in collapse prevention / augmentation strategy.<br>
Decoder: x&#x302; = sigmoid(&#x3A3;_d softplus(z[d]&#xB7;s[d]+e[d])&#xB7;exp(P[:,d]+c[d]) + b) — 200 epochs, Adam lr=3e-3.<br>
<span style="color:#3fb950">green</span> = better than random &nbsp;|&nbsp; <span style="color:#f85149">red</span> = worse than random.</p>
<table><thead>{header}</thead><tbody>{body}</tbody></table>
</body></html>"""

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_supervised_image("mnist")
    X_tr = data.X.reshape(data.n_samples, 784).astype(np.float32)
    X_te = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)

    rows = []
    for dim in LATENT_DIMS:
        print(f"\n── LATENT={dim} ────────────────────────────")

        ema_mse = noaug_mse = sigreg_mse = None

        if dim in GRAM_EMA_DIMS:
            enc = load_gram_encoder(SSL_DIR / f"params_exp_ema_{dim}.pkl")
            Z_tr = encode_gram(enc, X_tr)
            Z_te = encode_gram(enc, X_te)
            ema_mse = train_and_eval(Z_tr, X_tr, Z_te, X_te, dim)
            print(f"  gram-EMA (aug)   MSE: {ema_mse:.4f}")

        if dim in GRAM_EMA_NOAUG_DIMS:
            enc = load_gram_encoder(SSL_DIR / f"params_exp_ema_noaug_{dim}.pkl")
            Z_tr = encode_gram(enc, X_tr)
            Z_te = encode_gram(enc, X_te)
            noaug_mse = train_and_eval(Z_tr, X_tr, Z_te, X_te, dim)
            print(f"  gram-EMA (noaug) MSE: {noaug_mse:.4f}")

        if dim in SIGREG_DIMS:
            enc = load_gram_encoder(SSL_DIR / "params_exp16a.pkl")
            Z_tr = encode_gram(enc, X_tr)
            Z_te = encode_gram(enc, X_te)
            sigreg_mse = train_and_eval(Z_tr, X_tr, Z_te, X_te, dim)
            print(f"  SIGReg           MSE: {sigreg_mse:.4f}")

        Z_tr_rand = encode_random(X_tr, dim)
        Z_te_rand = encode_random(X_te, dim)
        rand_mse = train_and_eval(Z_tr_rand, X_tr, Z_te_rand, X_te, dim)
        print(f"  random           MSE: {rand_mse:.4f}")

        rows.append((dim, ema_mse, noaug_mse, sigreg_mse, rand_mse))

    html = build_table_html(rows)
    url  = save_media("ssl_vs_random_mse_table.html", html.encode(), content_type="text/html")
    print(f"\n→ {url}")
