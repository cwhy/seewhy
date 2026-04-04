"""
clf_table_ssl_vs_random.py — Linear probe + KNN accuracy table.

Same encoder set as mse_table_ssl_vs_random.py:
  gram-EMA (aug)   — exp_ema_{dim}.pkl          (512, 1024, 2048)
  gram-EMA (noaug) — exp_ema_noaug_{dim}.pkl    (512, 1024, 2048, 4096)
  SIGReg           — exp16a.pkl                 (4096 only)
  random           — frozen random projection   (all dims)

Evaluations:
  linear probe — logistic regression on frozen embeddings (200 epochs, Adam)
  KNN          — k=1,5,10,20 using FAISS L2 index

Usage:
    uv run python projects/ema-viz/scripts/tmp/clf_table_ssl_vs_random.py
"""
import pickle, sys
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np, optax, faiss

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_lib.datasets import load_supervised_image
from shared_lib.media import save_media

SSL_DIR = Path(__file__).parent.parent.parent.parent / "ssl"

LATENT_DIMS          = [512, 1024, 2048, 4096]
GRAM_EMA_DIMS        = {512, 1024, 2048}
GRAM_EMA_NOAUG_DIMS  = {512, 1024, 2048, 4096}
SIGREG_DIMS          = {4096}
KS                   = [1, 5, 10, 20]

# ── Pixel positions (gram encoder) ────────────────────────────────────────────

_ROWS = jnp.array(np.arange(784) // 28)
_COLS = jnp.array(np.arange(784) % 28)

# ── Encoders ──────────────────────────────────────────────────────────────────

def load_gram_encoder(path):
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {k: jnp.array(v) for k, v in raw.items()}

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

def encode_gram(p, X, bs=256):
    return np.concatenate([
        np.array(_encode_gram(p, jnp.array(X[i:i+bs].astype(np.float32))))
        for i in range(0, len(X), bs)
    ])

def encode_random(X, latent_dim, seed=1):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((784, latent_dim)).astype(np.float32) / np.sqrt(784)
    return (X / 255.0) @ W

# ── Linear probe ──────────────────────────────────────────────────────────────

def linear_probe(Z_tr, Y_tr, Z_te, Y_te, epochs=200, lr=1e-3, batch=512, seed=0):
    n_classes = 10
    dim = Z_tr.shape[1]

    # L2-normalize embeddings
    Z_tr = Z_tr / (np.linalg.norm(Z_tr, axis=1, keepdims=True) + 1e-8)
    Z_te = Z_te / (np.linalg.norm(Z_te, axis=1, keepdims=True) + 1e-8)

    key = jax.random.PRNGKey(seed)
    W = jax.random.normal(key, (dim, n_classes)) * 0.01
    b = jnp.zeros(n_classes)
    params = {"W": W, "b": b}
    opt   = optax.adam(lr)
    state = opt.init(params)

    n  = (len(Z_tr) // batch) * batch
    idx = np.random.default_rng(seed).permutation(len(Z_tr))[:n]
    Zb = jnp.array(Z_tr[idx].reshape(-1, batch, dim))
    Yb = jnp.array(Y_tr[idx].reshape(-1, batch))

    @jax.jit
    def step(params, state, z, y):
        def loss(p):
            logits = z @ p["W"] + p["b"]
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        l, g = jax.value_and_grad(loss)(params)
        u, s = opt.update(g, state, params)
        return optax.apply_updates(params, u), s, l

    for ep in range(epochs):
        for bi in range(len(Zb)):
            params, state, _ = step(params, state, Zb[bi], Yb[bi])

    logits = jnp.array(Z_te) @ params["W"] + params["b"]
    preds  = np.array(jnp.argmax(logits, axis=1))
    return float((preds == Y_te).mean())

# ── KNN ───────────────────────────────────────────────────────────────────────

def knn_accuracies(Z_tr, Y_tr, Z_te, Y_te):
    Z_tr = Z_tr.astype(np.float32)
    Z_te = Z_te.astype(np.float32)
    # L2-normalize
    Z_tr = Z_tr / (np.linalg.norm(Z_tr, axis=1, keepdims=True) + 1e-8)
    Z_te = Z_te / (np.linalg.norm(Z_te, axis=1, keepdims=True) + 1e-8)

    index = faiss.IndexFlatL2(Z_tr.shape[1])
    index.add(Z_tr)
    _, I = index.search(Z_te, max(KS))
    results = {}
    for k in KS:
        preds = np.array([
            np.bincount(Y_tr[I[i, :k]], minlength=10).argmax()
            for i in range(len(Z_te))
        ])
        results[k] = float((preds == Y_te).mean())
    return results

# ── HTML table ────────────────────────────────────────────────────────────────

def pct(v):
    return f"{v*100:.2f}%" if v is not None else "—"

def build_html(rows):
    """rows: list of (label, dim, probe_acc, knn_accs_dict|None)"""
    _CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d1117; color: #e6edf3; font-family: ui-monospace, monospace;
       font-size: 13px; padding: 32px 40px; }
h1 { font-size: 15px; font-weight: normal; color: #e6edf3; margin-bottom: 8px; }
p  { color: #8b949e; font-size: 11px; margin-bottom: 24px; line-height: 1.7; }
table { border-collapse: collapse; margin-bottom: 40px; }
th, td { padding: 7px 16px; text-align: right; border-bottom: 1px solid #21262d; }
th { color: #58a6ff; font-weight: normal; }
th:first-child, td:first-child { text-align: left; }
th:nth-child(2), td:nth-child(2) { text-align: left; }
.hi { color: #3fb950; }
.lo { color: #f85149; }
.na { color: #484f58; }
tr.rand { color: #8b949e; }
"""

    # Build per-dim grouped rows
    # Group by dim
    from collections import defaultdict
    by_dim = defaultdict(list)
    for label, dim, probe, knns in rows:
        by_dim[dim].append((label, probe, knns))

    probe_section = _build_section(
        "Linear probe accuracy (frozen embeddings, 200 epochs Adam)",
        by_dim, mode="probe"
    )
    knn_sections = "".join(
        _build_section(f"KNN accuracy — k={k}", by_dim, mode="knn", k=k)
        for k in KS
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>SSL classification: linear probe + KNN</title>
<style>{_CSS}</style></head>
<body>
<h1>MNIST classification: linear probe + KNN on frozen embeddings</h1>
<p>All SSL encoders share the same Hebbian gram + Hadamard readout architecture.<br>
Embeddings L2-normalized before probe and KNN.<br>
Higher is better.</p>
{probe_section}
{knn_sections}
</body></html>"""


def _build_section(title, by_dim, mode, k=None):
    header = "<tr><th>encoder</th><th>latent dim</th><th>accuracy</th></tr>"
    body = ""
    # find best non-random per dim for coloring
    best_per_dim = {}
    for dim, entries in sorted(by_dim.items()):
        vals = []
        for label, probe, knns in entries:
            if label == "random":
                continue
            v = probe if mode == "probe" else (knns[k] if knns else None)
            if v is not None:
                vals.append(v)
        best_per_dim[dim] = max(vals) if vals else None

    rand_per_dim = {}
    for dim, entries in sorted(by_dim.items()):
        for label, probe, knns in entries:
            if label == "random":
                v = probe if mode == "probe" else (knns[k] if knns else None)
                rand_per_dim[dim] = v

    prev_dim = None
    for dim, entries in sorted(by_dim.items()):
        for label, probe, knns in entries:
            v = probe if mode == "probe" else (knns[k] if knns else None)
            dim_cell = f"<td>{dim}</td>" if dim != prev_dim else "<td></td>"
            prev_dim = dim

            if v is None:
                body += f"<tr><td>{label}</td>{dim_cell}<td class='na'>—</td></tr>"
                continue

            rand_v = rand_per_dim.get(dim, 0)
            if label == "random":
                cls = "na"
                row_cls = " class='rand'"
            elif v == best_per_dim.get(dim):
                cls = "hi"
                row_cls = ""
            elif v < rand_v:
                cls = "lo"
                row_cls = ""
            else:
                cls = ""
                row_cls = ""

            body += f"<tr{row_cls}><td>{label}</td>{dim_cell}<td class='{cls}'>{pct(v)}</td></tr>"

    return f"<h2 style='font-size:13px;color:#f0883e;margin:24px 0 8px;'>{title}</h2><table><thead>{header}</thead><tbody>{body}</tbody></table>"


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_supervised_image("mnist")
    X_tr = data.X.reshape(data.n_samples, 784).astype(np.float32)
    Y_tr = np.array(data.y, dtype=np.int32)
    X_te = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)
    Y_te = np.array(data.y_test, dtype=np.int32)

    rows = []  # (label, dim, probe_acc, knn_accs|None)

    # Pixel space baseline (784-dim, X/255 normalized)
    print("\n── PIXEL SPACE (784-dim) ────────────────────────────")
    Z_px_tr = (X_tr / 255.0).astype(np.float32)
    Z_px_te = (X_te / 255.0).astype(np.float32)
    probe_px = linear_probe(Z_px_tr, Y_tr, Z_px_te, Y_te)
    knns_px  = knn_accuracies(Z_px_tr, Y_tr, Z_px_te, Y_te)
    print(f"  {'pixel space':<22} probe={probe_px*100:.2f}%  knn1={knns_px[1]*100:.2f}%  knn5={knns_px[5]*100:.2f}%")
    rows.append(("pixel space", 784, probe_px, knns_px))

    for dim in LATENT_DIMS:
        print(f"\n── LATENT={dim} ────────────────────────────")

        encoders = []
        if dim in GRAM_EMA_DIMS:
            encoders.append(("gram-EMA (aug)", SSL_DIR / f"params_exp_ema_{dim}.pkl"))
        if dim in GRAM_EMA_NOAUG_DIMS:
            encoders.append(("gram-EMA (noaug)", SSL_DIR / f"params_exp_ema_noaug_{dim}.pkl"))
        if dim in SIGREG_DIMS:
            encoders.append(("SIGReg", SSL_DIR / "params_exp16a.pkl"))

        for label, path in encoders:
            enc = load_gram_encoder(path)
            Z_tr = encode_gram(enc, X_tr)
            Z_te = encode_gram(enc, X_te)
            probe = linear_probe(Z_tr, Y_tr, Z_te, Y_te)
            knns  = knn_accuracies(Z_tr, Y_tr, Z_te, Y_te)
            print(f"  {label:<22} probe={probe*100:.2f}%  knn1={knns[1]*100:.2f}%  knn5={knns[5]*100:.2f}%")
            rows.append((label, dim, probe, knns))

        # Random baseline
        Z_tr_r = encode_random(X_tr, dim)
        Z_te_r = encode_random(X_te, dim)
        probe_r = linear_probe(Z_tr_r, Y_tr, Z_te_r, Y_te)
        knns_r  = knn_accuracies(Z_tr_r, Y_tr, Z_te_r, Y_te)
        print(f"  {'random':<22} probe={probe_r*100:.2f}%  knn1={knns_r[1]*100:.2f}%  knn5={knns_r[5]*100:.2f}%")
        rows.append(("random", dim, probe_r, knns_r))

    html = build_html(rows)
    url  = save_media("clf_table_ssl_vs_random.html", html.encode(), content_type="text/html")
    print(f"\n→ {url}")
