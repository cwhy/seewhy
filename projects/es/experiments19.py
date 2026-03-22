"""
exp19 — Domain-invariant initialization strategies for f3(4,4,4) bilinear.

Problem (exp18): scale-fix gives large *random* M from ep0 — strong signal
but no structure → slower convergence than baseline despite earlier escape.

Goal: initializations that give M with *structure* from ep0, domain-invariantly.

Three strategies:
  B — Sinusoidal PE: row/col embeddings use sin/cos at geometric frequencies
      (transformer-style). M immediately encodes spatial proximity.
      row_emb[i, 2k]   = sin(i · π · (k+1) / d_r)
      row_emb[i, 2k+1] = cos(i · π · (k+1) / d_r)
      Normalized to have consistent scale.

  C — Orthogonal frame: row_emb and col_emb initialized as scaled orthogonal
      frames so Σ_r row[r]⊗row[r] = I_{d_r} (tight frame). Then M_pos has
      Kronecker structure M_pos = I_{d_r} ⊗ I_{d_c}. val_emb also orthogonal.
      Combined M ≈ (n_pixels/d) · I_d, a scaled identity from ep0.

  D — Sinusoidal PE + orthogonal val: combine sinusoidal position structure
      with orthogonal val_emb. Best of both: spatial structure + frame property.

Control:
  A — Baseline (scale=0.02, current default)

All: Standard ES, σ=0.001, rank=8, n_pop=128, 50 epochs.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python projects/es/experiments19.py
"""

import json, time, sys
from pathlib import Path
from datetime import datetime

import jax
import jax.numpy as jnp
import optax
import logging
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image
from shared_lib.media import save_matplotlib_figure

N_VAL_BINS   = 32
BATCH_SIZE   = 128
SEED         = 42
MAX_EPOCHS   = 50
ADAM_LR_INIT = 1e-3
ADAM_LR_END  = 3e-5

JSONL       = Path(__file__).parent / "results_mnist.jsonl"
REPORT_PATH = Path(__file__).parent / "report19.md"

ROWS = jnp.arange(784) // 28
COLS = jnp.arange(784) % 28


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def random_noise_batch(imgs, rng, scale=30.0):
    noise = jax.random.uniform(rng, imgs.shape, minval=-scale, maxval=scale)
    return jnp.clip(imgs + noise, 0.0, 255.0)


# ── Initialization strategies ─────────────────────────────────────────────

def sinusoidal_emb(n_pos, d):
    """Transformer-style sin/cos positional encoding, shape (n_pos, d)."""
    pos = np.arange(n_pos)[:, None]          # (n_pos, 1)
    dim = np.arange(d // 2)[None, :]         # (1, d//2)
    freq = np.pi * (dim + 1) / d             # geometric frequencies
    emb = np.zeros((n_pos, d))
    emb[:, 0::2] = np.sin(pos * freq)
    emb[:, 1::2] = np.cos(pos * freq[:, :d - d // 2])
    # normalize rows to unit norm
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-8)
    return emb.astype(np.float32)


def orthogonal_frame(n_pos, d, rng_key):
    """
    Scaled orthogonal frame: rows of shape (n_pos, d).
    Tight frame property: Σ_i v_i⊗v_i = I_d  iff columns are orthonormal.
    We take d random orthonormal vectors in R^n_pos, transposed.
    Scale: 1/sqrt(n_pos/d) so that the Gram matrix has unit diagonal.
    """
    mat = jax.random.normal(rng_key, (n_pos, d))
    q, _ = jnp.linalg.qr(mat)          # (n_pos, d) with orthonormal columns
    # Frame property: Σ_i q[i]⊗q[i] = q.T@q = I_d (if n_pos >= d)
    # Scale rows so that each row has norm sqrt(d/n_pos)
    # → Σ_i row[i]·row[i].T = (d/n_pos) * n_pos = d ... hmm
    # Actually we want Σ_i (scale*q[i])⊗(scale*q[i]) = I_d
    # scale² * Σ_i q[i]⊗q[i] = scale² * I_d = I_d → scale = 1
    # So q already satisfies the frame property. Return as-is.
    return q


def make_f3_bilinear(d_r, d_c, d_v, mlp_dim, init="baseline"):
    """
    init: "baseline" | "sinusoidal" | "ortho_frame" | "sinusoidal_ortho_val"
    """
    d  = d_r * d_c * d_v
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))

    if init == "baseline":
        row_emb = jax.random.normal(next(kg).get(), (28,        d_r)) * 0.02
        col_emb = jax.random.normal(next(kg).get(), (28,        d_c)) * 0.02
        val_emb = jax.random.normal(next(kg).get(), (N_VAL_BINS, d_v)) * 0.02

    elif init == "sinusoidal":
        _ = next(kg).get(); _ = next(kg).get(); _ = next(kg).get()  # keep key gen in sync
        row_emb = jnp.array(sinusoidal_emb(28,        d_r))
        col_emb = jnp.array(sinusoidal_emb(28,        d_c))
        val_emb = jnp.array(sinusoidal_emb(N_VAL_BINS, d_v))

    elif init == "ortho_frame":
        k1 = next(kg).get(); k2 = next(kg).get(); k3 = next(kg).get()
        row_emb = orthogonal_frame(28,        d_r, k1)
        col_emb = orthogonal_frame(28,        d_c, k2)
        val_emb = orthogonal_frame(N_VAL_BINS, d_v, k3)

    elif init == "sinusoidal_ortho_val":
        _ = next(kg).get(); _ = next(kg).get(); k3 = next(kg).get()
        row_emb = jnp.array(sinusoidal_emb(28,        d_r))
        col_emb = jnp.array(sinusoidal_emb(28,        d_c))
        val_emb = orthogonal_frame(N_VAL_BINS, d_v, k3)

    else:
        raise ValueError(f"Unknown init: {init}")

    p = {
        "row_emb":       row_emb,
        "col_emb":       col_emb,
        "val_emb":       val_emb,
        "H_answer":      jax.random.normal(next(kg).get(), (1, d))   * (1.0 / d**0.5),
        "ln_out_gamma":  jnp.ones(d),
        "ln_out_beta":   jnp.zeros(d),
        "W1":            jax.random.normal(next(kg).get(), (mlp_dim, d)) / d**0.5,
        "b1":            jnp.zeros(mlp_dim),
        "W2":            jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2":            jnp.zeros(10),
        "ln_hop0_gamma": jnp.ones(d),
        "ln_hop0_beta":  jnp.zeros(d),
    }

    def fwd(params, x):
        B    = x.shape[0]
        bins = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
        re   = params["row_emb"][ROWS];  ce = params["col_emb"][COLS]
        ve   = params["val_emb"][bins]
        pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r * d_c)
        emb  = (pos[None, :, :, None] * ve[:, :, None, :]).reshape(B, 784, d)
        M    = jnp.einsum("bti,btj->bij", emb, emb)
        h    = jnp.broadcast_to(params["H_answer"][0], (B, d))
        h    = jnp.einsum("bij,bj->bi", M, h)
        h    = layer_norm(h, params["ln_hop0_gamma"], params["ln_hop0_beta"])
        h    = layer_norm(h, params["ln_out_gamma"],  params["ln_out_beta"])
        h    = jax.nn.gelu(h @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p


def acc_fn(fwd, params, x, y):
    preds = jnp.concatenate([
        jnp.argmax(fwd(params, x[i:i + BATCH_SIZE]), 1)
        for i in range(0, len(x), BATCH_SIZE)])
    return float(jnp.mean(preds == y))


# ── ES ─────────────────────────────────────────────────────────────────────

def make_es_step(fwd, sigma, n_pop, tx, rank=8):
    def make_perturbation(pop_key, params):
        eps = {}
        for idx, (k, v) in enumerate(params.items()):
            subkey = jax.random.fold_in(pop_key, idx)
            if v.ndim == 2:
                m, n = v.shape
                ab = jax.random.normal(subkey, (m + n, rank))
                a, b = ab[:m], ab[m:]
                eps[k] = (a @ b.T) * sigma
            else:
                eps[k] = jax.random.normal(subkey, v.shape) * sigma
        return eps

    def pair_estimate(pop_key, params, bx, by):
        eps   = make_perturbation(pop_key, params)
        p_pos = {k: v + eps[k] for k, v in params.items()}
        p_neg = {k: v - eps[k] for k, v in params.items()}
        ce    = lambda p: jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(fwd(p, bx), by))
        return ce(p_pos) - ce(p_neg), eps

    @jax.jit
    def es_step(params, opt_state, bx, by, pop_keys):
        fitnesses, epses = jax.vmap(
            pair_estimate, in_axes=(0, None, None, None)
        )(pop_keys, params, bx, by)
        grad = jax.tree_util.tree_map(
            lambda e: jnp.einsum("i,i...->...", fitnesses, e) / (n_pop * rank * sigma**2),
            epses,
        )
        updates, new_opt_state = tx.update(grad, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, jnp.mean(jnp.abs(fitnesses))

    return es_step


def train_es(name, fwd, params, X_tr, y_tr, X_te, y_te,
             sigma=0.001, n_pop=128, rank=8, max_epochs=MAX_EPOCHS):
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = max_epochs * num_batches
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT, total_steps,
                                              alpha=ADAM_LR_END / ADAM_LR_INIT)
    tx        = optax.adam(lr_sched)
    es_step   = make_es_step(fwd, sigma, n_pop, tx, rank=rank)
    opt_state = tx.init(params)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    history = {"train": [], "test": [], "fitness": []}
    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        key  = next(key_gen).get()
        perm = jax.random.permutation(key, len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        aug_rng = key
        epoch_fit = []
        for b in range(num_batches):
            s, e   = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx, by = Xs[s:e], ys[s:e]
            aug_rng, sub = jax.random.split(aug_rng)
            bx = random_noise_batch(bx, sub)
            pop_keys = jax.random.split(jax.random.fold_in(key, b), n_pop)
            params, opt_state, fit = es_step(params, opt_state, bx, by, pop_keys)
            epoch_fit.append(float(fit))

        train_acc = acc_fn(fwd, params, X_tr, y_tr)
        test_acc  = acc_fn(fwd, params, X_te, y_te)
        mean_fit  = sum(epoch_fit) / len(epoch_fit)
        history["train"].append(train_acc)
        history["test"].append(test_acc)
        history["fitness"].append(mean_fit)
        logging.info(
            f"[{name}] ep {epoch:3d}  train={train_acc:.4f}  test={test_acc:.4f}"
            f"  |fit|={mean_fit:.6f}  {time.perf_counter()-t0:.1f}s"
        )

    return train_acc, test_acc, time.perf_counter() - t0, history


def find_plateau_escape(history, threshold=0.20):
    for i, v in enumerate(history["test"]):
        if v > threshold:
            return i
    return None


# ── Plots ──────────────────────────────────────────────────────────────────

def make_plots(variants):
    """variants: list of (label, color, history)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(MAX_EPOCHS))
    urls   = {}

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, hist in variants:
        ax.plot(epochs, [v * 100 for v in hist["train"]],
                color=color, linestyle="--", alpha=0.5)
        ax.plot(epochs, [v * 100 for v in hist["test"]],
                color=color, linewidth=2, label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("exp19 — Init strategies: test accuracy (50 epochs)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    urls["accuracy"] = save_matplotlib_figure("exp19_accuracy", fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, color, hist in variants:
        ax.plot(epochs, hist["fitness"], color=color, linewidth=2, label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("|fitness|")
    ax.set_title("exp19 — Mean |fitness| per epoch")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    urls["fitness"] = save_matplotlib_figure("exp19_fitness", fig)
    plt.close(fig)

    return urls


# ── Report ─────────────────────────────────────────────────────────────────

def write_report(urls, results):
    """results: list of (label, init_desc, tr, te, t_s, plateau_ep)"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for label, init_desc, tr, te, t_s, plateau_ep in results:
        p = f"ep{plateau_ep}" if plateau_ep is not None else "none"
        rows.append(f"| {label} | {init_desc} | {tr*100:.2f}% | {te*100:.2f}% | {t_s:.0f}s | {p} |")

    lines = [
        "# exp19 — Domain-invariant Initialization Report",
        "",
        f"**Date:** {date_str}  **Epochs:** {MAX_EPOCHS}  **n_pop:** 128  **σ:** 0.001  **rank:** 8",
        "",
        "## Hypothesis",
        "",
        "exp18 showed scale-fix eliminates the plateau but converges slower.",
        "Cause: large *random* M has strong signal but no structure.",
        "Here we test inits that give M *structured* signal from ep0.",
        "",
        "## Results",
        "",
        "| Variant | Init | Train | Test | Time | Plateau escape |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        "## Accuracy curves",
        "",
        f"![accuracy]({urls['accuracy']})",
        "",
        "## |fitness| over time",
        "",
        f"![fitness]({urls['fitness']})",
    ]
    REPORT_PATH.write_text("\n".join(lines))
    logging.info(f"Report written to {REPORT_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────

VARIANTS = [
    ("A-baseline",            "baseline",           "scale=0.02 (random)",         "steelblue"),
    ("B-sinusoidal",          "sinusoidal",          "sin/cos PE (row+col+val)",     "darkorange"),
    ("C-ortho-frame",         "ortho_frame",         "orthogonal frame (row+col+val)", "green"),
    ("D-sinusoidal+ortho-val","sinusoidal_ortho_val","sin/cos PE + ortho val",       "purple"),
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info(f"JAX devices: {jax.devices()}")

    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    all_histories = []
    all_results   = []

    for var_name, init_key, init_desc, color in VARIANTS:
        logging.info(f"\n{'='*60}\n{var_name}: {init_desc}\n{'='*60}")
        fwd, params = make_f3_bilinear(4, 4, 4, 256, init=init_key)
        tr, te, t_s, hist = train_es(
            var_name, fwd, params, X_train, y_train, X_test, y_test,
            sigma=0.001, n_pop=128, rank=8)
        plateau_ep = find_plateau_escape(hist)
        logging.info(f"  → train={tr:.4f}  test={te:.4f}  {t_s:.0f}s  plateau_escape=ep{plateau_ep}")

        all_histories.append((var_name, color, hist))
        all_results.append((var_name, init_desc, tr, te, t_s, plateau_ep))
        append_result({"experiment": "exp19", "variant": var_name, "init": init_key,
                       "train_acc": round(tr, 6), "test_acc": round(te, 6),
                       "time_s": round(t_s, 1), "epochs": MAX_EPOCHS,
                       "plateau_escape_ep": plateau_ep})

    logging.info("Generating plots and uploading...")
    urls = make_plots(all_histories)
    write_report(urls, all_results)

    print(f"\n{'='*80}")
    print(f"{'Variant':<30} {'Init':<30} {'Test':>7} {'Escape':>8}")
    print(f"{'-'*80}")
    for var_name, init_desc, tr, te, t_s, plateau_ep in all_results:
        p = f"ep{plateau_ep}" if plateau_ep is not None else "none"
        print(f"{var_name:<30} {init_desc:<30} {te:>7.4f} {p:>8}")
    print(f"{'='*80}")
    print(f"\nReport: {REPORT_PATH}")
    for k, u in urls.items():
        print(f"  {k}: {u}")
