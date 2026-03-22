"""
exp2 — ES with adaptive σ, larger population, and cosine LR decay on MNIST.

Lesson from exp1: fixed σ=0.001 was too conservative for W1/embeddings (which
have larger weight magnitude) while just barely safe for W2 (std≈0.01 at init).
This caused weak gradient signal for most of the network for the first 10 epochs.

Adaptive σ fix: σ_k = sigma_rel * RMS(params_k)
  → 10% relative perturbation for every parameter regardless of magnitude
  → as weights grow during training, σ scales up → stronger signal

Improvements over exp1:
  - Adaptive sigma (sigma_rel=0.10, relative to weight RMS per param)
  - n_pop=64 antithetic pairs (2× more gradient signal)
  - Cosine LR decay (1e-3 → 1e-4)
  - No explicit σ² normalisation — Adam absorbs the scale

Reference: https://eshyperscale.github.io/
Usage:
    uv run python projects/es/experiments2.py
"""

import json, time, sys
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import optax
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

N_VAL_BINS = 32
BATCH_SIZE = 128
SEED       = 42

ES_SIGMA_REL = 0.10    # perturbation as fraction of weight RMS (10% relative)
ES_N_POP     = 64      # antithetic pairs per step
ES_RANK      = 1       # rank of low-rank perturbation for 2D params

ADAM_LR_INIT = 1e-3
ADAM_LR_END  = 1e-4
MAX_EPOCHS   = 60

JSONL = Path(__file__).parent / "results_mnist.jsonl"

ROWS = jnp.arange(784) // 28
COLS = jnp.arange(784) % 28


# ── Utilities ─────────────────────────────────────────────────────────────────

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta

def n_nonemb(p, emb_keys):
    return sum(v.size for k, v in p.items() if k not in emb_keys)

def n_total(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")

def random_noise_batch(imgs, rng, scale=30.0):
    noise = jax.random.uniform(rng, imgs.shape, minval=-scale, maxval=scale)
    return jnp.clip(imgs + noise, 0.0, 255.0)


# ── Architecture (f3 bilinear — same as RNN project) ─────────────────────────

F3_EMB_KEYS = frozenset({"row_emb", "col_emb", "val_emb"})


def make_f3_bilinear(d_r, d_c, d_v, mlp_dim, n_queries=1, depth=1):
    d        = d_r * d_c * d_v
    feat_dim = n_queries * d
    kg       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "row_emb": jax.random.normal(next(kg).get(), (28, d_r))         * 0.02,
        "col_emb": jax.random.normal(next(kg).get(), (28, d_c))         * 0.02,
        "val_emb": jax.random.normal(next(kg).get(), (N_VAL_BINS, d_v)) * 0.02,
        "H_answer":     jax.random.normal(next(kg).get(), (n_queries, d)) * 0.02,
        "ln_out_gamma": jnp.ones(feat_dim),
        "ln_out_beta":  jnp.zeros(feat_dim),
        "W1": jax.random.normal(next(kg).get(), (mlp_dim, feat_dim)) / feat_dim**0.5,
        "b1": jnp.zeros(mlp_dim),
        "W2": jax.random.normal(next(kg).get(), (10, mlp_dim))       * 0.01,
        "b2": jnp.zeros(10),
    }
    for i in range(depth):
        p[f"ln_hop{i}_gamma"] = jnp.ones(d)
        p[f"ln_hop{i}_beta"]  = jnp.zeros(d)

    def fwd(params, x):
        B    = x.shape[0]
        bins = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
        re   = params["row_emb"][ROWS]
        ce   = params["col_emb"][COLS]
        ve   = params["val_emb"][bins]
        pos  = (re[:, :, None] * ce[:, None, :]).reshape(784, d_r * d_c)
        emb  = (pos[None, :, :, None] * ve[:, :, None, :]).reshape(B, 784, d)
        M    = jnp.einsum("bti,btj->bij", emb, emb)
        outs = []
        for q in range(n_queries):
            h = jnp.broadcast_to(params["H_answer"][q], (B, d))
            for i in range(depth):
                h = jnp.einsum("bij,bj->bi", M, h)
                h = layer_norm(h, params[f"ln_hop{i}_gamma"],
                               params[f"ln_hop{i}_beta"])
            outs.append(h)
        features = jnp.concatenate(outs, axis=-1)
        features = layer_norm(features, params["ln_out_gamma"], params["ln_out_beta"])
        h = jax.nn.gelu(features @ params["W1"].T + params["b1"])
        return h @ params["W2"].T + params["b2"]

    return fwd, p, F3_EMB_KEYS


# ── ES with EGGROLL adaptive-σ low-rank mutations ─────────────────────────────

def make_es_step(fwd, sigma_rel, n_pop, tx, rank=1):
    """ES step with adaptive σ and Adam outer optimizer.

    Perturbation: σ_k = sigma_rel * RMS(params_k)  (10% relative per param)
      - 2D: ε_k = A @ B.T * σ_k   (rank-1 EGGROLL)
      - 1D: ε_k = v * σ_k

    Gradient estimate fed to Adam (no explicit σ² normalisation — Adam handles scale):
        grad_k = (1/n_pop) * Σᵢ fitness_i * ε_i_k
    """

    def make_perturbation(pop_key, params):
        eps = {}
        for idx, (k, v) in enumerate(params.items()):
            subkey   = jax.random.fold_in(pop_key, idx)
            rms      = jnp.sqrt(jnp.mean(v ** 2) + 1e-8)
            sig      = sigma_rel * rms
            if v.ndim == 2:
                m, n = v.shape
                ab   = jax.random.normal(subkey, (m + n, rank))
                a, b = ab[:m], ab[m:]
                eps[k] = (a @ b.T) * sig
            else:
                eps[k] = jax.random.normal(subkey, v.shape) * sig
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

        # Raw gradient: Σᵢ fitness_i * ε_i_k / n_pop  (Adam normalises the rest)
        grad = jax.tree_util.tree_map(
            lambda e: jnp.einsum("i,i...->...", fitnesses, e) / n_pop,
            epses,
        )
        updates, new_opt_state = tx.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, jnp.mean(jnp.abs(fitnesses))

    return es_step


# ── Training loop ─────────────────────────────────────────────────────────────

def train(name, fwd, params, emb_keys, X_tr, y_tr, X_te, y_te, max_epochs=MAX_EPOCHS):
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = max_epochs * num_batches
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT, total_steps,
                                               alpha=ADAM_LR_END / ADAM_LR_INIT)
    tx          = optax.adam(lr_sched)
    es_step     = make_es_step(fwd, ES_SIGMA_REL, ES_N_POP, tx, rank=ES_RANK)
    opt_state   = tx.init(params)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    def acc_fn(p, x, y):
        preds = jnp.concatenate([
            jnp.argmax(fwd(p, x[i:i + BATCH_SIZE]), 1)
            for i in range(0, len(x), BATCH_SIZE)])
        return float(jnp.mean(preds == y))

    t0 = time.perf_counter()
    train_acc = test_acc = 0.0
    for epoch in range(max_epochs):
        key  = next(key_gen).get()
        perm = jax.random.permutation(key, len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        aug_rng = key
        epoch_fitness = []
        for b in range(num_batches):
            s, e   = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx, by = Xs[s:e], ys[s:e]
            aug_rng, sub = jax.random.split(aug_rng)
            bx = random_noise_batch(bx, sub)
            pop_keys = jax.random.split(jax.random.fold_in(key, b), ES_N_POP)
            params, opt_state, mean_fit = es_step(params, opt_state, bx, by, pop_keys)
            epoch_fitness.append(float(mean_fit))

        train_acc = acc_fn(params, X_tr, y_tr)
        test_acc  = acc_fn(params, X_te, y_te)
        mean_fit  = sum(epoch_fitness) / len(epoch_fitness)
        logging.info(
            f"[{name}] epoch {epoch:2d}  train={train_acc:.4f}  test={test_acc:.4f}"
            f"  |fitness|={mean_fit:.4f}  {time.perf_counter() - t0:.1f}s"
        )

    return train_acc, test_acc, time.perf_counter() - t0


# ── Experiments ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    experiments = [
        ("f3(4,4,4) mlp=128 ES sigma_rel=0.10 n_pop=64 cosine_lr",
         *make_f3_bilinear(4, 4, 4, 128)),
    ]

    for label, fwd, params, emb_keys in experiments:
        nonemb = n_nonemb(params, emb_keys)
        ntotal = n_total(params)
        logging.info(f"\n{'='*60}\n{label}\n"
                     f"  non-emb={nonemb:,}  total={ntotal:,}\n{'='*60}")
        tr_acc, te_acc, elapsed = train(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test, max_epochs=MAX_EPOCHS)
        row = {
            "name": label, "experiment": "exp2",
            "family": "es_f3",
            "n_nonemb_params": nonemb,
            "n_total_params":  ntotal,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": MAX_EPOCHS, "time_s": round(elapsed, 1),
            "es_sigma_rel": ES_SIGMA_REL,
            "es_n_pop":     ES_N_POP,
            "es_rank":      ES_RANK,
            "adam_lr_init": ADAM_LR_INIT,
            "adam_lr_end":  ADAM_LR_END,
            "notes": "adaptive sigma (10% rel), n_pop=64, cosine LR decay, noise30",
        }
        append_result(row)
        logging.info(f"  → train={tr_acc:.4f}  test={te_acc:.4f}")
