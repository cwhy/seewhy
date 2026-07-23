"""
exp16 — Meta-ES: evolve sigma alongside model params using the same antithetic mechanism.

Core idea:
  Standard ES: fixed sigma, perturb theta only.
  Meta-ES: sigma is a learnable scalar (stored as log_sigma).
    Each pair i draws its own sigma_i = exp(log_sigma + eps_log_sigma_i),
    then generates eps_theta using that sigma_i.
    The same fitness drives both the theta gradient and the log_sigma gradient.

Gradient derivation:
  fitness_i = f(theta + eps_i) - f(theta - eps_i),  eps_i ~ rank-k scaled by sigma_i
  grad_theta    = sum_i fitness_i * eps_i / (n_pop * rank * sigma_i^2)
  grad_log_sigma = sum_i fitness_i * eps_log_sigma_i / (n_pop * meta_tau^2)

Both go through the same Adam optimizer.

Signal for log_sigma: a larger sigma_i generates larger eps_i -> potentially larger |fitness_i|.
So fitness_i and eps_log_sigma_i are causally correlated -> real gradient signal.

Small-scale test (single GPU, 100 epochs):
  A: Meta-ES   (adaptive log_sigma, meta_tau=0.3, rank=8, n_pop=128)
  B: Standard ES (fixed sigma=0.001, rank=8, n_pop=128)   <- control

Key diagnostics:
  - sigma trace: does it self-organise? (should be large early, shrink later)
  - Does meta-ES escape warm-up faster or reach a higher plateau?

Usage:
    uv run python projects/es/experiments16.py
"""

import json, time, sys, os
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image

N_VAL_BINS   = 32
BATCH_SIZE   = 128
SEED         = 42
MAX_EPOCHS   = 100
ADAM_LR_INIT = 1e-3
ADAM_LR_END  = 3e-5

JSONL = Path(__file__).parent / "results_mnist.jsonl"

ROWS = jnp.arange(784) // 28
COLS = jnp.arange(784) % 28


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


# ── Standard ES (control) ──────────────────────────────────────────────────

def make_es_step(fwd, sigma, n_pop, tx, rank=8):
    def make_perturbation(pop_key, params):
        eps = {}
        for idx, (k, v) in enumerate(params.items()):
            subkey = jax.random.fold_in(pop_key, idx)
            if v.ndim == 2:
                m, n = v.shape
                ab   = jax.random.normal(subkey, (m + n, rank))
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
            lambda e: jnp.einsum("i,i...->...", fitnesses, e) / (n_pop * rank * sigma ** 2),
            epses,
        )
        updates, new_opt_state = tx.update(grad, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, jnp.mean(jnp.abs(fitnesses))

    return es_step


# ── Meta-ES: adaptive log_sigma ────────────────────────────────────────────

def make_meta_es_step(fwd, log_sigma_init, n_pop, tx, rank=8, meta_tau=0.3):
    """
    log_sigma is a scalar that's updated alongside model params.
    Each antithetic pair uses its own sigma_i = exp(log_sigma + eps_log_sigma_i).

    grad_theta     = sum_i fitness_i * eps_i   / (n_pop * rank * sigma_i^2)
    grad_log_sigma = sum_i fitness_i * z_i     / (n_pop * meta_tau^2)
    where z_i = eps_log_sigma_i / meta_tau  (the raw N(0,1) draw)
    """

    def make_perturbation(pop_key, params, sigma_i):
        eps = {}
        for idx, (k, v) in enumerate(params.items()):
            subkey = jax.random.fold_in(pop_key, idx)
            if v.ndim == 2:
                m, n = v.shape
                ab   = jax.random.normal(subkey, (m + n, rank))
                a, b = ab[:m], ab[m:]
                eps[k] = (a @ b.T) * sigma_i
            else:
                eps[k] = jax.random.normal(subkey, v.shape) * sigma_i
        return eps

    def pair_estimate(pop_key, params, log_sigma, bx, by):
        k_meta, k_theta = jax.random.split(pop_key)

        # Sample per-pair sigma
        z_i       = jax.random.normal(k_meta, ())          # N(0,1)
        sigma_i   = jnp.exp(log_sigma + meta_tau * z_i)

        # Generate theta perturbation with this pair's sigma
        eps_theta = make_perturbation(k_theta, params, sigma_i)

        p_pos = {k: v + eps_theta[k] for k, v in params.items()}
        p_neg = {k: v - eps_theta[k] for k, v in params.items()}
        ce    = lambda p: jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(fwd(p, bx), by))
        fitness = ce(p_pos) - ce(p_neg)

        return fitness, eps_theta, z_i, sigma_i

    @jax.jit
    def meta_es_step(params, log_sigma, opt_state, bx, by, pop_keys):
        fitnesses, epses, z_is, sigma_is = jax.vmap(
            pair_estimate, in_axes=(0, None, None, None, None)
        )(pop_keys, params, log_sigma, bx, by)

        # Theta gradient: each pair's eps was drawn with sigma_i, so normalise by sigma_i^2
        # eps_i = sigma_i * noise_i  =>  eps_i / sigma_i^2 = noise_i / sigma_i
        # Using einsum: (fitnesses / sigma_is^2) · epses, summed over pop dim
        scale_i = fitnesses / (sigma_is ** 2)          # shape (n_pop,)
        grad_theta = jax.tree_util.tree_map(
            lambda e: jnp.einsum("i,i...->...", scale_i, e) / (n_pop * rank),
            epses,
        )

        # log_sigma gradient: score-function estimator via the z draws
        grad_log_sigma = jnp.dot(fitnesses, z_is) / (n_pop * meta_tau)

        # Pack log_sigma as a single-element array so Adam can track its moment
        all_params = {**params, "__log_sigma": jnp.array([log_sigma])}
        all_grads  = {**grad_theta, "__log_sigma": jnp.array([grad_log_sigma])}

        updates, new_opt_state = tx.update(all_grads, opt_state, all_params)
        new_all     = optax.apply_updates(all_params, updates)

        new_params     = {k: v for k, v in new_all.items() if k != "__log_sigma"}
        new_log_sigma  = new_all["__log_sigma"][0]

        mean_abs_fit = jnp.mean(jnp.abs(fitnesses))
        mean_sigma   = jnp.mean(sigma_is)
        return new_params, new_log_sigma, new_opt_state, mean_abs_fit, mean_sigma

    return meta_es_step


# ── Training loops ─────────────────────────────────────────────────────────

def acc_fn(fwd, params, x, y):
    preds = jnp.concatenate([
        jnp.argmax(fwd(params, x[i:i + BATCH_SIZE]), 1)
        for i in range(0, len(x), BATCH_SIZE)])
    return float(jnp.mean(preds == y))


def train_standard_es(name, fwd, params, X_tr, y_tr, X_te, y_te,
                      sigma=0.001, n_pop=128, rank=8, max_epochs=MAX_EPOCHS):
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = max_epochs * num_batches
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT, total_steps,
                                               alpha=ADAM_LR_END / ADAM_LR_INIT)
    tx        = optax.adam(lr_sched)
    es_step   = make_es_step(fwd, sigma, n_pop, tx, rank=rank)
    opt_state = tx.init(params)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    sigma_trace = []

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
        elapsed   = time.perf_counter() - t0
        sigma_trace.append(sigma)
        logging.info(
            f"[{name}] ep {epoch:3d}  train={train_acc:.4f}  test={test_acc:.4f}"
            f"  |fit|={mean_fit:.6f}  sigma={sigma:.5f}  {elapsed:.1f}s"
        )

    return train_acc, test_acc, time.perf_counter() - t0, sigma_trace


def train_meta_es(name, fwd, params, X_tr, y_tr, X_te, y_te,
                  log_sigma_init, n_pop=128, rank=8, meta_tau=0.3,
                  max_epochs=MAX_EPOCHS):
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = max_epochs * num_batches
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT, total_steps,
                                               alpha=ADAM_LR_END / ADAM_LR_INIT)
    tx         = optax.adam(lr_sched)
    meta_step  = make_meta_es_step(fwd, log_sigma_init, n_pop, tx,
                                    rank=rank, meta_tau=meta_tau)
    log_sigma  = jnp.array(log_sigma_init)

    # Init opt_state with the augmented param dict (includes __log_sigma)
    all_params = {**params, "__log_sigma": jnp.array([log_sigma_init])}
    opt_state  = tx.init(all_params)

    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    sigma_trace = []

    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        key  = next(key_gen).get()
        perm = jax.random.permutation(key, len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        aug_rng = key
        epoch_fit   = []
        epoch_sigma = []

        for b in range(num_batches):
            s, e   = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx, by = Xs[s:e], ys[s:e]
            aug_rng, sub = jax.random.split(aug_rng)
            bx = random_noise_batch(bx, sub)
            pop_keys = jax.random.split(jax.random.fold_in(key, b), n_pop)
            params, log_sigma, opt_state, fit, mean_sigma = meta_step(
                params, log_sigma, opt_state, bx, by, pop_keys)
            epoch_fit.append(float(fit))
            epoch_sigma.append(float(mean_sigma))

        train_acc  = acc_fn(fwd, params, X_tr, y_tr)
        test_acc   = acc_fn(fwd, params, X_te, y_te)
        mean_fit   = sum(epoch_fit) / len(epoch_fit)
        cur_sigma  = float(jnp.exp(log_sigma))
        elapsed    = time.perf_counter() - t0
        sigma_trace.append(cur_sigma)
        logging.info(
            f"[{name}] ep {epoch:3d}  train={train_acc:.4f}  test={test_acc:.4f}"
            f"  |fit|={mean_fit:.6f}  sigma={cur_sigma:.5f}  {elapsed:.1f}s"
        )

    return train_acc, test_acc, time.perf_counter() - t0, sigma_trace


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info(f"JAX devices: {jax.devices()}")

    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    results = {}

    # ── A: Standard ES ──
    name_std = "standard-ES  sigma=0.001 rank=8"
    logging.info(f"\n{'='*60}\n{name_std}\n{'='*60}")
    fwd, params, emb_keys = make_f3_bilinear(4, 4, 4, 256)
    tr, te, elapsed, sigma_trace_std = train_standard_es(
        name_std, fwd, params, X_train, y_train, X_test, y_test,
        sigma=0.001, n_pop=128, rank=8, max_epochs=MAX_EPOCHS)
    results["standard"] = dict(train=tr, test=te, time=elapsed, sigma_trace=sigma_trace_std)
    logging.info(f"  → train={tr:.4f}  test={te:.4f}  {elapsed:.0f}s")

    # ── B: Meta-ES (same init sigma, adaptive) ──
    name_meta = "meta-ES      sigma_init=0.001 rank=8 meta_tau=0.3"
    logging.info(f"\n{'='*60}\n{name_meta}\n{'='*60}")
    fwd, params, emb_keys = make_f3_bilinear(4, 4, 4, 256)
    tr, te, elapsed, sigma_trace_meta = train_meta_es(
        name_meta, fwd, params, X_train, y_train, X_test, y_test,
        log_sigma_init=-6.9078,   # log(0.001)
        n_pop=128, rank=8, meta_tau=0.3, max_epochs=MAX_EPOCHS)
    results["meta"] = dict(train=tr, test=te, time=elapsed, sigma_trace=sigma_trace_meta)
    logging.info(f"  → train={tr:.4f}  test={te:.4f}  {elapsed:.0f}s")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"{'Variant':<45} {'Train':>7} {'Test':>7} {'Time':>7}")
    print(f"{'-'*70}")
    print(f"{'standard ES  (sigma=0.001, fixed)':<45} {results['standard']['train']:>7.4f}"
          f" {results['standard']['test']:>7.4f} {results['standard']['time']:>6.0f}s")
    print(f"{'meta-ES      (sigma adaptive)':<45} {results['meta']['train']:>7.4f}"
          f" {results['meta']['test']:>7.4f} {results['meta']['time']:>6.0f}s")
    print(f"{'='*70}")

    print("\nSigma trace (every 10 epochs):")
    print(f"  {'ep':>4}  {'standard':>10}  {'meta':>10}")
    for ep in range(0, MAX_EPOCHS, 10):
        s_std  = results["standard"]["sigma_trace"][ep]
        s_meta = results["meta"]["sigma_trace"][ep]
        print(f"  {ep:>4}  {s_std:>10.5f}  {s_meta:>10.5f}")

    # Save to JSONL
    for variant, r in results.items():
        append_result({
            "experiment": "exp16",
            "variant": variant,
            "name": name_std if variant == "standard" else name_meta,
            "train_acc": round(r["train"], 6),
            "test_acc":  round(r["test"],  6),
            "time_s":    round(r["time"],  1),
            "epochs":    MAX_EPOCHS,
            "sigma_final": round(r["sigma_trace"][-1], 6),
        })
