"""
Seed stability check for A-baseline (scale=0.02, standard ES).
Runs 5 seeds × 50 epochs to measure variance.
"""
import json, time, sys
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import logging
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image

N_VAL_BINS   = 32
BATCH_SIZE   = 128
MAX_EPOCHS   = 50
ADAM_LR_INIT = 1e-3
ADAM_LR_END  = 3e-5

ROWS = jnp.arange(784) // 28
COLS = jnp.arange(784) % 28

SEEDS = [42, 7, 123, 999, 2025]


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


def random_noise_batch(imgs, rng, scale=30.0):
    noise = jax.random.uniform(rng, imgs.shape, minval=-scale, maxval=scale)
    return jnp.clip(imgs + noise, 0.0, 255.0)


def make_f3_bilinear(d_r, d_c, d_v, mlp_dim, seed):
    d  = d_r * d_c * d_v
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(seed))
    p = {
        "row_emb":       jax.random.normal(next(kg).get(), (28,        d_r)) * 0.02,
        "col_emb":       jax.random.normal(next(kg).get(), (28,        d_c)) * 0.02,
        "val_emb":       jax.random.normal(next(kg).get(), (N_VAL_BINS, d_v)) * 0.02,
        "H_answer":      jax.random.normal(next(kg).get(), (1,          d))  * (1.0 / d**0.5),
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


def train_es(seed, fwd, params, X_tr, y_tr, X_te, y_te,
             sigma=0.001, n_pop=128, rank=8):
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = MAX_EPOCHS * num_batches
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT, total_steps,
                                              alpha=ADAM_LR_END / ADAM_LR_INIT)
    tx        = optax.adam(lr_sched)
    es_step   = make_es_step(fwd, sigma, n_pop, tx, rank=rank)
    opt_state = tx.init(params)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(seed + 1))

    test_per_epoch = []
    plateau_escape = None
    t0 = time.perf_counter()
    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = jax.random.permutation(key, len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        aug_rng = key
        for b in range(num_batches):
            s, e   = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx, by = Xs[s:e], ys[s:e]
            aug_rng, sub = jax.random.split(aug_rng)
            bx = random_noise_batch(bx, sub)
            pop_keys = jax.random.split(jax.random.fold_in(key, b), n_pop)
            params, opt_state, _ = es_step(params, opt_state, bx, by, pop_keys)

        te = acc_fn(fwd, params, X_te, y_te)
        test_per_epoch.append(te)
        if plateau_escape is None and te > 0.20:
            plateau_escape = epoch
        logging.info(f"  [seed={seed}] ep {epoch:3d}  test={te:.4f}  {time.perf_counter()-t0:.1f}s")

    return test_per_epoch, plateau_escape


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info(f"JAX devices: {jax.devices()}")

    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    results = []
    for seed in SEEDS:
        logging.info(f"\n{'='*50}\nSEED={seed}\n{'='*50}")
        fwd, params = make_f3_bilinear(4, 4, 4, 256, seed=seed)
        test_curve, plateau_ep = train_es(
            seed, fwd, params, X_train, y_train, X_test, y_test)
        final = test_curve[-1]
        best  = max(test_curve)
        results.append((seed, final, best, plateau_ep))
        logging.info(f"  seed={seed}  final={final:.4f}  best={best:.4f}  plateau=ep{plateau_ep}")

    finals = [r[1] for r in results]
    print(f"\n{'='*55}")
    print(f"{'Seed':>6}  {'Final@50':>9}  {'Best@50':>9}  {'Plateau':>8}")
    print(f"{'-'*55}")
    for seed, final, best, plateau_ep in results:
        print(f"{seed:>6}  {final:>9.4f}  {best:>9.4f}  ep{plateau_ep:>6}")
    print(f"{'-'*55}")
    print(f"{'mean':>6}  {np.mean(finals):>9.4f}  {'':>9}  {'':>8}")
    print(f"{'std':>6}  {np.std(finals):>9.4f}")
    print(f"{'min':>6}  {np.min(finals):>9.4f}")
    print(f"{'max':>6}  {np.max(finals):>9.4f}")
    print(f"{'='*55}")
