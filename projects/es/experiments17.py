"""
exp17 — Meta-ES-EA: evolve sigma AND rank per-individual using truncation selection + mutation.

Each antithetic pair carries its own (sigma_i, rank_i).
After each batch:
  1. Sort pairs by |fitness_i| descending.
  2. Keep top-half (sigma, rank) pairs (survivors).
  3. Generate new bottom half by independent log-normal mutation of each gene:
       sigma_child = sigma_parent * exp(tau_sigma * N(0,1))
       rank_child  = rank_parent  * exp(tau_rank  * N(0,1))   [then rounded, clamped]
  4. New population = survivors ++ children.

Theta gradient: sum_i fitness_i * eps_i / (n_pop * rank_i * sigma_i^2)

Variable rank in JAX (no recompilation):
  Always allocate MAX_RANK=16 columns for A, B. Mask out columns beyond rank_i:
    mask = (arange(MAX_RANK) < rank_i)  — dynamic, works under vmap

Initial population: log-uniform spread
  sigma: [3e-4, 3e-3]   rank: [1, 16]

Variants:
  A: Standard ES  (sigma=0.001, rank=8 fixed)  — control
  B: Meta-ES-EA   (sigma+rank population, EA)

Outputs (uploaded to R2 or saved locally):
  exp17b_accuracy.png   — train/test accuracy curves both variants
  exp17b_fitness.png    — |fitness| over epochs both variants
  exp17b_sigma_pop.png  — sigma population (median ± quartiles) vs reference σ=0.001
  exp17b_rank_pop.png   — rank  population (median ± quartiles) vs reference rank=8
  report17.md          — concise summary referencing the figures

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python projects/es/experiments17.py
"""

import json, time, sys, os
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
from shared_lib.media import save_matplotlib_figure, save_media

N_VAL_BINS   = 32
BATCH_SIZE   = 128
SEED         = 42
MAX_EPOCHS   = 100
ADAM_LR_INIT = 1e-3
ADAM_LR_END  = 3e-5
MAX_RANK     = 16

JSONL        = Path(__file__).parent / "results_mnist.jsonl"
REPORT_PATH  = Path(__file__).parent / "report17b.md"

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

def make_f3_bilinear(d_r, d_c, d_v, mlp_dim):
    d   = d_r * d_c * d_v
    kg  = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    p = {
        "row_emb":      jax.random.normal(next(kg).get(), (28, d_r))         * 0.02,
        "col_emb":      jax.random.normal(next(kg).get(), (28, d_c))         * 0.02,
        "val_emb":      jax.random.normal(next(kg).get(), (N_VAL_BINS, d_v)) * 0.02,
        "H_answer":     jax.random.normal(next(kg).get(), (1, d))            * 0.02,
        "ln_out_gamma": jnp.ones(d),
        "ln_out_beta":  jnp.zeros(d),
        "W1":           jax.random.normal(next(kg).get(), (mlp_dim, d)) / d**0.5,
        "b1":           jnp.zeros(mlp_dim),
        "W2":           jax.random.normal(next(kg).get(), (10, mlp_dim)) * 0.01,
        "b2":           jnp.zeros(10),
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


# ── Standard ES ───────────────────────────────────────────────────────────

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


# ── Meta-ES-EA ────────────────────────────────────────────────────────────
#
# Revised design: small candidate population (n_cands=8), each candidate gets
# group_size = n_pop // n_cands pairs. Selection uses GROUP-AVERAGED normalised
# fitness, so each candidate's quality is estimated from 16 pairs — reliable
# enough to distinguish good from bad sigma/rank even during the plateau.
#
# Layout of pop_keys:  [cand0_pair0 ... cand0_pair15 | cand1_pair0 ... | ...]
# sigma_per_pair:      repeat(sigma_cands, group_size)   shape (n_pop,)
# Selection criterion: mean(|fitness_group|) / (sigma_cand * sqrt(rank_cand))

def make_meta_ea_step(fwd, n_pop, tx,
                       n_cands=8,
                       tau_sigma=0.2, tau_rank=0.3,
                       sigma_min=1e-4, sigma_max=5e-2):
    assert n_pop % n_cands == 0
    group_size = n_pop // n_cands   # pairs per candidate (e.g. 16)
    half_cands = n_cands // 2

    def make_perturbation(pop_key, params, sigma_i, rank_i):
        col_mask = (jnp.arange(MAX_RANK) < rank_i).astype(jnp.float32)
        eps = {}
        for idx, (k, v) in enumerate(params.items()):
            subkey = jax.random.fold_in(pop_key, idx)
            if v.ndim == 2:
                m, n = v.shape
                ab = jax.random.normal(subkey, (m + n, MAX_RANK))
                a  = ab[:m] * col_mask[None, :]
                b  = ab[m:] * col_mask[None, :]
                eps[k] = (a @ b.T) * sigma_i
            else:
                eps[k] = jax.random.normal(subkey, v.shape) * sigma_i
        return eps

    def pair_estimate(pop_key, params, sigma_i, rank_i, bx, by):
        eps   = make_perturbation(pop_key, params, sigma_i, rank_i)
        p_pos = {k: v + eps[k] for k, v in params.items()}
        p_neg = {k: v - eps[k] for k, v in params.items()}
        ce    = lambda p: jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(fwd(p, bx), by))
        return ce(p_pos) - ce(p_neg), eps

    @jax.jit
    def theta_step(params, sigma_cands, rank_cands, opt_state, bx, by, pop_keys):
        """Inner step: update theta only. Returns per-group summed |fitness|."""
        sigma_per_pair = jnp.repeat(sigma_cands, group_size)
        rank_per_pair  = jnp.repeat(rank_cands,  group_size)

        fitnesses, epses = jax.vmap(
            pair_estimate, in_axes=(0, None, 0, 0, None, None)
        )(pop_keys, params, sigma_per_pair, rank_per_pair, bx, by)

        norm_i = rank_per_pair * sigma_per_pair ** 2
        grad   = jax.tree_util.tree_map(
            lambda e: jnp.einsum("i,i...->...", fitnesses / norm_i, e) / n_pop,
            epses,
        )
        updates, new_opt_state = tx.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Accumulate per-group |fitness| sum (to be averaged over batches in Python)
        group_fit_sum = jnp.sum(
            jnp.abs(fitnesses).reshape(n_cands, group_size), axis=1)   # (n_cands,)

        return new_params, new_opt_state, jnp.mean(jnp.abs(fitnesses)), group_fit_sum

    def ea_epoch_update(sigma_cands, rank_cands, group_fit_accum, n_batches, mut_key):
        """Outer step (once per epoch): EA selection + mutation on accumulated fitness."""
        group_fit  = group_fit_accum / (n_batches * group_size)          # mean per candidate
        norm_group = group_fit / (sigma_cands * jnp.sqrt(rank_cands))   # normalise

        order      = jnp.argsort(norm_group)[::-1]
        surv_sigma = sigma_cands[order[:half_cands]]
        surv_rank  = rank_cands[order[:half_cands]]

        k1, k2      = jax.random.split(mut_key)
        child_sigma = jnp.clip(
            surv_sigma * jnp.exp(tau_sigma * jax.random.normal(k1, (half_cands,))),
            sigma_min, sigma_max)
        child_rank  = jnp.clip(
            surv_rank * jnp.exp(tau_rank * jax.random.normal(k2, (half_cands,))),
            1.0, float(MAX_RANK))

        new_sigma_cands = jnp.concatenate([surv_sigma, child_sigma])
        new_rank_cands  = jnp.concatenate([surv_rank,  child_rank])
        return new_sigma_cands, new_rank_cands

    return theta_step, ea_epoch_update


# ── Training loops (return full history) ─────────────────────────────────

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


def train_meta_ea(name, fwd, params, X_tr, y_tr, X_te, y_te,
                   n_pop=128, tau_sigma=0.2, tau_rank=0.3,
                   sigma_lo=3e-4, sigma_hi=3e-3,
                   rank_lo=1.0, rank_hi=16.0,
                   max_epochs=MAX_EPOCHS):
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = max_epochs * num_batches
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT, total_steps,
                                               alpha=ADAM_LR_END / ADAM_LR_INIT)
    n_cands    = 8
    tx         = optax.adam(lr_sched)
    theta_step, ea_epoch_update = make_meta_ea_step(
        fwd, n_pop, tx, n_cands=n_cands,
        tau_sigma=tau_sigma, tau_rank=tau_rank)
    opt_state  = tx.init(params)

    # Candidate population (size n_cands=8): log-uniform spread
    sigma_cands = jnp.exp(jnp.linspace(jnp.log(sigma_lo), jnp.log(sigma_hi), n_cands))
    rank_cands  = jnp.exp(jnp.linspace(jnp.log(rank_lo),  jnp.log(rank_hi),  n_cands))

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    mut_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 99))

    history = {
        "train": [], "test": [], "fitness": [],
        "sigma_med": [], "sigma_q25": [], "sigma_q75": [], "sigma_mean": [],
        "rank_med":  [], "rank_q25":  [], "rank_q75":  [], "rank_mean":  [],
    }
    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        key  = next(key_gen).get()
        perm = jax.random.permutation(key, len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        aug_rng = key
        epoch_fit = []
        group_fit_accum = np.zeros(n_cands)
        for b in range(num_batches):
            s, e   = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx, by = Xs[s:e], ys[s:e]
            aug_rng, sub = jax.random.split(aug_rng)
            bx = random_noise_batch(bx, sub)
            pop_keys = jax.random.split(jax.random.fold_in(key, b), n_pop)
            params, opt_state, fit, gf = theta_step(
                params, sigma_cands, rank_cands, opt_state, bx, by, pop_keys)
            epoch_fit.append(float(fit))
            group_fit_accum += np.array(gf)

        # EA sigma update once per epoch using accumulated fitness
        mut_key = next(mut_gen).get()
        sigma_cands, rank_cands = ea_epoch_update(
            sigma_cands, rank_cands,
            jnp.array(group_fit_accum), num_batches, mut_key)

        train_acc = acc_fn(fwd, params, X_tr, y_tr)
        test_acc  = acc_fn(fwd, params, X_te, y_te)
        mean_fit  = sum(epoch_fit) / len(epoch_fit)

        s_np = np.array(sigma_cands);  r_np = np.array(rank_cands)
        history["train"].append(train_acc);   history["test"].append(test_acc)
        history["fitness"].append(mean_fit)
        history["sigma_med"].append(float(np.median(s_np)))
        history["sigma_q25"].append(float(np.percentile(s_np, 25)))
        history["sigma_q75"].append(float(np.percentile(s_np, 75)))
        history["sigma_mean"].append(float(np.mean(s_np)))
        history["rank_med"].append(float(np.median(r_np)))
        history["rank_q25"].append(float(np.percentile(r_np, 25)))
        history["rank_q75"].append(float(np.percentile(r_np, 75)))
        history["rank_mean"].append(float(np.mean(r_np)))

        logging.info(
            f"[{name}] ep {epoch:3d}  train={train_acc:.4f}  test={test_acc:.4f}"
            f"  |fit|={mean_fit:.6f}"
            f"  σ med={history['sigma_med'][-1]:.5f}"
            f"  rank med={history['rank_med'][-1]:.1f}"
            f"  {time.perf_counter()-t0:.1f}s"
        )

    return (train_acc, test_acc, time.perf_counter() - t0, history,
            float(np.median(s_np)), float(np.median(r_np)))


# ── Plotting ──────────────────────────────────────────────────────────────

def make_plots(hist_std, hist_meta, tr_std, te_std, tr_meta, te_meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(MAX_EPOCHS))
    urls   = {}

    # ── Fig 1: Accuracy curves ──
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [v * 100 for v in hist_std["train"]],
            color="steelblue", linestyle="--", alpha=0.7, label="Standard ES — train")
    ax.plot(epochs, [v * 100 for v in hist_std["test"]],
            color="steelblue", linewidth=2, label="Standard ES — test")
    ax.plot(epochs, [v * 100 for v in hist_meta["train"]],
            color="darkorange", linestyle="--", alpha=0.7, label="Meta-ES-EA — train")
    ax.plot(epochs, [v * 100 for v in hist_meta["test"]],
            color="darkorange", linewidth=2, label="Meta-ES-EA — test")
    ax.axhline(te_std  * 100, color="steelblue",  linestyle=":", alpha=0.5)
    ax.axhline(te_meta * 100, color="darkorange", linestyle=":", alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("exp17 — Accuracy: Standard ES vs Meta-ES-EA")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    urls["accuracy"] = save_matplotlib_figure("exp17b_accuracy", fig)
    plt.close(fig)

    # ── Fig 2: |fitness| ──
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, hist_std["fitness"],  color="steelblue",  label="Standard ES")
    ax.plot(epochs, hist_meta["fitness"], color="darkorange", label="Meta-ES-EA")
    ax.set_xlabel("Epoch"); ax.set_ylabel("|fitness|")
    ax.set_title("exp17 — Mean |fitness| per epoch")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    urls["fitness"] = save_matplotlib_figure("exp17b_fitness", fig)
    plt.close(fig)

    # ── Fig 3: Sigma population trend ──
    fig, ax = plt.subplots(figsize=(9, 4))
    s_med = np.array(hist_meta["sigma_med"])
    s_q25 = np.array(hist_meta["sigma_q25"])
    s_q75 = np.array(hist_meta["sigma_q75"])
    ax.fill_between(epochs, s_q25, s_q75, alpha=0.25, color="darkorange", label="IQR (25–75%)")
    ax.plot(epochs, s_med,  color="darkorange", linewidth=2, label="Median σ")
    ax.plot(epochs, hist_meta["sigma_mean"], color="darkorange", linestyle="--",
            alpha=0.6, label="Mean σ")
    ax.axhline(0.001, color="steelblue", linewidth=1.5, linestyle="--",
               label="Reference σ=0.001 (best fixed)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("σ (mutation scale)")
    ax.set_title("exp17 — Sigma population evolution (Meta-ES-EA)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    urls["sigma"] = save_matplotlib_figure("exp17b_sigma_pop", fig)
    plt.close(fig)

    # ── Fig 4: Rank population trend ──
    fig, ax = plt.subplots(figsize=(9, 4))
    r_med = np.array(hist_meta["rank_med"])
    r_q25 = np.array(hist_meta["rank_q25"])
    r_q75 = np.array(hist_meta["rank_q75"])
    ax.fill_between(epochs, r_q25, r_q75, alpha=0.25, color="purple", label="IQR (25–75%)")
    ax.plot(epochs, r_med,  color="purple", linewidth=2, label="Median rank")
    ax.plot(epochs, hist_meta["rank_mean"], color="purple", linestyle="--",
            alpha=0.6, label="Mean rank")
    ax.axhline(8.0, color="steelblue", linewidth=1.5, linestyle="--",
               label="Reference rank=8 (warm-up)")
    ax.axhline(2.0, color="teal", linewidth=1.5, linestyle=":",
               label="Reference rank=2 (convergence)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Rank")
    ax.set_title("exp17 — Rank population evolution (Meta-ES-EA)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    urls["rank"] = save_matplotlib_figure("exp17b_rank_pop", fig)
    plt.close(fig)

    return urls


# ── Report ────────────────────────────────────────────────────────────────

def write_report(urls, tr_std, te_std, tr_meta, te_meta,
                 t_std, t_meta, sigma_final, rank_final):
    date_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# exp17 — Meta-ES-EA Report",
        f"",
        f"**Date:** {date_str}  **Epochs:** {MAX_EPOCHS}  **n_pop:** 128",
        f"",
        f"## Results",
        f"",
        f"| Variant | Train | Test | Time |",
        f"|---|---|---|---|",
        f"| Standard ES (σ=0.001, rank=8, fixed) | {tr_std*100:.2f}% | {te_std*100:.2f}% | {t_std:.0f}s |",
        f"| Meta-ES-EA (σ+rank population, EA)   | {tr_meta*100:.2f}% | {te_meta*100:.2f}% | {t_meta:.0f}s |",
        f"",
        f"Final gene pool — σ median: `{sigma_final:.5f}` (ref: 0.001)  "
        f"rank median: `{rank_final:.1f}` (ref: 8→2)",
        f"",
        f"## Accuracy curves",
        f"",
        f"![accuracy]({urls['accuracy']})",
        f"",
        f"## |fitness| over time",
        f"",
        f"![fitness]({urls['fitness']})",
        f"",
        f"## Sigma population vs reference σ=0.001",
        f"",
        f"Shaded band = IQR (25–75%). Dashed blue = best known fixed sigma.",
        f"",
        f"![sigma population]({urls['sigma']})",
        f"",
        f"## Rank population vs reference ranks",
        f"",
        f"Shaded band = IQR (25–75%). Dashed blue = rank=8 (warm-up ref), dotted teal = rank=2 (convergence ref).",
        f"",
        f"![rank population]({urls['rank']})",
    ]
    REPORT_PATH.write_text("\n".join(lines))
    logging.info(f"Report written to {REPORT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info(f"JAX devices: {jax.devices()}")

    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    # ── A: Standard ES ──
    name_a = "standard-ES  σ=0.001 rank=8"
    logging.info(f"\n{'='*60}\n{name_a}\n{'='*60}")
    fwd, params = make_f3_bilinear(4, 4, 4, 256)
    tr_a, te_a, t_a, hist_a = train_standard_es(
        name_a, fwd, params, X_train, y_train, X_test, y_test,
        sigma=0.001, n_pop=128, rank=8)
    logging.info(f"  → train={tr_a:.4f}  test={te_a:.4f}  {t_a:.0f}s")
    append_result({"experiment": "exp17", "variant": "standard",
                   "train_acc": round(tr_a, 6), "test_acc": round(te_a, 6),
                   "time_s": round(t_a, 1), "epochs": MAX_EPOCHS})

    # ── B: Meta-ES-EA ──
    name_b = "meta-ES-EA   σ=[3e-4,3e-3] rank=[1,16]"
    logging.info(f"\n{'='*60}\n{name_b}\n{'='*60}")
    fwd, params = make_f3_bilinear(4, 4, 4, 256)
    tr_b, te_b, t_b, hist_b, sigma_f, rank_f = train_meta_ea(
        name_b, fwd, params, X_train, y_train, X_test, y_test,
        n_pop=128, tau_sigma=0.2, tau_rank=0.3,
        sigma_lo=3e-4, sigma_hi=3e-3, rank_lo=1.0, rank_hi=16.0)
    logging.info(f"  → train={tr_b:.4f}  test={te_b:.4f}  {t_b:.0f}s"
                 f"  σ_med={sigma_f:.5f}  rank_med={rank_f:.1f}")
    append_result({"experiment": "exp17", "variant": "meta-ea",
                   "train_acc": round(tr_b, 6), "test_acc": round(te_b, 6),
                   "time_s": round(t_b, 1), "epochs": MAX_EPOCHS,
                   "sigma_final": round(sigma_f, 6), "rank_final": round(rank_f, 1)})

    # ── Plots + report ──
    logging.info("Generating plots and uploading...")
    urls = make_plots(hist_a, hist_b, tr_a, te_a, tr_b, te_b)
    write_report(urls, tr_a, te_a, tr_b, te_b, t_a, t_b, sigma_f, rank_f)

    print(f"\n{'='*75}")
    print(f"{'Variant':<50} {'Train':>7} {'Test':>7} {'Time':>7}")
    print(f"{'-'*75}")
    print(f"{'standard ES  (σ=0.001, rank=8, fixed)':<50} {tr_a:>7.4f} {te_a:>7.4f} {t_a:>6.0f}s")
    print(f"{'meta-ES-EA   (σ+rank population, EA)':<50} {tr_b:>7.4f} {te_b:>7.4f} {t_b:>6.0f}s")
    print(f"{'='*75}")
    print(f"\nReport: {REPORT_PATH}")
    print(f"Figures:")
    for k, u in urls.items():
        print(f"  {k}: {u}")
