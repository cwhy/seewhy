"""
exp22 — Single-sided ES (no antithetic pairs).

Motivation: antithetic pairs collapse CE_+ and CE_- into a single fitness
scalar, destroying the individual CE values that DG needs as surprisal.
Without pairs, each sample has its own CE_i = -log π(y|x, params+eps_i),
which IS the action surprisal directly — enabling clean DG application.

Gradient estimator (single-sided):
  fitness_i = CE_i - baseline    where baseline = mean(CE_j)
  grad = Σ_i fitness_i * eps_i / (n_pop * rank * sigma²)

DG extension (single-sided):
  advantage  U_i = baseline - CE_i   (positive = this sample was better)
  surprisal  ℓ_i = CE_i              (model's own uncertainty)
  delight    χ_i = U_i × ℓ_i = (baseline - CE_i) × CE_i
  gate       w_i = sigmoid(χ_i / η)
  grad = Σ_i gate_i * fitness_i * eps_i / (n_pop * rank * sigma²)

Equal compute: antithetic n_pop=128 (256 evals) vs single-sided n_pop=256 (256 evals).

Variants (rank schedule 8→2 at ep12, 100 epochs):
  A: Antithetic ES  n_pop=128  (256 evals / batch) — control
  B: Single-sided   n_pop=256  (256 evals / batch)
  C: Single-sided + DG  η=0.01   n_pop=256
  D: Single-sided + DG  η=auto   (η = std of delight across population)

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python projects/es/experiments22.py
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
MAX_EPOCHS   = 100
ADAM_LR_INIT = 1e-3
ADAM_LR_END  = 3e-5
RANK_SWITCH_EP = 12

JSONL       = Path(__file__).parent / "results_mnist.jsonl"
REPORT_PATH = Path(__file__).parent / "report22.md"

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
    d  = d_r * d_c * d_v
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
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


def _make_perturbation(pop_key, params, sigma, rank):
    eps = {}
    for idx, (k, v) in enumerate(params.items()):
        subkey = jax.random.fold_in(pop_key, idx)
        if v.ndim == 2:
            m, n = v.shape
            ab = jax.random.normal(subkey, (m + n, rank))
            eps[k] = (ab[:m] @ ab[m:].T) * sigma
        else:
            eps[k] = jax.random.normal(subkey, v.shape) * sigma
    return eps


# ── A: Antithetic ES ───────────────────────────────────────────────────────

def make_antithetic_step(fwd, sigma, n_pop, tx, rank):
    def pair_estimate(pop_key, params, bx, by):
        eps = _make_perturbation(pop_key, params, sigma, rank)
        ce  = lambda p: jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(fwd(p, bx), by))
        return ce({k: v + eps[k] for k, v in params.items()}) \
             - ce({k: v - eps[k] for k, v in params.items()}), eps

    @jax.jit
    def step(params, opt_state, bx, by, pop_keys):
        fitnesses, epses = jax.vmap(
            pair_estimate, in_axes=(0, None, None, None)
        )(pop_keys, params, bx, by)
        grad = jax.tree_util.tree_map(
            lambda e: jnp.einsum("i,i...->...", fitnesses, e) / (n_pop * rank * sigma**2),
            epses)
        updates, new_opt_state = tx.update(grad, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, jnp.mean(jnp.abs(fitnesses))

    return step


# ── B & C & D: Single-sided ES ─────────────────────────────────────────────

def make_single_sided_step(fwd, sigma, n_pop, tx, rank, dg_eta=None):
    """
    Single-sided ES. Each sample i:
      CE_i = CE(params + eps_i)
      fitness_i = CE_i - mean(CE_j)   [baseline = population mean]

    If dg_eta is not None: apply DG gating.
      delight_i = (baseline - CE_i) * CE_i
      gate_i    = sigmoid(delight_i / eta)   if eta > 0
                  sigmoid(delight_i / std(delight))  if eta == 'auto'
    """
    def sample_estimate(pop_key, params, bx, by):
        eps  = _make_perturbation(pop_key, params, sigma, rank)
        ce_i = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                fwd({k: v + eps[k] for k, v in params.items()}, bx), by))
        return ce_i, eps

    @jax.jit
    def step(params, opt_state, bx, by, pop_keys):
        ce_vals, epses = jax.vmap(
            sample_estimate, in_axes=(0, None, None, None)
        )(pop_keys, params, bx, by)

        baseline  = jnp.mean(ce_vals)
        fitnesses = ce_vals - baseline      # positive = worse than average

        if dg_eta is None:
            # Standard single-sided
            weighted = fitnesses
            mean_gate = jnp.array(0.5)
        else:
            # DG: advantage = baseline - CE_i (positive = better than average)
            #     surprisal  = CE_i
            #     delight    = advantage * surprisal = (baseline - CE_i) * CE_i
            delight = (baseline - ce_vals) * ce_vals   # = -fitnesses * ce_vals
            if dg_eta == "auto":
                eta_val = jnp.std(delight) + 1e-8
            else:
                eta_val = dg_eta
            gates    = jax.nn.sigmoid(delight / eta_val)
            weighted = fitnesses * gates
            mean_gate = jnp.mean(gates)

        grad = jax.tree_util.tree_map(
            lambda e: jnp.einsum("i,i...->...", weighted, e) / (n_pop * rank * sigma**2),
            epses)
        updates, new_opt_state = tx.update(grad, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt_state,
                jnp.mean(jnp.abs(fitnesses)), mean_gate)

    return step


# ── Training loops ─────────────────────────────────────────────────────────

def train(name, fwd, params, X_tr, y_tr, X_te, y_te,
          step_fn_hi, step_fn_lo, opt_state, key_gen, has_gate=False):
    num_batches = len(X_tr) // BATCH_SIZE
    history = {"train": [], "test": [], "fitness": [], "gate": []}
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        step = step_fn_hi if epoch < RANK_SWITCH_EP else step_fn_lo
        key  = next(key_gen).get()
        perm = jax.random.permutation(key, len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        aug_rng = key
        epoch_fit, epoch_gate = [], []

        for b in range(num_batches):
            s, e   = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx, by = Xs[s:e], ys[s:e]
            aug_rng, sub = jax.random.split(aug_rng)
            bx = random_noise_batch(bx, sub)
            pop_keys = jax.random.split(jax.random.fold_in(key, b), step_fn_hi.__self_n_pop
                                        if hasattr(step_fn_hi, '__self_n_pop') else 1)
            # unpack based on return signature
            out = step(params, opt_state, bx, by, pop_keys)
            params, opt_state = out[0], out[1]
            epoch_fit.append(float(out[2]))
            if len(out) > 3:
                epoch_gate.append(float(out[3]))

        tr = acc_fn(fwd, params, X_tr, y_tr)
        te = acc_fn(fwd, params, X_te, y_te)
        mf = sum(epoch_fit) / len(epoch_fit)
        mg = sum(epoch_gate) / len(epoch_gate) if epoch_gate else 0.5
        history["train"].append(tr); history["test"].append(te)
        history["fitness"].append(mf); history["gate"].append(mg)
        rank_now = 8 if epoch < RANK_SWITCH_EP else 2
        gate_str = f"  gate={mg:.3f}" if epoch_gate else ""
        logging.info(f"[{name}] ep {epoch:3d}  train={tr:.4f}  test={te:.4f}"
                     f"  |fit|={mf:.6f}{gate_str}  rank={rank_now}  {time.perf_counter()-t0:.1f}s")

    return tr, te, time.perf_counter() - t0, history, params, opt_state


def build_and_train(name, fwd, params, X_tr, y_tr, X_te, y_te,
                    n_pop, sigma=0.001, dg_eta=None, is_antithetic=False):
    num_batches = len(X_tr) // BATCH_SIZE
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT,
                                              MAX_EPOCHS * num_batches,
                                              alpha=ADAM_LR_END / ADAM_LR_INIT)
    tx        = optax.adam(lr_sched)
    opt_state = tx.init(params)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    history   = {"train": [], "test": [], "fitness": [], "gate": []}
    t0 = time.perf_counter()

    if is_antithetic:
        step_hi = make_antithetic_step(fwd, sigma, n_pop, tx, rank=8)
        step_lo = make_antithetic_step(fwd, sigma, n_pop, tx, rank=2)
    else:
        step_hi = make_single_sided_step(fwd, sigma, n_pop, tx, rank=8, dg_eta=dg_eta)
        step_lo = make_single_sided_step(fwd, sigma, n_pop, tx, rank=2, dg_eta=dg_eta)

    for epoch in range(MAX_EPOCHS):
        step = step_hi if epoch < RANK_SWITCH_EP else step_lo
        key  = next(key_gen).get()
        perm = jax.random.permutation(key, len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        aug_rng = key
        epoch_fit, epoch_gate = [], []

        for b in range(num_batches):
            s, e   = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx, by = Xs[s:e], ys[s:e]
            aug_rng, sub = jax.random.split(aug_rng)
            bx = random_noise_batch(bx, sub)
            pop_keys = jax.random.split(jax.random.fold_in(key, b), n_pop)
            out = step(params, opt_state, bx, by, pop_keys)
            params, opt_state = out[0], out[1]
            epoch_fit.append(float(out[2]))
            if len(out) > 3:
                epoch_gate.append(float(out[3]))

        tr = acc_fn(fwd, params, X_tr, y_tr)
        te = acc_fn(fwd, params, X_te, y_te)
        mf = sum(epoch_fit) / len(epoch_fit)
        mg = (sum(epoch_gate) / len(epoch_gate)) if epoch_gate else 0.5
        history["train"].append(tr); history["test"].append(te)
        history["fitness"].append(mf); history["gate"].append(mg)
        rank_now = 8 if epoch < RANK_SWITCH_EP else 2
        gate_str = f"  gate={mg:.3f}" if epoch_gate else ""
        logging.info(f"[{name}] ep {epoch:3d}  train={tr:.4f}  test={te:.4f}"
                     f"  |fit|={mf:.6f}{gate_str}  rank={rank_now}  {time.perf_counter()-t0:.1f}s")

    return tr, te, time.perf_counter() - t0, history


# ── Plots ──────────────────────────────────────────────────────────────────

def make_plots(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(MAX_EPOCHS))
    urls   = {}

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, hist in results:
        ax.plot(epochs, [v*100 for v in hist["train"]], color=color, linestyle="--", alpha=0.4)
        ax.plot(epochs, [v*100 for v in hist["test"]],  color=color, linewidth=2, label=label)
    ax.axvline(RANK_SWITCH_EP, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("exp22 — Single-sided ES vs Antithetic ES (100 epochs, rank 8→2)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    urls["accuracy"] = save_matplotlib_figure("exp22_accuracy", fig)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, color, hist in results:
        axes[0].plot(epochs, hist["fitness"], color=color, linewidth=2, label=label)
        axes[1].plot(epochs, hist["gate"],    color=color, linewidth=2, label=label)
    axes[0].set_title("|fitness| per epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("mean gate (0.5=neutral)"); axes[1].axhline(0.5, color="gray", linestyle=":")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    urls["diagnostics"] = save_matplotlib_figure("exp22_diagnostics", fig)
    plt.close(fig)

    return urls


def write_report(urls, rows):
    date_str = datetime.now().strftime("%Y-%m-%d")
    table = "\n".join(
        f"| {v} | {cfg} | {n} | {tr*100:.2f}% | {te*100:.2f}% | {t:.0f}s |"
        for v, cfg, n, tr, te, t in rows)
    lines = [
        "# exp22 — Single-sided ES Report",
        "",
        f"**Date:** {date_str}  **Epochs:** {MAX_EPOCHS}  **σ:** 0.001  "
        f"**rank schedule:** 8→2 at ep{RANK_SWITCH_EP}",
        "",
        "## Hypothesis",
        "",
        "Antithetic pairs destroy individual CE values needed for DG.",
        "Single-sided ES with n_pop=256 has same compute as antithetic n_pop=128.",
        "With single-sided, ℓ_i = CE_i is directly available as action surprisal.",
        "",
        "## Results",
        "",
        "| Variant | Config | n_pop (evals/batch) | Train | Test | Time |",
        "|---|---|---|---|---|---|",
        table,
        "",
        "## Accuracy curves",
        "",
        f"![accuracy]({urls['accuracy']})",
        "",
        "## Diagnostics",
        "",
        f"![diagnostics]({urls['diagnostics']})",
    ]
    REPORT_PATH.write_text("\n".join(lines))
    logging.info(f"Report written to {REPORT_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info(f"JAX devices: {jax.devices()}")

    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    variants = [
        # (name, n_pop, dg_eta, is_antithetic, color)
        ("A-antithetic     n=128 (256 evals)", 128,  None,   True,   "steelblue"),
        ("B-single-sided   n=256 (256 evals)", 256,  None,   False,  "darkorange"),
        ("C-single+DG η=0.01  n=256",          256,  0.01,   False,  "green"),
        ("D-single+DG η=auto  n=256",          256,  "auto", False,  "purple"),
    ]

    all_results, plot_data = [], []

    for vname, n_pop, dg_eta, is_anti, color in variants:
        logging.info(f"\n{'='*60}\n{vname}\n{'='*60}")
        fwd, params = make_f3_bilinear(4, 4, 4, 256)
        cfg = "antithetic" if is_anti else f"single-sided DG={dg_eta}" if dg_eta else "single-sided"
        tr, te, t_s, hist = build_and_train(
            vname, fwd, params, X_train, y_train, X_test, y_test,
            n_pop=n_pop, dg_eta=dg_eta, is_antithetic=is_anti)
        logging.info(f"  → train={tr:.4f}  test={te:.4f}  {t_s:.0f}s")
        evals = n_pop if not is_anti else n_pop * 2
        all_results.append((vname, cfg, f"{n_pop} ({evals})", tr, te, t_s))
        plot_data.append((vname, color, hist))
        append_result({"experiment": "exp22", "variant": vname, "n_pop": n_pop,
                       "dg_eta": str(dg_eta), "antithetic": is_anti,
                       "train_acc": round(tr,6), "test_acc": round(te,6),
                       "time_s": round(t_s,1), "epochs": MAX_EPOCHS})

    logging.info("Generating plots and uploading...")
    urls = make_plots(plot_data)
    write_report(urls, all_results)

    print(f"\n{'='*75}")
    print(f"{'Variant':<38} {'n_pop':>8} {'Test':>7} {'Time':>7}")
    print(f"{'-'*75}")
    for v, cfg, n, tr, te, t_s in all_results:
        print(f"{v:<38} {n:>8} {te:>7.4f} {t_s:>6.0f}s")
    print(f"{'='*75}")
    for k, u in urls.items():
        print(f"  {k}: {u}")
