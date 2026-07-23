"""
exp14 — pure GD ceiling vs GD+ES with fine sigma.

exp13 results (breakthrough!):
  3 GD + 22 ES: 95.57% test (306s)  [prev best ES-only: 94.02% in 28 min]
  5 GD + 20 ES: 96.21% test (289s)  ← new all-time record

Key observation: GD peak at epoch 4 was 96.80%, but ES final is 96.21%.
  The ES phase (sigma=0.001, rank=8, std=0.00283) is slightly degrading the GD result.
  Two possible explanations:
    1. Sigma too large for fine-tuning near a well-converged GD optimum
    2. LR schedule designed for 25 ES epochs not optimal for 20 ES epochs starting mid-run

exp14 questions:
  GPU0: Pure GD for 25 epochs
    → What is the GD ceiling? Does it plateau or keep improving past 96.8%?
    → GD can do exact gradient descent with much finer steps than ES perturbations.
    → Expected: >97% by epoch 25 (each GD epoch ≈ 3.8s → 25 epochs ≈ 95s = 1.6 min!)

  GPU1: 5 GD + 20 ES with sigma=0.0003 (10× smaller than exp13)
    → std(eps) = 0.0003 * sqrt(8) = 0.000849 (vs 0.00283 in exp13)
    → Much smaller perturbations near the GD optimum → less destabilization
    → Hypothesis: ES can refine the GD-trained model without the noisy regression

Reference: https://eshyperscale.github.io/
Usage:
    uv run python projects/es/experiments14.py
"""

import json, time, sys, os, subprocess
from pathlib import Path

if __name__ == "__main__" and "--gpu" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp
import optax
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image

N_VAL_BINS  = 32
BATCH_SIZE  = 128
SEED        = 42
MAX_EPOCHS  = 25
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


def make_gd_step(fwd, tx):
    @jax.jit
    def gd_step(params, opt_state, bx, by):
        def loss_fn(p):
            return jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(fwd(p, bx), by))
        loss, grad = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = tx.update(grad, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss
    return gd_step


def make_es_step(fwd, sigma, n_pop, tx, rank=1):
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


def train_gd_then_es(name, fwd, params, emb_keys, X_tr, y_tr, X_te, y_te,
                     sigma=0.001, n_pop=128, rank=8,
                     gd_epochs=0, max_epochs=MAX_EPOCHS):
    """Train gd_epochs with GD, then (max_epochs - gd_epochs) with ES.
    If gd_epochs == max_epochs, runs pure GD."""
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = max_epochs * num_batches
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT, total_steps,
                                               alpha=ADAM_LR_END / ADAM_LR_INIT)
    tx          = optax.adam(lr_sched)
    gd_step     = make_gd_step(fwd, tx)
    es_step_fn  = None if gd_epochs >= max_epochs else make_es_step(fwd, sigma, n_pop, tx, rank=rank)
    opt_state   = tx.init(params)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    def acc_fn(p, x, y):
        preds = jnp.concatenate([
            jnp.argmax(fwd(p, x[i:i + BATCH_SIZE]), 1)
            for i in range(0, len(x), BATCH_SIZE)])
        return float(jnp.mean(preds == y))

    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        mode = "GD" if epoch < gd_epochs else "ES"
        key  = next(key_gen).get()
        perm = jax.random.permutation(key, len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        aug_rng = key
        epoch_metric = []

        for b in range(num_batches):
            s, e   = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            bx, by = Xs[s:e], ys[s:e]
            aug_rng, sub = jax.random.split(aug_rng)
            bx = random_noise_batch(bx, sub)

            if epoch < gd_epochs:
                params, opt_state, loss = gd_step(params, opt_state, bx, by)
                epoch_metric.append(float(loss))
            else:
                pop_keys = jax.random.split(jax.random.fold_in(key, b), n_pop)
                params, opt_state, mean_fit = es_step_fn(params, opt_state, bx, by, pop_keys)
                epoch_metric.append(float(mean_fit))

        train_acc = acc_fn(params, X_tr, y_tr)
        test_acc  = acc_fn(params, X_te, y_te)
        mean_metric = sum(epoch_metric) / len(epoch_metric)
        elapsed     = time.perf_counter() - t0
        metric_label = "loss" if mode == "GD" else "|fitness|"
        logging.info(
            f"[{name}] epoch {epoch:3d} [{mode}]  train={train_acc:.4f}  test={test_acc:.4f}"
            f"  {metric_label}={mean_metric:.6f}  {elapsed:.1f}s"
        )

    return train_acc, test_acc, time.perf_counter() - t0


EXPERIMENTS = [
    # GPU 0: pure GD for 25 epochs — what is the GD ceiling for this architecture?
    # GD epochs ≈ 3.8s each → 25 epochs ≈ 95s = 1.6 min. Very fast!
    # After epoch 0: ~90.8%, epoch 4: ~96.8%, epoch 24: expected >97%
    dict(label="f3(4,4,4) mlp=256 25ep pure-GD",
         mlp=256, sigma=None, n_pop=None, rank=None,
         gd_epochs=25, gpu=0),
    # GPU 1: 5 GD + 20 ES with sigma=0.0003 (10× smaller than exp13's sigma=0.003)
    # std(eps) = 0.0003 * sqrt(8) = 0.000849 — finer perturbations near GD optimum
    # Hypothesis: smaller sigma allows ES to explore without destabilizing the GD solution
    dict(label="f3(4,4,4) mlp=256 rank=8 n_pop=128 25ep gd5+es20-sigma0.0003",
         mlp=256, sigma=0.0003, n_pop=128, rank=8,
         gd_epochs=5, gpu=1),
]


def run_single(gpu_id):
    logging.basicConfig(level=logging.INFO,
                        format=f"[GPU{gpu_id}] %(asctime)s %(levelname)s %(message)s")
    logging.info(f"JAX devices: {jax.devices()}")
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    for cfg in EXPERIMENTS:
        if cfg["gpu"] != gpu_id:
            continue
        label = cfg["label"]
        fwd, params, emb_keys = make_f3_bilinear(4, 4, 4, cfg["mlp"])[:]
        nonemb = n_nonemb(params, emb_keys)
        ntotal = n_total(params)
        logging.info(f"\n{'='*60}\n{label}\n"
                     f"  non-emb={nonemb:,}  total={ntotal:,}\n{'='*60}")
        tr_acc, te_acc, elapsed = train_gd_then_es(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test,
            sigma=cfg.get("sigma"), n_pop=cfg.get("n_pop"), rank=cfg.get("rank"),
            gd_epochs=cfg["gd_epochs"], max_epochs=MAX_EPOCHS)
        row = {
            "name": label, "experiment": "exp14",
            "family": "es_f3_gd_ceiling",
            "n_nonemb_params": nonemb, "n_total_params": ntotal,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": MAX_EPOCHS, "time_s": round(elapsed, 1),
            "batch_size": BATCH_SIZE,
            "es_sigma": cfg.get("sigma"), "es_n_pop": cfg.get("n_pop"),
            "es_rank": cfg.get("rank"),
            "gd_epochs": cfg["gd_epochs"],
            "adam_lr_init": ADAM_LR_INIT, "adam_lr_end": ADAM_LR_END,
            "notes": f"gd_epochs={cfg['gd_epochs']} sigma={cfg.get('sigma')}",
        }
        append_result(row)
        logging.info(f"  → train={tr_acc:.4f}  test={te_acc:.4f}  {elapsed:.0f}s")


if __name__ == "__main__":
    if "--gpu" in sys.argv:
        gpu_id = int(sys.argv[sys.argv.index("--gpu") + 1])
        run_single(gpu_id)
    else:
        script = str(Path(__file__).resolve())
        procs  = []
        for gpu_id in [0, 1]:
            log_path = f"/tmp/exp14_gpu{gpu_id}.log"
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            cmd = [sys.executable, script, "--gpu", str(gpu_id)]
            print(f"Launching GPU {gpu_id} → {log_path}")
            with open(log_path, "w") as lf:
                p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf)
            procs.append((gpu_id, p, log_path))

        print("Both GPUs running (GPU0: ~2 min pure GD, GPU1: ~5 min GD+ES)...")
        for gpu_id, p, log_path in procs:
            p.wait()
            print(f"GPU {gpu_id} done (exit={p.returncode})")

        print(f"\n{'='*95}")
        print(f"{'Variant':<62} {'Train':>8} {'Test':>8} {'Time':>7}")
        print(f"{'-'*95}")
        seen = set()
        for line in open(JSONL):
            row = json.loads(line)
            if row.get("experiment") == "exp14" and row["name"] not in seen:
                seen.add(row["name"])
                print(f"{row['name']:<62} {row['train_acc']:>8.4f} "
                      f"{row['test_acc']:>8.4f} {row['time_s']:>6.0f}s")
        print(f"{'='*95}")
