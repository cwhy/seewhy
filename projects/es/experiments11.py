"""
exp11 — rank schedule: rank=8 for warm-up escape, rank=2 for clean convergence.

Key insight from exp9/exp10:
  - "Faster escape" architectures (depth=2, n_queries=2, sigma=0.002) escape the
    ~11% warm-up plateau earlier BUT converge to lower accuracy at 25 epochs.
  - The large perturbations that enable escape (rank=8: std=0.00283) also create
    noisy gradient estimates that prevent fine convergence.
  - Lower rank (rank=2: std=0.00141) gives cleaner gradients but escapes later:
    rank=2 at 25ep: 68.28% (still in warm-up!), rank=2 at 150ep: 94.02% (best ever).

Solution: rank schedule — use rank=8 for warm-up, switch to rank=2 after escape.
  - Pre-compile both JIT functions (avoid double-compile overhead).
  - Cost: ~1 extra recompilation (~24s) at switch point.
  - Benefit: rank=8 escape speed + rank=2 convergence quality in a single run.

GPU0: n_queries=1, rank 8→2 at epoch 12
  Rationale: epoch 12-13 is the empirical warm-up breakout point for depth=1, rank=8.
  Switch immediately when rank=8 is no longer needed for escape.
  17 productive epochs with rank=2 (vs 13 for baseline with rank=8 throughout).

GPU1: n_queries=2, rank 8→2 at epoch 6
  Rationale: n_queries=2 escapes at epoch 6 (confirmed from exp10).
  19 productive epochs with rank=2 (vs 13 for depth=1 baseline rank=8).
  n_queries=2 features may be richer, plus cleaner rank=2 gradients → higher ceiling?

Reference: https://eshyperscale.github.io/
Usage:
    uv run python projects/es/experiments11.py
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

N_VAL_BINS   = 32
BATCH_SIZE   = 128
SEED         = 42
MAX_EPOCHS   = 25
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


def make_es_step(fwd, sigma, n_pop, tx, rank):
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


def train_with_rank_schedule(name, fwd, params, emb_keys, X_tr, y_tr, X_te, y_te,
                              sigma=0.001, n_pop=128,
                              rank_warmup=8, rank_finetune=2, switch_epoch=12,
                              max_epochs=MAX_EPOCHS):
    """Train with rank_warmup for early epochs, rank_finetune after switch_epoch."""
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = max_epochs * num_batches
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT, total_steps,
                                               alpha=ADAM_LR_END / ADAM_LR_INIT)
    tx          = optax.adam(lr_sched)

    # Pre-compile both rank variants before training starts.
    # This triggers JIT compilation for both at epoch 0 (for rank_warmup)
    # and right before the switch (for rank_finetune), avoiding cold-start lag.
    es_step_warmup   = make_es_step(fwd, sigma, n_pop, tx, rank=rank_warmup)
    es_step_finetune = make_es_step(fwd, sigma, n_pop, tx, rank=rank_finetune)

    opt_state = tx.init(params)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    def acc_fn(p, x, y):
        preds = jnp.concatenate([
            jnp.argmax(fwd(p, x[i:i + BATCH_SIZE]), 1)
            for i in range(0, len(x), BATCH_SIZE)])
        return float(jnp.mean(preds == y))

    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        if epoch == switch_epoch:
            logging.info(f"[{name}] Switching rank {rank_warmup}→{rank_finetune} "
                         f"at epoch {epoch}")
        es_step = es_step_warmup if epoch < switch_epoch else es_step_finetune
        current_rank = rank_warmup if epoch < switch_epoch else rank_finetune

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
            pop_keys = jax.random.split(jax.random.fold_in(key, b), n_pop)
            params, opt_state, mean_fit = es_step(params, opt_state, bx, by, pop_keys)
            epoch_fitness.append(float(mean_fit))

        train_acc = acc_fn(params, X_tr, y_tr)
        test_acc  = acc_fn(params, X_te, y_te)
        mean_fit  = sum(epoch_fitness) / len(epoch_fitness)
        elapsed   = time.perf_counter() - t0
        logging.info(
            f"[{name}] epoch {epoch:3d} [r={current_rank}]"
            f"  train={train_acc:.4f}  test={test_acc:.4f}"
            f"  |fitness|={mean_fit:.6f}  {elapsed:.1f}s"
        )

    return train_acc, test_acc, time.perf_counter() - t0


EXPERIMENTS = [
    # GPU 0: n_queries=1, depth=1 — standard architecture with rank schedule 8→2 at epoch 12.
    # rank=8 epochs 0-11: std=0.00283, escapes warm-up at epoch 12-13.
    # rank=2 epochs 12-24: std=0.00141, 13 clean-gradient epochs (same count as baseline).
    # Expected improvement: rank=2 gradients are 4× lower-variance than rank=8 → better convergence.
    dict(label="f3(4,4,4) mlp=256 n_pop=128 25ep rank-sched-8to2-ep12",
         mlp=256, sigma=0.001, n_pop=128,
         rank_warmup=8, rank_finetune=2, switch_epoch=12,
         n_queries=1, depth=1, gpu=0),
    # GPU 1: n_queries=2, depth=1 — richer features with rank schedule 8→2 at epoch 6.
    # rank=8 epochs 0-5: std=0.00283, n_queries=2 escapes warm-up at epoch 6.
    # rank=2 epochs 6-24: std=0.00141, 19 clean-gradient epochs (6 more than baseline).
    # Combines early escape advantage of n_queries=2 with rank=2 convergence quality.
    dict(label="f3(4,4,4) mlp=256 n_pop=128 25ep n_queries=2 rank-sched-8to2-ep6",
         mlp=256, sigma=0.001, n_pop=128,
         rank_warmup=8, rank_finetune=2, switch_epoch=6,
         n_queries=2, depth=1, gpu=1),
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
        fwd, params, emb_keys = make_f3_bilinear(
            4, 4, 4, cfg["mlp"],
            n_queries=cfg.get("n_queries", 1),
            depth=cfg.get("depth", 1))[:]
        nonemb = n_nonemb(params, emb_keys)
        ntotal = n_total(params)
        logging.info(f"\n{'='*60}\n{label}\n"
                     f"  non-emb={nonemb:,}  total={ntotal:,}\n{'='*60}")
        tr_acc, te_acc, elapsed = train_with_rank_schedule(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test,
            sigma=cfg["sigma"], n_pop=cfg["n_pop"],
            rank_warmup=cfg["rank_warmup"],
            rank_finetune=cfg["rank_finetune"],
            switch_epoch=cfg["switch_epoch"],
            max_epochs=MAX_EPOCHS)
        row = {
            "name": label, "experiment": "exp11",
            "family": "es_f3_rank_schedule",
            "n_nonemb_params": nonemb, "n_total_params": ntotal,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": MAX_EPOCHS, "time_s": round(elapsed, 1),
            "batch_size": BATCH_SIZE,
            "es_sigma": cfg["sigma"], "es_n_pop": cfg["n_pop"],
            "es_rank_warmup": cfg["rank_warmup"],
            "es_rank_finetune": cfg["rank_finetune"],
            "rank_switch_epoch": cfg["switch_epoch"],
            "adam_lr_init": ADAM_LR_INIT, "adam_lr_end": ADAM_LR_END,
            "notes": (f"rank schedule {cfg['rank_warmup']}→{cfg['rank_finetune']} "
                      f"at ep{cfg['switch_epoch']}, n_queries={cfg.get('n_queries',1)}"),
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
            log_path = f"/tmp/exp11_gpu{gpu_id}.log"
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            cmd = [sys.executable, script, "--gpu", str(gpu_id)]
            print(f"Launching GPU {gpu_id} → {log_path}")
            with open(log_path, "w") as lf:
                p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf)
            procs.append((gpu_id, p, log_path))

        print("Both GPUs running (~5-6 min)...")
        for gpu_id, p, log_path in procs:
            p.wait()
            print(f"GPU {gpu_id} done (exit={p.returncode})")

        print(f"\n{'='*90}")
        print(f"{'Variant':<55} {'Train':>8} {'Test':>8} {'Time':>7}")
        print(f"{'-'*90}")
        seen = set()
        for line in open(JSONL):
            row = json.loads(line)
            if row.get("experiment") == "exp11" and row["name"] not in seen:
                seen.add(row["name"])
                print(f"{row['name']:<55} {row['train_acc']:>8.4f} "
                      f"{row['test_acc']:>8.4f} {row['time_s']:>6.0f}s")
        print(f"{'='*90}")
