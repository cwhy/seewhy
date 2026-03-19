"""
exp12 — smaller model + larger population tradeoff.

User insight: reduce model size to fit larger n_pop in the same memory footprint.

Current bottleneck: M matrix (BATCH=128, d=64, d=64) under vmap over n_pop pairs.
  - n_pop=128, d=64: M = 128×64×64 = 524k per pair → 67MB for 128 pairs (f32)
  - n_pop=256, d=64: tried in exp6 → 88.31% (-1.5% vs n_pop=128, 2× slower)

Key insight: M size scales as d². Reduce d=64→d=27 (3^3 vs 4^3):
  - M = 128×27×27 = 93k per pair → 7× smaller
  - n_pop=512, d=27: M ≈ 128×27×27×512 pairs ≈ 7GB in vmap → fits in 24GB GPU
  - Time per epoch ≈ (27/64)^2 × 4 × 12.7s ≈ 9s → 25 epochs ≈ 225s ≈ 3.75 min ✓

Tradeoff:
  - d=27 (3,3,3): 42% of feature dimensions, fewer bilinear interactions
  - n_pop=512: 4× more gradient samples → 4× lower gradient variance → cleaner steps
  - With larger n_pop, warm-up escape might also happen earlier (better gradient signal)

GPU0: f3(3,3,3) mlp=128 n_pop=512 rank=8
  → smaller features but 4× more gradient samples. Will gradient quality compensate?
  → mlp=128 (vs 256) to further reduce non-emb params: 128*27=3456 + 10*128=1280 ≈ 4.7k

GPU1: f3(3,3,3) mlp=256 n_pop=512 rank=8
  → same embedding config but mlp=256 for capacity comparison.
  → non-emb: 256*27=6912 + 10*256=2560 ≈ 9.5k

Both run in ~3.75 min. Compare with baseline f3(4,4,4) mlp=256 n_pop=128 → 89.76%.

Reference: https://eshyperscale.github.io/
Usage:
    uv run python projects/es/experiments12.py
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


def train(name, fwd, params, emb_keys, X_tr, y_tr, X_te, y_te,
          sigma=0.001, n_pop=128, rank=1, max_epochs=MAX_EPOCHS,
          d_r=4, d_c=4, d_v=4):
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = max_epochs * num_batches
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT, total_steps,
                                               alpha=ADAM_LR_END / ADAM_LR_INIT)
    tx          = optax.adam(lr_sched)
    es_step     = make_es_step(fwd, sigma, n_pop, tx, rank=rank)
    opt_state   = tx.init(params)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    def acc_fn(p, x, y):
        preds = jnp.concatenate([
            jnp.argmax(fwd(p, x[i:i + BATCH_SIZE]), 1)
            for i in range(0, len(x), BATCH_SIZE)])
        return float(jnp.mean(preds == y))

    t0 = time.perf_counter()
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
            pop_keys = jax.random.split(jax.random.fold_in(key, b), n_pop)
            params, opt_state, mean_fit = es_step(params, opt_state, bx, by, pop_keys)
            epoch_fitness.append(float(mean_fit))

        train_acc = acc_fn(params, X_tr, y_tr)
        test_acc  = acc_fn(params, X_te, y_te)
        mean_fit  = sum(epoch_fitness) / len(epoch_fitness)
        elapsed   = time.perf_counter() - t0
        logging.info(
            f"[{name}] epoch {epoch:3d}  train={train_acc:.4f}  test={test_acc:.4f}"
            f"  |fitness|={mean_fit:.6f}  {elapsed:.1f}s"
        )

    return train_acc, test_acc, time.perf_counter() - t0


EXPERIMENTS = [
    # GPU 0: d=27 (3,3,3), mlp=128, n_pop=512.
    # d=27 shrinks M from (B,64,64) to (B,27,27): 5.6× less memory.
    # n_pop=512: 4× more gradient samples → Var[grad] ∝ 1/(n_pop×rank) drops 4×.
    # Non-emb: 128*27=3456 + 10*128=1280 + biases = 4.7k params (vs 19.5k baseline).
    dict(label="f3(3,3,3) mlp=128 rank=8 n_pop=512 25ep",
         d_r=3, d_c=3, d_v=3, mlp=128, sigma=0.001, n_pop=512, rank=8, gpu=0),
    # GPU 1: d=27 (3,3,3), mlp=256, n_pop=512.
    # Same small embedding but larger MLP for more discriminative capacity.
    # Non-emb: 256*27=6912 + 10*256=2560 + biases = 9.5k params.
    dict(label="f3(3,3,3) mlp=256 rank=8 n_pop=512 25ep",
         d_r=3, d_c=3, d_v=3, mlp=256, sigma=0.001, n_pop=512, rank=8, gpu=1),
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
            cfg["d_r"], cfg["d_c"], cfg["d_v"], cfg["mlp"])[:]
        nonemb = n_nonemb(params, emb_keys)
        ntotal = n_total(params)
        logging.info(f"\n{'='*60}\n{label}\n"
                     f"  non-emb={nonemb:,}  total={ntotal:,}\n{'='*60}")
        tr_acc, te_acc, elapsed = train(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test,
            sigma=cfg["sigma"], n_pop=cfg["n_pop"], rank=cfg["rank"],
            max_epochs=MAX_EPOCHS,
            d_r=cfg["d_r"], d_c=cfg["d_c"], d_v=cfg["d_v"])
        row = {
            "name": label, "experiment": "exp12",
            "family": "es_f3_small_model_large_pop",
            "n_nonemb_params": nonemb, "n_total_params": ntotal,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": MAX_EPOCHS, "time_s": round(elapsed, 1),
            "batch_size": BATCH_SIZE,
            "es_sigma": cfg["sigma"], "es_n_pop": cfg["n_pop"], "es_rank": cfg["rank"],
            "adam_lr_init": ADAM_LR_INIT, "adam_lr_end": ADAM_LR_END,
            "notes": f"d=({cfg['d_r']},{cfg['d_c']},{cfg['d_v']}) mlp={cfg['mlp']} n_pop={cfg['n_pop']}",
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
            log_path = f"/tmp/exp12_gpu{gpu_id}.log"
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            cmd = [sys.executable, script, "--gpu", str(gpu_id)]
            print(f"Launching GPU {gpu_id} → {log_path}")
            with open(log_path, "w") as lf:
                p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf)
            procs.append((gpu_id, p, log_path))

        print("Both GPUs running (~4 min, smaller model + n_pop=512)...")
        for gpu_id, p, log_path in procs:
            p.wait()
            print(f"GPU {gpu_id} done (exit={p.returncode})")

        print(f"\n{'='*95}")
        print(f"{'Variant':<58} {'Train':>8} {'Test':>8} {'Time':>7}")
        print(f"{'-'*95}")
        seen = set()
        for line in open(JSONL):
            row = json.loads(line)
            if row.get("experiment") == "exp12" and row["name"] not in seen:
                seen.add(row["name"])
                print(f"{row['name']:<58} {row['train_acc']:>8.4f} "
                      f"{row['test_acc']:>8.4f} {row['time_s']:>6.0f}s")
        print(f"{'='*95}")
