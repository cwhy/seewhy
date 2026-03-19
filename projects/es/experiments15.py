"""
exp15 — overnight run: ~7 hours per GPU, best ES config pushed to convergence.

No time restriction. Goal: find the ES ceiling for this architecture.

Key lessons applied:
  1. rank=2 > rank=8 at long runs: 94.02% (rank=2, 150ep) > 92.77% (rank=8, 50ep)
     Lower rank = cleaner gradient estimates = better convergence asymptote.
  2. Rank schedule 8→2: use rank=8 to escape the ~11% warm-up plateau quickly,
     then switch to rank=2 for all productive epochs. Cost: ~1 recompilation (~25s).
  3. n_queries=2 escapes warm-up at epoch 6 (rank=8) — switch to rank=2 at epoch 6
     and get ~1994 clean-gradient epochs. Richer 128-dim features may push ES ceiling.
  4. sigma=0.001 is optimal: larger→noisy, smaller→fails warm-up escape.
  5. n_pop=128: sweet spot — n_pop=256 slightly worse (fewer epochs in budget).

Epoch estimates (7 hours = 25200s):
  GPU0 n_queries=1: ~11.2s/epoch → 25200/11.2 ≈ 2250 epochs → use 2000
  GPU1 n_queries=2: ~14.2s/epoch → 25200/14.2 ≈ 1774 epochs → use 1600

GPU0: f3(4,4,4) mlp=256, n_queries=1, rank 8→2 at ep12, n_pop=128, sigma=0.001, 2000ep
  → Best known pure-ES config extended to convergence.
  → How far does rank=2 go beyond its 94.02% @ 150ep?

GPU1: f3(4,4,4) mlp=256, n_queries=2, rank 8→2 at ep6, n_pop=128, sigma=0.001, 1600ep
  → Richer 128-dim bilinear features + clean rank=2 gradients.
  → n_queries=2 escaped warm-up at epoch 6 in exp10/exp11, giving ~1994 rank=2 epochs.
  → Larger model (36k non-emb) may find better optima given enough epochs.

Reference: https://eshyperscale.github.io/
Usage:
    uv run python projects/es/experiments15.py         # both GPUs, overnight
    uv run python projects/es/experiments15.py --gpu 0  # single GPU
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


def train(name, fwd, params, emb_keys, X_tr, y_tr, X_te, y_te,
          sigma, n_pop, rank_warmup, rank_finetune, switch_epoch,
          max_epochs, log_every=10):
    """Train with rank schedule: rank_warmup for early epochs, rank_finetune after switch_epoch."""
    num_batches = len(X_tr) // BATCH_SIZE
    total_steps = max_epochs * num_batches
    lr_sched    = optax.cosine_decay_schedule(ADAM_LR_INIT, total_steps,
                                               alpha=ADAM_LR_END / ADAM_LR_INIT)
    tx          = optax.adam(lr_sched)
    # Pre-compile both rank variants upfront to avoid cold recompile at switch point
    step_warmup   = make_es_step(fwd, sigma, n_pop, tx, rank=rank_warmup)
    step_finetune = make_es_step(fwd, sigma, n_pop, tx, rank=rank_finetune)
    opt_state     = tx.init(params)
    key_gen       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    def acc_fn(p, x, y):
        preds = jnp.concatenate([
            jnp.argmax(fwd(p, x[i:i + BATCH_SIZE]), 1)
            for i in range(0, len(x), BATCH_SIZE)])
        return float(jnp.mean(preds == y))

    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        if epoch == switch_epoch:
            logging.info(f"[{name}] Switching rank {rank_warmup}→{rank_finetune} at epoch {epoch}")
        step         = step_warmup if epoch < switch_epoch else step_finetune
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
            params, opt_state, mean_fit = step(params, opt_state, bx, by, pop_keys)
            epoch_fitness.append(float(mean_fit))

        # Only compute and log full eval every log_every epochs to save time
        if epoch % log_every == 0 or epoch < 20:
            train_acc = acc_fn(params, X_tr, y_tr)
            test_acc  = acc_fn(params, X_te, y_te)
            mean_fit  = sum(epoch_fitness) / len(epoch_fitness)
            elapsed   = time.perf_counter() - t0
            logging.info(
                f"[{name}] epoch {epoch:5d} [r={current_rank}]"
                f"  train={train_acc:.4f}  test={test_acc:.4f}"
                f"  |fitness|={mean_fit:.6f}  {elapsed:.0f}s"
            )

    # Final eval
    train_acc = acc_fn(params, X_tr, y_tr)
    test_acc  = acc_fn(params, X_te, y_te)
    elapsed   = time.perf_counter() - t0
    logging.info(f"[{name}] FINAL epoch {max_epochs-1}  train={train_acc:.4f}  test={test_acc:.4f}  {elapsed:.0f}s")
    return train_acc, test_acc, elapsed


EXPERIMENTS = [
    # GPU 0: n_queries=1, rank 8→2 at epoch 12, 2000 epochs (~6.2 hours)
    # Extends the best known ES config (rank=2 at 150ep=94.02%) to full convergence.
    # ~11.2s/epoch → 2000ep ≈ 22400s = 6.2 hours
    dict(label="f3(4,4,4) mlp=256 n_pop=128 2000ep rank-sched-8to2-ep12",
         n_queries=1, depth=1,
         sigma=0.001, n_pop=128, rank_warmup=8, rank_finetune=2,
         switch_epoch=12, max_epochs=2000, gpu=0),
    # GPU 1: n_queries=2, rank 8→2 at epoch 6, 1600 epochs (~6.3 hours)
    # Richer 128-dim bilinear features. n_queries=2 escapes warm-up at epoch 6.
    # rank=2 for all 1594 productive epochs → tests if richer features push ES ceiling.
    # ~14.2s/epoch → 1600ep ≈ 22720s = 6.3 hours
    dict(label="f3(4,4,4) mlp=256 n_queries=2 n_pop=128 1600ep rank-sched-8to2-ep6",
         n_queries=2, depth=1,
         sigma=0.001, n_pop=128, rank_warmup=8, rank_finetune=2,
         switch_epoch=6, max_epochs=1600, gpu=1),
]


def run_single(gpu_id):
    logging.basicConfig(
        level=logging.INFO,
        format=f"[GPU{gpu_id}] %(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
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
            4, 4, 4, 256,
            n_queries=cfg["n_queries"],
            depth=cfg["depth"])[:]
        nonemb = n_nonemb(params, emb_keys)
        ntotal = n_total(params)
        logging.info(f"\n{'='*70}\n{label}\n"
                     f"  n_queries={cfg['n_queries']}  non-emb={nonemb:,}  total={ntotal:,}\n"
                     f"  rank schedule: {cfg['rank_warmup']}→{cfg['rank_finetune']} at ep{cfg['switch_epoch']}\n"
                     f"  max_epochs={cfg['max_epochs']} (~{cfg['max_epochs']*12/3600:.1f}h)\n"
                     f"{'='*70}")
        tr_acc, te_acc, elapsed = train(
            label, fwd, params, emb_keys,
            X_train, y_train, X_test, y_test,
            sigma=cfg["sigma"], n_pop=cfg["n_pop"],
            rank_warmup=cfg["rank_warmup"],
            rank_finetune=cfg["rank_finetune"],
            switch_epoch=cfg["switch_epoch"],
            max_epochs=cfg["max_epochs"],
            log_every=50)
        row = {
            "name": label, "experiment": "exp15",
            "family": "es_f3_overnight",
            "n_nonemb_params": nonemb, "n_total_params": ntotal,
            "train_acc": round(tr_acc, 6), "test_acc": round(te_acc, 6),
            "epochs": cfg["max_epochs"], "time_s": round(elapsed, 1),
            "batch_size": BATCH_SIZE,
            "es_sigma": cfg["sigma"], "es_n_pop": cfg["n_pop"],
            "es_rank_warmup": cfg["rank_warmup"],
            "es_rank_finetune": cfg["rank_finetune"],
            "rank_switch_epoch": cfg["switch_epoch"],
            "n_queries": cfg["n_queries"],
            "adam_lr_init": ADAM_LR_INIT, "adam_lr_end": ADAM_LR_END,
            "notes": (f"overnight: rank {cfg['rank_warmup']}→{cfg['rank_finetune']} "
                      f"at ep{cfg['switch_epoch']}, n_queries={cfg['n_queries']}"),
        }
        append_result(row)
        logging.info(f"  → train={tr_acc:.4f}  test={te_acc:.4f}  {elapsed/3600:.2f}h")


if __name__ == "__main__":
    if "--gpu" in sys.argv:
        gpu_id = int(sys.argv[sys.argv.index("--gpu") + 1])
        run_single(gpu_id)
    else:
        script = str(Path(__file__).resolve())
        procs  = []
        for gpu_id in [0, 1]:
            log_path = f"/tmp/exp15_gpu{gpu_id}.log"
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            cmd = [sys.executable, script, "--gpu", str(gpu_id)]
            print(f"Launching GPU {gpu_id} → {log_path}")
            with open(log_path, "w") as lf:
                p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf)
            procs.append((gpu_id, p, log_path))

        print("Both GPUs running overnight (~6-7 hours)...")
        print("  GPU0: n_queries=1, rank 8→2 at ep12, 2000 epochs")
        print("  GPU1: n_queries=2, rank 8→2 at ep6,  1600 epochs")
        print(f"  Logs: /tmp/exp15_gpu0.log  /tmp/exp15_gpu1.log")
        for gpu_id, p, log_path in procs:
            p.wait()
            print(f"GPU {gpu_id} done (exit={p.returncode})")

        print(f"\n{'='*90}")
        print(f"{'Variant':<60} {'Train':>8} {'Test':>8} {'Time':>8}")
        print(f"{'-'*90}")
        seen = set()
        for line in open(JSONL):
            row = json.loads(line)
            if row.get("experiment") == "exp15" and row["name"] not in seen:
                seen.add(row["name"])
                h = row["time_s"] / 3600
                print(f"{row['name']:<60} {row['train_acc']:>8.4f} "
                      f"{row['test_acc']:>8.4f} {h:>7.2f}h")
        print(f"{'='*90}")
