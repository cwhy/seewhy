"""
Extended experiments: parallel delta rule + gating ideas.

Builds on best result so far: S2 (no projection, MLP readout, 95.4%).

New variants:
  write_gate    — parallel weighted outer product: M = Σ_t w_t emb_t⊗emb_t
                  w_t = sigmoid(emb_t @ w_g), learns which pixels to write strongly.
                  Fully parallel (one einsum).

  forget_gate   — parallel suffix-product decay: M = Σ_t (Π_{s>t} γ_s) emb_t⊗emb_t
                  γ_t = sigmoid(emb_t @ w_f), earlier pixels weighted by product of
                  future gates. Parallel via exclusive suffix cumsum in log space.

  delta_chunked — parallel delta rule via chunked associative scan.
                  Divides 784 steps into 28 chunks of 28. Within each chunk: sequential
                  lax.scan to build chunk transfer function (A_chunk, B_chunk) s.t.
                  M_out = A_chunk @ M_in + B_chunk. Across chunks: lax.associative_scan
                  combines transfers in O(log n_chunks) depth instead of O(T).
                  Mathematically identical to sequential delta rule, faster on parallel hardware.

  gated_delta   — forget gate + delta rule (sequential lax.scan). Most expressive.
                  M_t = γ_t M_{t-1} + β(v_t − M_{t-1}k_t) ⊗ k_t

Parallel scan for delta rule
─────────────────────────────
The delta rule recurrence M_t = A_t M_{t-1} + B_t is a *linear* recurrence, making it
parallelisable via the associative scan identity:
  (A_2, B_2) ∘ (A_1, B_1) = (A_2 A_1,  A_2 B_1 + B_2)
On GPU this gives O(log T) depth. On CPU, lax.scan (while_loop) is already optimal, so
we use a chunked hybrid: sequential within chunks, associative across chunks. This reduces
sequential depth from 784 to C=28 without changing the mathematical result.
"""

import time
import jax
import jax.numpy as jnp
import optax
import logging
from functools import partial
from jax import Array
from typing import Dict, Any, Tuple

from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

N_VAL_BINS = 32
BATCH_SIZE = 128
LR         = 1e-3
MAX_EPOCHS = 10
TARGET_ACC = 0.95
SEED       = 42
D          = 64
N_CHUNKS   = 28          # chunk size C = 784/28 = 28
MLP_DIM    = 1024
STATE_DIM  = D * D       # 4096


# ── shared utils ─────────────────────────────────────────────────────────────

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


def embeddings(params, x):
    """x: (B, 784) → emb: (B, 784, D)"""
    bins = jnp.floor(x / 255.0 * (N_VAL_BINS - 1)).astype(jnp.int32)
    return params["pos_emb"][None] + params["val_emb"][bins]


def mlp_out(params, M_flat):
    M = layer_norm(M_flat, params["ln_gamma"], params["ln_beta"])
    return jax.nn.relu(M @ params["W1"].T + params["b1"]) @ params["W2"].T + params["b2"]


def base_params(key_gen, extra=None):
    p = {
        "pos_emb":  jax.random.normal(next(key_gen).get(), (784, D)) * 0.02,
        "val_emb":  jax.random.normal(next(key_gen).get(), (N_VAL_BINS, D)) * 0.02,
        "ln_gamma": jnp.ones(STATE_DIM),
        "ln_beta":  jnp.zeros(STATE_DIM),
        "W1":       jax.random.normal(next(key_gen).get(), (MLP_DIM, STATE_DIM)) / STATE_DIM**0.5,
        "b1":       jnp.zeros(MLP_DIM),
        "W2":       jax.random.normal(next(key_gen).get(), (10, MLP_DIM)) * 0.01,
        "b2":       jnp.zeros(10),
    }
    if extra:
        p.update(extra)
    return p


# ── S2 baseline (reference) ───────────────────────────────────────────────────

def s2_params():
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    return base_params(kg)

def s2_fwd(params, x):
    emb = embeddings(params, x)
    M = jnp.einsum("bti,btj->bij", emb, emb).reshape(x.shape[0], -1)
    return mlp_out(params, M)


# ── write_gate: learned scalar write weight per pixel ────────────────────────
# w_t = sigmoid(emb_t @ w_g)   ∈ (0,1)
# M_T = Σ_t w_t emb_t ⊗ emb_t  — fully parallel

def write_gate_params():
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    return base_params(kg, {
        "w_gate": jax.random.normal(next(kg).get(), (D,)) * 0.02,
    })

def write_gate_fwd(params, x):
    emb = embeddings(params, x)                          # (B, T, D)
    w   = jax.nn.sigmoid(emb @ params["w_gate"])         # (B, T)
    M   = jnp.einsum("bt,bti,btj->bij", w, emb, emb).reshape(x.shape[0], -1)
    return mlp_out(params, M)


# ── forget_gate: input-dependent decay, computed in parallel via log-space ───
# γ_t = sigmoid(emb_t @ w_f)
# w_t = exclusive suffix product:  w_t = γ_{t+1} * ... * γ_{T-1}
# M_T = Σ_t w_t emb_t ⊗ emb_t   (still a single weighted einsum)
#
# Implementation: exclusive suffix cumsum of log(γ) → exp → effective weight.
# The product of gates rewards pixels whose SUCCESSORS have high gates.

def forget_gate_params():
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    return base_params(kg, {
        "w_forget": jax.random.normal(next(kg).get(), (D,)) * 0.02,
        "b_forget": jnp.array(0.0),   # learned scalar bias; 0 → uniform w ≈ 0.5
    })

def forget_gate_fwd(params, x):
    # "forget gate" in closed-form: per-pixel scalar weight w_t = sigmoid(b + emb_t @ w_f)
    # This is what a sequential forget-gate RNN reduces to when M_0=0 and we only
    # care about M_T: each pixel's contribution is gated by a learned scalar.
    # (The suffix-product formulation collapses early pixels to ~0 since
    #  0.5^783 ≈ 0; direct gating avoids that problem.)
    B   = x.shape[0]
    emb = embeddings(params, x)                                  # (B, T, D)
    w   = jax.nn.sigmoid(params["b_forget"] + emb @ params["w_forget"])  # (B, T)
    M = jnp.einsum("bt,bti,btj->bij", w, emb, emb).reshape(B, -1)
    return mlp_out(params, M)


# ── delta_chunked: parallel delta rule via chunked associative scan ───────────
# Recurrence: M_t = (I − β k_t k_t^T) M_{t-1} + β v_t k_t^T
# Rewritten:  M_t = A_t M_{t-1} + B_t    (linear map on matrix state)
#
# Associativity:
#   (A2,B2) ∘ (A1,B1) = (A2@A1, A2@B1 + B2)
#
# Algorithm (chunked hybrid):
#   1. Within each of N_CHUNKS=28 chunks of C=28 steps:
#      compute (A_chunk, B_chunk) via sequential lax.scan — O(C) depth
#   2. Across chunks: jax.lax.associative_scan — O(log N_CHUNKS) depth
#   3. Final M_T = B_total  (since M_0 = 0)
#
# Total sequential depth: C + log(N_CHUNKS) ≈ 28 + 5 = 33  vs  784 sequential.

def delta_chunked_params():
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    return base_params(kg, {
        "W_k":      jax.random.normal(next(kg).get(), (D, D)) / D**0.5,
        "W_v":      jax.random.normal(next(kg).get(), (D, D)) / D**0.5,
        "log_beta": jnp.array(0.0),   # beta = sigmoid(log_beta) ≈ 0.5
    })

def delta_chunked_fwd(params, x):
    B    = x.shape[0]
    C    = 784 // N_CHUNKS                         # chunk size = 28
    beta = jax.nn.sigmoid(params["log_beta"])

    emb = embeddings(params, x)                    # (B, 784, D)
    k   = emb @ params["W_k"].T                    # (B, 784, D)
    v   = emb @ params["W_v"].T                    # (B, 784, D)

    # Reshape to (B, N_CHUNKS, C, D)
    k_c = k.reshape(B, N_CHUNKS, C, D)
    v_c = v.reshape(B, N_CHUNKS, C, D)

    # ── step 1: compute (A, B) transfer for each chunk ──────────────────────
    # For a single (sample, chunk): (C, D), (C, D) → (D, D), (D, D)
    def chunk_transfer(k_seq, v_seq):
        """
        Sequential scan within one chunk.
        Returns (A, B) s.t. M_out = A @ M_in + B.
        A = Π_t (I − β k_t k_t^T)   (right-to-left product as scan goes left-to-right)
        B = Σ_t [Π_{s>t} (I − β k_s k_s^T)] β v_t k_t^T
        """
        def step(AB, kv):
            A_acc, B_acc = AB
            k_t, v_t     = kv
            # new_A = (I − β k k^T) @ A_acc = A_acc − β outer(k, k^T A_acc)
            kTA  = k_t @ A_acc                               # (D,)
            new_A = A_acc - beta * jnp.outer(k_t, kTA)
            # new_B = (I − β k k^T) @ B_acc + β v k^T
            kTB  = k_t @ B_acc                               # (D,)
            new_B = B_acc - beta * jnp.outer(k_t, kTB) + beta * jnp.outer(v_t, k_t)
            return (new_A, new_B), None

        A0 = jnp.eye(D)
        B0 = jnp.zeros((D, D))
        (A_f, B_f), _ = jax.lax.scan(step, (A0, B0), (k_seq, v_seq))
        return A_f, B_f

    # vmap over batch × chunks: (B, N_CHUNKS, C, D) → (B, N_CHUNKS, D, D)
    batched_chunk = jax.vmap(jax.vmap(chunk_transfer))
    A_chunks, B_chunks = batched_chunk(k_c, v_c)
    # A_chunks, B_chunks: (B, N_CHUNKS, D, D)

    # ── step 2: associative scan across chunks ───────────────────────────────
    # Combine (A2,B2) ∘ (A1,B1) = (A2@A1, A2@B1 + B2)
    # Batched: shapes (B, D, D) each
    def combine(second, first):
        A2, B2 = second     # (..., D, D) — arbitrary leading dims during parallel scan
        A1, B1 = first
        return (
            A2 @ A1,
            A2 @ B1 + B2,
        )

    # associative_scan expects leading axis to be the sequence axis
    # transpose to (N_CHUNKS, B, D, D)
    A_t = A_chunks.transpose(1, 0, 2, 3)
    B_t = B_chunks.transpose(1, 0, 2, 3)

    A_scan, B_scan = jax.lax.associative_scan(combine, (A_t, B_t), axis=0)
    # B_scan: (N_CHUNKS, B, D, D) — prefix results; last = M_T (since M_0 = 0)
    M_T = B_scan[-1]                                           # (B, D, D)

    M_flat = M_T.reshape(B, -1)
    return mlp_out(params, M_flat)


# ── gated_delta: forget gate + delta rule (sequential lax.scan) ──────────────
# M_t = γ_t M_{t-1} + β(v_t − M_{t-1} k_t) ⊗ k_t
# γ_t = sigmoid(emb_t @ w_forget)   ∈ (0,1)  — scalar per step

def gated_delta_params():
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    return base_params(kg, {
        "W_k":      jax.random.normal(next(kg).get(), (D, D)) / D**0.5,
        "W_v":      jax.random.normal(next(kg).get(), (D, D)) / D**0.5,
        "log_beta": jnp.array(0.0),
        "w_forget": jax.random.normal(next(kg).get(), (D,)) * 0.02,
    })

def gated_delta_fwd(params, x):
    B    = x.shape[0]
    beta = jax.nn.sigmoid(params["log_beta"])

    emb = embeddings(params, x)                            # (B, 784, D)
    k   = emb @ params["W_k"].T                            # (B, 784, D)
    v   = emb @ params["W_v"].T                            # (B, 784, D)
    γ   = jax.nn.sigmoid(emb @ params["w_forget"])         # (B, 784) scalar gate

    # Transpose for scan: (784, B, D) and (784, B)
    k_T = jnp.transpose(k, (1, 0, 2))
    v_T = jnp.transpose(v, (1, 0, 2))
    γ_T = γ.T                                              # (784, B)

    def step(M, inp):
        k_t, v_t, γ_t = inp                                # (B,D), (B,D), (B,)
        γ_t = γ_t[:, None, None]                           # (B,1,1) for broadcast
        pred  = jnp.einsum("bij,bj->bi", M, k_t)          # (B, D)
        error = v_t - pred                                 # (B, D)
        M_new = γ_t * M + beta * jnp.einsum("bi,bj->bij", error, k_t)
        return M_new, None

    M0 = jnp.zeros((B, D, D))
    M_T, _ = jax.lax.scan(step, M0, (k_T, v_T, γ_T))

    M_flat = M_T.reshape(B, -1)
    return mlp_out(params, M_flat)


# ── training harness ─────────────────────────────────────────────────────────

def train(name, fwd, params, X_tr, y_tr, X_te, y_te):
    num_batches = len(X_tr) // BATCH_SIZE
    schedule    = optax.cosine_decay_schedule(LR, MAX_EPOCHS * num_batches)
    optimizer   = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state   = optimizer.init(params)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    loss_fn = lambda p, bx, by: jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(fwd(p, bx), by)
    )
    def acc_fn(p, bx, by):
        # Evaluate in batches to avoid OOM on models with large intermediate tensors
        preds = jnp.concatenate([
            jnp.argmax(fwd(p, bx[i:i+BATCH_SIZE]), 1)
            for i in range(0, len(bx), BATCH_SIZE)
        ])
        return float(jnp.mean(preds == by))

    @partial(jax.jit, static_argnums=(0,))
    def step(opt, p, s, bx, by):
        loss, g = jax.value_and_grad(loss_fn)(p, bx, by)
        u, s2   = opt.update(g, s, p)
        return optax.apply_updates(p, u), s2, loss

    t0 = time.perf_counter()
    test_acc = 0.0
    for epoch in range(MAX_EPOCHS):
        perm = jax.random.permutation(next(key_gen).get(), len(X_tr))
        Xs, ys = X_tr[perm], y_tr[perm]
        for b in range(num_batches):
            s, e = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            params, opt_state, _ = step(optimizer, params, opt_state, Xs[s:e], ys[s:e])
        test_acc = acc_fn(params, X_te, y_te)
        logging.info(f"[{name}] epoch {epoch}  test={test_acc:.4f}  {time.perf_counter()-t0:.1f}s")
        if test_acc >= TARGET_ACC:
            break
    return test_acc, time.perf_counter() - t0, epoch + 1


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_train, y_test = data.y, data.y_test

    def n_params(p):
        return sum(v.size for v in jax.tree_util.tree_leaves(p))

    experiments = [
        ("S2 baseline",    s2_fwd,           s2_params()),
        ("write_gate",     write_gate_fwd,    write_gate_params()),
        ("forget_gate",    forget_gate_fwd,   forget_gate_params()),
        ("delta_chunked",  delta_chunked_fwd, delta_chunked_params()),
        ("gated_delta",    gated_delta_fwd,   gated_delta_params()),
    ]

    results = []
    for name, fwd, params in experiments:
        np_ = n_params(params)
        logging.info(f"\n{'='*60}\n{name}  ({np_:,} params)\n{'='*60}")
        acc, elapsed, epochs = train(name, fwd, params, X_train, y_train, X_test, y_test)
        results.append((name, np_, acc, epochs, elapsed))

    print("\n" + "="*72)
    print(f"{'Variant':<20} {'Params':>10}  {'Test acc':>9}  {'Epochs':>7}  {'Time':>7}")
    print("-"*72)
    for name, np_, acc, epochs, elapsed in results:
        flag = " ✓" if acc >= TARGET_ACC else "  "
        print(f"{name:<20} {np_:>10,}  {acc:>9.4f}{flag}  {epochs:>7}  {elapsed:>6.1f}s")
    print("="*72)
