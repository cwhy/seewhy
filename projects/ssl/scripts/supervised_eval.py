"""
supervised_eval.py — Train each encoder architecture directly for classification.

Tests whether observed SSL accuracy gaps are due to encoder bugs/capacity
or due to the EMA-JEPA training setup itself.

Encoders tested (all at LATENT_DIM=256, then linear head → 10 classes):
  - mlp1    : 784 → 512 → 256
  - mlp2    : 784 → 1024 → 1024 → 256
  - conv1   : [32, 64, 128] + direct GAP→256
  - conv2   : [64, 128, 256] + FC(512)→256
  - vit     : 3L, D=128, 4H, 49 patches → mean pool → 256
  - gram    : Hebbian gram + Hadamard dual → 256 (DEC_RANK=16, gram=256)
  - gram_s  : Hebbian gram + single-proj Hadamard → 256 (DEC_RANK=16)

Training: Adam + cosine decay, 100 epochs, batch=256, LR=1e-3.

Usage:
    uv run python projects/ssl/scripts/supervised_eval.py
"""

import json, time, sys
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BATCH_SIZE  = 256
SEED        = 42
MAX_EPOCHS  = 100
LATENT_DIM  = 256       # embedding size fed to classification head
ADAM_LR     = 1e-3

JSONL = Path(__file__).parent.parent / "results_supervised.jsonl"

# ── Utilities ─────────────────────────────────────────────────────────────────

def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Encoder definitions
# ═══════════════════════════════════════════════════════════════════════════════

def _n(key_gen, shape, scale=0.02):
    return jax.random.normal(next(key_gen).get(), shape) * scale

# ── Shallow MLP ───────────────────────────────────────────────────────────────

def mlp1_init(key_gen):
    return {
        "W1": _n(key_gen, (784, 512),         scale=784**-0.5),
        "b1": jnp.zeros(512),
        "W2": _n(key_gen, (512, LATENT_DIM),  scale=512**-0.5),
        "b2": jnp.zeros(LATENT_DIM),
    }

def mlp1_encode(p, x):
    h = jax.nn.gelu(x / 255.0 @ p["W1"] + p["b1"])
    return h @ p["W2"] + p["b2"]

# ── Deeper MLP ────────────────────────────────────────────────────────────────

def mlp2_init(key_gen):
    return {
        "W1": _n(key_gen, (784, 1024),          scale=784**-0.5),
        "b1": jnp.zeros(1024),
        "W2": _n(key_gen, (1024, 1024),         scale=1024**-0.5),
        "b2": jnp.zeros(1024),
        "W3": _n(key_gen, (1024, LATENT_DIM),   scale=1024**-0.5),
        "b3": jnp.zeros(LATENT_DIM),
    }

def mlp2_encode(p, x):
    h = jax.nn.gelu(x / 255.0 @ p["W1"] + p["b1"])
    h = jax.nn.gelu(h         @ p["W2"] + p["b2"])
    return h @ p["W3"] + p["b3"]

# ── ConvNet v1 [32,64,128] ────────────────────────────────────────────────────

_DN = ("NHWC", "HWIO", "NHWC")
def _conv(x, W, b): return jax.lax.conv_general_dilated(x, W, (1,1), "SAME", dimension_numbers=_DN) + b
def _maxpool(x): return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1,2,2,1), (1,2,2,1), "VALID")

def conv1_init(key_gen):
    return {
        "Wc1": _n(key_gen, (3,3,1,32),          scale=(9*1)**-0.5),   "bc1": jnp.zeros(32),
        "Wc2": _n(key_gen, (3,3,32,64),         scale=(9*32)**-0.5),  "bc2": jnp.zeros(64),
        "Wc3": _n(key_gen, (3,3,64,128),        scale=(9*64)**-0.5),  "bc3": jnp.zeros(128),
        "Wfc": _n(key_gen, (128, LATENT_DIM),   scale=128**-0.5),     "bfc": jnp.zeros(LATENT_DIM),
    }

def conv1_encode(p, x):
    h = (x / 255.0).reshape(-1, 28, 28, 1)
    h = jax.nn.gelu(_conv(h, p["Wc1"], p["bc1"])); h = _maxpool(h)
    h = jax.nn.gelu(_conv(h, p["Wc2"], p["bc2"])); h = _maxpool(h)
    h = jax.nn.gelu(_conv(h, p["Wc3"], p["bc3"]))
    return h.mean(axis=(1,2)) @ p["Wfc"] + p["bfc"]

# ── ConvNet v2 [64,128,256] + FC(512) ─────────────────────────────────────────

def conv2_init(key_gen):
    return {
        "Wc1":  _n(key_gen, (3,3,1,64),          scale=(9*1)**-0.5),   "bc1":  jnp.zeros(64),
        "Wc2":  _n(key_gen, (3,3,64,128),        scale=(9*64)**-0.5),  "bc2":  jnp.zeros(128),
        "Wc3":  _n(key_gen, (3,3,128,256),       scale=(9*128)**-0.5), "bc3":  jnp.zeros(256),
        "Wfc1": _n(key_gen, (256, 512),           scale=256**-0.5),     "bfc1": jnp.zeros(512),
        "Wfc2": _n(key_gen, (512, LATENT_DIM),   scale=512**-0.5),     "bfc2": jnp.zeros(LATENT_DIM),
    }

def conv2_encode(p, x):
    h = (x / 255.0).reshape(-1, 28, 28, 1)
    h = jax.nn.gelu(_conv(h, p["Wc1"], p["bc1"])); h = _maxpool(h)
    h = jax.nn.gelu(_conv(h, p["Wc2"], p["bc2"])); h = _maxpool(h)
    h = jax.nn.gelu(_conv(h, p["Wc3"], p["bc3"]))
    h = jax.nn.gelu(h.mean(axis=(1,2)) @ p["Wfc1"] + p["bfc1"])
    return h @ p["Wfc2"] + p["bfc2"]

# ── Mini ViT (D=128, 3L, 4H, 49 patches) ─────────────────────────────────────

D_MODEL = 128; N_HEADS = 4; DH = D_MODEL // N_HEADS; FFN_DIM = 256; N_LAYERS = 3
N_PATCHES = 49; PATCH_DIM = 16

def vit_init(key_gen):
    p = {
        "W_patch": _n(key_gen, (PATCH_DIM, D_MODEL), scale=PATCH_DIM**-0.5),
        "b_patch": jnp.zeros(D_MODEL),
        "pos_emb": _n(key_gen, (N_PATCHES, D_MODEL), scale=0.02),
    }
    for i in range(N_LAYERS):
        p.update({
            f"Wq{i}": _n(key_gen, (D_MODEL, D_MODEL), scale=D_MODEL**-0.5),
            f"Wk{i}": _n(key_gen, (D_MODEL, D_MODEL), scale=D_MODEL**-0.5),
            f"Wv{i}": _n(key_gen, (D_MODEL, D_MODEL), scale=D_MODEL**-0.5),
            f"Wo{i}": _n(key_gen, (D_MODEL, D_MODEL), scale=D_MODEL**-0.5),
            f"Wf1_{i}": _n(key_gen, (D_MODEL, FFN_DIM), scale=D_MODEL**-0.5),
            f"bf1_{i}": jnp.zeros(FFN_DIM),
            f"Wf2_{i}": _n(key_gen, (FFN_DIM, D_MODEL), scale=FFN_DIM**-0.5),
            f"bf2_{i}": jnp.zeros(D_MODEL),
            f"ln1g{i}": jnp.ones(D_MODEL), f"ln1b{i}": jnp.zeros(D_MODEL),
            f"ln2g{i}": jnp.ones(D_MODEL), f"ln2b{i}": jnp.zeros(D_MODEL),
        })
    p.update({
        "lng": jnp.ones(D_MODEL), "lnb": jnp.zeros(D_MODEL),
        "Wh": _n(key_gen, (D_MODEL, LATENT_DIM), scale=D_MODEL**-0.5),
        "bh": jnp.zeros(LATENT_DIM),
    })
    return p

def _ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x-m)**2).mean(-1, keepdims=True)
    return g * (x - m) / jnp.sqrt(v + eps) + b

def _mha(p, x, i):
    B, S, D = x.shape
    Q = (x @ p[f"Wq{i}"]).reshape(B,S,N_HEADS,DH).transpose(0,2,1,3)
    K = (x @ p[f"Wk{i}"]).reshape(B,S,N_HEADS,DH).transpose(0,2,1,3)
    V = (x @ p[f"Wv{i}"]).reshape(B,S,N_HEADS,DH).transpose(0,2,1,3)
    attn = jax.nn.softmax(Q @ K.transpose(0,1,3,2) / jnp.sqrt(DH), axis=-1)
    return (attn @ V).transpose(0,2,1,3).reshape(B,S,D) @ p[f"Wo{i}"]

def vit_encode(p, x):
    B = x.shape[0]
    patches = (x / 255.0).reshape(B, 7, 4, 7, 4).transpose(0,1,3,2,4).reshape(B, 49, 16)
    h = patches @ p["W_patch"] + p["b_patch"] + p["pos_emb"][None]
    for i in range(N_LAYERS):
        h = h + _mha(p, _ln(h, p[f"ln1g{i}"], p[f"ln1b{i}"]), i)
        h = h + jax.nn.gelu(_ln(h, p[f"ln2g{i}"], p[f"ln2b{i}"]) @ p[f"Wf1_{i}"] + p[f"bf1_{i}"]) @ p[f"Wf2_{i}"] + p[f"bf2_{i}"]
    h = _ln(h, p["lng"], p["lnb"]).mean(axis=1)
    return h @ p["Wh"] + p["bh"]

# ── Gram + Hadamard dual (DEC_RANK=16, LATENT=256) ────────────────────────────

DEC_RANK = 16  # DEC_RANK² = 256 = LATENT_DIM
D_R, D_C, D_V = 6, 6, 4; D = D_R*D_C*D_V; D_RC = D_R*D_C
ROWS = jnp.arange(784) // 28; COLS = jnp.arange(784) % 28

def gram_init(key_gen):
    return {
        "enc_row_emb": _n(key_gen, (28, D_R),           scale=D_R**-0.5),
        "enc_col_emb": _n(key_gen, (28, D_C),           scale=D_C**-0.5),
        "W_pos_enc"  : _n(key_gen, (D_RC, D),           scale=D_RC**-0.5),
        "W_enc"      : _n(key_gen, (D, DEC_RANK),       scale=D**-0.5),
        "W_enc_A"    : _n(key_gen, (DEC_RANK**2, LATENT_DIM), scale=(DEC_RANK**2)**-0.5),
        "W_enc_B"    : _n(key_gen, (DEC_RANK**2, LATENT_DIM), scale=(DEC_RANK**2)**-0.5),
    }

def gram_encode(p, x):
    x_norm = x / 255.0
    re = p["enc_row_emb"][ROWS]; ce = p["enc_col_emb"][COLS]
    pos_rc   = jax.vmap(lambda r,c: (r[:,None]*c[None,:]).reshape(D_RC))(re, ce)
    pos_feat = pos_rc @ p["W_pos_enc"]
    f        = pos_feat @ p["W_enc"]
    F        = x_norm[:,:,None] * f[None,:,:]
    H        = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK**2)
    return (H @ p["W_enc_A"]) * (H @ p["W_enc_B"])

# ── Gram-single (DEC_RANK=16, LATENT=256) ─────────────────────────────────────

def gram_single_init(key_gen):
    return {
        "enc_row_emb": _n(key_gen, (28, D_R),           scale=D_R**-0.5),
        "enc_col_emb": _n(key_gen, (28, D_C),           scale=D_C**-0.5),
        "W_pos_enc"  : _n(key_gen, (D_RC, D),           scale=D_RC**-0.5),
        "W_enc"      : _n(key_gen, (D, DEC_RANK),       scale=D**-0.5),
        "W_proj"     : _n(key_gen, (DEC_RANK**2, LATENT_DIM), scale=(DEC_RANK**2)**-0.5),
    }

def gram_single_encode(p, x):
    x_norm = x / 255.0
    re = p["enc_row_emb"][ROWS]; ce = p["enc_col_emb"][COLS]
    pos_rc   = jax.vmap(lambda r,c: (r[:,None]*c[None,:]).reshape(D_RC))(re, ce)
    pos_feat = pos_rc @ p["W_pos_enc"]
    f        = pos_feat @ p["W_enc"]
    F        = x_norm[:,:,None] * f[None,:,:]
    H        = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK**2)
    return (H @ p["W_proj"]) * H

# ═══════════════════════════════════════════════════════════════════════════════
# Supervised training
# ═══════════════════════════════════════════════════════════════════════════════

def make_train_fn(encode_fn, optimizer):
    def loss_fn(params, x, y):
        emb    = encode_fn(params["enc"], x)
        logits = emb @ params["W_cls"] + params["b_cls"]
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    def scan_body(carry, inputs):
        params, opt_state = carry
        x, y = inputs
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, new_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), new_state), loss

    def epoch_fn(params, opt_state, Xs, Ys, key):
        B = Xs.shape[0]
        keys = jax.vmap(lambda b: jax.random.fold_in(key, b))(jnp.arange(B))
        (params, opt_state), losses = jax.lax.scan(
            scan_body, (params, opt_state), (Xs, Ys)
        )
        return params, opt_state, losses.mean()

    return jax.jit(epoch_fn)

def eval_accuracy(params, encode_fn, X, Y):
    accs = []
    for i in range(0, len(X), 512):
        xb = jnp.array(np.array(X[i:i+512]))
        yb = jnp.array(np.array(Y[i:i+512]))
        emb    = encode_fn(params["enc"], xb)
        logits = emb @ params["W_cls"] + params["b_cls"]
        accs.append(float((jnp.argmax(logits, 1) == yb).mean()) * len(xb))
    return sum(accs) / len(X)

def run_supervised(name, enc_init_fn, enc_fn, X_tr, Y_tr, X_te, Y_te):
    logging.info(f"\n{'='*60}\nEncoder: {name}\n{'='*60}")
    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))

    enc_params = enc_init_fn(key_gen)
    params = {
        "enc"  : enc_params,
        "W_cls": jax.random.normal(next(key_gen).get(), (LATENT_DIM, 10)) * LATENT_DIM**-0.5,
        "b_cls": jnp.zeros(10),
    }
    total = n_params(params)
    logging.info(f"  Total params: {total:,}  (enc: {n_params(enc_params):,}  + head: {LATENT_DIM*10+10:,})")

    num_steps = MAX_EPOCHS * (len(X_tr) // BATCH_SIZE)
    optimizer = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.01))
    opt_state = optimizer.init(params)
    epoch_fn  = make_train_fn(enc_fn, optimizer)

    key_gen2   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 100))
    num_batches = len(X_tr) // BATCH_SIZE
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen2).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        Xs   = jnp.array(np.array(X_tr)[perm][:num_batches*BATCH_SIZE].reshape(num_batches, BATCH_SIZE, 784))
        Ys   = jnp.array(np.array(Y_tr)[perm][:num_batches*BATCH_SIZE].reshape(num_batches, BATCH_SIZE))
        params, opt_state, loss = epoch_fn(params, opt_state, Xs, Ys, key)

        if (epoch + 1) % 10 == 0:
            acc = eval_accuracy(params, enc_fn, X_te, Y_te)
            logging.info(f"  epoch {epoch+1:3d}  loss={float(loss):.4f}  test_acc={acc:.1%}  {time.perf_counter()-t0:.1f}s")

    final_acc = eval_accuracy(params, enc_fn, X_te, Y_te)
    train_acc  = eval_accuracy(params, enc_fn, X_tr, Y_tr)
    elapsed = time.perf_counter() - t0
    logging.info(f"  DONE  train_acc={train_acc:.1%}  test_acc={final_acc:.1%}  {elapsed:.1f}s")
    append_result({
        "encoder"    : name,
        "n_params"   : total,
        "epochs"     : MAX_EPOCHS,
        "latent_dim" : LATENT_DIM,
        "time_s"     : round(elapsed, 1),
        "train_acc"  : round(train_acc, 6),
        "test_acc"   : round(final_acc, 6),
    })
    return final_acc

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)
    Y_train = np.array(data.y)
    Y_test  = np.array(data.y_test)

    encoders = [
        ("mlp1",       mlp1_init,        mlp1_encode),
        ("mlp2",       mlp2_init,        mlp2_encode),
        ("conv1",      conv1_init,       conv1_encode),
        ("conv2",      conv2_init,       conv2_encode),
        ("vit",        vit_init,         vit_encode),
        ("gram",       gram_init,        gram_encode),
        ("gram_single",gram_single_init, gram_single_encode),
    ]

    results = {}
    for name, init_fn, enc_fn in encoders:
        results[name] = run_supervised(name, init_fn, enc_fn, X_train, Y_train, X_test, Y_test)

    print("\n" + "="*50)
    print("SUPERVISED CLASSIFICATION RESULTS (LATENT=256)")
    print("="*50)
    for name, acc in results.items():
        print(f"  {name:15s}  {acc:.2%}")
