"""
exp_masked_distill_vit_patch — ViT student (patch masking) + frozen ConvNet-v2 DAE teacher.

ViT: 4x4 patches (49 total), D_MODEL=128, 3 layers, 4 heads -> LATENT_DIM=1024.
Uses patch-level masking (whole 4x4 patches zeroed).
Teacher: exp_dae_conv2 (frozen). Loss: MSE(predictor(student(aug_x), action), teacher(clean_x)).
"""

import json, pickle, sys, time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME = "exp_masked_distill_vit_patch"

# ── Hyperparameters ──────────────────────────────────────────────

BATCH_SIZE      = 256
SEED            = 42
MAX_EPOCHS      = 200
ADAM_LR         = 3e-4
EVAL_EVERY      = 10
PROBE_EPOCHS    = 10

LATENT_DIM      = 1024
MLP_PRED_HIDDEN = 512

N_ACTIONS     = 4
ACTION_DIM    = 32
ACTION_LABELS = ["denoise_gaussian_75", "patch_mask_25", "patch_mask_50", "patch_mask_75"]

NOISE_STD   = 75.0
MASK_RATIOS = [0.25, 0.50, 0.75]

TEACHER_PKL = Path(__file__).parent / "params_exp_dae_conv2.pkl"

JSONL = Path(__file__).parent / "results.jsonl"

PATCH_SIZE      = 4
N_PATCHES       = 49
PATCH_DIM       = 16
D_MODEL         = 128
N_HEADS         = 4
DH              = D_MODEL // N_HEADS
FFN_DIM         = 256
N_LAYERS        = 3

# ── Utilities ──────────────────────────────────────────────────────

def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

# ── Teacher encoder (ConvNet-v2, frozen) ───────────────────────────────

_DN = ("NHWC", "HWIO", "NHWC")

def _conv(x, W, b):
    return jax.lax.conv_general_dilated(x, W, (1, 1), "SAME", dimension_numbers=_DN) + b

def _maxpool(x):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

def teacher_encode(params, x):
    h = (x / 255.0).reshape(-1, 28, 28, 1)
    h = jax.nn.gelu(_conv(h, params["W_conv1"], params["b_conv1"]))
    h = _maxpool(h)
    h = jax.nn.gelu(_conv(h, params["W_conv2"], params["b_conv2"]))
    h = _maxpool(h)
    h = jax.nn.gelu(_conv(h, params["W_conv3"], params["b_conv3"]))
    h = h.mean(axis=(1, 2))
    h = jax.nn.gelu(h @ params["W_fc1"] + params["b_fc1"])
    return h @ params["W_fc2"] + params["b_fc2"]

# ── Student helpers ───────────────────────────────────────────────

def _layer_norm(x, g, b, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var  = ((x - mean) ** 2).mean(-1, keepdims=True)
    return g * (x - mean) / jnp.sqrt(var + eps) + b

def _mha(params, x, i):
    B, S, D = x.shape
    Q = (x @ params[f"W_q_{i}"]).reshape(B, S, N_HEADS, DH).transpose(0, 2, 1, 3)
    K = (x @ params[f"W_k_{i}"]).reshape(B, S, N_HEADS, DH).transpose(0, 2, 1, 3)
    V = (x @ params[f"W_v_{i}"]).reshape(B, S, N_HEADS, DH).transpose(0, 2, 1, 3)
    attn = jax.nn.softmax(Q @ K.transpose(0, 1, 3, 2) / jnp.sqrt(DH), axis=-1)
    out  = (attn @ V).transpose(0, 2, 1, 3).reshape(B, S, D)
    return out @ params[f"W_o_{i}"]

def _ffn(params, x, i):
    h = jax.nn.gelu(x @ params[f"W_ff1_{i}"] + params[f"b_ff1_{i}"])
    return h @ params[f"W_ff2_{i}"] + params[f"b_ff2_{i}"]

def _transformer_layer(params, x, i):
    x = x + _mha(params, _layer_norm(x, params[f"ln1_g_{i}"], params[f"ln1_b_{i}"]), i)
    x = x + _ffn(params, _layer_norm(x, params[f"ln2_g_{i}"], params[f"ln2_b_{i}"]), i)
    return x

def _extract_patches(x):
    B = x.shape[0]
    return x.reshape(B, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(B, 49, 16)

def _patch_mask(x, key, mask_ratio):
    B       = x.shape[0]
    patches = x.reshape(B, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(B, 49, 16)
    keep    = jax.random.bernoulli(key, 1.0 - mask_ratio, shape=(B, 49))
    patches = patches * keep[:, :, None]
    return patches.reshape(B, 7, 7, 4, 4).transpose(0, 1, 3, 2, 4).reshape(B, 784)

# ── Student parameters ─────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    params = {
        "W_patch"   : n((PATCH_DIM, D_MODEL),  scale=PATCH_DIM**-0.5),
        "b_patch"   : jnp.zeros(D_MODEL),
        "pos_emb"   : n((N_PATCHES, D_MODEL),  scale=0.02),
    }
    for i in range(N_LAYERS):
        params.update({
            f"W_q_{i}"    : n((D_MODEL, D_MODEL), scale=D_MODEL**-0.5),
            f"W_k_{i}"    : n((D_MODEL, D_MODEL), scale=D_MODEL**-0.5),
            f"W_v_{i}"    : n((D_MODEL, D_MODEL), scale=D_MODEL**-0.5),
            f"W_o_{i}"    : n((D_MODEL, D_MODEL), scale=D_MODEL**-0.5),
            f"W_ff1_{i}"  : n((D_MODEL, FFN_DIM), scale=D_MODEL**-0.5),
            f"b_ff1_{i}"  : jnp.zeros(FFN_DIM),
            f"W_ff2_{i}"  : n((FFN_DIM, D_MODEL), scale=FFN_DIM**-0.5),
            f"b_ff2_{i}"  : jnp.zeros(D_MODEL),
            f"ln1_g_{i}"  : jnp.ones(D_MODEL),
            f"ln1_b_{i}"  : jnp.zeros(D_MODEL),
            f"ln2_g_{i}"  : jnp.ones(D_MODEL),
            f"ln2_b_{i}"  : jnp.zeros(D_MODEL),
        })
    params.update({
        "ln_g"      : jnp.ones(D_MODEL),
        "ln_b"      : jnp.zeros(D_MODEL),
        "W_head"    : n((D_MODEL, LATENT_DIM),  scale=D_MODEL**-0.5),
        "b_head"    : jnp.zeros(LATENT_DIM),
        "action_emb": n((N_ACTIONS, ACTION_DIM), scale=0.01),
        "W_pred1"   : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),
        "b_pred1"   : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_s"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_s"  : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_t"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_t"  : jnp.zeros(MLP_PRED_HIDDEN),
        "W_pred2"   : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),
        "b_pred2"   : jnp.zeros(LATENT_DIM),
    })
    return params

# ── Student encoder ──────────────────────────────────────────────

def encode(params, x):
    patches = _extract_patches(x / 255.0)
    h = patches @ params["W_patch"] + params["b_patch"]
    h = h + params["pos_emb"][None, :, :]
    for i in range(N_LAYERS):
        h = _transformer_layer(params, h, i)
    h = _layer_norm(h, params["ln_g"], params["ln_b"])
    h = h.mean(axis=1)
    return h @ params["W_head"] + params["b_head"]

# ── Predictor (FiLM) ─────────────────────────────────────────────

def predict(params, e, a):
    h     = jax.nn.gelu(e @ params["W_pred1"] + params["b_pred1"])
    scale = a @ params["W_film_s"] + params["b_film_s"]
    shift = a @ params["W_film_t"] + params["b_film_t"]
    h     = h * (1.0 + scale) + shift
    return h @ params["W_pred2"] + params["b_pred2"]

# ── Loss ──────────────────────────────────────────────────────

def total_loss(online_params, teacher_params, x, key):
    k_idx, k_noise, k_m1, k_m2, k_m3 = jax.random.split(key, 5)
    B = x.shape[0]

    idx   = jax.random.randint(k_idx, (B,), 0, N_ACTIONS)
    a_emb = online_params["action_emb"][idx]

    x_n75 = jnp.clip(x + jax.random.normal(k_noise, x.shape) * NOISE_STD, 0.0, 255.0)
    x_m25 = _patch_mask(x, k_m1, MASK_RATIOS[0])
    x_m50 = _patch_mask(x, k_m2, MASK_RATIOS[1])
    x_m75 = _patch_mask(x, k_m3, MASK_RATIOS[2])
    x_aug = jnp.where(idx[:, None] == 0, x_n75,
            jnp.where(idx[:, None] == 1, x_m25,
            jnp.where(idx[:, None] == 2, x_m50, x_m75)))

    e_target = jax.lax.stop_gradient(teacher_encode(teacher_params, x))
    e_aug    = encode(online_params, x_aug)
    e_pred   = predict(online_params, e_aug, a_emb)

    return ((e_pred - e_target) ** 2).mean(), ()

# ── Epoch function ───────────────────────────────────────────────

def make_epoch_fn(optimizer, teacher_params):
    def scan_body(carry, inputs):
        online_params, opt_state = carry
        x_batch, key = inputs
        (loss, _), grads = jax.value_and_grad(total_loss, has_aux=True)(
            online_params, teacher_params, x_batch, key
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, online_params)
        return (optax.apply_updates(online_params, updates), new_opt_state), loss

    def epoch_fn(online_params, opt_state, Xs_batched, epoch_key):
        num_batches = Xs_batched.shape[0]
        keys = jax.vmap(lambda b: jax.random.fold_in(epoch_key, b))(jnp.arange(num_batches))
        (online_params, opt_state), losses = jax.lax.scan(
            scan_body, (online_params, opt_state), (Xs_batched, keys)
        )
        return online_params, opt_state, losses.mean()

    return jax.jit(epoch_fn)

# ── Linear probe ─────────────────────────────────────────────────

def linear_probe_accuracy(params, X_tr, Y_tr, X_te, Y_te):
    def encode_all(X):
        parts = []
        for i in range(0, len(X), 512):
            parts.append(np.array(encode(params, jnp.array(np.array(X[i:i+512])))))
        return np.concatenate(parts, 0)

    E_tr = encode_all(X_tr)
    E_te = encode_all(X_te)
    n_tr = (len(E_tr) // BATCH_SIZE) * BATCH_SIZE
    E_b  = jnp.array(E_tr[:n_tr].reshape(-1, BATCH_SIZE, LATENT_DIM))
    Y_b  = jnp.array(np.array(Y_tr[:n_tr]).reshape(-1, BATCH_SIZE))

    W, b        = jnp.zeros((LATENT_DIM, 10)), jnp.zeros(10)
    probe_opt   = optax.adam(1e-3)
    probe_state = probe_opt.init((W, b))

    def probe_step(carry, inputs):
        (W, b), state = carry
        e, y = inputs
        def ce(wb):
            return optax.softmax_cross_entropy_with_integer_labels(e @ wb[0] + wb[1], y).mean()
        loss, grads = jax.value_and_grad(ce)((W, b))
        updates, new_state = probe_opt.update(grads, state, (W, b))
        return (optax.apply_updates((W, b), updates), new_state), loss

    @jax.jit
    def probe_epoch(W, b, state, E_b, Y_b):
        (W, b), state = jax.lax.scan(probe_step, ((W, b), state), (E_b, Y_b))[0]
        return W, b, state

    for _ in range(PROBE_EPOCHS):
        W, b, probe_state = probe_epoch(W, b, probe_state, E_b, Y_b)

    logits = jnp.array(E_te) @ W + b
    return float((jnp.argmax(logits, 1) == jnp.array(np.array(Y_te))).mean())

# ── Training loop ──────────────────────────────────────────────

def train(online_params, teacher_params, X_tr, Y_tr, X_te, Y_te):
    num_steps   = MAX_EPOCHS * (len(X_tr) // BATCH_SIZE)
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(online_params)
    epoch_fn    = make_epoch_fn(optimizer, teacher_params)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    num_batches = len(X_tr) // BATCH_SIZE
    history     = {"loss": [], "probe_acc": []}
    t0          = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        Xs   = jnp.array(
            np.array(X_tr)[perm][:num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE, 784)
        )
        online_params, opt_state, loss = epoch_fn(
            online_params, opt_state, Xs, jax.random.fold_in(key, epoch)
        )
        loss = float(loss)
        history["loss"].append(loss)

        probe_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            acc = linear_probe_accuracy(online_params, X_tr, Y_tr, X_te, Y_te)
            history["probe_acc"].append((epoch, acc))
            probe_str = f"  probe_acc={acc:.1%}"

        logging.info(f"epoch {epoch:3d}  loss={loss:.6f}{probe_str}  {time.perf_counter()-t0:.1f}s")

    return online_params, history, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")

    done = set()
    if JSONL.exists():
        with open(JSONL) as f:
            for line in f:
                try: done.add(json.loads(line).get("experiment"))
                except Exception: pass
    if EXP_NAME in done:
        logging.info(f"{EXP_NAME} already in results.jsonl — skipping")
        exit(0)

    logging.info(f"Loading teacher from {TEACHER_PKL.name}")
    teacher_params = {k: jnp.array(v) for k, v in pickle.load(open(TEACHER_PKL, "rb")).items()}
    logging.info(f"Teacher params: {n_params(teacher_params):,}")

    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)
    Y_train = np.array(data.y)
    Y_test  = np.array(data.y_test)

    key_gen       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    online_params = init_params(key_gen)
    logging.info(f"Student params: {n_params(online_params):,}")

    online_params, history, elapsed = train(
        online_params, teacher_params, X_train, Y_train, X_test, Y_test
    )

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in online_params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_acc = linear_probe_accuracy(online_params, X_train, Y_train, X_test, Y_test)
    append_result({
        "experiment"      : EXP_NAME,
        "n_params"        : n_params(online_params),
        "epochs"          : MAX_EPOCHS,
        "time_s"          : round(elapsed, 1),
        "final_probe_acc" : round(final_acc, 6),
        "loss_curve"      : [round(v, 6) for v in history["loss"]],
        "probe_acc_curve" : history["probe_acc"],
        "latent_dim"      : LATENT_DIM,
        "adam_lr"         : ADAM_LR,
        "batch_size"      : BATCH_SIZE,
        "encoder"         : "vit_patch",
        "teacher"         : "exp_dae_conv2",
        "method"          : "masked_distill",
        "d_model"         : D_MODEL,
        "n_layers"        : N_LAYERS,
    })
    logging.info(f"Done. probe_acc={final_acc:.1%}  {elapsed:.1f}s")
