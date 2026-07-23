"""
exp_distill_heatmap_dae — arch × arch masked distillation sweep.

For each (teacher_arch, student_arch) pair:
  - Teacher: frozen DAE encoder (LATENT_DIM=1024)
  - Student: freshly initialised encoder of student_arch
  - Loss:    MSE(predictor(student(aug_x), action), stop_grad(teacher(clean_x)))
  - Eval:    linear probe accuracy + 1-NN accuracy on MNIST

Results go to results_distill_heatmap_dae.jsonl.
Skip-if-done so the script is safe to restart.

Usage:
    uv run python projects/ssl/exp_distill_heatmap_dae.py
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

SSL = Path(__file__).parent

JSONL       = SSL / "results_distill_heatmap_dae.jsonl"
PARAMS_DIR  = SSL

# ── Shared hyperparameters ─────────────────────────────────────────────────────

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
ACTION_LABELS = ["denoise_gaussian_75", "mask_25", "mask_50", "mask_75"]

NOISE_STD   = 75.0
MASK_RATIOS = [0.25, 0.50, 0.75]

# ── Architecture-specific constants ───────────────────────────────────────────

# MLP-shallow
MLP_ENC_HIDDEN = 512

# MLP2
MLP2_H1 = 1024
MLP2_H2 = 1024

# ConvNet-v1
CONV1_CH = [32, 64, 128]

# ConvNet-v2
CONV2_CH = [64, 128, 256]
CONV2_FC = 512

# ViT (shared by vit + vit_patch)
N_PATCHES = 49
PATCH_DIM = 16
D_MODEL   = 128
N_HEADS   = 4
DH        = D_MODEL // N_HEADS   # 32
FFN_DIM   = 256
N_LAYERS  = 3

# Gram (shared by gram_dual, gram_single, gram_glu)
D_R, D_C, D_V = 6, 6, 4
D              = D_R * D_C * D_V   # 144
D_RC           = D_R * D_C         # 36
DEC_RANK       = 32    # DEC_RANK^2 = 1024 = LATENT_DIM
ROWS = jnp.arange(784) // 28
COLS = jnp.arange(784) % 28

# ── Conv / maxpool helpers (shared) ───────────────────────────────────────────

_DN = ("NHWC", "HWIO", "NHWC")

def _conv(x, W, b):
    return jax.lax.conv_general_dilated(x, W, (1, 1), "SAME", dimension_numbers=_DN) + b

def _maxpool(x):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

# ── ViT helpers ───────────────────────────────────────────────────────────────

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

# ── Utilities ──────────────────────────────────────────────────────────────────

def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

def encode_all(encode_fn, params, X):
    parts = []
    for i in range(0, len(X), 512):
        parts.append(np.array(encode_fn(params, jnp.array(np.array(X[i:i+512])))))
    return np.concatenate(parts, 0)

# ── 9 teacher encode functions ────────────────────────────────────────────────

def teacher_encode_mlp(params, x):
    x_norm = x / 255.0
    h = jax.nn.gelu(x_norm @ params["W_enc1"] + params["b_enc1"])
    return h @ params["W_enc2"] + params["b_enc2"]

def teacher_encode_mlp2(params, x):
    x_norm = x / 255.0
    h = jax.nn.gelu(x_norm @ params["W_enc1"] + params["b_enc1"])
    h = jax.nn.gelu(h     @ params["W_enc2"] + params["b_enc2"])
    return h @ params["W_enc3"] + params["b_enc3"]

def teacher_encode_conv(params, x):
    h = (x / 255.0).reshape(-1, 28, 28, 1)
    h = jax.nn.gelu(_conv(h, params["W_conv1"], params["b_conv1"]))
    h = _maxpool(h)
    h = jax.nn.gelu(_conv(h, params["W_conv2"], params["b_conv2"]))
    h = _maxpool(h)
    h = jax.nn.gelu(_conv(h, params["W_conv3"], params["b_conv3"]))
    h = h.mean(axis=(1, 2))
    return h @ params["W_fc"] + params["b_fc"]

def teacher_encode_conv2(params, x):
    h = (x / 255.0).reshape(-1, 28, 28, 1)
    h = jax.nn.gelu(_conv(h, params["W_conv1"], params["b_conv1"]))
    h = _maxpool(h)
    h = jax.nn.gelu(_conv(h, params["W_conv2"], params["b_conv2"]))
    h = _maxpool(h)
    h = jax.nn.gelu(_conv(h, params["W_conv3"], params["b_conv3"]))
    h = h.mean(axis=(1, 2))
    h = jax.nn.gelu(h @ params["W_fc1"] + params["b_fc1"])
    return h @ params["W_fc2"] + params["b_fc2"]

def teacher_encode_vit(params, x):
    patches = _extract_patches(x / 255.0)
    h = patches @ params["W_patch"] + params["b_patch"]
    h = h + params["pos_emb"][None, :, :]
    for i in range(N_LAYERS):
        h = _transformer_layer(params, h, i)
    h = _layer_norm(h, params["ln_g"], params["ln_b"])
    h = h.mean(axis=1)
    return h @ params["W_head"] + params["b_head"]

def teacher_encode_vit_patch(params, x):
    return teacher_encode_vit(params, x)   # same architecture

def teacher_encode_gram_dual(params, x):
    x_norm   = x / 255.0
    re       = params["enc_row_emb"][ROWS]
    ce       = params["enc_col_emb"][COLS]
    pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_RC))(re, ce)
    pos_feat = pos_rc @ params["W_pos_enc"]
    f        = pos_feat @ params["W_enc"]
    F        = x_norm[:, :, None] * f[None, :, :]
    dr       = params["W_enc"].shape[1]   # DEC_RANK may differ per checkpoint
    H        = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], dr ** 2)
    a        = H @ params["W_enc_A"]
    b        = H @ params["W_enc_B"]
    return a * b

def teacher_encode_gram_single(params, x):
    x_norm   = x / 255.0
    re       = params["enc_row_emb"][ROWS]
    ce       = params["enc_col_emb"][COLS]
    pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_RC))(re, ce)
    pos_feat = pos_rc @ params["W_pos_enc"]
    f        = pos_feat @ params["W_enc"]
    F        = x_norm[:, :, None] * f[None, :, :]
    H        = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK ** 2)
    z        = H @ params["W_proj"]
    return z * H

def teacher_encode_gram_glu(params, x):
    x_norm   = x / 255.0
    re       = params["enc_row_emb"][ROWS]
    ce       = params["enc_col_emb"][COLS]
    pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_RC))(re, ce)
    pos_feat = pos_rc @ params["W_pos_enc"]
    f        = pos_feat @ params["W_enc"]
    F        = x_norm[:, :, None] * f[None, :, :]
    H        = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK ** 2)
    gate     = jax.nn.sigmoid(H @ params["W_gate"])
    H_g      = H * gate
    a        = H_g @ params["W_enc_A"]
    b        = H_g @ params["W_enc_B"]
    return a * b

# ── 9 student init_params + encode functions ──────────────────────────────────

def init_params_mlp(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    return {
        "W_enc1"    : n((784, MLP_ENC_HIDDEN),         scale=784**-0.5),
        "b_enc1"    : jnp.zeros(MLP_ENC_HIDDEN),
        "W_enc2"    : n((MLP_ENC_HIDDEN, LATENT_DIM),  scale=MLP_ENC_HIDDEN**-0.5),
        "b_enc2"    : jnp.zeros(LATENT_DIM),
        "action_emb": n((N_ACTIONS, ACTION_DIM),        scale=0.01),
        "W_pred1"   : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),
        "b_pred1"   : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_s"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_s"  : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_t"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_t"  : jnp.zeros(MLP_PRED_HIDDEN),
        "W_pred2"   : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),
        "b_pred2"   : jnp.zeros(LATENT_DIM),
    }

def encode_mlp(params, x):
    x_norm = x / 255.0
    h = jax.nn.gelu(x_norm @ params["W_enc1"] + params["b_enc1"])
    return h @ params["W_enc2"] + params["b_enc2"]

def init_params_mlp2(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    return {
        "W_enc1"    : n((784, MLP2_H1),            scale=784**-0.5),
        "b_enc1"    : jnp.zeros(MLP2_H1),
        "W_enc2"    : n((MLP2_H1, MLP2_H2),        scale=MLP2_H1**-0.5),
        "b_enc2"    : jnp.zeros(MLP2_H2),
        "W_enc3"    : n((MLP2_H2, LATENT_DIM),     scale=MLP2_H2**-0.5),
        "b_enc3"    : jnp.zeros(LATENT_DIM),
        "action_emb": n((N_ACTIONS, ACTION_DIM),    scale=0.01),
        "W_pred1"   : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),
        "b_pred1"   : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_s"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_s"  : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_t"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_t"  : jnp.zeros(MLP_PRED_HIDDEN),
        "W_pred2"   : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),
        "b_pred2"   : jnp.zeros(LATENT_DIM),
    }

def encode_mlp2(params, x):
    x_norm = x / 255.0
    h = jax.nn.gelu(x_norm @ params["W_enc1"] + params["b_enc1"])
    h = jax.nn.gelu(h     @ params["W_enc2"] + params["b_enc2"])
    return h @ params["W_enc3"] + params["b_enc3"]

def init_params_conv(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    c1, c2, c3 = CONV1_CH
    return {
        "W_conv1"   : n((3, 3,  1, c1), scale=(3*3*1 )**-0.5),
        "b_conv1"   : jnp.zeros(c1),
        "W_conv2"   : n((3, 3, c1, c2), scale=(3*3*c1)**-0.5),
        "b_conv2"   : jnp.zeros(c2),
        "W_conv3"   : n((3, 3, c2, c3), scale=(3*3*c2)**-0.5),
        "b_conv3"   : jnp.zeros(c3),
        "W_fc"      : n((c3, LATENT_DIM),            scale=c3**-0.5),
        "b_fc"      : jnp.zeros(LATENT_DIM),
        "action_emb": n((N_ACTIONS, ACTION_DIM),     scale=0.01),
        "W_pred1"   : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),
        "b_pred1"   : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_s"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_s"  : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_t"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_t"  : jnp.zeros(MLP_PRED_HIDDEN),
        "W_pred2"   : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),
        "b_pred2"   : jnp.zeros(LATENT_DIM),
    }

def encode_conv(params, x):
    h = (x / 255.0).reshape(-1, 28, 28, 1)
    h = jax.nn.gelu(_conv(h, params["W_conv1"], params["b_conv1"]))
    h = _maxpool(h)
    h = jax.nn.gelu(_conv(h, params["W_conv2"], params["b_conv2"]))
    h = _maxpool(h)
    h = jax.nn.gelu(_conv(h, params["W_conv3"], params["b_conv3"]))
    h = h.mean(axis=(1, 2))
    return h @ params["W_fc"] + params["b_fc"]

def init_params_conv2(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    c1, c2, c3 = CONV2_CH
    return {
        "W_conv1"   : n((3, 3,  1, c1), scale=(3*3*1 )**-0.5),
        "b_conv1"   : jnp.zeros(c1),
        "W_conv2"   : n((3, 3, c1, c2), scale=(3*3*c1)**-0.5),
        "b_conv2"   : jnp.zeros(c2),
        "W_conv3"   : n((3, 3, c2, c3), scale=(3*3*c2)**-0.5),
        "b_conv3"   : jnp.zeros(c3),
        "W_fc1"     : n((c3, CONV2_FC),            scale=c3**-0.5),
        "b_fc1"     : jnp.zeros(CONV2_FC),
        "W_fc2"     : n((CONV2_FC, LATENT_DIM),    scale=CONV2_FC**-0.5),
        "b_fc2"     : jnp.zeros(LATENT_DIM),
        "action_emb": n((N_ACTIONS, ACTION_DIM),   scale=0.01),
        "W_pred1"   : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),
        "b_pred1"   : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_s"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_s"  : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_t"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_t"  : jnp.zeros(MLP_PRED_HIDDEN),
        "W_pred2"   : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),
        "b_pred2"   : jnp.zeros(LATENT_DIM),
    }

def encode_conv2(params, x):
    h = (x / 255.0).reshape(-1, 28, 28, 1)
    h = jax.nn.gelu(_conv(h, params["W_conv1"], params["b_conv1"]))
    h = _maxpool(h)
    h = jax.nn.gelu(_conv(h, params["W_conv2"], params["b_conv2"]))
    h = _maxpool(h)
    h = jax.nn.gelu(_conv(h, params["W_conv3"], params["b_conv3"]))
    h = h.mean(axis=(1, 2))
    h = jax.nn.gelu(h @ params["W_fc1"] + params["b_fc1"])
    return h @ params["W_fc2"] + params["b_fc2"]

def _init_vit_common(key_gen):
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

def init_params_vit(key_gen):
    return _init_vit_common(key_gen)

def init_params_vit_patch(key_gen):
    return _init_vit_common(key_gen)

def encode_vit(params, x):
    patches = _extract_patches(x / 255.0)
    h = patches @ params["W_patch"] + params["b_patch"]
    h = h + params["pos_emb"][None, :, :]
    for i in range(N_LAYERS):
        h = _transformer_layer(params, h, i)
    h = _layer_norm(h, params["ln_g"], params["ln_b"])
    h = h.mean(axis=1)
    return h @ params["W_head"] + params["b_head"]

def encode_vit_patch(params, x):
    return encode_vit(params, x)   # same architecture, different masking in loss

def _init_gram_common(key_gen, extra_keys):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    params = {
        "enc_row_emb" : n((28, D_R),   scale=D_R**-0.5),
        "enc_col_emb" : n((28, D_C),   scale=D_C**-0.5),
        "W_pos_enc"   : n((D_RC, D),   scale=D_RC**-0.5),
        "W_enc"       : n((D, DEC_RANK), scale=D**-0.5),
    }
    params.update({k: n(s, sc) for k, s, sc in extra_keys})
    params.update({
        "action_emb"  : n((N_ACTIONS, ACTION_DIM),     scale=0.01),
        "W_pred1"     : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),
        "b_pred1"     : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_s"    : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_s"    : jnp.zeros(MLP_PRED_HIDDEN),
        "W_film_t"    : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),
        "b_film_t"    : jnp.zeros(MLP_PRED_HIDDEN),
        "W_pred2"     : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),
        "b_pred2"     : jnp.zeros(LATENT_DIM),
    })
    return params

def init_params_gram_dual(key_gen):
    return _init_gram_common(key_gen, [
        ("W_enc_A", (DEC_RANK**2, LATENT_DIM), (DEC_RANK**2)**-0.5),
        ("W_enc_B", (DEC_RANK**2, LATENT_DIM), (DEC_RANK**2)**-0.5),
    ])

def init_params_gram_single(key_gen):
    return _init_gram_common(key_gen, [
        ("W_proj", (DEC_RANK**2, LATENT_DIM), (DEC_RANK**2)**-0.5),
    ])

def init_params_gram_glu(key_gen):
    return _init_gram_common(key_gen, [
        ("W_enc_A", (DEC_RANK**2, LATENT_DIM),  (DEC_RANK**2)**-0.5),
        ("W_enc_B", (DEC_RANK**2, LATENT_DIM),  (DEC_RANK**2)**-0.5),
        ("W_gate",  (DEC_RANK**2, DEC_RANK**2), (DEC_RANK**2)**-0.5),
    ])

def encode_gram_dual(params, x):
    x_norm   = x / 255.0
    re       = params["enc_row_emb"][ROWS]
    ce       = params["enc_col_emb"][COLS]
    pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_RC))(re, ce)
    pos_feat = pos_rc @ params["W_pos_enc"]
    f        = pos_feat @ params["W_enc"]
    F        = x_norm[:, :, None] * f[None, :, :]
    H        = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK ** 2)
    a        = H @ params["W_enc_A"]
    b        = H @ params["W_enc_B"]
    return a * b

def encode_gram_single(params, x):
    x_norm   = x / 255.0
    re       = params["enc_row_emb"][ROWS]
    ce       = params["enc_col_emb"][COLS]
    pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_RC))(re, ce)
    pos_feat = pos_rc @ params["W_pos_enc"]
    f        = pos_feat @ params["W_enc"]
    F        = x_norm[:, :, None] * f[None, :, :]
    H        = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK ** 2)
    z        = H @ params["W_proj"]
    return z * H

def encode_gram_glu(params, x):
    x_norm   = x / 255.0
    re       = params["enc_row_emb"][ROWS]
    ce       = params["enc_col_emb"][COLS]
    pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_RC))(re, ce)
    pos_feat = pos_rc @ params["W_pos_enc"]
    f        = pos_feat @ params["W_enc"]
    F        = x_norm[:, :, None] * f[None, :, :]
    H        = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK ** 2)
    gate     = jax.nn.sigmoid(H @ params["W_gate"])
    H_g      = H * gate
    a        = H_g @ params["W_enc_A"]
    b        = H_g @ params["W_enc_B"]
    return a * b

# ── Architecture dispatch tables ──────────────────────────────────────────────

ARCHS = ["mlp", "mlp2", "conv", "conv2", "vit", "vit_patch", "gram_dual", "gram_single", "gram_glu"]

TEACHER_ENCODE_FNS = {
    "mlp"        : teacher_encode_mlp,
    "mlp2"       : teacher_encode_mlp2,
    "conv"       : teacher_encode_conv,
    "conv2"      : teacher_encode_conv2,
    "vit"        : teacher_encode_vit,
    "vit_patch"  : teacher_encode_vit_patch,
    "gram_dual"  : teacher_encode_gram_dual,
    "gram_single": teacher_encode_gram_single,
    "gram_glu"   : teacher_encode_gram_glu,
}

TEACHER_PKLS = {
    "mlp"        : "params_exp_dae_mlp.pkl",
    "mlp2"       : "params_exp_dae_mlp2.pkl",
    "conv"       : "params_exp_dae_conv.pkl",
    "conv2"      : "params_exp_dae_conv2.pkl",
    "vit"        : "params_exp_dae_vit.pkl",
    "vit_patch"  : "params_exp_dae_vit_patch.pkl",
    "gram_dual"  : "params_exp_dae_gram_dual.pkl",
    "gram_single": "params_exp_dae_gram_single.pkl",
    "gram_glu"   : "params_exp_dae_gram_glu.pkl",
}

TEACHER_ARCHS = ARCHS

STUDENT_INIT_FNS = {
    "mlp"        : init_params_mlp,
    "mlp2"       : init_params_mlp2,
    "conv"       : init_params_conv,
    "conv2"      : init_params_conv2,
    "vit"        : init_params_vit,
    "vit_patch"  : init_params_vit_patch,
    "gram_dual"  : init_params_gram_dual,
    "gram_single": init_params_gram_single,
    "gram_glu"   : init_params_gram_glu,
}

STUDENT_ENCODE_FNS = {
    "mlp"        : encode_mlp,
    "mlp2"       : encode_mlp2,
    "conv"       : encode_conv,
    "conv2"      : encode_conv2,
    "vit"        : encode_vit,
    "vit_patch"  : encode_vit_patch,
    "gram_dual"  : encode_gram_dual,
    "gram_single": encode_gram_single,
    "gram_glu"   : encode_gram_glu,
}

# vit_patch uses patch masking, others use pixel masking
PATCH_MASK_STUDENTS = {"vit_patch"}

# ── Predictor (FiLM, shared) ──────────────────────────────────────────────────

def predict(params, e, a):
    h     = jax.nn.gelu(e @ params["W_pred1"] + params["b_pred1"])
    scale = a @ params["W_film_s"] + params["b_film_s"]
    shift = a @ params["W_film_t"] + params["b_film_t"]
    h     = h * (1.0 + scale) + shift
    return h @ params["W_pred2"] + params["b_pred2"]

# ── Epoch function ────────────────────────────────────────────────────────────

def make_epoch_fn(optimizer, teacher_encode_fn, teacher_params, student_encode_fn,
                  use_patch_mask=False):
    def total_loss(online_params, x, key):
        k_idx, k_noise, k_m1, k_m2, k_m3 = jax.random.split(key, 5)
        B = x.shape[0]
        idx   = jax.random.randint(k_idx, (B,), 0, N_ACTIONS)
        a_emb = online_params["action_emb"][idx]
        x_n75 = jnp.clip(x + jax.random.normal(k_noise, x.shape) * NOISE_STD, 0.0, 255.0)
        if use_patch_mask:
            x_m25 = _patch_mask(x, k_m1, MASK_RATIOS[0])
            x_m50 = _patch_mask(x, k_m2, MASK_RATIOS[1])
            x_m75 = _patch_mask(x, k_m3, MASK_RATIOS[2])
        else:
            x_m25 = x * jax.random.bernoulli(k_m1, 1.0 - MASK_RATIOS[0], shape=x.shape)
            x_m50 = x * jax.random.bernoulli(k_m2, 1.0 - MASK_RATIOS[1], shape=x.shape)
            x_m75 = x * jax.random.bernoulli(k_m3, 1.0 - MASK_RATIOS[2], shape=x.shape)
        x_aug = jnp.where(idx[:, None] == 0, x_n75,
                jnp.where(idx[:, None] == 1, x_m25,
                jnp.where(idx[:, None] == 2, x_m50, x_m75)))
        e_target = jax.lax.stop_gradient(teacher_encode_fn(teacher_params, x))
        e_aug    = student_encode_fn(online_params, x_aug)
        e_pred   = predict(online_params, e_aug, a_emb)
        return ((e_pred - e_target) ** 2).mean(), ()

    def scan_body(carry, inputs):
        online_params, opt_state = carry
        x_batch, key = inputs
        (loss, _), grads = jax.value_and_grad(total_loss, has_aux=True)(
            online_params, x_batch, key
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

# ── Linear probe ──────────────────────────────────────────────────────────────

def linear_probe_accuracy(encode_fn, params, X_tr, Y_tr, X_te, Y_te):
    E_tr = encode_all(encode_fn, params, X_tr)
    E_te = encode_all(encode_fn, params, X_te)
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

# ── 1-NN accuracy ─────────────────────────────────────────────────────────────

def knn_1nn_accuracy(encode_fn, params, X_tr, Y_tr, X_te, Y_te):
    E_tr = encode_all(encode_fn, params, X_tr)   # (N_tr, D)
    E_te = encode_all(encode_fn, params, X_te)   # (N_te, D)
    tr_sq = (E_tr ** 2).sum(1)                   # (N_tr,)
    te_sq = (E_te ** 2).sum(1)                   # (N_te,)
    correct = 0
    chunk = 200
    for i in range(0, len(E_te), chunk):
        e_c  = E_te[i:i+chunk]                    # (chunk, D)
        dists = te_sq[i:i+chunk, None] + tr_sq[None, :] - 2.0 * (e_c @ E_tr.T)
        nn_labels = Y_tr[dists.argmin(1)]
        correct  += int((nn_labels == Y_te[i:i+chunk]).sum())
    return float(correct / len(E_te))

# ── Train one pair ────────────────────────────────────────────────────────────

def train_one_pair(teacher_arch, student_arch, teacher_params,
                   X_tr, Y_tr, X_te, Y_te):
    teacher_encode_fn = TEACHER_ENCODE_FNS[teacher_arch]
    student_encode_fn = STUDENT_ENCODE_FNS[student_arch]
    init_fn           = STUDENT_INIT_FNS[student_arch]
    use_patch         = student_arch in PATCH_MASK_STUDENTS

    num_steps     = MAX_EPOCHS * (len(X_tr) // BATCH_SIZE)
    optimizer     = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    key_gen       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    online_params = init_fn(key_gen)
    opt_state     = optimizer.init(online_params)
    epoch_fn      = make_epoch_fn(optimizer, teacher_encode_fn, teacher_params,
                                  student_encode_fn, use_patch)

    key_gen2    = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    num_batches = len(X_tr) // BATCH_SIZE
    t0          = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen2).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        Xs   = jnp.array(
            np.array(X_tr)[perm][:num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE, 784)
        )
        online_params, opt_state, loss = epoch_fn(
            online_params, opt_state, Xs, jax.random.fold_in(key, epoch)
        )
        if (epoch + 1) % EVAL_EVERY == 0:
            logging.info(f"  [{teacher_arch}->{student_arch}] epoch {epoch:3d}  loss={float(loss):.5f}  {time.perf_counter()-t0:.0f}s")

    elapsed = time.perf_counter() - t0
    probe   = linear_probe_accuracy(student_encode_fn, online_params, X_tr, Y_tr, X_te, Y_te)
    knn     = knn_1nn_accuracy(student_encode_fn, online_params, X_tr, Y_tr, X_te, Y_te)
    logging.info(f"  [{teacher_arch}->{student_arch}] probe={probe:.1%}  knn={knn:.1%}  {elapsed:.0f}s")
    return online_params, probe, knn, elapsed

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")

    # Load done set
    done = set()
    if JSONL.exists():
        with open(JSONL) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["teacher_arch"], r["student_arch"]))
                except Exception:
                    pass
    logging.info(f"Already done: {len(done)}/81 pairs")

    # Load data once
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)
    Y_train = np.array(data.y)
    Y_test  = np.array(data.y_test)

    # Main loop — one teacher at a time (reload teacher params once per teacher)
    for teacher_arch in TEACHER_ARCHS:
        pkl = PARAMS_DIR / TEACHER_PKLS[teacher_arch]
        if not pkl.exists():
            logging.warning(f"Teacher params not found: {pkl.name} — skipping")
            continue
        teacher_params = {k: jnp.array(v)
                          for k, v in pickle.load(open(pkl, "rb")).items()}
        logging.info(f"Teacher {teacher_arch}: {n_params(teacher_params):,} params from {pkl.name}")

        for student_arch in ARCHS:
            if (teacher_arch, student_arch) in done:
                logging.info(f"  skip {teacher_arch}->{student_arch} (already done)")
                continue

            logging.info(f"Running {teacher_arch} -> {student_arch}")
            _, probe, knn, elapsed = train_one_pair(
                teacher_arch, student_arch, teacher_params,
                X_train, Y_train, X_test, Y_test
            )
            append_result({
                "teacher_arch"  : teacher_arch,
                "student_arch"  : student_arch,
                "probe_acc"     : round(probe, 6),
                "knn_1nn_acc"   : round(knn,   6),
                "time_s"        : round(elapsed, 1),
                "epochs"        : MAX_EPOCHS,
                "latent_dim"    : LATENT_DIM,
            })
            done.add((teacher_arch, student_arch))
            logging.info(f"  -> probe={probe:.1%}  knn={knn:.1%}  saved.")
