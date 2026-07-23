"""mk_exp_masked_distill.py — generate masked_distill for 8 encoder architectures.

Student learns to predict frozen ConvNet-v2 DAE teacher representations
from augmented inputs.  One file per encoder architecture.
Run: uv run python projects/ssl/scripts/tmp/mk_exp_masked_distill.py
"""
from pathlib import Path

SSL = Path(__file__).parent.parent.parent

# ── Shared loss blocks ────────────────────────────────────────────────────────

LOSS_PIXEL = (
    'def total_loss(online_params, teacher_params, x, key):\n'
    '    k_idx, k_noise, k_m1, k_m2, k_m3 = jax.random.split(key, 5)\n'
    '    B = x.shape[0]\n'
    '\n'
    '    idx   = jax.random.randint(k_idx, (B,), 0, N_ACTIONS)\n'
    '    a_emb = online_params["action_emb"][idx]\n'
    '\n'
    '    x_n75 = jnp.clip(x + jax.random.normal(k_noise, x.shape) * NOISE_STD, 0.0, 255.0)\n'
    '    x_m25 = x * jax.random.bernoulli(k_m1, 1.0 - MASK_RATIOS[0], shape=x.shape)\n'
    '    x_m50 = x * jax.random.bernoulli(k_m2, 1.0 - MASK_RATIOS[1], shape=x.shape)\n'
    '    x_m75 = x * jax.random.bernoulli(k_m3, 1.0 - MASK_RATIOS[2], shape=x.shape)\n'
    '    x_aug = jnp.where(idx[:, None] == 0, x_n75,\n'
    '            jnp.where(idx[:, None] == 1, x_m25,\n'
    '            jnp.where(idx[:, None] == 2, x_m50, x_m75)))\n'
    '\n'
    '    e_target = jax.lax.stop_gradient(teacher_encode(teacher_params, x))\n'
    '    e_aug    = encode(online_params, x_aug)\n'
    '    e_pred   = predict(online_params, e_aug, a_emb)\n'
    '\n'
    '    return ((e_pred - e_target) ** 2).mean(), ()\n'
)

LOSS_PATCH = (
    'def total_loss(online_params, teacher_params, x, key):\n'
    '    k_idx, k_noise, k_m1, k_m2, k_m3 = jax.random.split(key, 5)\n'
    '    B = x.shape[0]\n'
    '\n'
    '    idx   = jax.random.randint(k_idx, (B,), 0, N_ACTIONS)\n'
    '    a_emb = online_params["action_emb"][idx]\n'
    '\n'
    '    x_n75 = jnp.clip(x + jax.random.normal(k_noise, x.shape) * NOISE_STD, 0.0, 255.0)\n'
    '    x_m25 = _patch_mask(x, k_m1, MASK_RATIOS[0])\n'
    '    x_m50 = _patch_mask(x, k_m2, MASK_RATIOS[1])\n'
    '    x_m75 = _patch_mask(x, k_m3, MASK_RATIOS[2])\n'
    '    x_aug = jnp.where(idx[:, None] == 0, x_n75,\n'
    '            jnp.where(idx[:, None] == 1, x_m25,\n'
    '            jnp.where(idx[:, None] == 2, x_m50, x_m75)))\n'
    '\n'
    '    e_target = jax.lax.stop_gradient(teacher_encode(teacher_params, x))\n'
    '    e_aug    = encode(online_params, x_aug)\n'
    '    e_pred   = predict(online_params, e_aug, a_emb)\n'
    '\n'
    '    return ((e_pred - e_target) ** 2).mean(), ()\n'
)

# ── Shared ViT helpers ────────────────────────────────────────────────────────

VIT_HELPERS = (
    'def _layer_norm(x, g, b, eps=1e-5):\n'
    '    mean = x.mean(-1, keepdims=True)\n'
    '    var  = ((x - mean) ** 2).mean(-1, keepdims=True)\n'
    '    return g * (x - mean) / jnp.sqrt(var + eps) + b\n'
    '\n'
    'def _mha(params, x, i):\n'
    '    B, S, D = x.shape\n'
    '    Q = (x @ params[f"W_q_{i}"]).reshape(B, S, N_HEADS, DH).transpose(0, 2, 1, 3)\n'
    '    K = (x @ params[f"W_k_{i}"]).reshape(B, S, N_HEADS, DH).transpose(0, 2, 1, 3)\n'
    '    V = (x @ params[f"W_v_{i}"]).reshape(B, S, N_HEADS, DH).transpose(0, 2, 1, 3)\n'
    '    attn = jax.nn.softmax(Q @ K.transpose(0, 1, 3, 2) / jnp.sqrt(DH), axis=-1)\n'
    '    out  = (attn @ V).transpose(0, 2, 1, 3).reshape(B, S, D)\n'
    '    return out @ params[f"W_o_{i}"]\n'
    '\n'
    'def _ffn(params, x, i):\n'
    '    h = jax.nn.gelu(x @ params[f"W_ff1_{i}"] + params[f"b_ff1_{i}"])\n'
    '    return h @ params[f"W_ff2_{i}"] + params[f"b_ff2_{i}"]\n'
    '\n'
    'def _transformer_layer(params, x, i):\n'
    '    x = x + _mha(params, _layer_norm(x, params[f"ln1_g_{i}"], params[f"ln1_b_{i}"]), i)\n'
    '    x = x + _ffn(params, _layer_norm(x, params[f"ln2_g_{i}"], params[f"ln2_b_{i}"]), i)\n'
    '    return x\n'
    '\n'
    'def _extract_patches(x):\n'
    '    B = x.shape[0]\n'
    '    return x.reshape(B, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(B, 49, 16)\n'
)

VIT_PATCH_HELPERS = VIT_HELPERS + (
    '\n'
    'def _patch_mask(x, key, mask_ratio):\n'
    '    B       = x.shape[0]\n'
    '    patches = x.reshape(B, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(B, 49, 16)\n'
    '    keep    = jax.random.bernoulli(key, 1.0 - mask_ratio, shape=(B, 49))\n'
    '    patches = patches * keep[:, :, None]\n'
    '    return patches.reshape(B, 7, 7, 4, 4).transpose(0, 1, 3, 2, 4).reshape(B, 784)\n'
)

VIT_INIT_PARAMS = (
    'def init_params(key_gen):\n'
    '    def n(shape, scale=0.02):\n'
    '        return jax.random.normal(next(key_gen).get(), shape) * scale\n'
    '    params = {\n'
    '        "W_patch"   : n((PATCH_DIM, D_MODEL),  scale=PATCH_DIM**-0.5),\n'
    '        "b_patch"   : jnp.zeros(D_MODEL),\n'
    '        "pos_emb"   : n((N_PATCHES, D_MODEL),  scale=0.02),\n'
    '    }\n'
    '    for i in range(N_LAYERS):\n'
    '        params.update({\n'
    '            f"W_q_{i}"    : n((D_MODEL, D_MODEL), scale=D_MODEL**-0.5),\n'
    '            f"W_k_{i}"    : n((D_MODEL, D_MODEL), scale=D_MODEL**-0.5),\n'
    '            f"W_v_{i}"    : n((D_MODEL, D_MODEL), scale=D_MODEL**-0.5),\n'
    '            f"W_o_{i}"    : n((D_MODEL, D_MODEL), scale=D_MODEL**-0.5),\n'
    '            f"W_ff1_{i}"  : n((D_MODEL, FFN_DIM), scale=D_MODEL**-0.5),\n'
    '            f"b_ff1_{i}"  : jnp.zeros(FFN_DIM),\n'
    '            f"W_ff2_{i}"  : n((FFN_DIM, D_MODEL), scale=FFN_DIM**-0.5),\n'
    '            f"b_ff2_{i}"  : jnp.zeros(D_MODEL),\n'
    '            f"ln1_g_{i}"  : jnp.ones(D_MODEL),\n'
    '            f"ln1_b_{i}"  : jnp.zeros(D_MODEL),\n'
    '            f"ln2_g_{i}"  : jnp.ones(D_MODEL),\n'
    '            f"ln2_b_{i}"  : jnp.zeros(D_MODEL),\n'
    '        })\n'
    '    params.update({\n'
    '        "ln_g"      : jnp.ones(D_MODEL),\n'
    '        "ln_b"      : jnp.zeros(D_MODEL),\n'
    '        "W_head"    : n((D_MODEL, LATENT_DIM),  scale=D_MODEL**-0.5),\n'
    '        "b_head"    : jnp.zeros(LATENT_DIM),\n'
    '        "action_emb": n((N_ACTIONS, ACTION_DIM), scale=0.01),\n'
    '        "W_pred1"   : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),\n'
    '        "b_pred1"   : jnp.zeros(MLP_PRED_HIDDEN),\n'
    '        "W_film_s"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),\n'
    '        "b_film_s"  : jnp.zeros(MLP_PRED_HIDDEN),\n'
    '        "W_film_t"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),\n'
    '        "b_film_t"  : jnp.zeros(MLP_PRED_HIDDEN),\n'
    '        "W_pred2"   : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),\n'
    '        "b_pred2"   : jnp.zeros(LATENT_DIM),\n'
    '    })\n'
    '    return params\n'
)

VIT_ENCODE = (
    'def encode(params, x):\n'
    '    patches = _extract_patches(x / 255.0)\n'
    '    h = patches @ params["W_patch"] + params["b_patch"]\n'
    '    h = h + params["pos_emb"][None, :, :]\n'
    '    for i in range(N_LAYERS):\n'
    '        h = _transformer_layer(params, h, i)\n'
    '    h = _layer_norm(h, params["ln_g"], params["ln_b"])\n'
    '    h = h.mean(axis=1)\n'
    '    return h @ params["W_head"] + params["b_head"]\n'
)

GRAM_PRECOMPUTED = (
    'ROWS = jnp.arange(784) // 28\n'
    'COLS = jnp.arange(784) % 28\n'
)

# ── Architecture definitions ───────────────────────────────────────────────────

ARCHS = {
    "mlp2": dict(
        encoder_name="mlp2",
        docstring=(
            "exp_masked_distill_mlp2 — MLP2 (3-layer) student + frozen ConvNet-v2 DAE teacher.\n\n"
            "3-layer MLP: 784->1024->1024->LATENT_DIM.\n"
            "Teacher: exp_dae_conv2 (frozen). Loss: MSE(predictor(student(aug_x), action), teacher(clean_x))."
        ),
        arch_hypers="MLP_ENC_H1 = 1024\nMLP_ENC_H2 = 1024",
        precomputed="",
        student_helpers="",
        init_params=(
            'def init_params(key_gen):\n'
            '    def n(shape, scale=0.02):\n'
            '        return jax.random.normal(next(key_gen).get(), shape) * scale\n'
            '    return {\n'
            '        "W_enc1"    : n((784, MLP_ENC_H1),            scale=784**-0.5),\n'
            '        "b_enc1"    : jnp.zeros(MLP_ENC_H1),\n'
            '        "W_enc2"    : n((MLP_ENC_H1, MLP_ENC_H2),     scale=MLP_ENC_H1**-0.5),\n'
            '        "b_enc2"    : jnp.zeros(MLP_ENC_H2),\n'
            '        "W_enc3"    : n((MLP_ENC_H2, LATENT_DIM),     scale=MLP_ENC_H2**-0.5),\n'
            '        "b_enc3"    : jnp.zeros(LATENT_DIM),\n'
            '        "action_emb": n((N_ACTIONS, ACTION_DIM),       scale=0.01),\n'
            '        "W_pred1"   : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),\n'
            '        "b_pred1"   : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_film_s"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),\n'
            '        "b_film_s"  : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_film_t"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),\n'
            '        "b_film_t"  : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_pred2"   : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),\n'
            '        "b_pred2"   : jnp.zeros(LATENT_DIM),\n'
            '    }\n'
        ),
        encode=(
            'def encode(params, x):\n'
            '    x_norm = x / 255.0\n'
            '    h = jax.nn.gelu(x_norm @ params["W_enc1"] + params["b_enc1"])\n'
            '    h = jax.nn.gelu(h     @ params["W_enc2"] + params["b_enc2"])\n'
            '    return h @ params["W_enc3"] + params["b_enc3"]\n'
        ),
        loss=LOSS_PIXEL,
        extra_result_fields=(
            '        "mlp_enc_h1"      : MLP_ENC_H1,\n'
            '        "mlp_enc_h2"      : MLP_ENC_H2,'
        ),
        n_actions="4",
        action_labels='["denoise_gaussian_75", "mask_25", "mask_50", "mask_75"]',
    ),

    "conv": dict(
        encoder_name="conv",
        docstring=(
            "exp_masked_distill_conv — ConvNet-v1 student + frozen ConvNet-v2 DAE teacher.\n\n"
            "ConvNet-v1: Conv(32)+MaxPool -> Conv(64)+MaxPool -> Conv(128)+GAP -> Linear(LATENT_DIM).\n"
            "Teacher: exp_dae_conv2 (frozen). Loss: MSE(predictor(student(aug_x), action), teacher(clean_x))."
        ),
        arch_hypers="CONV_CH = [32, 64, 128]",
        precomputed="",
        student_helpers="",
        init_params=(
            'def init_params(key_gen):\n'
            '    def n(shape, scale=0.02):\n'
            '        return jax.random.normal(next(key_gen).get(), shape) * scale\n'
            '    c1, c2, c3 = CONV_CH\n'
            '    return {\n'
            '        "W_conv1"   : n((3, 3,  1, c1), scale=(3*3*1 )**-0.5),\n'
            '        "b_conv1"   : jnp.zeros(c1),\n'
            '        "W_conv2"   : n((3, 3, c1, c2), scale=(3*3*c1)**-0.5),\n'
            '        "b_conv2"   : jnp.zeros(c2),\n'
            '        "W_conv3"   : n((3, 3, c2, c3), scale=(3*3*c2)**-0.5),\n'
            '        "b_conv3"   : jnp.zeros(c3),\n'
            '        "W_fc"      : n((c3, LATENT_DIM),            scale=c3**-0.5),\n'
            '        "b_fc"      : jnp.zeros(LATENT_DIM),\n'
            '        "action_emb": n((N_ACTIONS, ACTION_DIM),     scale=0.01),\n'
            '        "W_pred1"   : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),\n'
            '        "b_pred1"   : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_film_s"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),\n'
            '        "b_film_s"  : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_film_t"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),\n'
            '        "b_film_t"  : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_pred2"   : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),\n'
            '        "b_pred2"   : jnp.zeros(LATENT_DIM),\n'
            '    }\n'
        ),
        encode=(
            'def encode(params, x):\n'
            '    h = (x / 255.0).reshape(-1, 28, 28, 1)\n'
            '    h = jax.nn.gelu(_conv(h, params["W_conv1"], params["b_conv1"]))\n'
            '    h = _maxpool(h)\n'
            '    h = jax.nn.gelu(_conv(h, params["W_conv2"], params["b_conv2"]))\n'
            '    h = _maxpool(h)\n'
            '    h = jax.nn.gelu(_conv(h, params["W_conv3"], params["b_conv3"]))\n'
            '    h = h.mean(axis=(1, 2))\n'
            '    return h @ params["W_fc"] + params["b_fc"]\n'
        ),
        loss=LOSS_PIXEL,
        extra_result_fields='        "conv_ch"         : CONV_CH,',
        n_actions="4",
        action_labels='["denoise_gaussian_75", "mask_25", "mask_50", "mask_75"]',
    ),

    "conv2": dict(
        encoder_name="conv2",
        docstring=(
            "exp_masked_distill_conv2 — ConvNet-v2 student + frozen ConvNet-v2 DAE teacher.\n\n"
            "ConvNet-v2: Conv(64)+MaxPool -> Conv(128)+MaxPool -> Conv(256)+GAP -> FC(512) -> LATENT_DIM.\n"
            "Same architecture as teacher, different random init.\n"
            "Teacher: exp_dae_conv2 (frozen). Loss: MSE(predictor(student(aug_x), action), teacher(clean_x))."
        ),
        arch_hypers="CONV_CH  = [64, 128, 256]\nFC_HIDDEN = 512",
        precomputed="",
        student_helpers="",
        init_params=(
            'def init_params(key_gen):\n'
            '    def n(shape, scale=0.02):\n'
            '        return jax.random.normal(next(key_gen).get(), shape) * scale\n'
            '    c1, c2, c3 = CONV_CH\n'
            '    return {\n'
            '        "W_conv1"   : n((3, 3,  1, c1), scale=(3*3*1 )**-0.5),\n'
            '        "b_conv1"   : jnp.zeros(c1),\n'
            '        "W_conv2"   : n((3, 3, c1, c2), scale=(3*3*c1)**-0.5),\n'
            '        "b_conv2"   : jnp.zeros(c2),\n'
            '        "W_conv3"   : n((3, 3, c2, c3), scale=(3*3*c2)**-0.5),\n'
            '        "b_conv3"   : jnp.zeros(c3),\n'
            '        "W_fc1"     : n((c3, FC_HIDDEN),            scale=c3**-0.5),\n'
            '        "b_fc1"     : jnp.zeros(FC_HIDDEN),\n'
            '        "W_fc2"     : n((FC_HIDDEN, LATENT_DIM),    scale=FC_HIDDEN**-0.5),\n'
            '        "b_fc2"     : jnp.zeros(LATENT_DIM),\n'
            '        "action_emb": n((N_ACTIONS, ACTION_DIM),    scale=0.01),\n'
            '        "W_pred1"   : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),\n'
            '        "b_pred1"   : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_film_s"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),\n'
            '        "b_film_s"  : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_film_t"  : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),\n'
            '        "b_film_t"  : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_pred2"   : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),\n'
            '        "b_pred2"   : jnp.zeros(LATENT_DIM),\n'
            '    }\n'
        ),
        encode=(
            'def encode(params, x):\n'
            '    h = (x / 255.0).reshape(-1, 28, 28, 1)\n'
            '    h = jax.nn.gelu(_conv(h, params["W_conv1"], params["b_conv1"]))\n'
            '    h = _maxpool(h)\n'
            '    h = jax.nn.gelu(_conv(h, params["W_conv2"], params["b_conv2"]))\n'
            '    h = _maxpool(h)\n'
            '    h = jax.nn.gelu(_conv(h, params["W_conv3"], params["b_conv3"]))\n'
            '    h = h.mean(axis=(1, 2))\n'
            '    h = jax.nn.gelu(h @ params["W_fc1"] + params["b_fc1"])\n'
            '    return h @ params["W_fc2"] + params["b_fc2"]\n'
        ),
        loss=LOSS_PIXEL,
        extra_result_fields=(
            '        "conv_ch"         : CONV_CH,\n'
            '        "fc_hidden"       : FC_HIDDEN,'
        ),
        n_actions="4",
        action_labels='["denoise_gaussian_75", "mask_25", "mask_50", "mask_75"]',
    ),

    "vit": dict(
        encoder_name="vit",
        docstring=(
            "exp_masked_distill_vit — ViT student (pixel masking) + frozen ConvNet-v2 DAE teacher.\n\n"
            "ViT: 4x4 patches (49 total), D_MODEL=128, 3 layers, 4 heads -> LATENT_DIM=1024.\n"
            "Teacher: exp_dae_conv2 (frozen). Loss: MSE(predictor(student(aug_x), action), teacher(clean_x))."
        ),
        arch_hypers=(
            "PATCH_SIZE      = 4\n"
            "N_PATCHES       = 49\n"
            "PATCH_DIM       = 16\n"
            "D_MODEL         = 128\n"
            "N_HEADS         = 4\n"
            "DH              = D_MODEL // N_HEADS\n"
            "FFN_DIM         = 256\n"
            "N_LAYERS        = 3"
        ),
        precomputed="",
        student_helpers=VIT_HELPERS,
        init_params=VIT_INIT_PARAMS,
        encode=VIT_ENCODE,
        loss=LOSS_PIXEL,
        extra_result_fields=(
            '        "d_model"         : D_MODEL,\n'
            '        "n_layers"        : N_LAYERS,'
        ),
        n_actions="4",
        action_labels='["denoise_gaussian_75", "mask_25", "mask_50", "mask_75"]',
    ),

    "vit_patch": dict(
        encoder_name="vit_patch",
        docstring=(
            "exp_masked_distill_vit_patch — ViT student (patch masking) + frozen ConvNet-v2 DAE teacher.\n\n"
            "ViT: 4x4 patches (49 total), D_MODEL=128, 3 layers, 4 heads -> LATENT_DIM=1024.\n"
            "Uses patch-level masking (whole 4x4 patches zeroed).\n"
            "Teacher: exp_dae_conv2 (frozen). Loss: MSE(predictor(student(aug_x), action), teacher(clean_x))."
        ),
        arch_hypers=(
            "PATCH_SIZE      = 4\n"
            "N_PATCHES       = 49\n"
            "PATCH_DIM       = 16\n"
            "D_MODEL         = 128\n"
            "N_HEADS         = 4\n"
            "DH              = D_MODEL // N_HEADS\n"
            "FFN_DIM         = 256\n"
            "N_LAYERS        = 3"
        ),
        precomputed="",
        student_helpers=VIT_PATCH_HELPERS,
        init_params=VIT_INIT_PARAMS,
        encode=VIT_ENCODE,
        loss=LOSS_PATCH,
        extra_result_fields=(
            '        "d_model"         : D_MODEL,\n'
            '        "n_layers"        : N_LAYERS,'
        ),
        n_actions="4",
        action_labels='["denoise_gaussian_75", "patch_mask_25", "patch_mask_50", "patch_mask_75"]',
    ),

    "gram_dual": dict(
        encoder_name="gram_dual",
        docstring=(
            "exp_masked_distill_gram_dual — Gram-dual student + frozen ConvNet-v2 DAE teacher.\n\n"
            "Gram-dual: Hebbian gram matrix + dual Hadamard readout (W_enc_A * W_enc_B).\n"
            "Teacher: exp_dae_conv2 (frozen). Loss: MSE(predictor(student(aug_x), action), teacher(clean_x))."
        ),
        arch_hypers=(
            "D_R, D_C, D_V = 6, 6, 4\n"
            "D              = D_R * D_C * D_V   # 144\n"
            "D_RC           = D_R * D_C         # 36\n"
            "DEC_RANK       = 32    # DEC_RANK^2 = 1024 = LATENT_DIM"
        ),
        precomputed=GRAM_PRECOMPUTED,
        student_helpers="",
        init_params=(
            'def init_params(key_gen):\n'
            '    def n(shape, scale=0.02):\n'
            '        return jax.random.normal(next(key_gen).get(), shape) * scale\n'
            '    return {\n'
            '        "enc_row_emb" : n((28, D_R),                   scale=D_R**-0.5),\n'
            '        "enc_col_emb" : n((28, D_C),                   scale=D_C**-0.5),\n'
            '        "W_pos_enc"   : n((D_RC, D),                   scale=D_RC**-0.5),\n'
            '        "W_enc"       : n((D, DEC_RANK),               scale=D**-0.5),\n'
            '        "W_enc_A"     : n((DEC_RANK**2, LATENT_DIM),   scale=(DEC_RANK**2)**-0.5),\n'
            '        "W_enc_B"     : n((DEC_RANK**2, LATENT_DIM),   scale=(DEC_RANK**2)**-0.5),\n'
            '        "action_emb"  : n((N_ACTIONS, ACTION_DIM),     scale=0.01),\n'
            '        "W_pred1"     : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),\n'
            '        "b_pred1"     : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_film_s"    : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),\n'
            '        "b_film_s"    : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_film_t"    : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),\n'
            '        "b_film_t"    : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_pred2"     : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),\n'
            '        "b_pred2"     : jnp.zeros(LATENT_DIM),\n'
            '    }\n'
        ),
        encode=(
            'def encode(params, x):\n'
            '    x_norm   = x / 255.0\n'
            '    re       = params["enc_row_emb"][ROWS]\n'
            '    ce       = params["enc_col_emb"][COLS]\n'
            '    pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_RC))(re, ce)\n'
            '    pos_feat = pos_rc @ params["W_pos_enc"]\n'
            '    f        = pos_feat @ params["W_enc"]\n'
            '    F        = x_norm[:, :, None] * f[None, :, :]\n'
            '    H        = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK ** 2)\n'
            '    a        = H @ params["W_enc_A"]\n'
            '    b        = H @ params["W_enc_B"]\n'
            '    return a * b\n'
        ),
        loss=LOSS_PIXEL,
        extra_result_fields='        "dec_rank"        : DEC_RANK,',
        n_actions="4",
        action_labels='["denoise_gaussian_75", "mask_25", "mask_50", "mask_75"]',
    ),

    "gram_single": dict(
        encoder_name="gram_single",
        docstring=(
            "exp_masked_distill_gram_single — Gram-single student + frozen ConvNet-v2 DAE teacher.\n\n"
            "Gram-single: Hebbian gram matrix + single-projection Hadamard readout (z * H).\n"
            "Teacher: exp_dae_conv2 (frozen). Loss: MSE(predictor(student(aug_x), action), teacher(clean_x))."
        ),
        arch_hypers=(
            "D_R, D_C, D_V = 6, 6, 4\n"
            "D              = D_R * D_C * D_V   # 144\n"
            "D_RC           = D_R * D_C         # 36\n"
            "DEC_RANK       = 32    # DEC_RANK^2 = 1024 = LATENT_DIM"
        ),
        precomputed=GRAM_PRECOMPUTED,
        student_helpers="",
        init_params=(
            'def init_params(key_gen):\n'
            '    def n(shape, scale=0.02):\n'
            '        return jax.random.normal(next(key_gen).get(), shape) * scale\n'
            '    return {\n'
            '        "enc_row_emb" : n((28, D_R),                   scale=D_R**-0.5),\n'
            '        "enc_col_emb" : n((28, D_C),                   scale=D_C**-0.5),\n'
            '        "W_pos_enc"   : n((D_RC, D),                   scale=D_RC**-0.5),\n'
            '        "W_enc"       : n((D, DEC_RANK),               scale=D**-0.5),\n'
            '        "W_proj"      : n((DEC_RANK**2, LATENT_DIM),   scale=(DEC_RANK**2)**-0.5),\n'
            '        "action_emb"  : n((N_ACTIONS, ACTION_DIM),     scale=0.01),\n'
            '        "W_pred1"     : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),\n'
            '        "b_pred1"     : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_film_s"    : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),\n'
            '        "b_film_s"    : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_film_t"    : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),\n'
            '        "b_film_t"    : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_pred2"     : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),\n'
            '        "b_pred2"     : jnp.zeros(LATENT_DIM),\n'
            '    }\n'
        ),
        encode=(
            'def encode(params, x):\n'
            '    x_norm   = x / 255.0\n'
            '    re       = params["enc_row_emb"][ROWS]\n'
            '    ce       = params["enc_col_emb"][COLS]\n'
            '    pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_RC))(re, ce)\n'
            '    pos_feat = pos_rc @ params["W_pos_enc"]\n'
            '    f        = pos_feat @ params["W_enc"]\n'
            '    F        = x_norm[:, :, None] * f[None, :, :]\n'
            '    H        = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK ** 2)\n'
            '    z        = H @ params["W_proj"]\n'
            '    return z * H\n'
        ),
        loss=LOSS_PIXEL,
        extra_result_fields='        "dec_rank"        : DEC_RANK,',
        n_actions="4",
        action_labels='["denoise_gaussian_75", "mask_25", "mask_50", "mask_75"]',
    ),

    "gram_glu": dict(
        encoder_name="gram_glu",
        docstring=(
            "exp_masked_distill_gram_glu — Gram-GLU student + frozen ConvNet-v2 DAE teacher.\n\n"
            "Gram-GLU: Hebbian gram matrix + sigmoid gate + dual Hadamard readout.\n"
            "Teacher: exp_dae_conv2 (frozen). Loss: MSE(predictor(student(aug_x), action), teacher(clean_x))."
        ),
        arch_hypers=(
            "D_R, D_C, D_V = 6, 6, 4\n"
            "D              = D_R * D_C * D_V   # 144\n"
            "D_RC           = D_R * D_C         # 36\n"
            "DEC_RANK       = 32    # DEC_RANK^2 = 1024 = LATENT_DIM"
        ),
        precomputed=GRAM_PRECOMPUTED,
        student_helpers="",
        init_params=(
            'def init_params(key_gen):\n'
            '    def n(shape, scale=0.02):\n'
            '        return jax.random.normal(next(key_gen).get(), shape) * scale\n'
            '    return {\n'
            '        "enc_row_emb" : n((28, D_R),                   scale=D_R**-0.5),\n'
            '        "enc_col_emb" : n((28, D_C),                   scale=D_C**-0.5),\n'
            '        "W_pos_enc"   : n((D_RC, D),                   scale=D_RC**-0.5),\n'
            '        "W_enc"       : n((D, DEC_RANK),               scale=D**-0.5),\n'
            '        "W_enc_A"     : n((DEC_RANK**2, LATENT_DIM),   scale=(DEC_RANK**2)**-0.5),\n'
            '        "W_enc_B"     : n((DEC_RANK**2, LATENT_DIM),   scale=(DEC_RANK**2)**-0.5),\n'
            '        "W_gate"      : n((DEC_RANK**2, DEC_RANK**2),  scale=(DEC_RANK**2)**-0.5),\n'
            '        "action_emb"  : n((N_ACTIONS, ACTION_DIM),     scale=0.01),\n'
            '        "W_pred1"     : n((LATENT_DIM, MLP_PRED_HIDDEN), scale=LATENT_DIM**-0.5),\n'
            '        "b_pred1"     : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_film_s"    : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),\n'
            '        "b_film_s"    : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_film_t"    : n((ACTION_DIM, MLP_PRED_HIDDEN), scale=ACTION_DIM**-0.5),\n'
            '        "b_film_t"    : jnp.zeros(MLP_PRED_HIDDEN),\n'
            '        "W_pred2"     : n((MLP_PRED_HIDDEN, LATENT_DIM), scale=MLP_PRED_HIDDEN**-0.5),\n'
            '        "b_pred2"     : jnp.zeros(LATENT_DIM),\n'
            '    }\n'
        ),
        encode=(
            'def encode(params, x):\n'
            '    x_norm   = x / 255.0\n'
            '    re       = params["enc_row_emb"][ROWS]\n'
            '    ce       = params["enc_col_emb"][COLS]\n'
            '    pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_RC))(re, ce)\n'
            '    pos_feat = pos_rc @ params["W_pos_enc"]\n'
            '    f        = pos_feat @ params["W_enc"]\n'
            '    F        = x_norm[:, :, None] * f[None, :, :]\n'
            '    H        = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK ** 2)\n'
            '    gate     = jax.nn.sigmoid(H @ params["W_gate"])\n'
            '    H_g      = H * gate\n'
            '    a        = H_g @ params["W_enc_A"]\n'
            '    b        = H_g @ params["W_enc_B"]\n'
            '    return a * b\n'
        ),
        loss=LOSS_PIXEL,
        extra_result_fields='        "dec_rank"        : DEC_RANK,',
        n_actions="4",
        action_labels='["denoise_gaussian_75", "mask_25", "mask_50", "mask_75"]',
    ),
}

# ── Template ───────────────────────────────────────────────────────────────────
# Markers: EXP_NAME_M, ENCODER_NAME_M, DOCSTRING_M, ARCH_HYPERS_M, PRECOMPUTED_M,
#          STUDENT_HELPERS_M, INIT_PARAMS_M, ENCODE_M, LOSS_M,
#          N_ACTIONS_M, ACTION_LABELS_M, EXTRA_RESULT_FIELDS_M

TEMPLATE = (
    '"""\n'
    'DOCSTRING_M\n'
    '"""\n'
    '\n'
    'import json, pickle, sys, time\n'
    'from pathlib import Path\n'
    '\n'
    'import jax\n'
    'import jax.numpy as jnp\n'
    'import optax\n'
    'import numpy as np\n'
    'import logging\n'
    '\n'
    'sys.path.insert(0, str(Path(__file__).parent.parent.parent))\n'
    'from shared_lib.random_utils import infinite_safe_keys_from_key\n'
    'from shared_lib.datasets import load_supervised_image\n'
    '\n'
    'logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")\n'
    '\n'
    'EXP_NAME = "EXP_NAME_M"\n'
    '\n'
    '# \u2500\u2500 Hyperparameters \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
    '\n'
    'BATCH_SIZE      = 256\n'
    'SEED            = 42\n'
    'MAX_EPOCHS      = 200\n'
    'ADAM_LR         = 3e-4\n'
    'EVAL_EVERY      = 10\n'
    'PROBE_EPOCHS    = 10\n'
    '\n'
    'LATENT_DIM      = 1024\n'
    'MLP_PRED_HIDDEN = 512\n'
    '\n'
    'N_ACTIONS     = N_ACTIONS_M\n'
    'ACTION_DIM    = 32\n'
    'ACTION_LABELS = ACTION_LABELS_M\n'
    '\n'
    'NOISE_STD   = 75.0\n'
    'MASK_RATIOS = [0.25, 0.50, 0.75]\n'
    '\n'
    'TEACHER_PKL = Path(__file__).parent / "params_exp_dae_conv2.pkl"\n'
    '\n'
    'JSONL = Path(__file__).parent / "results.jsonl"\n'
    '\n'
    'ARCH_HYPERS_M\n'
    '\n'
    'PRECOMPUTED_M'
    '# \u2500\u2500 Utilities \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
    '\n'
    'def n_params(p):\n'
    '    return sum(v.size for v in jax.tree_util.tree_leaves(p))\n'
    '\n'
    'def append_result(row):\n'
    '    with open(JSONL, "a") as f:\n'
    '        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\\n")\n'
    '\n'
    '# \u2500\u2500 Teacher encoder (ConvNet-v2, frozen) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
    '\n'
    '_DN = ("NHWC", "HWIO", "NHWC")\n'
    '\n'
    'def _conv(x, W, b):\n'
    '    return jax.lax.conv_general_dilated(x, W, (1, 1), "SAME", dimension_numbers=_DN) + b\n'
    '\n'
    'def _maxpool(x):\n'
    '    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")\n'
    '\n'
    'def teacher_encode(params, x):\n'
    '    h = (x / 255.0).reshape(-1, 28, 28, 1)\n'
    '    h = jax.nn.gelu(_conv(h, params["W_conv1"], params["b_conv1"]))\n'
    '    h = _maxpool(h)\n'
    '    h = jax.nn.gelu(_conv(h, params["W_conv2"], params["b_conv2"]))\n'
    '    h = _maxpool(h)\n'
    '    h = jax.nn.gelu(_conv(h, params["W_conv3"], params["b_conv3"]))\n'
    '    h = h.mean(axis=(1, 2))\n'
    '    h = jax.nn.gelu(h @ params["W_fc1"] + params["b_fc1"])\n'
    '    return h @ params["W_fc2"] + params["b_fc2"]\n'
    '\n'
    '# \u2500\u2500 Student helpers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
    '\n'
    'STUDENT_HELPERS_M'
    '# \u2500\u2500 Student parameters \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
    '\n'
    'INIT_PARAMS_M\n'
    '# \u2500\u2500 Student encoder \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
    '\n'
    'ENCODE_M\n'
    '# \u2500\u2500 Predictor (FiLM) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
    '\n'
    'def predict(params, e, a):\n'
    '    h     = jax.nn.gelu(e @ params["W_pred1"] + params["b_pred1"])\n'
    '    scale = a @ params["W_film_s"] + params["b_film_s"]\n'
    '    shift = a @ params["W_film_t"] + params["b_film_t"]\n'
    '    h     = h * (1.0 + scale) + shift\n'
    '    return h @ params["W_pred2"] + params["b_pred2"]\n'
    '\n'
    '# \u2500\u2500 Loss \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
    '\n'
    'LOSS_M\n'
    '# \u2500\u2500 Epoch function \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
    '\n'
    'def make_epoch_fn(optimizer, teacher_params):\n'
    '    def scan_body(carry, inputs):\n'
    '        online_params, opt_state = carry\n'
    '        x_batch, key = inputs\n'
    '        (loss, _), grads = jax.value_and_grad(total_loss, has_aux=True)(\n'
    '            online_params, teacher_params, x_batch, key\n'
    '        )\n'
    '        updates, new_opt_state = optimizer.update(grads, opt_state, online_params)\n'
    '        return (optax.apply_updates(online_params, updates), new_opt_state), loss\n'
    '\n'
    '    def epoch_fn(online_params, opt_state, Xs_batched, epoch_key):\n'
    '        num_batches = Xs_batched.shape[0]\n'
    '        keys = jax.vmap(lambda b: jax.random.fold_in(epoch_key, b))(jnp.arange(num_batches))\n'
    '        (online_params, opt_state), losses = jax.lax.scan(\n'
    '            scan_body, (online_params, opt_state), (Xs_batched, keys)\n'
    '        )\n'
    '        return online_params, opt_state, losses.mean()\n'
    '\n'
    '    return jax.jit(epoch_fn)\n'
    '\n'
    '# \u2500\u2500 Linear probe \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
    '\n'
    'def linear_probe_accuracy(params, X_tr, Y_tr, X_te, Y_te):\n'
    '    def encode_all(X):\n'
    '        parts = []\n'
    '        for i in range(0, len(X), 512):\n'
    '            parts.append(np.array(encode(params, jnp.array(np.array(X[i:i+512])))))\n'
    '        return np.concatenate(parts, 0)\n'
    '\n'
    '    E_tr = encode_all(X_tr)\n'
    '    E_te = encode_all(X_te)\n'
    '    n_tr = (len(E_tr) // BATCH_SIZE) * BATCH_SIZE\n'
    '    E_b  = jnp.array(E_tr[:n_tr].reshape(-1, BATCH_SIZE, LATENT_DIM))\n'
    '    Y_b  = jnp.array(np.array(Y_tr[:n_tr]).reshape(-1, BATCH_SIZE))\n'
    '\n'
    '    W, b        = jnp.zeros((LATENT_DIM, 10)), jnp.zeros(10)\n'
    '    probe_opt   = optax.adam(1e-3)\n'
    '    probe_state = probe_opt.init((W, b))\n'
    '\n'
    '    def probe_step(carry, inputs):\n'
    '        (W, b), state = carry\n'
    '        e, y = inputs\n'
    '        def ce(wb):\n'
    '            return optax.softmax_cross_entropy_with_integer_labels(e @ wb[0] + wb[1], y).mean()\n'
    '        loss, grads = jax.value_and_grad(ce)((W, b))\n'
    '        updates, new_state = probe_opt.update(grads, state, (W, b))\n'
    '        return (optax.apply_updates((W, b), updates), new_state), loss\n'
    '\n'
    '    @jax.jit\n'
    '    def probe_epoch(W, b, state, E_b, Y_b):\n'
    '        (W, b), state = jax.lax.scan(probe_step, ((W, b), state), (E_b, Y_b))[0]\n'
    '        return W, b, state\n'
    '\n'
    '    for _ in range(PROBE_EPOCHS):\n'
    '        W, b, probe_state = probe_epoch(W, b, probe_state, E_b, Y_b)\n'
    '\n'
    '    logits = jnp.array(E_te) @ W + b\n'
    '    return float((jnp.argmax(logits, 1) == jnp.array(np.array(Y_te))).mean())\n'
    '\n'
    '# \u2500\u2500 Training loop \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
    '\n'
    'def train(online_params, teacher_params, X_tr, Y_tr, X_te, Y_te):\n'
    '    num_steps   = MAX_EPOCHS * (len(X_tr) // BATCH_SIZE)\n'
    '    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))\n'
    '    opt_state   = optimizer.init(online_params)\n'
    '    epoch_fn    = make_epoch_fn(optimizer, teacher_params)\n'
    '    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))\n'
    '    num_batches = len(X_tr) // BATCH_SIZE\n'
    '    history     = {"loss": [], "probe_acc": []}\n'
    '    t0          = time.perf_counter()\n'
    '\n'
    '    for epoch in range(MAX_EPOCHS):\n'
    '        key  = next(key_gen).get()\n'
    '        perm = np.array(jax.random.permutation(key, len(X_tr)))\n'
    '        Xs   = jnp.array(\n'
    '            np.array(X_tr)[perm][:num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE, 784)\n'
    '        )\n'
    '        online_params, opt_state, loss = epoch_fn(\n'
    '            online_params, opt_state, Xs, jax.random.fold_in(key, epoch)\n'
    '        )\n'
    '        loss = float(loss)\n'
    '        history["loss"].append(loss)\n'
    '\n'
    '        probe_str = ""\n'
    '        if (epoch + 1) % EVAL_EVERY == 0:\n'
    '            acc = linear_probe_accuracy(online_params, X_tr, Y_tr, X_te, Y_te)\n'
    '            history["probe_acc"].append((epoch, acc))\n'
    '            probe_str = f"  probe_acc={acc:.1%}"\n'
    '\n'
    '        logging.info(f"epoch {epoch:3d}  loss={loss:.6f}{probe_str}  {time.perf_counter()-t0:.1f}s")\n'
    '\n'
    '    return online_params, history, time.perf_counter() - t0\n'
    '\n'
    '# \u2500\u2500 Main \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
    '\n'
    'if __name__ == "__main__":\n'
    '    logging.info(f"JAX devices: {jax.devices()}")\n'
    '\n'
    '    done = set()\n'
    '    if JSONL.exists():\n'
    '        with open(JSONL) as f:\n'
    '            for line in f:\n'
    '                try: done.add(json.loads(line).get("experiment"))\n'
    '                except Exception: pass\n'
    '    if EXP_NAME in done:\n'
    '        logging.info(f"{EXP_NAME} already in results.jsonl \u2014 skipping")\n'
    '        exit(0)\n'
    '\n'
    '    logging.info(f"Loading teacher from {TEACHER_PKL.name}")\n'
    '    teacher_params = {k: jnp.array(v) for k, v in pickle.load(open(TEACHER_PKL, "rb")).items()}\n'
    '    logging.info(f"Teacher params: {n_params(teacher_params):,}")\n'
    '\n'
    '    data    = load_supervised_image("mnist")\n'
    '    X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)\n'
    '    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)\n'
    '    Y_train = np.array(data.y)\n'
    '    Y_test  = np.array(data.y_test)\n'
    '\n'
    '    key_gen       = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))\n'
    '    online_params = init_params(key_gen)\n'
    '    logging.info(f"Student params: {n_params(online_params):,}")\n'
    '\n'
    '    online_params, history, elapsed = train(\n'
    '        online_params, teacher_params, X_train, Y_train, X_test, Y_test\n'
    '    )\n'
    '\n'
    '    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:\n'
    '        pickle.dump({k: np.array(v) for k, v in online_params.items()}, f)\n'
    '    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:\n'
    '        pickle.dump(history, f)\n'
    '\n'
    '    final_acc = linear_probe_accuracy(online_params, X_train, Y_train, X_test, Y_test)\n'
    '    append_result({\n'
    '        "experiment"      : EXP_NAME,\n'
    '        "n_params"        : n_params(online_params),\n'
    '        "epochs"          : MAX_EPOCHS,\n'
    '        "time_s"          : round(elapsed, 1),\n'
    '        "final_probe_acc" : round(final_acc, 6),\n'
    '        "loss_curve"      : [round(v, 6) for v in history["loss"]],\n'
    '        "probe_acc_curve" : history["probe_acc"],\n'
    '        "latent_dim"      : LATENT_DIM,\n'
    '        "adam_lr"         : ADAM_LR,\n'
    '        "batch_size"      : BATCH_SIZE,\n'
    '        "encoder"         : "ENCODER_NAME_M",\n'
    '        "teacher"         : "exp_dae_conv2",\n'
    '        "method"          : "masked_distill",\n'
    'EXTRA_RESULT_FIELDS_M\n'
    '    })\n'
    '    logging.info(f"Done. probe_acc={final_acc:.1%}  {elapsed:.1f}s")\n'
)

# ── Generate files ─────────────────────────────────────────────────────────────

for arch, d in ARCHS.items():
    exp_name = f"exp_masked_distill_{arch}"

    # Build precomputed + helpers sections (add trailing newline for spacing)
    precomputed_block = d["precomputed"] + "\n" if d["precomputed"] else ""
    helpers_block     = d["student_helpers"] + "\n" if d["student_helpers"] else ""

    content = TEMPLATE
    content = content.replace("DOCSTRING_M",            d["docstring"])
    content = content.replace("EXP_NAME_M",             exp_name)
    content = content.replace("ENCODER_NAME_M",         d["encoder_name"])
    content = content.replace("N_ACTIONS_M",            d["n_actions"])
    content = content.replace("ACTION_LABELS_M",        d["action_labels"])
    content = content.replace("ARCH_HYPERS_M",          d["arch_hypers"])
    content = content.replace("PRECOMPUTED_M",          precomputed_block)
    content = content.replace("STUDENT_HELPERS_M",      helpers_block)
    content = content.replace("INIT_PARAMS_M",          d["init_params"])
    content = content.replace("ENCODE_M",               d["encode"])
    content = content.replace("LOSS_M",                 d["loss"])
    content = content.replace("EXTRA_RESULT_FIELDS_M",  d["extra_result_fields"])

    out = SSL / f"exp_masked_distill_{arch}.py"
    out.write_text(content)
    print(f"wrote {out.name}")
