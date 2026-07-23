"""
Generate two new encoder variants + their MNIST/Fashion-MNIST LATENT=1024 runs:

  exp_vit_patch_ema      — ViT with patch-level masking augmentation
  exp_gram_glu_ema       — Gram + sigmoid GLU gate on H before Hadamard

  fmnist_exp_vit_patch_ema
  fmnist_exp_gram_glu_ema
"""

from pathlib import Path

SSL = Path(__file__).parent.parent.parent

# ══════════════════════════════════════════════════════════════════════════════
# 1.  ViT with patch-level masking
# ══════════════════════════════════════════════════════════════════════════════

src = (SSL / "exp_vit_ema.py").read_text()

# a) docstring / description
src = src.replace(
    "exp_vit_ema — LeWM JEPA on MNIST: mini Vision Transformer encoder + EMA, LATENT_DIM=1024.",
    "exp_vit_patch_ema — ViT encoder + EMA with PATCH-LEVEL masking augmentation, LATENT_DIM=1024.\n\n"
    "Replaces flat per-pixel Bernoulli masking with 4×4 patch-level masking so that\n"
    "the ViT's spatial inductive bias is properly challenged during SSL training."
)
src = src.replace('EXP_NAME = "exp_vit_ema"', 'EXP_NAME = "exp_vit_patch_ema"')
src = src.replace(
    '    uv run python projects/ssl/exp_vit_ema.py',
    '    uv run python projects/ssl/exp_vit_patch_ema.py'
)

# b) action labels
src = src.replace(
    'ACTION_LABELS = ["denoise_gaussian_75", "mask_25", "mask_50", "mask_75"]',
    'ACTION_LABELS = ["denoise_gaussian_75", "patch_mask_25", "patch_mask_50", "patch_mask_75"]'
)

# c) inject _patch_mask helper just before total_loss
patch_mask_fn = '''
# ── Patch-level masking ───────────────────────────────────────────────────────

def _patch_mask(x, key, mask_ratio):
    """Zero entire 4×4 patches at given ratio.  x: (B, 784) → (B, 784)."""
    B       = x.shape[0]
    patches = x.reshape(B, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(B, 49, 16)
    keep    = jax.random.bernoulli(key, 1.0 - mask_ratio, shape=(B, 49))
    patches = patches * keep[:, :, None]
    return patches.reshape(B, 7, 7, 4, 4).transpose(0, 1, 3, 2, 4).reshape(B, 784)

'''
src = src.replace("# ── Loss ──", patch_mask_fn + "# ── Loss ──")

# d) swap flat pixel masking for patch masking inside total_loss
src = src.replace(
    "    x_m25 = x * jax.random.bernoulli(k_m1, 1.0 - MASK_RATIOS[0], shape=x.shape)\n"
    "    x_m50 = x * jax.random.bernoulli(k_m2, 1.0 - MASK_RATIOS[1], shape=x.shape)\n"
    "    x_m75 = x * jax.random.bernoulli(k_m3, 1.0 - MASK_RATIOS[2], shape=x.shape)",
    "    x_m25 = _patch_mask(x, k_m1, MASK_RATIOS[0])\n"
    "    x_m50 = _patch_mask(x, k_m2, MASK_RATIOS[1])\n"
    "    x_m75 = _patch_mask(x, k_m3, MASK_RATIOS[2])"
)

# e) update result name
src = src.replace(
    '"name"            : f"jepa-vit-ema aug LATENT={LATENT_DIM} D={D_MODEL} L={N_LAYERS}"',
    '"name"            : f"jepa-vit-patch-ema LATENT={LATENT_DIM} D={D_MODEL} L={N_LAYERS}"'
)
src = src.replace('"encoder"         : "vit"', '"encoder"         : "vit_patch"')

(SSL / "exp_vit_patch_ema.py").write_text(src)
print("wrote exp_vit_patch_ema.py")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Gram + GLU gate on H before Hadamard
# ══════════════════════════════════════════════════════════════════════════════

src = (SSL / "exp_ema.py").read_text()

# a) pin LATENT to 1024 (base file is 4096)
src = src.replace("LATENT_DIM   = 4096", "LATENT_DIM   = 1024")

# b) docstring / description / EXP_NAME
src = src.replace(
    "exp_ema — LeWM JEPA on MNIST: EMA target encoder baseline, LATENT_DIM=4096.",
    "exp_gram_glu_ema — Gram+Hadamard with sigmoid GLU gate on H, EMA, LATENT_DIM=1024.\n\n"
    "Adds a learned input-dependent gate on the flattened gram H before the\n"
    "Hadamard readout:\n"
    "  gate = sigmoid(H @ W_gate)   # (B, DEC_RANK²)\n"
    "  H_g  = H * gate              # selectively suppress gram co-activations\n"
    "  out  = (H_g @ W_A) ⊙ (H_g @ W_B)"
)
src = src.replace('EXP_NAME = "exp_ema"', 'EXP_NAME = "exp_gram_glu_ema"')
src = src.replace(
    '    uv run python projects/ssl/exp_ema.py',
    '    uv run python projects/ssl/exp_gram_glu_ema.py'
)

# c) add W_gate to init_params (after W_enc_B line)
src = src.replace(
    '        "W_enc_B"     : n((DEC_RANK**2, LATENT_DIM),   scale=(DEC_RANK**2)**-0.5),',
    '        "W_enc_B"     : n((DEC_RANK**2, LATENT_DIM),   scale=(DEC_RANK**2)**-0.5),\n'
    '        # GLU gate — operates in gram space (DEC_RANK² → DEC_RANK²)\n'
    '        "W_gate"      : n((DEC_RANK**2, DEC_RANK**2),  scale=(DEC_RANK**2)**-0.5),'
)

# d) modify encode() to apply gate
src = src.replace(
    "    H = jnp.einsum(\"bpi,bpj->bij\", F, F).reshape(x.shape[0], DEC_RANK ** 2)\n"
    "    a = H @ params[\"W_enc_A\"]\n"
    "    b = H @ params[\"W_enc_B\"]\n"
    "    return a * b",
    "    H    = jnp.einsum(\"bpi,bpj->bij\", F, F).reshape(x.shape[0], DEC_RANK ** 2)\n"
    "    gate = jax.nn.sigmoid(H @ params[\"W_gate\"])  # input-dependent gram gate\n"
    "    H_g  = H * gate\n"
    "    a    = H_g @ params[\"W_enc_A\"]\n"
    "    b    = H_g @ params[\"W_enc_B\"]\n"
    "    return a * b"
)

# e) update result name and encoder tag
src = src.replace(
    '"name"            : f"jepa-ema LATENT={LATENT_DIM} DEC_RANK={DEC_RANK} tau={EMA_TAU}"',
    '"name"            : f"jepa-gram-glu-ema LATENT={LATENT_DIM} DEC_RANK={DEC_RANK}"'
)
src = src.replace('"encoder"         : "gram"', '"encoder"         : "gram_glu"')

(SSL / "exp_gram_glu_ema.py").write_text(src)
print("wrote exp_gram_glu_ema.py")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LATENT=1024 variants (ViT-patch is already 1024; gram_glu too)
#     Generate Fashion-MNIST variants of both
# ══════════════════════════════════════════════════════════════════════════════

def fmnist_variant(src_file, src_exp):
    src = (SSL / f"{src_file}.py").read_text()
    src = src.replace('load_supervised_image("mnist")',  'load_supervised_image("fashion_mnist")')
    src = src.replace(f'EXP_NAME = "{src_exp}"',         f'EXP_NAME = "fmnist_{src_exp}"')
    src = src.replace(f'    uv run python projects/ssl/{src_file}.py',
                      f'    uv run python projects/ssl/fmnist_{src_exp}.py')
    return src

for src_exp in ["exp_vit_patch_ema", "exp_gram_glu_ema"]:
    text = fmnist_variant(src_exp, src_exp)
    out  = f"fmnist_{src_exp}"
    (SSL / f"{out}.py").write_text(text)
    print(f"wrote {out}.py")
