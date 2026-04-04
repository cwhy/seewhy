"""
Generate experiment 15b: 4 actions (noise75 + mask25/50/75), FiLM, LATENT_DIM=2048.

exp14b baseline: 2 actions (noise75/mask50), FiLM, LATENT=2048 → 97.1%
"""

from pathlib import Path

SSL = Path(__file__).parent.parent.parent
src = (SSL / "exp14b.py").read_text()
text = src

text = text.replace(
    "exp14b — LeWM JEPA on MNIST: Hebbian gram + Hadamard readout, two actions, FiLM predictor, LATENT_DIM=2048.",
    "exp15b — LeWM JEPA on MNIST: Hebbian gram + Hadamard readout, 4 actions, FiLM predictor, LATENT_DIM=2048."
)
text = text.replace(
    "vs exp13b (96.6%): scale LATENT_DIM 1024→2048, same FiLM two-action setup:\n  action 0: denoise_gaussian_75  (Gaussian noise σ=75)\n  action 1: mask_50              (random 50% pixel masking)\nPredictor conditioned on action via FiLM.",
    "vs exp14b (97.1%): expand to 4 actions:\n  action 0: denoise_gaussian_75  (Gaussian noise σ=75)\n  action 1: mask_25              (random 25% pixel masking)\n  action 2: mask_50              (random 50% pixel masking)\n  action 3: mask_75              (random 75% pixel masking)\nPredictor conditioned on action via FiLM."
)
text = text.replace(
    "    uv run python projects/ssl/exp14b.py",
    "    uv run python projects/ssl/exp15b.py"
)
text = text.replace('EXP_NAME = "exp14b"', 'EXP_NAME = "exp15b"')
text = text.replace(
    "N_ACTIONS    = 2\nACTION_DIM   = 32\nACTION_LABELS = [\"denoise_gaussian_75\", \"mask_50\"]\n\nNOISE_STD    = 75.0\nMASK_RATIO   = 0.5   # fraction of pixels to zero",
    "N_ACTIONS    = 4\nACTION_DIM   = 32\nACTION_LABELS = [\"denoise_gaussian_75\", \"mask_25\", \"mask_50\", \"mask_75\"]\n\nNOISE_STD    = 75.0\nMASK_RATIOS  = [0.25, 0.50, 0.75]   # fraction of pixels zeroed per masking action"
)
text = text.replace(
    """def total_loss(params, x, key):
    k_idx, k_noise, k_mask, k_sigreg = jax.random.split(key, 4)
    B = x.shape[0]

    # Sample augmentation type per image
    idx   = jax.random.randint(k_idx, (B,), 0, N_ACTIONS)   # (B,)
    a_emb = params["action_emb"][idx]                         # (B, ACTION_DIM)

    # Augmentation 0: Gaussian noise σ=75
    noise   = jax.random.normal(k_noise, x.shape) * NOISE_STD
    x_noisy = jnp.clip(x + noise, 0.0, 255.0)

    # Augmentation 1: random 50% pixel masking
    keep    = jax.random.bernoulli(k_mask, 1.0 - MASK_RATIO, shape=x.shape)
    x_mskd  = x * keep

    # Select per image
    x_aug = jnp.where(idx[:, None] == 0, x_noisy, x_mskd)

    e_full  = encode(params, x)
    e_aug   = encode(params, x_aug)
    e_pred  = predict(params, e_aug, a_emb)

    pred_loss = ((e_pred - e_full) ** 2).mean()
    sreg_loss = sigreg(e_full, k_sigreg)

    return pred_loss + LAMBDA * sreg_loss, (pred_loss, sreg_loss)""",
    """def total_loss(params, x, key):
    k_idx, k_noise, k_m1, k_m2, k_m3, k_sigreg = jax.random.split(key, 6)
    B = x.shape[0]

    # Sample augmentation type per image
    idx   = jax.random.randint(k_idx, (B,), 0, N_ACTIONS)   # (B,)
    a_emb = params["action_emb"][idx]                         # (B, ACTION_DIM)

    # Augmentation 0: Gaussian noise σ=75
    noise   = jax.random.normal(k_noise, x.shape) * NOISE_STD
    x_n75   = jnp.clip(x + noise, 0.0, 255.0)

    # Augmentations 1/2/3: random 25/50/75% pixel masking
    x_m25 = x * jax.random.bernoulli(k_m1, 1.0 - MASK_RATIOS[0], shape=x.shape)
    x_m50 = x * jax.random.bernoulli(k_m2, 1.0 - MASK_RATIOS[1], shape=x.shape)
    x_m75 = x * jax.random.bernoulli(k_m3, 1.0 - MASK_RATIOS[2], shape=x.shape)

    # Select per image
    x_aug = jnp.where(idx[:, None] == 0, x_n75,
            jnp.where(idx[:, None] == 1, x_m25,
            jnp.where(idx[:, None] == 2, x_m50, x_m75)))

    e_full  = encode(params, x)
    e_aug   = encode(params, x_aug)
    e_pred  = predict(params, e_aug, a_emb)

    pred_loss = ((e_pred - e_full) ** 2).mean()
    sreg_loss = sigreg(e_full, k_sigreg)

    return pred_loss + LAMBDA * sreg_loss, (pred_loss, sreg_loss)"""
)
text = text.replace(
    '"name"            : f"jepa-hebbian+hadamard+FiLM noise75+mask50 LATENT={LATENT_DIM} DEC_RANK={DEC_RANK}"',
    '"name"            : f"jepa-hebbian+hadamard+FiLM 4actions LATENT={LATENT_DIM} DEC_RANK={DEC_RANK}"'
)
text = text.replace(
    '"mask_ratio"      : MASK_RATIO,',
    '"mask_ratios"     : MASK_RATIOS,'
)

(SSL / "exp15b.py").write_text(text)
print("wrote exp15b.py")
