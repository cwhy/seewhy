"""
Generate experiment 16b: 4 actions mixing noise levels + masking.

exp15b: 4 actions (noise75 + mask25/50/75), LATENT=2048 → 97.2%
exp16b: 4 actions (noise25/50/75 + mask50),  LATENT=2048

Hypothesis: mixing augmentation types (noise vs mask) at multiple intensities
            provides richer signal than varying only one type.
"""

from pathlib import Path

SSL = Path(__file__).parent.parent.parent
src = (SSL / "exp15b.py").read_text()
text = src

text = text.replace(
    "exp15b — LeWM JEPA on MNIST: Hebbian gram + Hadamard readout, 4 actions, FiLM predictor, LATENT_DIM=2048.",
    "exp16b — LeWM JEPA on MNIST: Hebbian gram + Hadamard readout, 4 actions (noise25/50/75+mask50), FiLM, LATENT_DIM=2048."
)
text = text.replace(
    """vs exp14b (97.1%): expand to 4 actions:
  action 0: denoise_gaussian_75  (Gaussian noise σ=75)
  action 1: mask_25              (random 25% pixel masking)
  action 2: mask_50              (random 50% pixel masking)
  action 3: mask_75              (random 75% pixel masking)
Predictor conditioned on action via FiLM.""",
    """vs exp15b (97.2%): swap action set — mix noise levels with masking:
  action 0: denoise_gaussian_25  (Gaussian noise σ=25)
  action 1: denoise_gaussian_50  (Gaussian noise σ=50)
  action 2: denoise_gaussian_75  (Gaussian noise σ=75)
  action 3: mask_50              (random 50% pixel masking)
Predictor conditioned on action via FiLM."""
)
text = text.replace("    uv run python projects/ssl/exp15b.py", "    uv run python projects/ssl/exp16b.py")
text = text.replace('EXP_NAME = "exp15b"', 'EXP_NAME = "exp16b"')
text = text.replace(
    'ACTION_LABELS = ["denoise_gaussian_75", "mask_25", "mask_50", "mask_75"]',
    'ACTION_LABELS = ["denoise_gaussian_25", "denoise_gaussian_50", "denoise_gaussian_75", "mask_50"]'
)
text = text.replace(
    "NOISE_STD    = 75.0\nMASK_RATIOS  = [0.25, 0.50, 0.75]   # fraction of pixels zeroed per masking action",
    "NOISE_STDS   = [25.0, 50.0, 75.0]   # noise std per noise action\nMASK_RATIO   = 0.5"
)
text = text.replace(
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

    return pred_loss + LAMBDA * sreg_loss, (pred_loss, sreg_loss)""",
    """def total_loss(params, x, key):
    k_idx, k_n1, k_n2, k_n3, k_mask, k_sigreg = jax.random.split(key, 6)
    B = x.shape[0]

    # Sample augmentation type per image
    idx   = jax.random.randint(k_idx, (B,), 0, N_ACTIONS)   # (B,)
    a_emb = params["action_emb"][idx]                         # (B, ACTION_DIM)

    # Augmentations 0/1/2: Gaussian noise at σ=25/50/75
    x_n25 = jnp.clip(x + jax.random.normal(k_n1, x.shape) * NOISE_STDS[0], 0.0, 255.0)
    x_n50 = jnp.clip(x + jax.random.normal(k_n2, x.shape) * NOISE_STDS[1], 0.0, 255.0)
    x_n75 = jnp.clip(x + jax.random.normal(k_n3, x.shape) * NOISE_STDS[2], 0.0, 255.0)

    # Augmentation 3: random 50% pixel masking
    x_m50 = x * jax.random.bernoulli(k_mask, 1.0 - MASK_RATIO, shape=x.shape)

    # Select per image
    x_aug = jnp.where(idx[:, None] == 0, x_n25,
            jnp.where(idx[:, None] == 1, x_n50,
            jnp.where(idx[:, None] == 2, x_n75, x_m50)))

    e_full  = encode(params, x)
    e_aug   = encode(params, x_aug)
    e_pred  = predict(params, e_aug, a_emb)

    pred_loss = ((e_pred - e_full) ** 2).mean()
    sreg_loss = sigreg(e_full, k_sigreg)

    return pred_loss + LAMBDA * sreg_loss, (pred_loss, sreg_loss)"""
)
text = text.replace(
    '"name"            : f"jepa-hebbian+hadamard+FiLM 4actions LATENT={LATENT_DIM} DEC_RANK={DEC_RANK}"',
    '"name"            : f"jepa-hebbian+hadamard+FiLM noise25/50/75+mask50 LATENT={LATENT_DIM} DEC_RANK={DEC_RANK}"'
)
text = text.replace(
    '"mask_ratios"     : MASK_RATIOS,',
    '"noise_stds"      : NOISE_STDS,\n        "mask_ratio"      : MASK_RATIO,'
)

(SSL / "exp16b.py").write_text(text)
print("wrote exp16b.py")
