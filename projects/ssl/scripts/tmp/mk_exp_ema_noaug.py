"""
Generate EMA + no-augmentation ablations at LATENT_DIM = 512, 1024, 2048, 4096.
Based on exp_ema.py but x_aug = x (no corruption).
"""

from pathlib import Path

SSL = Path(__file__).parent.parent.parent
src = (SSL / "exp_ema.py").read_text()

# Patch: replace augmentation block with identity
src = src.replace(
    """    idx   = jax.random.randint(k_idx, (B,), 0, N_ACTIONS)
    a_emb = online_params["action_emb"][idx]

    # Augmentations
    x_n75 = jnp.clip(x + jax.random.normal(k_noise, x.shape) * NOISE_STD, 0.0, 255.0)
    x_m25 = x * jax.random.bernoulli(k_m1, 1.0 - MASK_RATIOS[0], shape=x.shape)
    x_m50 = x * jax.random.bernoulli(k_m2, 1.0 - MASK_RATIOS[1], shape=x.shape)
    x_m75 = x * jax.random.bernoulli(k_m3, 1.0 - MASK_RATIOS[2], shape=x.shape)
    x_aug = jnp.where(idx[:, None] == 0, x_n75,
            jnp.where(idx[:, None] == 1, x_m25,
            jnp.where(idx[:, None] == 2, x_m50, x_m75)))""",
    """    # Ablation: no augmentation — context = target
    idx   = jnp.zeros(B, dtype=jnp.int32)
    a_emb = online_params["action_emb"][idx]
    x_aug = x"""
)
src = src.replace(
    "    k_idx, k_noise, k_m1, k_m2, k_m3 = jax.random.split(key, 5)",
    "    _ = key  # no augmentation keys needed"
)

configs = [
    ("exp_ema_noaug_512",  512,  "DEC_RANK² = 1024 → projected to LATENT_DIM=512"),
    ("exp_ema_noaug_1024", 1024, "DEC_RANK² = 1024 = LATENT_DIM"),
    ("exp_ema_noaug_2048", 2048, "DEC_RANK² = 1024 → projected to LATENT_DIM=2048"),
    ("exp_ema_noaug_4096", 4096, "DEC_RANK² = 1024 → projected to LATENT_DIM=4096"),
]

for exp_name, latent, rank_comment in configs:
    text = src
    text = text.replace(
        "exp_ema — LeWM JEPA on MNIST: EMA target encoder baseline, LATENT_DIM=4096.",
        f"{exp_name} — ABLATION: EMA, no augmentation, LATENT_DIM={latent}."
    )
    text = text.replace(
        "    uv run python projects/ssl/exp_ema.py",
        f"    uv run python projects/ssl/{exp_name}.py"
    )
    text = text.replace('EXP_NAME = "exp_ema"', f'EXP_NAME = "{exp_name}"')
    text = text.replace("LATENT_DIM   = 4096", f"LATENT_DIM   = {latent}")
    text = text.replace(
        "DEC_RANK     = 32    # DEC_RANK² = 1024 → projected to LATENT_DIM=4096",
        f"DEC_RANK     = 32    # {rank_comment}"
    )
    (SSL / f"{exp_name}.py").write_text(text)
    print(f"wrote {exp_name}.py")
