"""
Generate MLP encoder + EMA variants:
  - exp_mlp_ema_{512,1024,2048,4096}.py       (with augmentation)
  - exp_mlp_ema_noaug_{512,1024,2048,4096}.py (no augmentation)

Based on exp_mlp_ema.py (LATENT_DIM=1024 baseline).
"""

from pathlib import Path

SSL = Path(__file__).parent.parent.parent
src = (SSL / "exp_mlp_ema.py").read_text()

# ── Aug variants ──────────────────────────────────────────────────────────────

aug_configs = [
    ("exp_mlp_ema_512",  512),
    ("exp_mlp_ema_1024", 1024),
    ("exp_mlp_ema_2048", 2048),
    ("exp_mlp_ema_4096", 4096),
]

for exp_name, latent in aug_configs:
    text = src
    text = text.replace(
        'exp_mlp_ema — LeWM JEPA on MNIST: naive MLP encoder + EMA, LATENT_DIM=1024.',
        f'{exp_name} — MLP encoder + EMA + augmentation, LATENT_DIM={latent}.'
    )
    text = text.replace(
        '    uv run python projects/ssl/exp_mlp_ema.py',
        f'    uv run python projects/ssl/{exp_name}.py'
    )
    text = text.replace('EXP_NAME = "exp_mlp_ema"', f'EXP_NAME = "{exp_name}"')
    text = text.replace('LATENT_DIM      = 1024', f'LATENT_DIM      = {latent}')
    (SSL / f"{exp_name}.py").write_text(text)
    print(f"wrote {exp_name}.py")

# ── No-aug variants ───────────────────────────────────────────────────────────

noaug_configs = [
    ("exp_mlp_ema_noaug_512",  512),
    ("exp_mlp_ema_noaug_1024", 1024),
    ("exp_mlp_ema_noaug_2048", 2048),
    ("exp_mlp_ema_noaug_4096", 4096),
]

# Augmentation block to replace
aug_block = """    k_idx, k_noise, k_m1, k_m2, k_m3 = jax.random.split(key, 5)
    B = x.shape[0]

    idx   = jax.random.randint(k_idx, (B,), 0, N_ACTIONS)
    a_emb = online_params["action_emb"][idx]

    x_n75 = jnp.clip(x + jax.random.normal(k_noise, x.shape) * NOISE_STD, 0.0, 255.0)
    x_m25 = x * jax.random.bernoulli(k_m1, 1.0 - MASK_RATIOS[0], shape=x.shape)
    x_m50 = x * jax.random.bernoulli(k_m2, 1.0 - MASK_RATIOS[1], shape=x.shape)
    x_m75 = x * jax.random.bernoulli(k_m3, 1.0 - MASK_RATIOS[2], shape=x.shape)
    x_aug = jnp.where(idx[:, None] == 0, x_n75,
            jnp.where(idx[:, None] == 1, x_m25,
            jnp.where(idx[:, None] == 2, x_m50, x_m75)))"""

noaug_block = """    _ = key  # no augmentation keys needed
    B = x.shape[0]

    # Ablation: no augmentation — context = target
    idx   = jnp.zeros(B, dtype=jnp.int32)
    a_emb = online_params["action_emb"][idx]
    x_aug = x"""

for exp_name, latent in noaug_configs:
    text = src
    text = text.replace(
        'exp_mlp_ema — LeWM JEPA on MNIST: naive MLP encoder + EMA, LATENT_DIM=1024.',
        f'{exp_name} — MLP encoder + EMA + NO augmentation, LATENT_DIM={latent}.'
    )
    text = text.replace(
        '    uv run python projects/ssl/exp_mlp_ema.py',
        f'    uv run python projects/ssl/{exp_name}.py'
    )
    text = text.replace('EXP_NAME = "exp_mlp_ema"', f'EXP_NAME = "{exp_name}"')
    text = text.replace('LATENT_DIM      = 1024', f'LATENT_DIM      = {latent}')
    text = text.replace(aug_block, noaug_block)
    (SSL / f"{exp_name}.py").write_text(text)
    print(f"wrote {exp_name}.py")
