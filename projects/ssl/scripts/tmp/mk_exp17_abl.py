"""
Generate ablation experiments 17a–17b, based on exp16a (97.4%).

  exp17a: no SIGReg (LAMBDA=0)           — how much does regularization help?
  exp17b: no augmentation (x_aug = x)    — how much does input corruption help?
"""

from pathlib import Path

SSL = Path(__file__).parent.parent.parent
src = (SSL / "exp16a.py").read_text()

# ── exp17a: LAMBDA=0 (disable SIGReg) ─────────────────────────────────────────
text = src
text = text.replace(
    "exp16a — LeWM JEPA on MNIST: Hebbian gram + Hadamard readout, 4 actions, FiLM predictor, LATENT_DIM=4096.",
    "exp17a — ABLATION: exp16a without SIGReg (LAMBDA=0), LATENT_DIM=4096."
)
text = text.replace(
    "vs exp15b (97.2%): same 4-action FiLM config, scale LATENT_DIM 2048→4096.",
    "ablation of exp16a (97.4%): disable SIGReg by setting LAMBDA=0."
)
text = text.replace("    uv run python projects/ssl/exp16a.py", "    uv run python projects/ssl/exp17a.py")
text = text.replace('EXP_NAME = "exp16a"', 'EXP_NAME = "exp17a"')
text = text.replace("LAMBDA       = 0.1", "LAMBDA       = 0.0   # ablation: SIGReg disabled")
(SSL / "exp17a.py").write_text(text)
print("wrote exp17a.py")

# ── exp17b: no augmentation (x_aug = x) ───────────────────────────────────────
text = src
text = text.replace(
    "exp16a — LeWM JEPA on MNIST: Hebbian gram + Hadamard readout, 4 actions, FiLM predictor, LATENT_DIM=4096.",
    "exp17b — ABLATION: exp16a without augmentation (context = target), LATENT_DIM=4096."
)
text = text.replace(
    "vs exp15b (97.2%): same 4-action FiLM config, scale LATENT_DIM 2048→4096.",
    "ablation of exp16a (97.4%): no augmentation applied — context and target are identical."
)
text = text.replace("    uv run python projects/ssl/exp16a.py", "    uv run python projects/ssl/exp17b.py")
text = text.replace('EXP_NAME = "exp16a"', 'EXP_NAME = "exp17b"')
text = text.replace(
    """    # Augmentation 0: Gaussian noise σ=75
    x_n75   = jnp.clip(x + jax.random.normal(k_noise, x.shape) * NOISE_STDS[2], 0.0, 255.0)

    # Augmentations 1/2/3: random 25/50/75% pixel masking
    x_m25 = x * jax.random.bernoulli(k_m1, 1.0 - MASK_RATIOS[0], shape=x.shape)
    x_m50 = x * jax.random.bernoulli(k_m2, 1.0 - MASK_RATIOS[1], shape=x.shape)
    x_m75 = x * jax.random.bernoulli(k_m3, 1.0 - MASK_RATIOS[2], shape=x.shape)

    # Select per image
    x_aug = jnp.where(idx[:, None] == 0, x_n75,
            jnp.where(idx[:, None] == 1, x_m25,
            jnp.where(idx[:, None] == 2, x_m50, x_m75)))""",
    """    # Ablation: no augmentation — context equals target
    x_aug = x"""
)
(SSL / "exp17b.py").write_text(text)
print("wrote exp17b.py")
