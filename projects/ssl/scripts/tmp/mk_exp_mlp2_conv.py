"""
Generate latent-size variants for:
  - exp_mlp2_ema_{512,1024,2048,4096}   (deeper MLP + EMA + aug)
  - exp_conv_ema_{512,1024,2048,4096}   (ConvNet + EMA + aug)
"""

from pathlib import Path

SSL = Path(__file__).parent.parent.parent

# ── MLP2 variants ─────────────────────────────────────────────────────────────

src_mlp2 = (SSL / "exp_mlp2_ema.py").read_text()

for latent in [512, 1024, 2048, 4096]:
    exp_name = f"exp_mlp2_ema_{latent}"
    text = src_mlp2
    text = text.replace(
        "exp_mlp2_ema — LeWM JEPA on MNIST: deeper MLP encoder + EMA, LATENT_DIM=1024.",
        f"{exp_name} — deeper MLP encoder + EMA + augmentation, LATENT_DIM={latent}."
    )
    text = text.replace(
        "    uv run python projects/ssl/exp_mlp2_ema.py",
        f"    uv run python projects/ssl/{exp_name}.py"
    )
    text = text.replace('EXP_NAME = "exp_mlp2_ema"', f'EXP_NAME = "{exp_name}"')
    text = text.replace("LATENT_DIM      = 1024", f"LATENT_DIM      = {latent}")
    (SSL / f"{exp_name}.py").write_text(text)
    print(f"wrote {exp_name}.py")

# ── ConvNet variants ──────────────────────────────────────────────────────────

src_conv = (SSL / "exp_conv_ema.py").read_text()

for latent in [512, 1024, 2048, 4096]:
    exp_name = f"exp_conv_ema_{latent}"
    text = src_conv
    text = text.replace(
        "exp_conv_ema — LeWM JEPA on MNIST: ConvNet encoder + EMA, LATENT_DIM=1024.",
        f"{exp_name} — ConvNet encoder + EMA + augmentation, LATENT_DIM={latent}."
    )
    text = text.replace(
        "    uv run python projects/ssl/exp_conv_ema.py",
        f"    uv run python projects/ssl/{exp_name}.py"
    )
    text = text.replace('EXP_NAME = "exp_conv_ema"', f'EXP_NAME = "{exp_name}"')
    text = text.replace("LATENT_DIM      = 1024", f"LATENT_DIM      = {latent}")
    (SSL / f"{exp_name}.py").write_text(text)
    print(f"wrote {exp_name}.py")
