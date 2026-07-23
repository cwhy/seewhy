"""
Generate latent-size variants for:
  - exp_conv2_ema_{512,1024,2048,4096}        (improved ConvNet)
  - exp_vit_ema_{512,1024,2048,4096}          (mini transformer)
  - exp_gram_single_ema_{256,1024,4096}       (single-projection Hadamard, LATENT=DEC_RANK²)
"""

from pathlib import Path
import math

SSL = Path(__file__).parent.parent.parent

# ── ConvNet v2 ────────────────────────────────────────────────────────────────

src = (SSL / "exp_conv2_ema.py").read_text()
for latent in [512, 1024, 2048, 4096]:
    exp = f"exp_conv2_ema_{latent}"
    text = src
    text = text.replace(
        "exp_conv2_ema — LeWM JEPA on MNIST: improved ConvNet encoder + EMA, LATENT_DIM=1024.",
        f"{exp} — improved ConvNet encoder + EMA, LATENT_DIM={latent}."
    )
    text = text.replace("    uv run python projects/ssl/exp_conv2_ema.py",
                        f"    uv run python projects/ssl/{exp}.py")
    text = text.replace('EXP_NAME = "exp_conv2_ema"', f'EXP_NAME = "{exp}"')
    text = text.replace("LATENT_DIM      = 1024", f"LATENT_DIM      = {latent}")
    (SSL / f"{exp}.py").write_text(text)
    print(f"wrote {exp}.py")

# ── Mini ViT ──────────────────────────────────────────────────────────────────

src = (SSL / "exp_vit_ema.py").read_text()
for latent in [512, 1024, 2048, 4096]:
    exp = f"exp_vit_ema_{latent}"
    text = src
    text = text.replace(
        "exp_vit_ema — LeWM JEPA on MNIST: mini Vision Transformer encoder + EMA, LATENT_DIM=1024.",
        f"{exp} — mini ViT encoder + EMA, LATENT_DIM={latent}."
    )
    text = text.replace("    uv run python projects/ssl/exp_vit_ema.py",
                        f"    uv run python projects/ssl/{exp}.py")
    text = text.replace('EXP_NAME = "exp_vit_ema"', f'EXP_NAME = "{exp}"')
    text = text.replace("LATENT_DIM      = 1024", f"LATENT_DIM      = {latent}")
    (SSL / f"{exp}.py").write_text(text)
    print(f"wrote {exp}.py")

# ── Gram-single (LATENT = DEC_RANK²) ─────────────────────────────────────────

src = (SSL / "exp_gram_single_ema.py").read_text()
for latent in [256, 1024, 4096]:
    dec_rank = int(round(math.sqrt(latent)))
    assert dec_rank * dec_rank == latent, f"latent={latent} is not a perfect square"
    exp = f"exp_gram_single_ema_{latent}"
    text = src
    text = text.replace(
        "exp_gram_single_ema — LeWM JEPA on MNIST: Gram encoder + single-projection Hadamard + EMA.",
        f"{exp} — Gram encoder + single-projection Hadamard + EMA, LATENT={latent}."
    )
    text = text.replace("    uv run python projects/ssl/exp_gram_single_ema.py",
                        f"    uv run python projects/ssl/{exp}.py")
    text = text.replace('EXP_NAME = "exp_gram_single_ema"', f'EXP_NAME = "{exp}"')
    text = text.replace("LATENT_DIM   = 1024", f"LATENT_DIM   = {latent}")
    text = text.replace("DEC_RANK     = 32    # DEC_RANK² = LATENT_DIM",
                        f"DEC_RANK     = {dec_rank}    # DEC_RANK² = {latent} = LATENT_DIM")
    (SSL / f"{exp}.py").write_text(text)
    print(f"wrote {exp}.py  (DEC_RANK={dec_rank})")
