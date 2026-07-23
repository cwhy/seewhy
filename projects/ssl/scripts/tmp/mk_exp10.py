"""
Generate experiments 10a–10b: push LATENT_DIM to 512.

exp9c baseline: DEC_RANK=64, LATENT_DIM=256, σ=75, 200ep → 95.8%

  exp10a: DEC_RANK=32, LATENT_DIM=512
  exp10b: DEC_RANK=64, LATENT_DIM=512
"""

from pathlib import Path

SRC = Path(__file__).parent.parent.parent / "exp9c.py"
src = SRC.read_text()

configs = [
    ("exp10a", 32, 512),
    ("exp10b", 64, 512),
]

for exp_name, dec_rank, latent_dim in configs:
    text = src
    text = text.replace(
        "exp9c — LeWM JEPA on MNIST: encoder capacity — DEC_RANK=64 LATENT_DIM=256.",
        f"{exp_name} — LeWM JEPA on MNIST: encoder capacity — DEC_RANK={dec_rank} LATENT_DIM={latent_dim}."
    )
    text = text.replace(
        "vs exp7c (95.4%): DEC_RANK=64 LATENT_DIM=256, else identical.",
        f"vs exp9c (95.8%): DEC_RANK={dec_rank} LATENT_DIM={latent_dim}, else identical."
    )
    text = text.replace(f"    uv run python projects/ssl/exp9c.py",
                        f"    uv run python projects/ssl/{exp_name}.py")
    text = text.replace('EXP_NAME = "exp9c"',  f'EXP_NAME = "{exp_name}"')
    text = text.replace("DEC_RANK     = 64",   f"DEC_RANK     = {dec_rank}")
    text = text.replace("LATENT_DIM   = 256",  f"LATENT_DIM   = {latent_dim}")

    out = SRC.parent / f"{exp_name}.py"
    out.write_text(text)
    print(f"wrote {out}")
