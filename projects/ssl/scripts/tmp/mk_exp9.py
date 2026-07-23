"""
Generate experiments 9a–9c: encoder capacity sweep from exp7c baseline.

exp7c baseline: Hebbian gram, DEC_RANK=32, LATENT_DIM=128, MLP_HIDDEN=512,
                noise σ=75, 200 epochs → 95.4%

Grid:
  exp9a: DEC_RANK=64, LATENT_DIM=128  (scale gram rank only)
  exp9b: DEC_RANK=32, LATENT_DIM=256  (scale embedding only)
  exp9c: DEC_RANK=64, LATENT_DIM=256  (scale both)
"""

from pathlib import Path

SRC = Path(__file__).parent.parent.parent / "exp7c.py"
src = SRC.read_text()

configs = [
    ("exp9a", 64, 128, "DEC_RANK=64 LATENT_DIM=128"),
    ("exp9b", 32, 256, "DEC_RANK=32 LATENT_DIM=256"),
    ("exp9c", 64, 256, "DEC_RANK=64 LATENT_DIM=256"),
]

for exp_name, dec_rank, latent_dim, desc in configs:
    text = src
    text = text.replace(
        "exp7c — LeWM JEPA on MNIST: Gaussian noise sweep NOISE_STD=75 epochs=200.",
        f"{exp_name} — LeWM JEPA on MNIST: encoder capacity — {desc}."
    )
    text = text.replace(
        "vs exp7 (93.5%, std=50, 100ep): NOISE_STD=75, MAX_EPOCHS=200.",
        f"vs exp7c (95.4%): {desc}, else identical."
    )
    text = text.replace("    uv run python projects/ssl/exp7c.py",
                        f"    uv run python projects/ssl/{exp_name}.py")
    text = text.replace('EXP_NAME = "exp7c"',   f'EXP_NAME = "{exp_name}"')
    text = text.replace("DEC_RANK     = 32",     f"DEC_RANK     = {dec_rank}")
    text = text.replace("LATENT_DIM   = 128",    f"LATENT_DIM   = {latent_dim}")

    out = SRC.parent / f"{exp_name}.py"
    out.write_text(text)
    print(f"wrote {out}")
