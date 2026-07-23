"""
Generate experiments 7a–7d: noise std sweep at 200 epochs.

Base: exp7 (Hebbian encoder, LATENT=128, DEC_RANK=32, MLP_HIDDEN=512)
Grid: NOISE_STD ∈ {25, 50, 75, 100}, MAX_EPOCHS=200
"""

from pathlib import Path

SRC  = Path(__file__).parent.parent.parent / "exp7.py"
src  = SRC.read_text()

configs = [
    ("exp7a", 25),
    ("exp7b", 50),
    ("exp7c", 75),
    ("exp7d", 100),
]

for exp_name, noise_std in configs:
    text = src
    text = text.replace(
        "exp7 — LeWM JEPA on MNIST: Gaussian noise augmentation instead of masking.",
        f"{exp_name} — LeWM JEPA on MNIST: Gaussian noise sweep NOISE_STD={noise_std} epochs=200."
    )
    text = text.replace(
        "vs exp6 (94.3%): replace random 50% pixel masking with additive Gaussian noise.\n"
        "  x_noisy = clip(x + N(0, NOISE_STD²), 0, 255).\n"
        "  The predictor maps noisy-image embedding → clean-image embedding.",
        f"vs exp7 (93.5%, std=50, 100ep): NOISE_STD={noise_std}, MAX_EPOCHS=200."
    )
    text = text.replace("    uv run python projects/ssl/exp7.py",
                        f"    uv run python projects/ssl/{exp_name}.py")
    text = text.replace('EXP_NAME = "exp7"',   f'EXP_NAME = "{exp_name}"')
    text = text.replace("NOISE_STD  = 50.0",   f"NOISE_STD  = {float(noise_std)}")
    text = text.replace("MAX_EPOCHS   = 100",  "MAX_EPOCHS   = 200")

    out = SRC.parent / f"{exp_name}.py"
    out.write_text(text)
    print(f"wrote {out}")
