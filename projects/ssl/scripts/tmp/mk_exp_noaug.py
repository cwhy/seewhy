"""
Generate no-augmentation ablations at LATENT_DIM = 512, 1024, 2048.
(exp17b already covers LATENT=4096)

All based on exp17b (no augmentation, SIGReg, Hebbian+Hadamard+FiLM).
"""

from pathlib import Path

SSL = Path(__file__).parent.parent.parent
src = (SSL / "exp17b.py").read_text()

configs = [
    ("exp_noaug_512",  512,  "DEC_RANK² = 1024 → projected to LATENT_DIM=512"),
    ("exp_noaug_1024", 1024, "DEC_RANK² = 1024 = LATENT_DIM"),
    ("exp_noaug_2048", 2048, "DEC_RANK² = 1024 → projected to LATENT_DIM=2048"),
]

for exp_name, latent, rank_comment in configs:
    text = src
    text = text.replace(
        "exp17b — ABLATION: exp16a without augmentation (context = target), LATENT_DIM=4096.",
        f"{exp_name} — ABLATION: no augmentation (context = target), LATENT_DIM={latent}."
    )
    text = text.replace(
        "    uv run python projects/ssl/exp17b.py",
        f"    uv run python projects/ssl/{exp_name}.py"
    )
    text = text.replace('EXP_NAME = "exp17b"', f'EXP_NAME = "{exp_name}"')
    text = text.replace(
        "LATENT_DIM   = 4096",
        f"LATENT_DIM   = {latent}"
    )
    text = text.replace(
        "DEC_RANK     = 32    # DEC_RANK² = 1024 → projected to LATENT_DIM=4096",
        f"DEC_RANK     = 32    # {rank_comment}"
    )
    (SSL / f"{exp_name}.py").write_text(text)
    print(f"wrote {exp_name}.py")
