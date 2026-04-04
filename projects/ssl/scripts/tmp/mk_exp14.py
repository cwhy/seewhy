"""
Generate experiments 14a–14b.

exp13a baseline: Hebbian gram + Hadamard readout, DEC_RANK=32, LATENT_DIM=2048 → 96.9%

  exp14a: same, scale LATENT_DIM 2048→4096
  exp14b: same encoder, two actions (denoise σ=75 / mask 50%), FiLM predictor, LATENT_DIM=2048
"""

from pathlib import Path

SSL = Path(__file__).parent.parent.parent

# ── exp14a: based on exp13a, LATENT_DIM 2048→4096 ─────────────────────────────
src_a = (SSL / "exp13a.py").read_text()
text = src_a
text = text.replace(
    "exp13a — LeWM JEPA on MNIST: Hebbian gram + Hadamard readout, LATENT_DIM=2048.",
    "exp14a — LeWM JEPA on MNIST: Hebbian gram + Hadamard readout, LATENT_DIM=4096."
)
text = text.replace("    uv run python projects/ssl/exp13a.py", "    uv run python projects/ssl/exp14a.py")
text = text.replace('EXP_NAME = "exp13a"', 'EXP_NAME = "exp14a"')
text = text.replace("LATENT_DIM   = 2048", "LATENT_DIM   = 4096")
(SSL / "exp14a.py").write_text(text)
print("wrote exp14a.py")

# ── exp14b: based on exp13b (FiLM two-action), LATENT_DIM 1024→2048 ───────────
src_b = (SSL / "exp13b.py").read_text()
text = src_b
text = text.replace(
    "exp13b — LeWM JEPA on MNIST: Hebbian gram + Hadamard readout, two actions, FiLM predictor.",
    "exp14b — LeWM JEPA on MNIST: Hebbian gram + Hadamard readout, two actions, FiLM predictor, LATENT_DIM=2048."
)
text = text.replace(
    "vs exp11a (96.4%): add action conditioning — two augmentation types sampled per image:",
    "vs exp13b (96.6%): scale LATENT_DIM 1024→2048, same FiLM two-action setup:"
)
text = text.replace("    uv run python projects/ssl/exp13b.py", "    uv run python projects/ssl/exp14b.py")
text = text.replace('EXP_NAME = "exp13b"', 'EXP_NAME = "exp14b"')
text = text.replace("LATENT_DIM   = 1024", "LATENT_DIM   = 2048")
text = text.replace("DEC_RANK     = 32    # DEC_RANK² = 1024 = LATENT_DIM", "DEC_RANK     = 32    # DEC_RANK² = 1024 → projected to LATENT_DIM=2048")
(SSL / "exp14b.py").write_text(text)
print("wrote exp14b.py")
