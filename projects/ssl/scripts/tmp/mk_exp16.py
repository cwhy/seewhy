"""
Generate experiments 16a and 16c.

exp15b baseline: Hadamard+FiLM, 4 actions, LATENT=2048, 200ep → 97.2%

  exp16a: same, scale LATENT_DIM 2048→4096
  exp16c: same, double epochs 200→400
"""

from pathlib import Path

SSL = Path(__file__).parent.parent.parent
src = (SSL / "exp15b.py").read_text()

# ── exp16a: LATENT_DIM 2048→4096 ──────────────────────────────────────────────
text = src
text = text.replace(
    "exp15b — LeWM JEPA on MNIST: Hebbian gram + Hadamard readout, 4 actions, FiLM predictor, LATENT_DIM=2048.",
    "exp16a — LeWM JEPA on MNIST: Hebbian gram + Hadamard readout, 4 actions, FiLM predictor, LATENT_DIM=4096."
)
text = text.replace(
    "vs exp14b (97.1%): expand to 4 actions:",
    "vs exp15b (97.2%): same 4-action FiLM config, scale LATENT_DIM 2048→4096."
)
text = text.replace("    uv run python projects/ssl/exp15b.py", "    uv run python projects/ssl/exp16a.py")
text = text.replace('EXP_NAME = "exp15b"', 'EXP_NAME = "exp16a"')
text = text.replace("LATENT_DIM   = 2048", "LATENT_DIM   = 4096")
text = text.replace("DEC_RANK     = 32    # DEC_RANK² = 1024 → projected to LATENT_DIM=2048",
                    "DEC_RANK     = 32    # DEC_RANK² = 1024 → projected to LATENT_DIM=4096")
(SSL / "exp16a.py").write_text(text)
print("wrote exp16a.py")

# ── exp16c: 400 epochs ─────────────────────────────────────────────────────────
text = src
text = text.replace(
    "exp15b — LeWM JEPA on MNIST: Hebbian gram + Hadamard readout, 4 actions, FiLM predictor, LATENT_DIM=2048.",
    "exp16c — LeWM JEPA on MNIST: Hebbian gram + Hadamard readout, 4 actions, FiLM predictor, LATENT_DIM=2048, 400ep."
)
text = text.replace(
    "vs exp14b (97.1%): expand to 4 actions:",
    "vs exp15b (97.2%): same config, double epochs 200→400."
)
text = text.replace("    uv run python projects/ssl/exp15b.py", "    uv run python projects/ssl/exp16c.py")
text = text.replace('EXP_NAME = "exp15b"', 'EXP_NAME = "exp16c"')
text = text.replace("MAX_EPOCHS   = 200", "MAX_EPOCHS   = 400")
(SSL / "exp16c.py").write_text(text)
print("wrote exp16c.py")
