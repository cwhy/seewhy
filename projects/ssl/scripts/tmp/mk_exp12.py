"""
Generate experiments 12a–12c: push beyond 96.4% (exp11a/11b).

exp11a/11b baseline: DEC_RANK=32, LATENT_DIM=1024, 200ep → 96.4%

  exp12a: Hadamard readout,  DEC_RANK=64, LATENT_DIM=1024   (bigger gram → project down)
  exp12b: Hebbian+MLP,       DEC_RANK=64, LATENT_DIM=1024, MLP_HIDDEN=1024
  exp12c: Hebbian+MLP,       DEC_RANK=32, LATENT_DIM=2048,  MLP_HIDDEN=1024
"""

from pathlib import Path

SSL = Path(__file__).parent.parent.parent

# ── exp12a: based on exp11a (Hadamard readout), DEC_RANK 32→64 ────────────────
src_a = (SSL / "exp11a.py").read_text()
text = src_a
text = text.replace(
    'exp11a — LeWM JEPA on MNIST: Hebbian gram + Hadamard readout, LATENT_DIM=1024.',
    'exp12a — LeWM JEPA on MNIST: Hebbian gram + Hadamard readout, DEC_RANK=64 LATENT_DIM=1024.'
)
text = text.replace(
    'vs exp10a (96.0%): same Hebbian gram structure (DEC_RANK=32 → H of shape DEC_RANK²=1024),\n                   replace MLP readout with two parallel linear projections + hadamard:\n                     a = H @ W_enc_A   (B, 1024)\n                     b = H @ W_enc_B   (B, 1024)\n                     h = a ⊙ b         (B, 1024)\n                   DEC_RANK²=LATENT_DIM=1024 so no dimensionality change.',
    'vs exp11a (96.4%): scale DEC_RANK 32→64 (gram grows to 4096), project down to 1024\n                   via W_enc_A/W_enc_B: (4096, 1024) each.'
)
text = text.replace("    uv run python projects/ssl/exp11a.py", "    uv run python projects/ssl/exp12a.py")
text = text.replace('EXP_NAME = "exp11a"', 'EXP_NAME = "exp12a"')
text = text.replace("DEC_RANK     = 32    # DEC_RANK² = 1024 = LATENT_DIM", "DEC_RANK     = 64    # DEC_RANK² = 4096 → projected to LATENT_DIM=1024")
(SSL / "exp12a.py").write_text(text)
print("wrote exp12a.py")

# ── exp12b: based on exp11b (Hebbian+MLP), DEC_RANK 32→64, MLP_HIDDEN 512→1024 ─
src_b = (SSL / "exp11b.py").read_text()
text = src_b
text = text.replace(
    'exp11b — LeWM JEPA on MNIST: Hebbian gram encoder, LATENT_DIM=1024.',
    'exp12b — LeWM JEPA on MNIST: Hebbian gram encoder, DEC_RANK=64 LATENT_DIM=1024 MLP_HIDDEN=1024.'
)
text = text.replace(
    'vs exp10a (96.0%): scale LATENT_DIM 512→1024 with standard Hebbian gram (DEC_RANK=32).',
    'vs exp11b (96.4%): scale DEC_RANK 32→64 and MLP_HIDDEN 512→1024.'
)
text = text.replace("    uv run python projects/ssl/exp11b.py", "    uv run python projects/ssl/exp12b.py")
text = text.replace('EXP_NAME = "exp11b"', 'EXP_NAME = "exp12b"')
text = text.replace("DEC_RANK     = 32", "DEC_RANK     = 64")
text = text.replace("MLP_HIDDEN   = 512", "MLP_HIDDEN   = 1024")
(SSL / "exp12b.py").write_text(text)
print("wrote exp12b.py")

# ── exp12c: based on exp11b (Hebbian+MLP), LATENT_DIM 1024→2048, MLP_HIDDEN 512→1024
text = src_b
text = text.replace(
    'exp11b — LeWM JEPA on MNIST: Hebbian gram encoder, LATENT_DIM=1024.',
    'exp12c — LeWM JEPA on MNIST: Hebbian gram encoder, DEC_RANK=32 LATENT_DIM=2048 MLP_HIDDEN=1024.'
)
text = text.replace(
    'vs exp10a (96.0%): scale LATENT_DIM 512→1024 with standard Hebbian gram (DEC_RANK=32).',
    'vs exp11b (96.4%): scale LATENT_DIM 1024→2048 and MLP_HIDDEN 512→1024.'
)
text = text.replace("    uv run python projects/ssl/exp11b.py", "    uv run python projects/ssl/exp12c.py")
text = text.replace('EXP_NAME = "exp11b"', 'EXP_NAME = "exp12c"')
text = text.replace("LATENT_DIM   = 1024", "LATENT_DIM   = 2048")
text = text.replace("MLP_HIDDEN   = 512", "MLP_HIDDEN   = 1024")
(SSL / "exp12c.py").write_text(text)
print("wrote exp12c.py")
