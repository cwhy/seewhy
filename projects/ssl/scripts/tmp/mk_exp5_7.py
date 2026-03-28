"""
Generate experiments 5–7: sweep DEC_RANK and MLP_HIDDEN from exp4 baseline.

exp4 baseline: LATENT_DIM=128, DEC_RANK=16, MLP_HIDDEN=256, mask=random50%

Grid:
  exp5: DEC_RANK=32, MLP_HIDDEN=256  (scale rank only)
  exp6: DEC_RANK=16, MLP_HIDDEN=512  (scale hidden only)
  exp7: DEC_RANK=32, MLP_HIDDEN=512  (scale both)
"""

from pathlib import Path

SRC = Path(__file__).parent.parent.parent / "experiments4.py"
src = SRC.read_text()

configs = [
    ("exp5", 32, 256, "DEC_RANK=32 MLP_HIDDEN=256"),
    ("exp6", 16, 512, "DEC_RANK=16 MLP_HIDDEN=512"),
    ("exp7", 32, 512, "DEC_RANK=32 MLP_HIDDEN=512"),
]

for exp_name, dec_rank, mlp_hidden, desc in configs:
    text = src
    text = text.replace('exp4 — LeWM JEPA on MNIST: larger embedding (LATENT_DIM 64 → 128).',
                        f'{exp_name} — LeWM JEPA on MNIST: {desc}.')
    text = text.replace('vs exp2 (best so far, 89.4%): same random 50% masking, double the embedding size.\n'
                        '  More capacity in the bottleneck should give the encoder more room to encode\n'
                        '  class-discriminative features.',
                        f'vs exp4 (91.1%): same random 50% masking, LATENT_DIM=128. {desc}.')
    text = text.replace('    uv run python projects/ssl/experiments4.py',
                        f'    uv run python projects/ssl/{exp_name}.py')
    text = text.replace('EXP_NAME = "exp4"', f'EXP_NAME = "{exp_name}"')
    text = text.replace(f'DEC_RANK     = 16', f'DEC_RANK     = {dec_rank}')
    text = text.replace(f'MLP_HIDDEN   = 256', f'MLP_HIDDEN   = {mlp_hidden}')
    text = text.replace('jepa-gram mask=random50% LATENT={LATENT_DIM} LAMBDA={LAMBDA} DEC_RANK={DEC_RANK}',
                        'jepa-gram mask=random50% LATENT={LATENT_DIM} DEC_RANK={DEC_RANK} MLP_HIDDEN={MLP_HIDDEN}')

    out = SRC.parent / f"{exp_name}.py"
    out.write_text(text)
    print(f"wrote {out}")
