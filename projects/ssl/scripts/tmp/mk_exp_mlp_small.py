"""Generate exp_mlp_ema_{16,32,64,128}.py from exp_mlp_ema_512.py."""
from pathlib import Path

SRC   = Path(__file__).parent.parent.parent / "exp_mlp_ema_512.py"
DIMS  = [16, 32, 64, 128]

src = SRC.read_text()

for dim in DIMS:
    out = (src
        .replace('exp_mlp_ema_512', f'exp_mlp_ema_{dim}')
        .replace('LATENT_DIM      = 512', f'LATENT_DIM      = {dim}')
        .replace('MLP_ENC_HIDDEN  = 512', f'MLP_ENC_HIDDEN  = 512')
        # predictor hidden can stay 512 — only encoder output changes
    )
    dest = SRC.parent / f"exp_mlp_ema_{dim}.py"
    dest.write_text(out)
    print(f"wrote {dest.name}")
