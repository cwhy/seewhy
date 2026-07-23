"""Check DEC_RANK for gram encoders across all teacher types."""
import pickle, numpy as np
from pathlib import Path

SSL = Path("projects/ssl")

configs = [
    ("EMA",     "params_exp_ema_1024.pkl",          "gram_dual"),
    ("EMA",     "params_exp_gram_single_ema_1024.pkl", "gram_single"),
    ("EMA",     "params_exp_gram_glu_ema.pkl",       "gram_glu"),
    ("Sigreg",  "params_exp_sigreg_gram_single.pkl", "gram_single"),
    ("Sigreg",  "params_exp_sigreg_gram_glu.pkl",    "gram_glu"),
    ("DAE",     "params_exp_dae_gram_dual.pkl",      "gram_dual"),
    ("DAE",     "params_exp_dae_gram_single.pkl",    "gram_single"),
    ("DAE",     "params_exp_dae_gram_glu.pkl",       "gram_glu"),
]

for method, pkl, arch in configs:
    p = SSL / pkl
    if not p.exists():
        print(f"  {method:8} {arch:12} MISSING")
        continue
    params = pickle.load(open(p, "rb"))
    W_enc  = params["W_enc"]
    D_rank = W_enc.shape[1]
    enc_row = params["enc_row_emb"].shape  # (28, D_R)
    enc_col = params["enc_col_emb"].shape  # (28, D_C)
    W_pos  = params["W_pos_enc"].shape     # (D_RC, D)
    print(f"  {method:8} {arch:12} W_enc={W_enc.shape}  D_rank={D_rank}  "
          f"enc_row={enc_row}  enc_col={enc_col}  W_pos={W_pos}")
    # Check W_enc_A/B shape if present
    if "W_enc_A" in params:
        print(f"           W_enc_A={params['W_enc_A'].shape}")
    if "W_proj" in params:
        print(f"           W_proj={params['W_proj'].shape}")
    if "W_gate" in params:
        print(f"           W_gate={params['W_gate'].shape}")
