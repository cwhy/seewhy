"""
Generate Fashion-MNIST EMA-JEPA variants for all encoder architectures at LATENT=1024.

Encoders:
  fmnist_mlp_ema       — shallow MLP (784→512→1024)
  fmnist_mlp2_ema      — deeper MLP  (784→1024→1024→1024)
  fmnist_conv_ema      — ConvNet v1  [32,64,128]
  fmnist_conv2_ema     — ConvNet v2  [64,128,256]+FC512
  fmnist_vit_ema       — mini ViT    (3L, D=128, 4H)
  fmnist_gram_ema      — Gram+Hadamard dual (DEC_RANK=32)
  fmnist_gram_single_ema — Gram single-proj  (DEC_RANK=32)

Changes per file:
  - load_supervised_image("mnist") → load_supervised_image("fashion_mnist")
  - EXP_NAME prefixed with "fmnist_"
  - LATENT pinned to 1024 (for gram_single: DEC_RANK=32, LATENT=1024)
"""

from pathlib import Path

SSL = Path(__file__).parent.parent.parent

def fmnist(src, src_exp_name, latent=1024, extra_subs=None):
    """Patch src to produce a Fashion-MNIST variant."""
    t = src
    # Dataset
    t = t.replace('load_supervised_image("mnist")', 'load_supervised_image("fashion_mnist")')
    # EXP_NAME
    t = t.replace(f'EXP_NAME = "{src_exp_name}"', f'EXP_NAME = "fmnist_{src_exp_name}"')
    # LATENT pinned (replace whatever latent is set)
    for old_latent in [512, 1024, 2048, 4096]:
        t = t.replace(f"LATENT_DIM      = {old_latent}", f"LATENT_DIM      = {latent}")
        t = t.replace(f"LATENT_DIM   = {old_latent}", f"LATENT_DIM   = {latent}")
    # Usage line
    t = t.replace(f"    uv run python projects/ssl/{src_exp_name}.py",
                  f"    uv run python projects/ssl/fmnist_{src_exp_name}.py")
    if extra_subs:
        for old, new in extra_subs:
            t = t.replace(old, new)
    return t

configs = [
    ("exp_mlp_ema",          "exp_mlp_ema",         1024, None),
    ("exp_mlp2_ema",         "exp_mlp2_ema",        1024, None),
    ("exp_conv_ema",         "exp_conv_ema",         1024, None),
    ("exp_conv2_ema",        "exp_conv2_ema",        1024, None),
    ("exp_vit_ema",          "exp_vit_ema",          1024, None),
    # gram: base file is exp_ema.py (LATENT=4096, DEC_RANK=32 → use DEC_RANK=32, LATENT=1024)
    ("exp_ema",              "exp_ema",              1024, [
        ("LATENT_DIM   = 4096", "LATENT_DIM   = 1024"),
    ]),
    # gram_single: base file already at LATENT=1024, DEC_RANK=32
    ("exp_gram_single_ema",  "exp_gram_single_ema",  1024, None),
]

for src_file, src_exp, latent, extra in configs:
    src = (SSL / f"{src_file}.py").read_text()
    text = fmnist(src, src_exp, latent, extra)
    out_name = f"fmnist_{src_exp}"
    (SSL / f"{out_name}.py").write_text(text)
    print(f"wrote {out_name}.py")
