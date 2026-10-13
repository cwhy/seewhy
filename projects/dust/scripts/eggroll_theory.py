"""What cosine must a random-direction ES estimate have? Per-layer gradient shares from the true gradient.

For an estimator that averages n directional derivatives along random directions (EGGROLL's
perturbations, whatever their rank), the error in layer l is isotropic noise of total energy
d_l * |g|^2 / n, where d_l is the layer's parameter count and |g| the norm of the *whole*
gradient: every direction's score is a sum over all layers, so the other layers' signal is
noise to this one. With s_l = |g_l| / |g|, the cosine to the true gradient is

    cos_l(n) = s_l / sqrt(s_l^2 + d_l / n),        ~ s_l sqrt(n / d_l)   for n << d_l / s_l^2.

This script measures s_l and d_l at a checkpoint (true gradient, CPU, f32) and appends them to
results.jsonl (kind="theory"), so the prediction is a function of logged numbers and the
alignment report can set it beside EGGROLL's measured cosines.

    CUDA_VISIBLE_DEVICES= ~/.local/bin/uv run --no-sync python projects/dust/scripts/eggroll_theory.py
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))

from lib import data, diagnostics as dg
from lib.align import BATCH, load_params
from lib.model import SIZES

CFG = SIZES["L8d512"]
BLOCK_KEY = dict(mlp_in="fc", mlp_out="mproj", attn_out="proj", q="q", k="k", v="v", gate="gate", value_emb="ve")


def main(ckpts: list[str]) -> None:
    rows = data.load_rows("val")[:BATCH]
    ids, tg = [jnp.asarray(a) for a in data.split_xy(rows)]
    done = set()
    jsonl = PROJECT / "results.jsonl"
    for line in jsonl.read_text().splitlines():
        try:
            done.add(json.loads(line)["name"])
        except Exception:
            pass
    for ckpt in ckpts:
        name = f"theory|{ckpt}"
        if name in done:
            continue
        params = load_params(ckpt, CFG)
        g = dg.true_gradient(params, ids, tg, CFG, jnp.float32)
        total = float(np.sqrt(sum(float(jnp.sum(jnp.square(x))) for x in jax.tree_util.tree_leaves(g))))
        share, dim = {}, {}
        for t, key in BLOCK_KEY.items():
            share[t] = {str(l): float(jnp.linalg.norm(b[key])) / total for l, b in enumerate(g["blocks"]) if key in b}
            dim[t] = {str(l): int(b[key].size) for l, b in enumerate(params["blocks"]) if key in b}
        share["token_emb"], dim["token_emb"] = {"0": float(jnp.linalg.norm(g["wte"])) / total}, {"0": int(params["wte"].size)}
        share["head"], dim["head"] = {"0": float(jnp.linalg.norm(g["head"])) / total}, {"0": int(params["head"].size)}
        row = dict(experiment="theory", name=name, kind="theory", ckpt=ckpt, grad_norm=total, share=share, dim=dim,
                   n_params=int(sum(x.size for x in jax.tree_util.tree_leaves(params))))
        with open(jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(ckpt, "grad norm", round(total, 4), {t: round(float(np.mean(list(s.values()))), 3) for t, s in share.items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["init", "bp100M_step610", "bp100M_step6100"])
