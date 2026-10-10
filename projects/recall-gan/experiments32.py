"""exp32 — the method trained on recall-gen's synthetic prior (latent dimension >= 8), tested on MNIST.

User's idea (2026-10-11): train on synthetic data, test on MNIST. The plain
method unchanged (context-aware generator, real = context items, fake =
generated, gradient penalty 10, Adam lr 2e-4, beta1 0, beta2 0.99, 20k steps),
but the training data is synthetic: each episode's 16 context items come from
one random latent-factor world (lib/synth.py, from recall-gen). The networks
never see an MNIST image in training. Every evaluation still runs the usual
MNIST measures; `synth_*` measures use 128 held-out worlds. Seeds 50-55;
compare exp15 (trained on MNIST) on the same seeds.

Usage:
    uv run --no-sync python projects/recall-gan/scripts/run_experiments.py --parallel exp31 exp32
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from lib.train import Cell, run_cells

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))
EXP = "smoke32" if SMOKE else "exp32"
STEPS = 20 if SMOKE else 20_000
EVAL_EVERY = 10 if SMOKE else 500
SEEDS = (50,) if SMOKE else tuple(range(50, 56))

CONFIGS = {"synth_k8": dict(beta1=0.0, adam_b2=0.99, data="synth_k8")}
MINE = ["synth_k8"]

CELLS = [Cell(exp=EXP, arm="ctx", m_ctx=16, seed=s, steps=STEPS, eval_every=EVAL_EVERY,
              q_mem=4, q_non=0, q_gen=4, r1=10.0, tag=name, **CONFIGS[name])
         for name in MINE for s in SEEDS]

if __name__ == "__main__":
    run_cells(CELLS)
