"""Did the synthetic-trained discriminator learn "same world" rather than "same item"?

Evaluation only, exp31/32 final checkpoints, 128 held-out worlds. For each of
512 episodes (16 context items from one world), D's scores for: context items,
unseen items of the same world, items of a different world, and MNIST test
images. AUROC of context items against each of the other three. Appends one row
per run (experiment "synth_world").

    ~/.local/bin/uv run --no-sync python projects/recall-gan/scripts/analyse_synth_world.py
"""
import json
import pickle
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from lib.model import Cfg, disc_logits  # noqa: E402
from lib.synth import synth_pools  # noqa: E402
from lib.train import append_result, auroc  # noqa: E402
from shared_lib.datasets import load_supervised_image  # noqa: E402

ROWS = [json.loads(l) for l in (PROJECT / "results.jsonl").read_text().splitlines()]
d = load_supervised_image("mnist")
te = np.asarray(d.X_test, np.float32).reshape(len(d.X_test), -1)
if te.max() > 1.5:
    te = te / 255.0
rng = np.random.default_rng(0)
E = 512

for exp, k in (("exp31", 1.0), ("exp32", 8.0)):
    _, tst = synth_pools(k_min=k)
    W = tst.shape[0]
    w = rng.integers(0, W, E)
    perm = np.argsort(rng.random((E, tst.shape[1])), 1)
    items = tst[w[:, None], perm]
    ctx, same = jnp.asarray(items[:, :16]), jnp.asarray(items[:, 16:20])
    w2 = (w + 1 + rng.integers(0, W - 1, E)) % W
    other = jnp.asarray(tst[w2][:, :4])
    mn = jnp.asarray(te[rng.choice(len(te), (E, 4))])
    mem = jnp.take_along_axis(ctx, jnp.asarray(rng.integers(0, 16, (E, 4)))[..., None], axis=1)
    for r in sorted([r for r in ROWS if r["experiment"] == exp], key=lambda r: r["seed"]):
        ck = pickle.load(open(PROJECT / f"params_{r['run']}.pkl", "rb"))
        pd = jax.tree_util.tree_map(jnp.asarray, ck["disc"]); cfg = Cfg(**ck["cfg"])
        s = {n: disc_logits(pd, ctx, q, cfg) for n, q in (("mem", mem), ("same", same), ("other", other), ("mnist", mn))}
        row = dict(experiment="synth_world", run=f"synth_world_{r['run']}", source_run=r["run"], seed=r["seed"],
                   auroc_vs_same_world=float(auroc(s["mem"], s["same"])),
                   auroc_vs_other_world=float(auroc(s["mem"], s["other"])),
                   auroc_vs_mnist=float(auroc(s["mem"], s["mnist"])),
                   p_mem=float(jax.nn.sigmoid(s["mem"]).mean()), p_same=float(jax.nn.sigmoid(s["same"]).mean()),
                   p_other=float(jax.nn.sigmoid(s["other"]).mean()), p_mnist=float(jax.nn.sigmoid(s["mnist"]).mean()))
        append_result(row)
        print(f"{exp} s{r['seed']}  context vs: same-world unseen {row['auroc_vs_same_world']:.3f}  "
              f"other world {row['auroc_vs_other_world']:.3f}  MNIST {row['auroc_vs_mnist']:.3f}  | P: ctx {row['p_mem']:.2f} "
              f"same {row['p_same']:.2f} other {row['p_other']:.2f} mnist {row['p_mnist']:.2f}")
