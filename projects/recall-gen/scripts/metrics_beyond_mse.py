"""Does the frozen model's advantage survive a metric that does not reward blur?

MSE's minimiser is the conditional mean. Given the top half of a digit the true
bottom half is genuinely ambiguous, so the MSE-optimal output is a blur over all
plausible completions: a model emitting a sharp, specific, plausible-but-wrong
bottom half is punished, and a model emitting mush is rewarded. That is the same
axis as soft-versus-sharp, which means "the soft model generalises better" may be
a restatement of "MSE prefers conditional means" rather than a fact about
generalisation.

Three more metrics on the same checkpoints and the same episodes, all on the
absent-target condition where the answer must be predicted:

  realism    distance from the predicted hidden half to the NEAREST REAL hidden
             half in the training pool. Blur is far from every real image; a
             sharp copy is close to one. No training, no classifier, so nothing
             here is tuned on the models being judged.
  nn_label   the label of that nearest real image, against the query's true
             label. Asks "is this the right digit" without a classifier.
  clf        an independently trained MLP classifies the composited completion
             (true visible half + predicted hidden half) against the true label.

Reference rows bound every column: the true image is the ceiling, the mean image
is the degenerate blur, and the soft look-up at a sharp and a soft temperature
shows what each metric does to the two ends of the axis under test.

CAVEAT, stated because it cuts against the interesting result: the classifier is
trained on real images, so a blurry composite is out of distribution for it and
may be scored harshly for that alone. `realism` and `nn_label` need no classifier
and are the more neutral measures; read them first.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/metrics_beyond_mse.py
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT.parents[1]))          # repo root LAST — see workflow.md
sys.path.insert(0, str(PROJECT))

from lib.core import Cfg, row_mask, masked_mse
from lib import evalsets
from lib.train import Run, build_pools, make_eval, append_result, already_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP = "metrics_beyond_mse"
COND = "D_novel_absent"
CFG = Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=17)
MODELS = {
    "exp20_best":  "params_exp20_best.pkl",
    "exp20_final": "params_exp20.pkl",
    "exp24_best":  "params_exp24_best.pkl",
    "exp24_final": "params_exp24.pkl",
}


# ── an independent digit classifier (never sees any model's output in training)
def train_classifier(X, y, key, steps=3000, batch=256):
    d, h, c = X.shape[1], 512, 10
    k1, k2, k3 = jax.random.split(key, 3)
    p = {"W1": jax.random.normal(k1, (d, h)) / d ** 0.5, "b1": jnp.zeros(h),
         "W2": jax.random.normal(k2, (h, h)) / h ** 0.5, "b2": jnp.zeros(h),
         "W3": jax.random.normal(k3, (h, c)) / h ** 0.5, "b3": jnp.zeros(c)}
    Xj, yj = jnp.array(X), jnp.array(y)

    def logits(p, x):
        x = jax.nn.relu(x @ p["W1"] + p["b1"])
        x = jax.nn.relu(x @ p["W2"] + p["b2"])
        return x @ p["W3"] + p["b3"]

    def loss(p, x, y):
        return optax.softmax_cross_entropy_with_integer_labels(logits(p, x), y).mean()

    opt = optax.adam(1e-3)
    st = opt.init(p)

    @jax.jit
    def step(p, st, k):
        i = jax.random.randint(k, (batch,), 0, Xj.shape[0])
        g = jax.grad(loss)(p, Xj[i], yj[i])
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st

    k = jax.random.key(0)
    for _ in range(steps):
        k, ks = jax.random.split(k)
        p, st = step(p, st, ks)
    return p, jax.jit(lambda p, x: logits(p, x))


def main():
    if already_done(EXP):
        logging.info(f"{EXP} already in results.jsonl — skipping")
        return

    rn = Run(exp_name="", name="", M=16, Q=1, mask_rows=14, cfg=CFG)
    pools, labels = build_pools(rn)
    mask = row_mask(14)
    mask_j = jnp.array(mask)
    hid = mask > 0.5
    mean_img = pools["train"].mean(0)
    ev = evalsets.build(pools, mask, 16, 1, 512, mean_img, ctx_mode="knn", labels=labels)
    es = ev[COND]

    # Classifier and the realism reference pool both come from the TRAIN split;
    # the queries are held-out images, so an exact match is never available.
    clf_p, clf_fn = train_classifier(pools["train"], labels["train"], jax.random.key(0))
    acc = float((jnp.argmax(clf_fn(clf_p, jnp.array(pools["held"])), -1)
                 == jnp.array(labels["held"])).mean())
    logging.info(f"  classifier held-out accuracy {acc:.4f}")

    ref_h = jnp.array(pools["train"][:, hid])                 # (N, 392) real hidden halves
    ref_lab = jnp.array(labels["train"])
    ref_sq = (ref_h ** 2).sum(-1)[None, :]
    n_hid = int(hid.sum())

    # The query's true label, needed for nn_label and clf. evalsets draws queries
    # from the pool without returning indices, so recover them by exact match on
    # the full image — held-out MNIST has no duplicate rows.
    held = jnp.array(pools["held"])
    q = es.qry[:, 0, :]
    qi = jnp.argmin(((q[:, None, :] - held[None, :, :]) ** 2).sum(-1)
                    if q.shape[0] <= 64 else
                    ((q ** 2).sum(-1)[:, None] + (held ** 2).sum(-1)[None, :]
                     - 2.0 * (q @ held.T)), axis=-1)
    true_lab = jnp.array(labels["held"])[qi]

    def score(pred_full, tag):
        """pred_full: (E,1,784). Composite the true visible half back in."""
        ph = pred_full[:, 0, :][:, hid]
        d = ((ph ** 2).sum(-1)[:, None] + ref_sq - 2.0 * (ph @ ref_h.T))
        nn = jnp.argmin(d, axis=-1)
        realism = float(jnp.take_along_axis(d, nn[:, None], 1).mean()) / n_hid
        nn_lab = float((ref_lab[nn] == true_lab).mean())
        comp = q * (1.0 - mask_j) + pred_full[:, 0, :] * mask_j
        pr = jax.nn.softmax(clf_fn(clf_p, comp), -1)
        row = {"nmse": float(masked_mse(pred_full, es.qry, mask_j)) / es.mse_mean,
               "realism": realism, "nn_label_acc": nn_lab,
               "clf_acc": float((jnp.argmax(pr, -1) == true_lab).mean()),
               "clf_conf": float(pr.max(-1).mean())}
        logging.info(f"  {tag:<22} nmse={row['nmse']:.3f}  realism={realism:.4f}  "
                     f"nn_label={nn_lab:.3f}  clf={row['clf_acc']:.3f} "
                     f"(conf {row['clf_conf']:.3f})")
        return row

    @jax.jit
    def soft_lookup(ctx, qry, tau):
        vis = 1.0 - mask_j
        d = (((qry[:, :, None, :] - ctx[:, None, :, :]) ** 2) * vis).sum(-1) / vis.sum()
        return jnp.einsum("eqm,emp->eqp", jax.nn.softmax(-d / tau, -1), ctx)

    out = {}
    out["true_image"] = score(es.qry, "true image (ceiling)")
    out["mean_image"] = score(jnp.broadcast_to(jnp.array(mean_img), es.qry.shape),
                              "mean image (blur floor)")
    for tau in (0.003, 0.03):
        out[f"lookup_tau{tau}"] = score(soft_lookup(es.ctx, es.qry, tau),
                                        f"look-up tau={tau}")
    for mk, fname in MODELS.items():
        path = PROJECT / fname
        if not path.exists():
            logging.info(f"{mk}: no checkpoint, skipping")
            continue
        with open(path, "rb") as f:
            p = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
        pred, _ = make_eval(rn, mask_j)(p, es.ctx, es.qry)
        out[mk] = score(pred, mk)

    append_result(dict(experiment=EXP, M=16, Q=1, mask_rows=14, n_eval=512,
                       name="metrics that do not reward the conditional mean",
                       condition=COND, ctx_mode="knn", time_s=0.0,
                       classifier_heldout_acc=acc, metrics=out))
    logging.info(f"wrote {EXP}")


if __name__ == "__main__":
    main()
