"""Model-free reference points. Every accuracy this project reports is read
against these, and a bare accuracy on MNIST is meaningless without them.

    chance             1/V
    marginal           always predict the most common value bin
    position marginal  predict the most common bin AT THAT POSITION, over the
                       training split — uses no context at all, so it is the
                       honest "did the context matter" line
    copy ceiling       1.0, by construction, in the present arm

Reported for the bottom half (rows 14-27), which is what evaluation scores, and
for all pixels, which is what training scores.

    .venv/bin/python projects/ar-recall/scripts/baselines.py
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT.parents[1]))          # repo root LAST — see workflow.md
sys.path.insert(0, str(PROJECT))

import numpy as np

# recall-gen's domain loader, by path: `lib` is this project's package.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "rg_domains", PROJECT.parents[0] / "recall-gen" / "lib" / "domains.py")
rg_domains = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(rg_domains)

V = 16
MASK_ROWS = 14          # bottom 14 rows are the target half, as in recall-gen


def main():
    Xtr, _, Xte, _ = rg_domains.raw_pools("mnist")
    q = lambda X: np.minimum((np.asarray(X) * V).astype(np.int64), V - 1)
    Btr, Bte = q(Xtr), q(Xte)
    bottom = np.zeros(784, bool)
    bottom[(28 - MASK_ROWS) * 28:] = True

    glob = int(np.bincount(Btr.reshape(-1), minlength=V).argmax())
    pos = np.stack([np.bincount(Btr[:, p], minlength=V).argmax()
                    for p in range(784)])
    hit = Bte == pos[None, :]

    print(f"V = {V} uniform bins, target half = bottom {MASK_ROWS} rows\n")
    print(f"  chance (1/V)                        {1.0 / V:.4f}")
    print(f"  most common bin overall             {glob}"
          f"   ({(Btr == 0).mean():.1%} of pixels are in bin 0)")
    print(f"  marginal,          all pixels       {(Bte == glob).mean():.4f}")
    print(f"  marginal,          target half      {(Bte[:, bottom] == glob).mean():.4f}")
    print(f"  position marginal, all pixels       {hit.mean():.4f}")
    print(f"  position marginal, target half      {hit[:, bottom].mean():.4f}")
    print(f"  copy ceiling (present arm)          1.0000")
    d = np.abs(Bte[:, bottom] - pos[None, bottom]).mean()
    print(f"\n  position marginal, mean |bin error| on the target half   {d:.4f}")

    # Overall accuracy is nearly useless here: predicting background always
    # scores ~0.81, so the whole dynamic range is 0.81 to 1.0. These are the
    # candidates for a metric that can actually be read.
    T = Bte[:, bottom]
    Pm = np.broadcast_to(pos[None, bottom], T.shape)
    fg = T > 0
    print(f"\n  fraction of target-half pixels that are foreground (bin > 0)  "
          f"{fg.mean():.4f}")
    print(f"  marginal,          accuracy on foreground pixels only   "
          f"{(T[fg] == glob).mean():.4f}")
    print(f"  position marginal, accuracy on foreground pixels only   "
          f"{(Pm[fg] == T[fg]).mean():.4f}")
    print(f"  position marginal, mean |bin error| on foreground       "
          f"{np.abs(Pm[fg] - T[fg]).mean():.4f}")

    # Cross-entropy of the position-conditional distribution, which is the
    # strongest context-free predictor and the natural AR reference.
    cnt = np.stack([np.bincount(Btr[:, p], minlength=V) for p in range(784)])
    prob = (cnt + 1.0) / (cnt.sum(1, keepdims=True) + V)          # Laplace
    lp = np.log2(prob[None, :, :].repeat(1, 0))
    idx = np.arange(784)
    ce_all = -np.log2(prob[idx[None, :], Bte]).mean()
    ce_bot = -np.log2(prob[idx[None, bottom], T]).mean()
    ce_uni = np.log2(V)
    print(f"\n  cross-entropy, uniform                    {ce_uni:.4f} bits")
    print(f"  cross-entropy, position-conditional, all  {ce_all:.4f} bits")
    print(f"  cross-entropy, position-conditional, tgt  {ce_bot:.4f} bits")


if __name__ == "__main__":
    main()
