"""Invariants of episode construction. Each one is a way the task could be
silently wrong in a way no training curve would reveal.

    1  lengths are what TaskCfg promises, for both training and evaluation
    2  a TRAINING stream never contains a held-out (label, position) pair
    3  an EVALUATION stream does contain them — the hold-out is a property of
       training, not of the task
    4  present arm: every target triple appears verbatim earlier in the stream
    5  absent arm: the query label never appears before the prompt
    6  slot structure — token i is a label / position / value by i mod 3
    7  a decoded micro-episode, printed, so the shape can be read by eye

    .venv/bin/python projects/ar-recall/scripts/test_task.py
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT.parents[1]))          # repo root LAST — see workflow.md
sys.path.insert(0, str(PROJECT))

import numpy as np

from lib.task import TaskCfg, build_holdout, make, quantise

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "rg_domains", PROJECT.parents[0] / "recall-gen" / "lib" / "domains.py")
rg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(rg)

OK = lambda c: "ok" if c else "FAIL"


def main():
    cfg = TaskCfg()
    H = build_holdout(cfg)
    Xtr, _, _, _ = rg.raw_pools("mnist")
    pool = quantise(Xtr[:4000], cfg)
    rng = np.random.default_rng(0)
    print(f"vocab {cfg.vocab}  (labels {cfg.n_labels}, positions {cfg.n_pos}, "
          f"values {cfg.n_values})")
    print(f"held out per label {cfg.n_hold} of {cfg.n_pos} "
          f"({cfg.n_hold / cfg.n_pos:.1%})\n")

    print("1. lengths")
    for tr in (True, False):
        ep = make(pool, H, cfg, 8, rng, present=True, training=tr)
        want = cfg.n_tokens(tr)
        print(f"   training={str(tr):<5s} tokens {ep.tokens.shape[1]:>6d} "
              f"expected {want:>6d}  {OK(ep.tokens.shape[1] == want)}")

    print("2/3. the hold-out is respected in training and not at evaluation")
    for tr in (True, False):
        ep = make(pool, H, cfg, 32, rng, present=True, training=tr)
        n_held = int(ep.is_held.sum())
        want = (n_held == 0) if tr else (n_held > 0)
        print(f"   training={str(tr):<5s} held-out triples in stream {n_held:>6d}  "
              f"{OK(want)}")

    print("4. present arm: every target triple is verbatim earlier in the stream")
    ep = make(pool, H, cfg, 16, rng, present=True, training=False)
    bad = 0
    for e in range(ep.tokens.shape[0]):
        row, tg = ep.tokens[e], ep.is_target[e]
        head = int(np.argmax(tg))                        # first target value slot
        ctx = {tuple(row[i:i + 3]) for i in range(0, head - 2, 3)}
        for i in np.flatnonzero(tg):
            if tuple(row[i - 2:i + 1]) not in ctx:
                bad += 1
    print(f"   target triples not found earlier: {bad}  {OK(bad == 0)}")

    print("5. absent arm: the query label does not appear before the prompt")
    ep = make(pool, H, cfg, 16, rng, present=False, training=False)
    bad = 0
    for e in range(ep.tokens.shape[0]):
        row, tg = ep.tokens[e], ep.is_target[e]
        head = int(np.argmax(tg))
        q_lab = row[head - 2]
        n_ctx_tri = cfg.n_context * cfg.n_pos
        if q_lab in row[:3 * n_ctx_tri:3]:
            bad += 1
    print(f"   episodes where the query label leaked into context: {bad}  "
          f"{OK(bad == 0)}")
    # and the counterpart: in the present arm it MUST appear
    epp = make(pool, H, cfg, 16, rng, present=True, training=False)
    seen = 0
    for e in range(epp.tokens.shape[0]):
        row, tg = epp.tokens[e], epp.is_target[e]
        head = int(np.argmax(tg))
        if row[head - 2] in row[:3 * cfg.n_context * cfg.n_pos:3]:
            seen += 1
    print(f"   present arm, query label IS in context: {seen}/16  "
          f"{OK(seen == 16)}")

    print("6. slot structure")
    row = ep.tokens[0]
    a = (row[0::3] < cfg.pos_base).all()
    b = ((row[1::3] >= cfg.pos_base) & (row[1::3] < cfg.val_base)).all()
    c = (row[2::3] >= cfg.val_base).all()
    print(f"   labels/positions/values in range: {OK(a)} {OK(b)} {OK(c)}")

    print("\n7. a micro-episode (2x2 images, 4 labels, 1 context image)")
    small = TaskCfg(n_labels=4, n_context=1, n_values=4, side=2, mask_rows=1,
                    n_hold=0)
    Hs = build_holdout(small)
    pl = np.array([[1, 0, 3, 2], [0, 3, 1, 2]])
    es = make(pl, Hs, small, 1, np.random.default_rng(3), present=True,
              training=True)
    r = es.tokens[0]
    for i in range(0, len(r), 3):
        kind = "TARGET" if es.is_target[0][i + 2] else "      "
        print(f"   {kind}  L{r[i] - small.lab_base}  P{r[i+1] - small.pos_base}"
              f"  V{r[i+2] - small.val_base}")


if __name__ == "__main__":
    main()
