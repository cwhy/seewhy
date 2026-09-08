"""Episode construction. See `concepts.md` for what every term here means.

An episode is recall-gen's, tokenised:

    A(full) B(full) C(full) D(full)  halfA  ->  predict the other half of A

emitted as `(label, position, value)` triples, three tokens per pixel.

Two shape decisions worth knowing before reading the code.

**Held-out pairs are omitted from training streams, so lengths would vary.** The
hold-out set `H` is therefore built with exactly `n_hold` positions per label,
split evenly between the top and bottom halves. Then every training episode has
exactly `(L + 1) (784 - n_hold)` triples and every evaluation episode exactly
`(L + 1) * 784`, both fixed, and neither needs padding.

**Evaluation streams omit nothing.** The hold-out is a property of training, not
of the task: at evaluation a held-out pairing is emitted like any other, and what
is measured is whether the model can handle a pairing it was never trained on.
"""

from typing import NamedTuple

import numpy as np


class TaskCfg(NamedTuple):
    n_labels: int = 32        # LAMBDA — the label symbol vocabulary
    n_context: int = 4        # L — context images per episode
    n_values: int = 16        # V — value bins
    side: int = 28
    mask_rows: int = 14       # bottom rows = the target half, as in recall-gen
    n_hold: int = 48          # held-out positions per label; even across halves

    @property
    def n_pos(self) -> int:
        return self.side * self.side

    # Disjoint ranges in one vocabulary, so a token id determines its class.
    @property
    def lab_base(self) -> int:
        return 0

    @property
    def pos_base(self) -> int:
        return self.n_labels

    @property
    def val_base(self) -> int:
        return self.n_labels + self.n_pos

    @property
    def vocab(self) -> int:
        return self.n_labels + self.n_pos + self.n_values

    @property
    def n_top(self) -> int:
        return (self.side - self.mask_rows) * self.side

    def n_triples(self, training: bool) -> int:
        per = self.n_pos - (self.n_hold if training else 0)
        return (self.n_context + 1) * per

    def n_tokens(self, training: bool) -> int:
        return 3 * self.n_triples(training)


def quantise(X: np.ndarray, cfg: TaskCfg) -> np.ndarray:
    """(N, 784) in [0,1] -> value bin indices in [0, V)."""
    return np.minimum((np.asarray(X) * cfg.n_values).astype(np.int64),
                      cfg.n_values - 1)


def build_holdout(cfg: TaskCfg, seed: int = 20260908) -> np.ndarray:
    """(n_labels, n_pos) bool. True = this (label, position) never appears in a
    training stream.

    Exactly `n_hold` per label, half drawn from the top half of the image and
    half from the bottom, so a training episode's length does not depend on which
    labels it happened to draw.
    """
    assert cfg.n_hold % 2 == 0, "n_hold is split evenly across the two halves"
    rng = np.random.default_rng(seed)
    H = np.zeros((cfg.n_labels, cfg.n_pos), bool)
    top = np.arange(cfg.n_top)
    bot = np.arange(cfg.n_top, cfg.n_pos)
    for l in range(cfg.n_labels):
        H[l, rng.choice(top, cfg.n_hold // 2, replace=False)] = True
        H[l, rng.choice(bot, cfg.n_hold // 2, replace=False)] = True
    return H


def _kept(H: np.ndarray, cfg: TaskCfg, training: bool):
    """Per label, the positions a stream may emit: all of them at evaluation,
    everything outside `H` during training. Fixed length either way."""
    if not training:
        allp = np.arange(cfg.n_pos)
        full = np.broadcast_to(allp, (cfg.n_labels, cfg.n_pos))
        return full, full[:, :cfg.n_top], full[:, cfg.n_top:]
    keep = np.stack([np.flatnonzero(~H[l]) for l in range(cfg.n_labels)])
    tops = np.stack([k[k < cfg.n_top] for k in keep])
    bots = np.stack([k[k >= cfg.n_top] for k in keep])
    return keep, tops, bots


def _run(label: int, positions: np.ndarray, img: np.ndarray, cfg: TaskCfg):
    """One label's triples: (3 * len(positions),) token ids."""
    t = np.empty((len(positions), 3), np.int32)
    t[:, 0] = cfg.lab_base + label
    t[:, 1] = cfg.pos_base + positions
    t[:, 2] = cfg.val_base + img[positions]
    return t.reshape(-1)


class Episodes(NamedTuple):
    tokens: np.ndarray        # (E, T) int32
    is_value: np.ndarray      # (E, T) bool — value slots; every one is a target
    is_target: np.ndarray     # (E, T) bool — value slots of the TARGET half
    is_held: np.ndarray       # (E, T) bool — this (label, pos) is in H
    present: bool             # whether the query image is among the context


def make(pool: np.ndarray, H: np.ndarray, cfg: TaskCfg, n_ep: int, rng,
         present: bool, training: bool) -> Episodes:
    """Build `n_ep` episodes from a pool of quantised images (N, n_pos).

    `present` selects axis 1: the query image is one of the context images, so
    its target half is verbatim in the stream, or it is a separate image, so the
    target half has to be inferred. `L` context images either way, so both arms
    have the same length and their numbers are comparable.
    """
    keep, tops, bots = _kept(H, cfg, training)
    L, T = cfg.n_context, cfg.n_tokens(training)
    tok = np.empty((n_ep, T), np.int32)
    tgt = np.zeros((n_ep, T), bool)

    for e in range(n_ep):
        # L+1 distinct labels; the present arm leaves the last one unused.
        labs = rng.choice(cfg.n_labels, L + 1, replace=False)
        if present:
            q_slot = int(rng.integers(L))                  # which context image
            imgs = rng.choice(len(pool), L, replace=False)
            q_img, q_lab = imgs[q_slot], labs[q_slot]
        else:
            picks = rng.choice(len(pool), L + 1, replace=False)
            imgs, q_img, q_lab = picks[:L], picks[L], labs[L]

        order = rng.permutation(L)
        parts = [_run(labs[i], keep[labs[i]], pool[imgs[i]], cfg) for i in order]
        parts.append(_run(q_lab, tops[q_lab], pool[q_img], cfg))
        head = sum(len(p) for p in parts)
        parts.append(_run(q_lab, bots[q_lab], pool[q_img], cfg))
        row = np.concatenate(parts)
        assert len(row) == T, f"episode {e}: {len(row)} tokens, expected {T}"
        tok[e] = row
        tgt[e, head:] = True

    slot = np.arange(T) % 3
    is_value = np.broadcast_to(slot == 2, (n_ep, T))
    # A value token's (label, position) is the two tokens before it.
    lab = tok[:, 0::3] - cfg.lab_base
    pos = tok[:, 1::3] - cfg.pos_base
    held_tri = H[lab, pos]                                  # (E, n_triples)
    is_held = np.zeros((n_ep, T), bool)
    is_held[:, 2::3] = held_tri
    return Episodes(np.asarray(tok), np.array(is_value), tgt & is_value,
                    is_held, present)


def value_class(tokens: np.ndarray, cfg: TaskCfg) -> np.ndarray:
    """Token ids -> value class indices in [0, V). Meaningless off value slots."""
    return tokens - cfg.val_base
