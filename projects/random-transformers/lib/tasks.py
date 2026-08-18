"""The seven tasks, encoded exactly as Appendix D.1 of the paper specifies them.

Every task exposes the same shape:

    task.vocab       int   vocabulary size
    task.n_ctx       int   positional-embedding rows (the paper's E_pos height)
    task.seq_len     int   length of a padded example
    task.sample(key, batch)      -> (toks int32 (B, L), mask bool (B, L))
    task.test_set()              -> (toks, mask) — fixed across every run

``mask[b, t]`` marks the positions whose *prediction* is scored: the model reads
``toks[b, :t+1]`` and must emit ``toks[b, t+1]``. Prompt and padding positions
are masked out, so both the loss and the accuracy see only the answer.

Where the paper pins a number we use it, including the positional-embedding
heights (5 / 100 / 40 / 80), which are larger than the tasks strictly need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

TEST_SIZE = 4096          # fixed heldout set for the dynamically generated tasks
TEST_SEED = 12345         # shared by every run and every architecture


@dataclass
class Task:
    name: str
    vocab: int
    n_ctx: int
    seq_len: int
    sample: Callable            # (key, batch) -> (toks, mask)
    test_set: Callable          # () -> (toks, mask)
    chance: float               # sequence-accuracy of the best constant guesser
    describe: str


# ── modular addition ──────────────────────────────────────────────────────────
# [a, b, (a+b) mod p], p = 199. All p^2 pairs, a fixed 95/5 split shared by
# every run — so "test accuracy" measures generalisation, not memorisation.

MOD_P = 199


def _mod_add_split():
    rng = np.random.default_rng(TEST_SEED)
    a, b = np.divmod(np.arange(MOD_P * MOD_P), MOD_P)
    toks = np.stack([a, b, (a + b) % MOD_P], axis=1).astype(np.int32)
    perm = rng.permutation(len(toks))
    n_train = int(round(0.95 * len(toks)))
    return toks[perm[:n_train]], toks[perm[n_train:]]


def make_mod_add() -> Task:
    train, test = _mod_add_split()
    train_j, test_j = jnp.asarray(train), jnp.asarray(test)
    mask = jnp.array([0, 1, 0], dtype=bool)          # predict position 1 -> token 2

    def sample(key, batch):
        idx = jax.random.randint(key, (batch,), 0, train_j.shape[0])
        t = train_j[idx]
        return t, jnp.broadcast_to(mask, t.shape)

    def test_set():
        return test_j, jnp.broadcast_to(mask, test_j.shape)

    return Task(
        name="mod_add", vocab=MOD_P, n_ctx=5, seq_len=3,
        sample=sample, test_set=test_set, chance=1.0 / MOD_P,
        describe=f"[a, b, (a+b) mod {MOD_P}]; {train.shape[0]}/{test.shape[0]} train/test split",
    )


# ── needle in a haystack ──────────────────────────────────────────────────────
# [m1, c1, ..., mk, ck, mu] -> cu. k ~ U[1,30]; values in [1,127]; markers are
# distinct in [128,157]; the query token is the asked marker + 30.

NEEDLE_MAX_K = 30
NEEDLE_LEN = 2 * NEEDLE_MAX_K + 2         # 60 pairs-worth + query + answer


def _needle_one(key):
    k_key, m_key, v_key, u_key = jax.random.split(key, 4)
    k = jax.random.randint(k_key, (), 1, NEEDLE_MAX_K + 1)
    markers = 128 + jax.random.permutation(m_key, NEEDLE_MAX_K)      # distinct
    values = jax.random.randint(v_key, (NEEDLE_MAX_K,), 1, 128)

    body = jnp.stack([markers, values], axis=1).reshape(-1)          # (60,)
    idx = jnp.arange(2 * NEEDLE_MAX_K)
    body = jnp.where(idx < 2 * k, body, 0)

    u = jax.random.randint(u_key, (), 0, k)
    query = markers[u] + 30
    answer = values[u]

    seq = jnp.concatenate([body, jnp.zeros(2, dtype=body.dtype)])
    pos = jnp.arange(NEEDLE_LEN)
    seq = jnp.where(pos == 2 * k, query, seq)
    seq = jnp.where(pos == 2 * k + 1, answer, seq)
    mask = pos == 2 * k                                              # predict the answer
    return seq.astype(jnp.int32), mask


def make_needle() -> Task:
    sample_batch = jax.jit(lambda key, batch: jax.vmap(_needle_one)(
        jax.random.split(key, batch)), static_argnums=1)
    test = sample_batch(jax.random.key(TEST_SEED), TEST_SIZE)

    return Task(
        name="needle", vocab=256, n_ctx=100, seq_len=NEEDLE_LEN,
        sample=lambda key, batch: sample_batch(key, batch),
        test_set=lambda: test, chance=1.0 / 127,
        describe="[m1,c1,...,mk,ck,mu] -> cu; k~U[1,30], values [1,127], markers [128,157]",
    )


# ── decimal addition ──────────────────────────────────────────────────────────
# Two 10-digit numbers, digits reversed. Input digits use tokens 0-9, '+' is 10,
# '=' is 11; output digits use 20-29 and 30 ends the output.

DEC_L = 10
DEC_SEQ = 2 * DEC_L + 2 + (DEC_L + 2)     # a, '+', b, '=', up to 11 digits + EOS


def _decimal_one(key):
    ka, kb = jax.random.split(key)
    # digits reversed: index 0 is the ones digit, index L-1 the leading digit,
    # which must be nonzero so both operands really have 10 digits.
    def digits(k):
        low = jax.random.randint(k, (DEC_L - 1,), 0, 10)
        high = jax.random.randint(jax.random.fold_in(k, 1), (1,), 1, 10)
        return jnp.concatenate([low, high])

    a, b = digits(ka), digits(kb)

    def step(carry, x):
        s = x[0] + x[1] + carry
        return s // 10, s % 10

    carry, out = jax.lax.scan(step, 0, jnp.stack([a, b], axis=1))
    out = jnp.concatenate([out, carry[None]])                    # (11,) reversed digits
    n_out = jnp.where(carry > 0, DEC_L + 1, DEC_L)               # 10 or 11 digits

    prompt = jnp.concatenate([a, jnp.array([10]), b, jnp.array([11])])   # (22,)
    tail = jnp.concatenate([out + 20, jnp.array([30])])                  # (12,)
    tail_pos = jnp.arange(DEC_L + 2)
    tail = jnp.where(tail_pos < n_out, tail, 30)                 # EOS, then padding

    seq = jnp.concatenate([prompt, tail])
    pos = jnp.arange(DEC_SEQ)
    # scored positions: '=' at index 21 predicts the first output digit, through
    # to the position that predicts EOS.
    mask = (pos >= 2 * DEC_L + 1) & (pos <= 2 * DEC_L + 1 + n_out)
    return seq.astype(jnp.int32), mask


def make_decimal() -> Task:
    sample_batch = jax.jit(lambda key, batch: jax.vmap(_decimal_one)(
        jax.random.split(key, batch)), static_argnums=1)
    test = sample_batch(jax.random.key(TEST_SEED + 1), TEST_SIZE)

    return Task(
        name="decimal", vocab=31, n_ctx=40, seq_len=DEC_SEQ,
        sample=lambda key, batch: sample_batch(key, batch),
        test_set=lambda: test, chance=0.0,
        describe="10-digit + 10-digit, digits reversed; output tokens 20-29, EOS 30",
    )


# ── parenthesis balancing ─────────────────────────────────────────────────────
# '(' = 1, ')' = 2, '?' = 3; the answer is 2 for balanced and 1 for unbalanced.
# Vocabulary 4, so only four token embeddings are ever trained.
#
# The paper's generate-then-mutate recipe is recursive and does not vectorise
# under jit, so this pool is built once in NumPy and sampled from. With four
# token embeddings and eighty positional ones there is nothing to memorise, so
# a finite pool is not a shortcut — see concepts.md.

PAREN_MAX = 60
PAREN_SEQ = PAREN_MAX + 2                 # parens + '?' + label
PAREN_POOL = 500_000


def _uniform_dyck(rng, t, n):
    """``n`` uniform Dyck words of ``t`` pairs, by the cycle lemma.

    A uniformly random arrangement of t opens and t+1 closes has exactly one
    cyclic rotation whose first 2t steps never go below zero; rotating to just
    after the first minimum of the prefix sums finds it.
    """
    arr = np.concatenate([np.ones(t, np.int8), -np.ones(t + 1, np.int8)])
    arr = np.tile(arr, (n, 1))
    order = np.argsort(rng.random((n, 2 * t + 1)), axis=1)
    arr = np.take_along_axis(arr, order, axis=1)
    s = np.cumsum(arr, axis=1)
    j = np.argmin(s, axis=1)                       # first occurrence of the minimum
    cols = (np.arange(2 * t + 1)[None, :] + (j[:, None] + 1)) % (2 * t + 1)
    rot = np.take_along_axis(arr, cols, axis=1)
    return rot[:, : 2 * t]


def _is_balanced(seq, length):
    """Exact Dyck test on a (n, PAREN_MAX) array of 1/2 tokens with row lengths."""
    step = np.where(seq == 1, 1, -1)
    valid = np.arange(PAREN_MAX)[None, :] < length[:, None]
    step = np.where(valid, step, 0)
    run = np.cumsum(step, axis=1)
    never_negative = (np.where(valid, run, 0) >= 0).all(axis=1)
    return never_negative & (run[:, -1] == 0)


def _paren_pool(seed, n):
    rng = np.random.default_rng(seed)
    seq = np.zeros((n, PAREN_MAX), np.int8)
    length = np.zeros(n, np.int64)

    uniform = rng.random(n) < 1 / 3
    n_u = int(uniform.sum())
    # (a) uniformly random parentheses, length uniform in [1, 60]
    lu = rng.integers(1, PAREN_MAX + 1, n_u)
    body = rng.integers(1, 3, (n_u, PAREN_MAX)).astype(np.int8)
    seq[uniform] = np.where(np.arange(PAREN_MAX)[None, :] < lu[:, None], body, 0)
    length[uniform] = lu

    # (b) balanced sequences, t pairs with t uniform in [1, 30]
    idx_b = np.flatnonzero(~uniform)
    t_all = rng.integers(1, PAREN_MAX // 2 + 1, idx_b.size)
    for t in range(1, PAREN_MAX // 2 + 1):
        rows = idx_b[t_all == t]
        if rows.size == 0:
            continue
        d = _uniform_dyck(rng, t, rows.size)
        assert (np.cumsum(d, 1) >= 0).all() and (d.sum(1) == 0).all(), "cycle lemma broken"
        seq[rows, : 2 * t] = np.where(d > 0, 1, 2)
        length[rows] = 2 * t

    # Mutate: swap random index pairs, then flip random indices. The counts are
    # geometric(1/2), so a handful of vectorised rounds covers every row —
    # round r touches only the rows that drew more than r operations. MAX_OPS
    # truncates a tail of probability 2^-16.
    MAX_OPS = 16
    n_swap = np.where(rng.random(n) < 0.5, rng.geometric(0.5, n), 0)
    n_flip = np.where(rng.random(n) < 0.5, rng.geometric(0.5, n), 0)
    n_swap = np.minimum(n_swap, np.minimum(length, MAX_OPS))
    n_flip = np.minimum(n_flip, np.minimum(length, MAX_OPS))

    for r in range(MAX_OPS):
        rows = np.flatnonzero(n_swap > r)
        if rows.size == 0:
            break
        L = length[rows]
        i = (rng.random(rows.size) * L).astype(np.int64)
        j = (rng.random(rows.size) * L).astype(np.int64)
        vi, vj = seq[rows, i].copy(), seq[rows, j].copy()
        seq[rows, i], seq[rows, j] = vj, vi

    for r in range(MAX_OPS):
        rows = np.flatnonzero(n_flip > r)
        if rows.size == 0:
            break
        L = length[rows]
        i = (rng.random(rows.size) * L).astype(np.int64)
        seq[rows, i] = 3 - seq[rows, i]

    label = np.where(_is_balanced(seq, length), 2, 1).astype(np.int8)

    toks = np.zeros((n, PAREN_SEQ), np.int32)
    toks[:, :PAREN_MAX] = seq
    rows = np.arange(n)
    toks[rows, length] = 3                         # '?'
    toks[rows, length + 1] = label
    mask = np.zeros((n, PAREN_SEQ), bool)
    mask[rows, length] = True                      # predict the label at '?'
    return toks, mask, label


def make_parens(pool_size: int = PAREN_POOL) -> Task:
    toks, mask, label = _paren_pool(TEST_SEED + 2, pool_size + TEST_SIZE)
    test = (jnp.asarray(toks[:TEST_SIZE]), jnp.asarray(mask[:TEST_SIZE]))
    # The training pool is held as int8 and its mask recomputed from the '?'
    # token rather than stored: at 500k rows an int32 pool plus a bool mask
    # would be a ~150 MB constant captured inside every jitted step.
    tr_t = jnp.asarray(toks[TEST_SIZE:].astype(np.int8))
    # majority-class rate on the heldout set is the honest floor here
    maj = max(np.mean(label[:TEST_SIZE] == 2), np.mean(label[:TEST_SIZE] == 1))

    def sample(key, batch):
        idx = jax.random.randint(key, (batch,), 0, tr_t.shape[0])
        t = tr_t[idx].astype(jnp.int32)
        return t, t == 3

    return Task(
        name="parens", vocab=4, n_ctx=80, seq_len=PAREN_SEQ,
        sample=sample, test_set=lambda: test, chance=float(maj),
        describe=f"generate-then-mutate Dyck, length <= {PAREN_MAX}; pool of {pool_size}",
    )


# ── memorization ──────────────────────────────────────────────────────────────
# Every (x, y) with x in [0,511], y in [512,1023] gets an independent uniform
# label z in [0,511]. 512^2 = 262144 associations, 9 bits each. No test split:
# the whole point is how much fits in the trainable parameters.

MEM_N = 512


def make_memorization() -> Task:
    rng = np.random.default_rng(TEST_SEED + 3)
    x, y = np.divmod(np.arange(MEM_N * MEM_N), MEM_N)
    z = rng.integers(0, MEM_N, MEM_N * MEM_N)
    toks = jnp.asarray(np.stack([x, y + MEM_N, z], axis=1).astype(np.int32))
    mask = jnp.broadcast_to(jnp.array([0, 1, 0], dtype=bool), toks.shape)

    def sample(key, batch):
        idx = jax.random.randint(key, (batch,), 0, toks.shape[0])
        return toks[idx], mask[idx]

    return Task(
        name="memorization", vocab=2 * MEM_N, n_ctx=5, seq_len=3,
        sample=sample, test_set=lambda: (toks, mask), chance=1.0 / MEM_N,
        describe=f"{MEM_N * MEM_N} random (x, y) -> z associations, 9 bits each",
    )


TASKS = {
    "mod_add": make_mod_add,
    "needle": make_needle,
    "decimal": make_decimal,
    "parens": make_parens,
    "memorization": make_memorization,
}


def get_task(name: str) -> Task:
    return TASKS[name]()
