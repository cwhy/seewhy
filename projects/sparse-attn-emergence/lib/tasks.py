"""
Synthetic task generators.

Linear map (paper §3.1): A in {0,1}^{SxS} with exactly s ones per row, transition
f(x) = Ax mod 2. A sequence is concat(x0, x1) of S*T tokens with T=2, vocab C=2.

The point of the construction: predicting token S+i requires attending to exactly
the s positions where row i of A is 1. The ground-truth attention support is known
by construction, so "did the model find the pattern" is directly measurable
(see support_iou in lib/models.py callers).

The first half of every sequence is i.i.d. uniform, so its CE is exactly ln 2 and
carries no signal — all metrics use the second half only.
"""

import jax
import jax.numpy as jnp


def linear_map_matrix(key, S: int, s: int) -> jnp.ndarray:
    """(S, S) int32 matrix with exactly s ones per row, columns chosen uniformly."""
    # argsort of uniform noise gives an independent random permutation per row;
    # taking the first s columns picks s distinct positions without replacement.
    idx = jnp.argsort(jax.random.uniform(key, (S, S)), axis=1)[:, :s]
    return jnp.zeros((S, S), jnp.int32).at[jnp.arange(S)[:, None], idx].set(1)


def linear_map_batch(key, A: jnp.ndarray, batch: int) -> jnp.ndarray:
    """(batch, 2S) int32 sequences concat(x0, A x0 mod 2), x0 ~ U{0,1}^S."""
    S = A.shape[0]
    x0 = jax.random.bernoulli(key, 0.5, (batch, S)).astype(jnp.int32)
    x1 = (x0 @ A.T) % 2
    return jnp.concatenate([x0, x1], axis=1)


def linear_map_traj_batch(key, A: jnp.ndarray, batch: int, T: int) -> jnp.ndarray:
    """(batch, S*T) — T states of a trajectory x_{t+1} = A x_t mod 2, flattened.

    T=2 reduces to linear_map_batch. Larger T puts several applications of the SAME A in
    one sequence, i.e. more worked examples of the map per sequence. The paper fixes T=2
    for the linear map ("We always use C=2 and T=2") and only varies trajectory length on
    the cellular automata task, so this axis is untested for the linear map.
    """
    S = A.shape[0]
    x0 = jax.random.bernoulli(key, 0.5, (batch, S)).astype(jnp.int32)

    def step(x, _):
        nx = (x @ A.T) % 2
        return nx, nx

    _, rest = jax.lax.scan(step, x0, None, length=T - 1)          # (T-1, batch, S)
    return jnp.concatenate([x0[None], rest], 0).transpose(1, 0, 2).reshape(batch, S * T)


def induction_batch(key, batch: int, n_pairs: int, V: int):
    """Associative recall / induction. Returns (seq (batch, 2*n_pairs), recallable mask).

    A sequence is pairs `a, f(a), a', f(a'), ...` where **f is a fresh random permutation
    per sequence**. When a key repeats, its value is recoverable only by matching the
    earlier occurrence of that key and copying the token after it — the canonical induction
    circuit.

    This is the axis both of the paper's synthetic tasks miss. There, the correct attention
    pattern is a fixed set of POSITIONS (row support, or a local window), so it can be
    expressed by position information alone. Here the correct position depends on the
    CONTENT and moves from sequence to sequence, which is what IOI, copying and in-context
    repetition actually require.

    `recallable[b, i]` marks pairs whose key already appeared, i.e. the positions where the
    answer is determined rather than a 1/V guess.
    """
    k_perm, k_keys = jax.random.split(key)
    f = jax.vmap(lambda k: jax.random.permutation(k, V))(jax.random.split(k_perm, batch))
    a = jax.random.randint(k_keys, (batch, n_pairs), 0, V)
    fa = jnp.take_along_axis(f, a, axis=1)
    seq = jnp.stack([a, fa], axis=-1).reshape(batch, 2 * n_pairs)

    same = (a[:, :, None] == a[:, None, :])
    earlier = jnp.tril(jnp.ones((n_pairs, n_pairs)), -1)[None]
    return seq, (same * earlier).sum(-1) > 0


def kofm_subset(key, m: int, k: int) -> jnp.ndarray:
    """(k,) sorted attribute indices — which k of m attributes the match depends on."""
    return jnp.sort(jnp.argsort(jax.random.uniform(key, (m,)))[:k])


def kofm_recall_batch(key, batch: int, n_blocks: int, m: int, k: int, A: int, V: int,
                      subset: jnp.ndarray):
    """k-of-m associative recall: a content-keyed task with a tunable candidate space.

    Each block is `m` attribute tokens followed by one value token. The value of a query
    block equals the value of the earlier context block that agrees with it on the `k`
    RELEVANT attributes; the other `m-k` attributes are re-randomised, so matching on the
    wrong subset retrieves the wrong block.

    Two things have to be learned, and they are exactly the two halves this project has
    been separating:

      * WHICH k of m attributes matter — a subset choice out of C(m, k), fixed for the run
        and learned into the weights, the direct analogue of the linear map's row support
      * the match itself — content-keyed, recomputed per sequence, the induction half

    So difficulty can be plotted against C(m, k) exactly as the linear map is plotted
    against C(S, s), which is what puts the position-keyed and content-keyed families on
    one axis.

    Tokens: attributes are 0..A-1, values are A..A+V-1. Returns (seq, value_positions,
    is_query) so the caller can score only the determined positions.
    """
    k_attr, k_val, k_src = jax.random.split(key, 3)
    n_ctx = n_blocks // 2
    attrs = jax.random.randint(k_attr, (batch, n_blocks, m), 0, A)
    values = jax.random.randint(k_val, (batch, n_blocks), 0, V)

    # every query block copies the relevant attributes, and the value, of one context block
    src = jax.random.randint(k_src, (batch, n_blocks - n_ctx), 0, n_ctx)
    ctx_attrs = jnp.take_along_axis(attrs[:, :n_ctx], src[:, :, None], axis=1)   # (b, nq, m)
    q_attrs = attrs[:, n_ctx:].at[:, :, subset].set(ctx_attrs[:, :, subset])
    attrs = jnp.concatenate([attrs[:, :n_ctx], q_attrs], axis=1)
    values = jnp.concatenate(
        [values[:, :n_ctx], jnp.take_along_axis(values[:, :n_ctx], src, axis=1)], axis=1)

    tokens = jnp.concatenate([attrs, (values + A)[:, :, None]], axis=2)
    seq = tokens.reshape(batch, n_blocks * (m + 1))
    value_pos = jnp.arange(n_blocks) * (m + 1) + m
    is_query = jnp.arange(n_blocks) >= n_ctx
    return seq, value_pos, is_query


def kofm_recall_unique(key, batch: int, n_blocks: int, m: int, k: int, A: int, V: int,
                       subset: jnp.ndarray):
    """k-of-m recall with a UNIQUELY identified match at every k.

    The plain version samples context attributes independently, so a non-source block also
    matches the query on the relevant subset with probability A^-k. At k=1, A=4 and four
    context blocks that is 0.75 expected spurious matches — the answer is genuinely
    ambiguous and no model can be right. Difficulty then falls with k mostly because the
    ambiguity does, which has nothing to do with finding the pattern.

    Here each context block gets a DISTINCT tuple on the relevant attributes (drawn without
    replacement from the A^k grid), so exactly one block ever matches. Difficulty across k
    then reflects the search, not the well-posedness of the retrieval.
    """
    k_code, k_rest, k_val, k_src = jax.random.split(key, 4)
    n_ctx = n_blocks // 2
    grid = A**k

    # distinct codes for the context blocks, decoded into base-A digits
    codes = jax.vmap(lambda kk: jax.random.permutation(kk, grid)[:n_ctx])(
        jax.random.split(k_code, batch))                                   # (b, n_ctx)
    digits = jnp.stack([(codes // (A**j)) % A for j in range(k)], axis=-1)  # (b, n_ctx, k)

    attrs = jax.random.randint(k_rest, (batch, n_blocks, m), 0, A)
    attrs = attrs.at[:, :n_ctx, subset].set(digits)
    values = jax.random.randint(k_val, (batch, n_blocks), 0, V)

    src = jax.random.randint(k_src, (batch, n_blocks - n_ctx), 0, n_ctx)
    ctx_attrs = jnp.take_along_axis(attrs[:, :n_ctx], src[:, :, None], axis=1)
    q_attrs = attrs[:, n_ctx:].at[:, :, subset].set(ctx_attrs[:, :, subset])
    attrs = jnp.concatenate([attrs[:, :n_ctx], q_attrs], axis=1)
    values = jnp.concatenate(
        [values[:, :n_ctx], jnp.take_along_axis(values[:, :n_ctx], src, axis=1)], axis=1)

    tokens = jnp.concatenate([attrs, (values + A)[:, :, None]], axis=2)
    return (tokens.reshape(batch, n_blocks * (m + 1)),
            jnp.arange(n_blocks) * (m + 1) + m,
            jnp.arange(n_blocks) >= n_ctx)


def ca_rule_pool(key, n_rules: int, C: int = 4, W: int = 3) -> jnp.ndarray:
    """(n_rules, C**W) int32 lookup tables. Sampled once per run; one rule per example.

    Per the paper's appendix: "N: Number of rules; one rule is sampled per training
    example". So unlike the linear map's single fixed A, the active rule changes every
    sequence — the model has to infer it IN CONTEXT before it can predict.
    """
    return jax.random.randint(key, (n_rules, C**W), 0, C)


def _ca_rollout(R, x0, T: int, k: int, C: int) -> jnp.ndarray:
    """Roll a per-sequence lookup table forward: T states, wrapped boundaries, flattened.

    k is the composition depth — the rule is applied k times per state transition, so the
    span of x_{t+1}[i] over x_t is 2k+1 wide. k is a Python int (static).
    """
    def apply_once(x):
        idx = jnp.roll(x, 1, axis=1) * C * C + x * C + jnp.roll(x, -1, axis=1)
        return jnp.take_along_axis(R, idx, axis=1)

    def step(x, _):
        for _ in range(k):
            x = apply_once(x)
        return x, x

    _, rest = jax.lax.scan(step, x0, None, length=T - 1)                  # (T-1, batch, S)
    batch, S = x0.shape
    return jnp.concatenate([x0[None], rest], 0).transpose(1, 0, 2).reshape(batch, S * T)


def ca_batch(key, rules: jnp.ndarray, batch: int, S: int, T: int, k: int,
             C: int = 4) -> jnp.ndarray:
    """(batch, S*T) int32 — one rule drawn per sequence from a POOL fixed for the run.

    The pool is what makes this memorisable: with N tables drawn once per run, a model can
    store all of them and only has to infer WHICH is active from the context.
    """
    k_rule, k_state = jax.random.split(key)
    R = rules[jax.random.randint(k_rule, (batch,), 0, rules.shape[0])]    # (batch, C**W)
    x0 = jax.random.randint(k_state, (batch, S), 0, C)
    return _ca_rollout(R, x0, T, k, C)


def ca_fresh_batch(key, batch: int, S: int, T: int, k: int, C: int = 4,
                   W: int = 3) -> jnp.ndarray:
    """(batch, S*T) int32 — a FRESH lookup table per sequence, drawn from all C^(C^W).

    No pool, so nothing can be memorised: the rule in play has almost certainly never been
    seen before. This is the genuine in-context version of the task, and the control that
    separates "learned the rules" from "learned to infer a rule".
    """
    k_rule, k_state = jax.random.split(key)
    R = jax.random.randint(k_rule, (batch, C**W), 0, C)
    x0 = jax.random.randint(k_state, (batch, S), 0, C)
    return _ca_rollout(R, x0, T, k, C)
