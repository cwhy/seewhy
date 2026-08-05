# Task setup, drawn out

Both synthetic tasks in the paper are next-token prediction problems built so that **the
attention pattern needed to solve them is known in advance**. This page shows how each token
is produced and what the prediction problem looks like on top of it — every step, with real
numbers.

Nothing here is a result. [Methods](sparse_attn_emergence_methods.html) has the metric
definitions and the deviations; [Findings](sparse_attn_emergence_findings.html) has the
outcomes.

---

## Task 1 — the linear map

One secret matrix, held fixed for a whole training run. Each output bit is the XOR of a few
specific input bits, and *which* few is the pattern the model has to discover.

### The matrix

![sampling A](https://media.tanh.xyz/seewhy/26-08-05/sparse_attn_emergence_task_lm1_matrix.svg)

### One token

![producing one token](https://media.tanh.xyz/seewhy/26-08-05/sparse_attn_emergence_task_lm2_token.svg)

Every token in the second half is produced this way: pick the `s` positions where the
corresponding row of `A` is 1, XOR those input bits, emit the result. No other input bit
affects it — which is exactly why an attention head that reads the wrong positions cannot
compensate with a cleverer MLP.

### The sequence

![the sequence](https://media.tanh.xyz/seewhy/26-08-05/sparse_attn_emergence_task_lm3_sequence.svg)

Note the asymmetry this creates. The first half is uniform noise, so no model can do better
than chance on it, and its cross-entropy is pinned at `ln 2` forever. Only the second half
carries signal — which is why every metric on this site is second-half-only, and why a
full-sequence loss (what the paper plots) has a floor of `(S−1)/ST · ln C` rather than zero.

### The pattern to be found

![ground truth attention](https://media.tanh.xyz/seewhy/26-08-05/sparse_attn_emergence_task_lm4_attention.svg)

This is the object the whole paper is about. It is a fixed, sparse, query-dependent set of
key positions — known to us, unknown to the model, and reachable only by search.

---

## Task 2 — cellular automata

Same idea, different shape, and one structural difference that changes what is being tested:
the rule is **not** fixed per run, so the model has to identify it from the sequence.

### The rule

![the rule table](https://media.tanh.xyz/seewhy/26-08-05/sparse_attn_emergence_task_ca1_rule.svg)

### One transition

![one transition](https://media.tanh.xyz/seewhy/26-08-05/sparse_attn_emergence_task_ca2_transition.svg)

### The sequence

![the trajectory](https://media.tanh.xyz/seewhy/26-08-05/sparse_attn_emergence_task_ca3_sequence.svg)

The consequence of drawing a fresh rule per sequence is that **early tokens are ambiguous by
design, not merely unlearned**. With 256 candidate rules, state 1 is close to unpredictable
from state 0 alone; by the last state the model has seen enough transitions to pin the rule
down. So the loss profile *within* a sequence is itself the measurement, and the headline
number is the final state.

---

## The two together

![the two tasks compared](https://media.tanh.xyz/seewhy/26-08-05/sparse_attn_emergence_task_ca4_compare.svg)

The pair is well chosen. If only the linear map showed the effect, the obvious objection would
be that XOR-of-a-subset is a peculiar function and the difficulty is algebraic. The cellular
automaton shares almost nothing with it — different vocabulary, different sequence length, a
rule that changes every example, four layers instead of one — but the same wall appears when
the required pattern widens: [exp5](sparse_attn_emergence_exp5.html) solves 4/8 at span 3 and
0/8 at span 5.

## Reproducing the data

Both generators are about twenty lines, in `lib/tasks.py`:

```python
A = linear_map_matrix(key, S, s)        # (S, S) int32, exactly s ones per row
seq = linear_map_batch(key, A, batch)   # (batch, 2S) — concat(x0, A x0 mod 2)

rules = ca_rule_pool(key, 256, C=4, W=3)          # (256, 64) lookup tables
seq = ca_batch(key, rules, batch, S, T, k, C=4)   # (batch, S*T)
```

Samples are drawn fresh at every training step rather than from a fixed dataset, so training
loss is an unbiased estimate of test loss and there is no train/test gap to reason about.
