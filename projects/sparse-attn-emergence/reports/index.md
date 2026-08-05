# Emergent capabilities from sparse attention

A small-scale replication of **[Emergent Capabilities Arise Randomly from Learning
Sparse Attention Patterns](https://arxiv.org/abs/2606.25010)** (Baherwani, Chen, Qiu,
Wilson, Izmailov — NYU, June 2026). Synthetic half only; the Pythia / IOI half is out
of scope.

## The claim

Transformer capabilities appear abruptly, at training steps that vary wildly between
otherwise identical runs, while pretraining loss falls smoothly. The paper's mechanistic
explanation: each capability needs **one sparse, task-relevant attention pattern**, and
finding it by SGD is a search. Until the search succeeds the model sits at the
marginal-entropy loss; when it succeeds, loss falls in a few hundred steps.

Synthetic tasks make this testable rather than anecdotal: the correct attention pattern
is known **by construction**, so "did the capability emerge" and "did a head find the
right support" are both directly measurable.

*New to this? [The paper in plain terms](sparse_attn_emergence_paper.html) explains the whole
setup with no background assumed.*

## Status

| | Claim | Verdict |
|---|---|---|
| **H1** | Emergence is abrupt, and its timing is seed-random | [**partially replicated**](sparse_attn_emergence_exp1.html) — timing yes (twice over, see [exp4](sparse_attn_emergence_exp4.html)), cliff-like abruptness no |
| **H2** | Difficulty is non-monotone in sparsity and grows with context length | [**partially replicated**](sparse_attn_emergence_exp2.html) — the unlearnable band and its widening yes; the dense-end "recovery" is a task artifact |
| **H3** | The loss jump *is* the attention pattern being found | [**supported**](sparse_attn_emergence_exp4.html) — causal ablation, 0.00 → 4.23 nats |
| **H4** | More heads help; head dimension saturates | [**replicated**](sparse_attn_emergence_exp3.html) — on the strict metric |
| **H5** | A non-attention mixer learns the linear map faster | [**direction only**](sparse_attn_emergence_exp67.html) — wins where attention fails, loses elsewhere |
| — | Not specific to the linear map | [**holds**](sparse_attn_emergence_exp5.html) — same wall on cellular automata |

**All seven experiments are complete.** The whole thing in one place, with diagrams:
[**Findings**](sparse_attn_emergence_findings.html). The errors made along the way, and how
each was caught: [**Mistakes**](sparse_attn_emergence_mistakes.html).

## Two results worth your attention

**The mechanism claim survives a causal test.** Zeroing the one head whose attention matches
the target matrix takes second-half loss from 0.0000 to **4.23 nats** — six times the
no-knowledge plateau, i.e. confidently wrong — while zeroing the least-aligned head costs
0.08. Details in [exp4](sparse_attn_emergence_exp4.html).

**Difficulty is quantitative, and one of the paper's cells is an artifact.** Learnability
tracks `C(S,s)`, the number of candidate supports per row: reliably solved below ~500,
never above ~8,000, at both context lengths. But the apparent recovery at maximum density
is a **copying shortcut**, not a solve — verified per-position. See
[exp2](sparse_attn_emergence_exp2.html).

## Headline result so far

At fixed difficulty — same matrix `A`, same hyperparameters, same token budget, 16 seeds
differing only in initialisation and data order — time-to-emergence ranged over

> **469 to 2521 steps: a 5.4× spread.**

At the paper's 3 seeds this is invisible. `[469, 563, 566]` and `[1196, 2187, 2521]` are
both plausible 3-seed draws from our 16, and they tell opposite stories about when a
capability "should" appear. The stochasticity is the finding, so the seed count is not a
detail.

## Read in order

0. **[The paper in plain terms](sparse_attn_emergence_paper.html)** — start here if you
   haven't read the paper. What emergence is, why sparse attention would explain it, and what
   the authors did. No prior background assumed.
1. **[Methods](sparse_attn_emergence_methods.html)** — the task construction, the model,
   and exactly what each metric means. Worth reading before the numbers.
2. **[exp1](sparse_attn_emergence_exp1.html)** — abruptness and seed-randomness (H1).
3. **exp2** — the difficulty surface over context length × sparsity (H2).
4. **exp3, exp4, exp5, exp6** — heads vs head dim, the mechanism plus a causal ablation,
   cellular automata, and the architecture comparison.
5. **Verdict** — claim-by-claim, with deviations stated.

## How this was run

One layer, `D=128`, ~200k parameters, 32-token sequences. Every seed of a configuration
trains **simultaneously** under a single `jax.vmap` over a leading parameter axis, which
is what makes 16-seed statistics affordable: the full 16-seed, 10,000-step exp1 run took
**167 seconds** on one RTX 4090. Sixteen seeds instead of the paper's three is a
deliberate choice — H1 and H2 are claims about a *distribution* over seeds, and a mean
curve hides them.

Everything is reproducible from `projects/sparse-attn-emergence/` in the
[seewhy](https://github.com/cwhy/seewhy) repo: one file per experiment, one
`results.jsonl` row per configuration carrying hyperparameters and per-seed curves, and
these pages generated from committed markdown.
