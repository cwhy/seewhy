# ICL-AR — Formulation

*Abandoned mid-design, 2026-08-11. Branch preserved so the formulation isn't lost;
the code is a partial scaffold and is mostly superseded by what's written here.
See "State of the code" at the bottom before touching anything.*

---

## The question

How do autoregressive models perform on in-context learning — and what does
"in-context learning" even denote, precisely enough to put a number on?

The intended deliverable was a report organised as a **task × architecture
grid**. The formulation below is what that grid should have been built on. It
arrived late in the design discussion and replaced an earlier, weaker setup
(supervised in-context regression, Garg et al. style); the earlier setup is a
special case of it.

## The setup

A prior `π` over tasks `θ`. Each `θ` names a distribution `p(·|θ)` over `X`.
One episode:

```
θ ~ π,    x₁, …, x_N  ~iid  p(·|θ)
```

The model never sees `θ`. The sequence is not iid — it is iid *conditional on
θ*, i.e. exchangeable. Its marginal law is

```
m(x₁…x_N) = ∫ ∏ᵢ p(xᵢ|θ) π(dθ)
```

Train an autoregressive `q_φ` with ordinary next-token loss on sequences drawn
this way.

## The central fact

```
argmin_q  E[ −log q(xₙ | x_<ₙ) ]  =  m(xₙ | x_<ₙ)  =  ∫ p(xₙ|θ) · π(dθ | x_<ₙ)
```

The loss-minimising AR model **is** the Bayesian posterior predictive. Not
approximately, not as an emergent surprise — it is the pointwise minimiser of
the training objective. In-context learning is not a capability that appears on
top of autoregressive training on exchangeable data; it is what that training
optimises for. Everything empirical then reduces to: how close does a given
architecture actually get, and what stops it?

The informal intuition this came from — "learn a generic P(X), then adapt to
this episode's P(Xᵢ)" — is exactly the two endpoints of the posterior
predictive:

| n | the optimal predictor is | |
|---|---|---|
| 0 | `∫ p(x\|θ) π(dθ)` | the prior predictive — the "generic P(X)" |
| → ∞ | `p(·\|θ_true)` | this episode's own distribution |

**In-context learning is the transit between those two.** That is the
definition this project would have used.

Two structural notes:

- By **de Finetti**, every infinitely exchangeable sequence is a mixture of iid.
  So the latent-task form is not an assumption being imposed — it is the general
  shape of any exchangeable autoregressive problem.
- The supervised setting is the restriction `X = (u, v)` where only the `v`
  tokens are scored. Regression, classification, and TabPFN are all instances.
  TabPFN specifically is this with a structured prior over tabular causal models,
  scoring only the label — so "start simple, work toward TabPFN" is a path of
  progressively enriching `π`, with no change to the machinery.

## Why this formulation: everything is measurable in nats

Because `xₙ ⊥ x_<ₙ | θ`, the per-step entropy splits:

```
H(xₙ | x_<ₙ)  =  H(X|θ)  +  I(θ ; xₙ | x_<ₙ)
                 └──┬──┘     └──────┬──────┘
              irreducible      the only part
                  noise       context can reduce
```

Write `gₙ = I(θ ; xₙ | x_<ₙ)` and let `Δₙ = E[ KL( m(·|x_<ₙ) ‖ q_φ(·|x_<ₙ) ) ]`
be the model's excess loss. Three consequences, and they are the reason to
prefer this framing over anything measured in raw MSE:

**1. An exact optimum and an exact zero.** `Δₙ ≥ 0`, attainable. So

```
ICL efficiency at n  =  (gₙ − Δₙ) / gₙ  ∈ [0, 1]
```

`0` = ignores the context entirely; `1` = exactly Bayes. Unitful, calibrated,
and comparable across tasks *and* architectures — which is what a task ×
architecture grid needs and almost never has.

**2. The area under the ICL curve is known in advance.**
`Σₙ gₙ = I(θ ; x₁…x_N)` — the total task information a context of length `N`
can possibly convey. A hard ceiling, computable before any training run.

**3. Theory predicts the curve's shape.** For a regular `d`-parameter family,
Clarke–Barron gives `I(θ; x_{1:N}) = (d/2)·log N + O(1)`, hence

```
gₙ ≈ d / (2n)
```

so `log gₙ` against `log n` is a line of **slope −1, intercept log(d/2)**. A
trained model's context curve can be checked against that slope, and the
effective `d` it behaves as though it inferred can be read off the intercept.

**Practical corollary, and the reason published ICL curves are so often
unreadable:** `H(X|θ)` is usually most of the loss. Raw loss curves are
dominated by irreducible noise and look nearly flat. Plot excess-over-Bayes,
never raw loss.

## The architecture axis, made sharp

Bayes-optimality depends on `x_<ₙ` *only through the posterior* `π(·|x_<ₙ)`. So
"what must a model's hidden state hold?" is literally "how many numbers describe
the posterior?"

- For an **exponential family**, the posterior is pinned by a fixed-dimensional
  sufficient statistic `(n, Σ T(xᵢ))`. A recurrent model with fixed state *can*
  be exactly Bayes-optimal at every `N`.
- **Koopman–Pitman–Darmois**: under regularity, exponential families are the
  only ones whose sufficient statistic has dimension bounded in `n`. Cauchy
  location is the textbook counterexample — all `n` points are needed.

Prediction: on Gaussian-location episodes, GRU ≈ transformer. On Cauchy-location
episodes, the GRU's excess KL grows with `n` and the transformer's does not.
Falsifiable, theory-derived, with an exactly computable optimum.

**Caveat that improves the design.** Put `θ` on a finite grid and the posterior
is always finite-dimensional, so KPD does not strictly bite. The honest
reframing is quantitative: **effective rank of the log-posterior**, measured by
PCA over episode trajectories.

| family | posterior determined by | predicted effective rank |
|---|---|---|
| Gaussian location | `(n, Σx)` | ~3 (span of 1, θ, θ²) |
| categorical / Dirichlet | bin counts | ~K |
| Cauchy location | all xᵢ | high; grows with grid resolution |

Then sweep recurrent state size against measured rank and look for the predicted
threshold. Stronger than the binary claim, and it survives discretisation.

## The simplest thing that would have run

Put `X` on a discrete grid (~128 bins) and define the generative process
**directly on the grid**. Then:

- the model is a tiny AR LM over a 128-token vocabulary;
- the loss is cross-entropy in nats;
- the posterior predictive is *exact* — a cumulative log-likelihood gather plus
  a softmax-weighted mixture, one einsum, no quadrature error to defend;
- every number in the report is a true KL.

Family ladder as the task axis: Gaussian location → categorical/Dirichlet →
Cauchy location. Architecture axis: softmax attention, linear attention, GRU,
static causal positional mixer — same depth/width/norm/optimiser, differing in
exactly one operator (see `lib/models.py`, which is written and reusable).

## Known risks

- Density modelling has no deterministic-`y` shortcut, so it is genuinely harder
  to fit than in-context regression. 1-D discrete should be tractable, but this
  is the thing most likely to bite first.
- Everything above assumes exchangeability. The AR model sees one *ordering*,
  and nothing forces it to learn the order-invariance that the target has.
  Measurable, and possibly a finding in its own right.
- The `d/(2n)` law is asymptotic and regular-family. Small `n` and the
  non-regular families will deviate; the deviation needs to be reported rather
  than smoothed over.

## State of the code

Scaffolded from `projects/TEMPLATES/project_start.md` (v2). Nothing has ever
been executed — there is no `.venv` on the Mac and nothing was pushed to the
GPU box. No `concepts.md`, no experiments, no report tree, `results.jsonl` empty.

| file | status |
|---|---|
| `lib/models.py` | **usable as-is.** Four architectures differing in one operator; arch-agnostic `Config` / `init_params` / `forward`. Independent of the formulation change. |
| `lib/encoding.py` | superseded. Interleaved `x y x y …` layout for *supervised* episodes. The density formulation needs a plain token sequence instead; keep only if the supervised restriction is revisited. |
| `lib/tasks.py` | superseded. Function-class priors for in-context regression. The replacement is a prior over grid distributions plus an exact posterior-predictive routine. |
| `lib/references.py` | superseded. Ridge / lasso / kNN baselines. Under the density formulation the single reference is the exact posterior predictive. |
| `workflow.md`, `scripts/*` | template copies, placeholders substituted, otherwise untouched. |
| `report/` | empty directories only (untracked by git). |

Resume by writing the grid prior + exact posterior predictive module, then
pointing `lib/models.py` at it. The architecture code and the workflow scaffold
are the parts worth keeping.
