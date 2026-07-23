# ROSA — Concepts

This document covers the **architecture, training signal, and key
hyperparameters** of the ROSA-MNIST pipeline. For the experiment roadmap, see
[plan.md](plan.md).

---

## High-level idea

A non-differentiable statistical next-token predictor (ROSA, a suffix-automaton
language model from [bcml-labs/rosa-plus](https://github.com/bcml-labs/rosa-plus))
sits in the middle of an otherwise neural pipeline. The neural input module
discretizes images and labels into tokens; ROSA accumulates the full token
stream across all training data and answers "what comes next?" queries. Its
predicted next-token is fed back as a feature for a neural output module that
performs the actual classification.

The pipeline is **continual by construction**: ROSA never resets, never
forgets — every token of every training sample is committed to its SAM. The
neural modules are trained by gradient descent (with STE around the
discretization), making the system jointly differentiable end-to-end except
for ROSA itself, which is treated as a stop-gradient feature provider.

---

## The token stream

The pipeline operates on a single global stream of integer tokens. Each MNIST
sample contributes a burst:

```
... <SAMPLE> s_1 s_2 ... s_K  <LABEL> l_1 l_2 ... l_K' ...
       ▲       ▲                ▲       ▲
       |       |                |       |
   indicator   image            indicator label
   token       tokens           token     tokens
   (ID = 0)    (variable K via  (ID = 1)  (variable K' via
               ACT, ≤ K_max)              ACT, ≤ K'_max)
```

**Vocabulary.** A single fixed vocabulary `V` shared across sample and label
tokens. Two indicator IDs reserved (`<SAMPLE>=0`, `<LABEL>=1`); the remaining
`V−2` IDs are "content" tokens chosen via codebook nearest-neighbor.

**Headline sizes:** `V=1024`, token embedding dim `D=64`, sample
`K_max=16`, label `K'_max=4`.

**Continuity.** During training, every emitted sample-token and every emitted
label-token is appended to ROSA's master stream. ROSA's SAM grows
monotonically across all epochs; example boundaries (`<SAMPLE>` indicators)
are marked so the fallback LM does not interpolate Witten–Bell counts across
sample boundaries.

---

## Modules

### Input module — shared recurrent MLP (fixed K in exp1)

A single recurrent MLP block converts either an image or a label into a
**fixed-length** burst of tokens. exp1 uses fixed K=16 for sample and K=4
for label (no ACT halt head). exp1b adds the UTM-style halt head and
variable burst length on top.

```
init state h_0:
    h_0 = proj_image(x)        if x is an image (28×28 → flatten 784)
    h_0 = proj_label(y)        if y is a one-hot label (10 → D_h)

for k in 1..K (fixed):
    h_k = MLP(h_{k-1})          # shared block
    z_k = head_emit(h_k)         # -> R^D, the token's continuous embedding
```

Outputs: `z[1..K]` — `K` continuous embeddings, all alive.

The shared block is **the only place trainable weights process per-token
state**. Two differences between sample and label paths:
- Different input projections (`proj_image` vs `proj_label`)
- Different `K` (16 vs. 4)

**ACT extension (exp1b only).** Add a `head_halt(h_k) → R` halting logit
per step; cumulative halt probability decides the live `K_actual` per
sample. ROSA receives only `z[1..K_actual]`. At training time, run all
K_max steps but mask outputs by halt-prob (UTM-style); at inference,
`while_loop` with hard halt. Halt-bias init = -3 (deep-start). Default
`λ_ponder = 0` for exp1b; turn on for exp9.

### Discretization — DiVeQ reparameterization (headline)

Each continuous emission `z_k` is mapped to a vocab ID by nearest neighbor
against the learnable codebook `C ∈ R^{V × D}`. Instead of the standard
straight-through estimator, we use the **DiVeQ** reparameterization
validated in `projects/diveq/`:

```
# Find nearest entry
dists  = ||z_k - C||²           # for all V codebook entries
v_k    = argmin_v dists         # hard assignment, ROSA gets v_k
c_star = C[v_k]

# Reparameterized output (training)
d_tilde   = stop_grad(c_star - z_k)            # displacement, no grad
v         = Gaussian(0, SIGMA² I)              # noise
direction = stop_grad((v + d_tilde) / ||v + d_tilde||)
magnitude = ||c_star - z_k||                   # scalar with gradient via z_k
c_k       = z_k + magnitude · direction         # ≈ c_star (perturbed by SIGMA)

# Eval
c_k = c_star
```

Forward output `c_k ≈ c_star` (with a small noise perturbation); backward,
gradient flows through `magnitude = ||c_star - z_k||`, which IS the
commitment-loss gradient direction baked into the reparameterization.

**Why DiVeQ over plain STE.** No separate `λ_commit` term needed — the
reparam puts the commitment gradient directly in the forward pass.
Empirically lower codebook-collapse rate (validated in diveq exp35).
One hyperparameter: `SIGMA = 0.01` (diveq-tested).

The codebook itself is trained with a small `λ_codebook · ||stop_grad(z_k) - C[v_k]||²`
to keep entries near their assigned cluster, identical to VQ-VAE.

Indicator tokens `<SAMPLE>` and `<LABEL>` are *fixed* codebook entries (not
trained) at IDs 0 and 1.

**Alternatives kept on the shelf** (single-flag swaps in the experiment file):

- **Plain VQ-STE.** `c_k = c_star + stop_grad(z_k - c_star)` (identity
  backward) plus explicit commitment loss `λ_commit · ||z_k - sg(c_star)||²`.
  More biased gradient, no exploration, but the simplest baseline. Falls
  back to here if DiVeQ has issues we can't diagnose.
- **Gumbel-softmax-ST.** Logits over V instead of nearest-neighbor; hard
  sample on forward, soft on backward; temperature τ annealed. Adds
  stochasticity at the cost of a tuning schedule. Worth a one-shot
  ablation in Phase 2.

### ROSA core

ROSA = ROSACharPredictor (suffix-automaton next-token predictor) + ROSAFallbackLM
(Witten–Bell smoothed n-gram). At each query position, ROSA:

1. Walks the SAM with the recent token suffix.
2. If a deterministic next-token rule fires (rightmost-occurrence in training),
   return that ID.
3. Otherwise sample / argmax from the fallback Witten–Bell distribution.

**Implementation.** Port `rosaplus.py` to `lib/rosa_core.py` with these
adaptations:
- Tokens are **integer IDs** (not characters). Replace the `char->state`
  transition dicts with `int->state` dicts (still Python dicts; the SAM is
  vocab-agnostic).
- Add a `predict_batch(token_ids: List[List[int]])` method that, given a list
  of partial streams, returns a list of next-token argmaxes. Vectorizing
  inside Python is fine — the cost is dominated by the SAM walk.
- Add a `commit_batch(token_ids: List[List[int]])` method that appends each
  burst to the master stream, marking a boundary at the end of each.

**ROSA's contribution to the output module.** At the position immediately
*after* the `<LABEL>` indicator, before any label tokens have been emitted,
ROSA produces a single argmax-predicted token ID `v*`. This ID is re-embedded
via the codebook (`C[v*]`) and passed — with `stop_grad` — as an extra
feature to the output module.

### Output module — 11-class classifier (10 digits + NO_OUTPUT)

The output module fires at **every position** in the stream. At each
position it produces an 11-way distribution over `{0, 1, ..., 9, NO_OUTPUT}`.
The supervision targets are:

| Position | Supervision target |
|---|---|
| `<LABEL>` indicator (just before `l_1`) | the true MNIST digit (class 0–9) |
| All other positions | `NO_OUTPUT` (class 10) |

This gives the output module two simultaneous jobs: learn **when** to fire
(timing — read off the recent token pattern, the `<LABEL>` indicator, and
ROSA's prediction) and **what** to fire (the digit). Importantly, the
"when" signal is learned *from context alone* — the output module is not
told externally that it's at a `<LABEL>` position; it has to infer it
from the recent stream. ROSA's predicted token is a strong cue: at a
`<LABEL>` position, ROSA's database-style memory may already be predicting
the right digit's first label-token.

**Architecture.** A small MLP over a fixed-size context summary at each
position:

```
context = pool(recent_token_embs, position_features)
rosa_feat = stop_grad(C[rosa_pred_id_at_this_position])
features = concat([context, rosa_feat])
logits   = MLP_out(features)                       # -> R^11
```

**Pooling.** At each position `t` along the stream, take the last `W=8`
token embeddings (sliding window). Padded with a learned `<PAD>` embedding
when fewer than `W` are available. Concrete feature:
`[concat(last_W_tokens), rosa_feat]` of dim `(W+1)·D = 9·D = 576`.
Position within the burst (e.g., "I am the 3rd sample-token of this
burst") is encoded by an additive learned positional embedding to keep
the output module aware of structure.

**Loss.** Cross-entropy at every position with class re-weighting to
handle the strong imbalance — per sample only ~1 position carries the
`DIGIT` target while ~21 positions carry `NO_OUTPUT`:

```
L_classify = mean(CE_no_output_positions) + λ_digit_boost · mean(CE_digit_positions)
```

Default `λ_digit_boost = 20.0` (roughly inverse-frequency). Without this,
the model converges to "always predict `NO_OUTPUT`" — a 95.4% accuracy
on the position-level task that is 0% useful for MNIST classification.

**Eval-time digit readout.** At test time, the digit prediction for sample
`i` is `argmax(logits[0..9])` at the `<LABEL>` position of sample `i`'s
burst — i.e., we ignore the `NO_OUTPUT` class at that one position. The
test accuracy is the fraction of test samples whose `<LABEL>`-position
argmax-over-digits matches the true label.

---

## Training signal

The full per-batch loss is:

```
L_batch = L_classify  +  λ_codebook · L_codebook
                      +  λ_ponder   · L_ponder
                      +  λ_commit   · L_commit                   # 0 for DiVeQ; only for STE ablation
                      − λ_entropy   · H(soft_assignments)        # optional
```

Where:
- `L_classify` — dense 11-class CE on output-module logits at every stream
  position, with `λ_digit_boost = 20.0` upweighting the `DIGIT`-target
  positions to counter `NO_OUTPUT`-class imbalance (see Output module).
- `L_codebook` — pulls codebook entries toward their assigned z's:
  `||stop_grad(z_k) - C[v_k]||²`.
- `L_commit` — only used in the VQ-STE ablation; DiVeQ's reparameterization
  bakes this gradient into the forward pass.
- `L_ponder` — UTM-style ACT cost: `mean(remainder + n_steps)` over both
  sample and label ACT loops. Default `λ_ponder = 0` for the baseline.
- `H(soft_assignments)` — entropy of the temperature-softmaxed codebook
  affinity `softmax(−||z - C||² / τ)`; encourages spread early. Default off.

**Gradient flow.**
- `L_classify` flows into: output module weights, indicator and `<PAD>`
  embeddings, the per-position pooled token embeddings, and via STE into
  the input module's `z` outputs and the codebook entries those `z`'s
  mapped to.
- ROSA's predicted-token embedding is `stop_grad`'d at every position —
  `L_classify` does not shape the codebook or input module via that path.
- `L_commit` flows into the input module only (codebook side is stop_grad'd).
- `L_codebook` flows into the codebook only (input side is stop_grad'd).
- `L_ponder` flows into the halt head.

**ROSA timing within a batch.** ROSA is **frozen during the forward pass of
each minibatch** — all `B` samples query the same SAM state. After the
optimizer step, all `B` samples' bursts (sample tokens + label tokens) are
committed to the master stream in a single `commit_batch` call. This means
sample `i+1` cannot benefit from sample `i`'s contribution within the same
batch, which is acceptable noise at batch sizes 128–256.

---

## ROSA at evaluation

**Frozen.** When training ends, ROSA's SAM and fallback LM are saved
(`rosa_expN.json`) and never modified again. Test samples are queried but
their tokens are *not* committed. This makes the test-set accuracy
reproducible and free of test-sample-order dependence.

A separate ablation (later) is the **continual-eval** setting where ROSA
keeps growing on the test stream — interesting for online/deployment scenarios
but not the headline benchmark.

---

## Hyperparameters (headline run, exp1)

| Group | Knob | Value |
|---|---|---|
| Vocab | `V` | 1024 |
| | `D` (token dim) | 64 |
| | `<SAMPLE>` ID, `<LABEL>` ID | 0, 1 |
| Input MLP | hidden width | 256 |
| | depth (per recurrent step) | 2 |
| | sample `K` (exp1, fixed) | 16 |
| | label `K` (exp1, fixed) | 4 |
| | sample `K_max` (exp1b, ACT) | 16 |
| | label `K_max` (exp1b, ACT) | 4 |
| | halt bias init (exp1b) | −3 (deep-start) |
| Output MLP | hidden width | 256 |
| | depth | 2 |
| Codebook | init scale | `±1 / V` (uniform) |
| Output MLP | context window `W` | 8 last token embeddings |
| | `λ_digit_boost` | 20.0 (counters `NO_OUTPUT` class imbalance) |
| DiVeQ | `SIGMA` | 0.01 (diveq-validated) |
| Loss | `λ_codebook` | 1.0 |
| | `λ_commit` | 0 (DiVeQ reparam handles this; turn on only for VQ-STE ablation) |
| | `λ_ponder` | 0 (off in baseline) |
| | `λ_entropy` | 0 (off in baseline) |
| Optimizer | Adam lr | 3e-4 |
| | cosine decay alpha | 0.05 |
| | epochs | 50 |
| | batch size | 128 |
| ROSA | max SAM order | 1,048,576 (no limit in practice) |
| | use EOT | False |
| | boundary at `<SAMPLE>` | True |

`n_params` (rough): codebook `V·D = 65,536` + input MLP ~270K + output
MLP ~70K + halt heads ~1K. Total ~400K trainable.
SAM state count grows roughly linearly with token count; expect a few million
states by the end of training.

---

## Why this might (or might not) work

**Pros.**
- ROSA captures literal "I have seen this exact suffix before" memory, which
  is exactly what enables one-shot recall — useful for MNIST's
  visually-distinctive digits.
- The neural input module learns a tokenization that *makes ROSA's job easy*
  (because ROSA's predicted-token, fed back to the output module, is a
  classification feature). This is a self-supervised pressure on the
  tokenization toward "cluster-like" structure.
- Continual SAM means later epochs strictly accumulate signal — no forgetting.

**Cons / failure modes.**
- **Stop-grad on ROSA's prediction means the input module has no incentive
  to make ROSA accurate.** The output module may learn to ignore ROSA's
  feature (it's stop-grad'd anyway) and just classify from the recent-window
  pool — degenerating to a vanilla VQ-MLP classifier. The ROSA-hit-rate metric
  monitors this.
- **`NO_OUTPUT` collapse.** With ~21:1 imbalance per sample, the easy
  fixed point is "always predict `NO_OUTPUT`." `λ_digit_boost = 20.0`
  in the loss should counter this; if not, increase further or move to
  per-position class-balanced sampling.
- **Output module learns indicator-position from `<LABEL>` token alone.**
  If the output module just learns "fire when last-seen token is
  `<LABEL>`-emb" and ignores ROSA + image content, accuracy floors at the
  vanilla VQ-MLP baseline. The `exp5` lesion (`rosa_feat = 0` at eval)
  measures this directly.
- **Stale tokens.** Tokens emitted by epoch-1 weights are different from
  tokens emitted by epoch-50 weights, but ROSA records both. Late counts
  dominate by frequency, but early "contamination" might bias the SAM toward
  meaningless suffixes. The `V=256` ablation tests sensitivity.
- **Codebook collapse.** Standard VQ-VAE failure: a few entries dominate.
  The entropy regularizer is a knob if needed.
- **ACT instability.** Halting can collapse to "always halt at step 1" or
  "never halt." Halt-bias = −3 (deep-start) discourages early collapse.
  Ponder cost off in baseline; turn on if K stabilizes too high.

If end-to-end STE plateaus, the EGGROLL fallback (exp4) treats the whole
pipeline as a black box and applies ES gradients on classification accuracy
with ROSA in the loop — bypassing the stop-grad bottleneck on ROSA.

---

## Glossary

- **SAM** — suffix automaton; the data structure ROSA uses to record all
  observed contiguous substrings of the training stream.
- **Witten–Bell** — a smoothing scheme for n-gram LMs that interpolates higher-
  and lower-order context counts; used as ROSA's fallback when the SAM has
  no deterministic answer.
- **STE** — straight-through estimator; identity gradient through a
  non-differentiable forward op (e.g. argmin codebook lookup).
- **ACT** — adaptive computation time; a halt head decides per step whether
  to continue iterating a recurrent block.
- **EGGROLL** — low-rank ES (evolution strategies) for gradient estimation
  via antithetic perturbations.
