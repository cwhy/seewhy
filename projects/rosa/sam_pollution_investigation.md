# Why ROSA's SAM fails when its alphabet drifts

*An investigation into the late-epoch "collapse" of ROSA-as-classifier
(`projects/rosa/experiments20b.py`), and why it isn't really a collapse.*

---

## TL;DR

The exp20b run on MNIST appeared to collapse from **82.41% (epoch 23) → 65.30% (epoch 29)**. We diagnosed this and the failure mechanism turned out to be subtle and informative:

- **The model's final parameters are good.** With the same params and a freshly-rebuilt SAM, test accuracy is **81.79%** — essentially the peak. The "collapse" is a measurement artifact in the (params, SAM) joint state, not a degradation of the learned policy.
- **The accumulated SAM's transitions encode multiple, mutually-inconsistent codebook geometries** laid down over 30 epochs of continual training. Late-epoch params produced different tokenizations than mid-epoch params; both ended up in the same SAM.
- **At eval time, ROSA's `rightmost-match` deterministic rule returns the next token from the *latest* matching position in `sam.text` — placed there under a drifted codebook.** That token ID is then re-embedded via the *current* codebook. The two codebooks disagree about what each ID *means*.
- **The failure is class-specific.** Per-class accuracy shows digits 3, 4, and 8 catastrophically collapsing under the polluted SAM (−53pp, −70pp, −46pp), while digits 0, 1, 6, 7 are unaffected or slightly *better*. This is the signature of codebook reassignment among visually-similar classes (curved digits) — not a global degradation.
- **Theoretical framing.** ROSA was designed for *static symbolic alphabets* (characters, BPE tokens). Here the alphabet is **learned and drifting**. The discrete ID is stable as a label, but its embedding interpretation is moving. This is essentially the *Ship-of-Theseus* problem for discrete intermediate representations in continual learning: ROSA accumulates symbol-sequences whose *symbolic identity* is unchanged but whose *semantic content* depends on the codebook at the time of the commit. The two — encoder ↔ symbol-stream ↔ embedding-lookup — must be kept temporally coherent, and naive continual ROSA accumulation breaks that.

The mitigation is trivial in this case (rebuild the SAM at eval time from the current encoder) but the broader implication is that **continual ROSA + learned tokens is fundamentally a credit-assignment-across-time problem**.

---

## 1. Setup recap

`experiments20b.py` on MNIST: ES (n_pop=128, rank=4, σ=10⁻³, 60K, 30 epochs).
Per-sample stream of 17 tokens: `[s_1..s_16, y_token]`. The first 16 tokens
come from a learned input MLP + DiVeQ argmin against a 1024-entry codebook
(content slots are IDs 11..1023; IDs 0..9 are reserved as digit-class
"output slots"; ID 10 is the null slot). At inference, the digit prediction
is:

```
predicted_id  = ROSA.predict_at(stream[:16])             ← argmax of SAM-deterministic rule
rosa_pred_emb = codebook[predicted_id]                   ← lookup the current codebook
logits        = rosa_pred_emb @ codebook[:11].T          ← 11-class linear head
predicted_digit = argmax(logits[:10])                    ← ignore null class
```

Crucially: **the entire classifier's only learned signal is in the codebook**.
The 11 reserved entries `codebook[0..10]` serve as the linear classifier
weights; the entries `codebook[11..1023]` serve double duty as (a) DiVeQ
quantization targets and (b) the embeddings ROSA's predicted IDs get looked
up to.

ROSA's training: every commit appends 17 integers to a shared `sam.text`,
marking a boundary after each. After 30 epochs × 60K samples × 17 tokens
≈ 30.6M committed tokens.

---

## 2. The observation that needed explaining

Late-epoch test accuracy on exp20b oscillated, peaked at 82.41% at epoch
23, then dropped to 65.30% by epoch 29. The final model params + final SAM
report 65.30%.

When we **rebuild a fresh SAM from scratch** with the *same final params*
(one pass over the training set), test accuracy jumps back to **81.79%**.
This rules out "the model itself collapsed" as the explanation. The
final-params encoder/codebook is fine; the joint (params, SAM) state is the
problem.

| Run     | as-trained SAM | fresh SAM | Δ        |
|---------|---------------:|----------:|---------:|
| exp20b  |         65.30% |   81.79% |  +16.49pp |
| exp20c  |         79.60% |   74.96% |   −4.64pp |

(Note that exp20c, which stopped at 24 epochs and didn't collapse, sees
the *opposite* sign — its as-trained SAM is *better* than a fresh rebuild.
We'll come back to this.)

---

## 3. Diagnostic D1 — codebooks drift

The two runs (exp20b ran 30 epochs, exp20c ran 24 epochs) start from the
*same* seed (42), the *same* architecture, the *same* hyperparameters
except for `N_EPOCHS`. They produced two *very different* final codebooks.

We sample 5000 training images and encode each with both final-param
codebooks. Token-level disagreement:

- **Overall ID disagreement rate: 78.85%** of (image, position) pairs map to a different ID under the two codebooks.
- **Per-position ID-set overlap: ~8–14%** of IDs used at a given position appear in both vocabularies.
- **Codebook entry L2 movement:** mean = 1.185, p90 = 1.323, max = 1.606. Compare to mean codebook entry norm of ~1.3: each entry has moved by ~90% of its own norm.

![Codebook drift histogram](https://media.tanh.xyz/seewhy/26-05-15/rosa_diag_codebook_drift.svg)

What this means: the two codebooks aren't small perturbations of each
other — they're nearly *different vocabularies*. Same architecture, same
seed, slightly different LR schedule (cosine annealing over 24 vs 30
epochs) → essentially disjoint embedding spaces.

This is the first clue: codebooks under ES training are not stable across
runs *or* across time within a run. Discrete IDs are a relabeling of a
moving continuous geometry; the relabeling can rotate, permute, or
collapse onto different content as training proceeds.

---

## 4. Diagnostic D2 — `sam.text` drifts within a single run

Now confine attention to exp20b's single SAM. It accumulated 30.58M tokens
across all 30 epochs of training. We slice `sam.text` into 10 equal-time
chunks (≈180K bursts each, ≈3M tokens each) and look at the distribution
of the *first sample-token* of each burst within each chunk.

| chunk (time) | unique first-pos tokens | KL(chunk \|\| chunk_0) | KL(chunk \|\| chunk_last) |
|---|---:|---:|---:|
| 0 (earliest) | 513 | 0.000 | **4.518** |
| 1 | 278 | 0.521 | 2.487 |
| 2 | 286 | 0.632 | 1.644 |
| 3 | 270 | 0.685 | 0.487 |
| 4 | 253 | 0.791 | 0.167 |
| 5 | 218 | 0.893 | 0.055 |
| 6 | 201 | 0.846 | 0.030 |
| 7 | 177 | 0.823 | 0.015 |
| 8 | 186 | 0.854 | 0.004 |
| 9 (latest)   | 205 | 0.863 | 0.000 |

![Within-SAM KL drift](https://media.tanh.xyz/seewhy/26-05-15/rosa_diag_kl_curve.svg)

Two effects are visible:

1. **Monotonically increasing KL to the start.** Each subsequent chunk diverges further from the first-chunk distribution. The codebook is rotating; tokens that *used to* be "first sample-token of digit-3 bursts" are not the same tokens after a few epochs of training.

2. **Sharply decreasing KL to the end.** The last few chunks (8, 9) are very similar to each other — the codebook converges in late epochs (small Adam steps, low LR). But chunks 0–3 differ from the final distribution by huge amounts (KL up to **4.5 nats**, ≈ 6.5 bits — the distributions are nearly disjoint on dominant tokens).

3. **Vocabulary contraction.** Unique first-position tokens drop from 513 (epoch ≈1) to ~200 (epochs 20+). The codebook is collapsing onto fewer entries, which is normal VQ behavior — but it means *late commits are quoting from a much smaller vocabulary than early commits*. The union of all 10 chunks contains 665 distinct first-position tokens; the last chunk uses 205 of them.

The SAM's text is, in effect, a record of every codebook geometry the
model has held. The "rightmost match" rule (which ROSA's deterministic
predictor uses) systematically prefers the **latest** position in `sam.text`,
i.e. the **latest codebook geometry's** transitions. But the latest codebook
geometry may not be the same as the current (post-final-step) codebook
geometry, especially if the very last training step shifted things.

---

## 5. Diagnostic D3 — as-trained vs fresh SAM head-to-head

We classify the 10000 MNIST test samples twice with the same final params
— once using the polluted as-trained SAM, once using a fresh single-pass
SAM.

| Outcome (10000 test samples) | count |
|---|---:|
| Both SAMs correct | 6066 |
| Both SAMs wrong   | 1357 |
| **Polluted only correct (fresh wrong)** | **464** |
| **Fresh only correct (polluted wrong)** | **2113** |
| Predicted-ID agreement | 68.74% |
| Digit-prediction agreement | 68.98% |

When the two SAMs disagree, the **fresh SAM wins ~4.5× as often** as the
polluted one. The polluted SAM is right on a few cases the fresh one
misses (probably because multi-epoch repetition strengthened some
correct-class statistics for those samples), but on net it loses 2113
samples for 464 gained — a net cost of ~16pp, matching the headline
accuracy gap.

Both SAMs produce IDs *in the digit-class range* (0..9) for >96% of test
queries, and neither ever predicts the null class. So this isn't an
"abstain vs answer" issue — both SAMs confidently emit digit-IDs; the
polluted one just emits the wrong digit more often.

---

## 6. Diagnostic D4 — where does the rightmost match live?

Hypothesis: failed test samples' matches live deeper in `sam.text` (later
in time, hence laid down by drifted codebooks).

| | failed (n=500) | correct (n=500) |
|---|---:|---:|
| mean rightmost-match position (fraction of `sam.text`) | 0.980 | 0.996 |
| median rightmost-match position | 1.000 | 1.000 |
| fraction with match in the last decile of `sam.text` | 95.4% | 98.4% |
| matched suffix length (out of K_SAMPLE=16) | 15.9 / 16 | 16.0 / 16 |

![Rightmost-match positions](https://media.tanh.xyz/seewhy/26-05-15/rosa_diag_rightmost.svg)

The result is surprising in a useful way: **both correct and failed
matches are overwhelmingly in the last decile of `sam.text`** — the latest
epochs' commits. So failure isn't due to matches falling back to *very early*
training. Failed and correct cases differ only marginally in position
(95% vs 98% in the last decile).

The match lengths are nearly identical (16 / 16) — the failure isn't from
shorter matches that fall through to noisier suffix-link ancestors.

So **the failure happens within the latest codebook geometry**, not by
reaching back to early epochs. Even when the rightmost match is in the
very last training epoch, the *content* of that match has shifted enough
in the last few steps that it returns the wrong follow-up.

This is consistent with D2's finding that the very-last chunks (8, 9) are
distributionally similar to each other (KL → 0 between them), but each
contains ~200 unique tokens — meaning many distinct test images are
collapsing onto the same small set of suffixes. The rightmost match thus
returns whichever class happened to commit *most recently* with that exact
suffix, even though several different classes may have committed that
suffix during late training.

---

## 7. Diagnostic D5 — the smoking gun: per-class collapse

The most striking single finding. Per-digit test accuracy:

| digit | polluted SAM | fresh SAM | Δ (polluted − fresh) |
|---|---:|---:|---:|
| 0 | 92.3% | 91.6% | +0.7pp |
| 1 | 97.4% | 96.7% | +0.6pp |
| 2 | 79.9% | 78.4% | +1.6pp |
| **3** | **23.4%** | **76.4%** | **−53.1pp** |
| **4** | **11.6%** | **81.1%** | **−69.5pp** |
| 5 | 65.1% | 64.1% | +1.0pp |
| 6 | 88.4% | 88.3% | +0.1pp |
| 7 | 85.1% | 83.8% | +1.4pp |
| **8** | **26.5%** | **72.3%** | **−45.8pp** |
| 9 | 77.7% | 81.6% | −3.9pp |

For seven of the ten digits, the polluted SAM is **slightly *better*** than
the fresh one (+0.6 to +1.6pp). For three digits (3, 4, 8), it
catastrophically collapses by 45–70pp.

This shape is the signature of **codebook reassignment among visually-similar
classes**, not a global breakdown. Digits 3, 4, and 8 share curved /
loopy structure — they're plausibly close in the input MLP's encoder
output before quantization, so they share codebook neighborhoods. When
late-epoch codebook drift caused these neighborhoods to reshuffle, the
last-committed sample-token-suffix → label-token transitions for these
classes got swapped (a digit-3 suffix's last commit may have ended in
label-token=4, or 8, etc.).

Conversely, digits 0, 1, 6, 7 occupy distant regions of the codebook space
(distinctive shapes, no near-duplicates) — their tokens didn't drift onto
neighboring classes' territory. The slight bonus the polluted SAM gives
those classes (+~1pp) probably comes from **multi-epoch repetition
strengthening their statistics** — each digit-0 burst was committed ~24×
with consistent tokens, so the rightmost-match for a digit-0 test sample
robustly returns label-token=0.

So the polluted SAM has *two opposing effects*:

1. **Helpful**: multi-epoch repetition reinforces statistics for stable
   class neighborhoods.
2. **Harmful**: codebook drift creates inconsistent neighbor-class
   commitments at the boundary of unstable class neighborhoods.

For exp20c (24 epochs, less late drift), (1) > (2) → polluted SAM is
overall better. For exp20b (30 epochs, more late drift), (2) > (1) →
polluted SAM is overall worse, but only for the unstable classes.

---

## 8. Theoretical framing — the Ship of Theseus

ROSA was designed for **static symbolic alphabets**. In its native domain
(character-level language modeling), the symbol `'a'` always means `'a'`.
The SAM's accumulated suffix statistics can be safely treated as durable
facts about the data because the alphabet has a fixed external definition.

In our setup, the alphabet is **learned**. Each codebook entry has an
identity (a discrete ID `v ∈ 0..1023`) and a meaning (a continuous
embedding `codebook[v] ∈ ℝ^64`). The identity is stable across time but
the meaning is *trained* and therefore drifts. ROSA accumulates a stream of
identities, but its downstream consumer (the linear head `codebook[:11].T`)
interprets via the *current* meaning.

Symbolically:

```
   commit time t:        encode_t(x)  → id_t          codebook_t[id_t] ≈ "digit-3-ish-feature"
   commit time t+ε:      encode_{t+ε}(x') → id_t      codebook_{t+ε}[id_t] ≈ "digit-4-ish-feature"  ← reassignment!

   sam.text stores:    [..., id_t, label_t, ..., id_t, label_{t+ε}, ...]
                              same ID, different commit time,
                              different underlying meaning,
                              different label outcome.

   at eval time T:     ROSA returns predicted_id = id_t   (rightmost match's next token)
                       codebook_T[id_t] ≈ "digit-?-feature" — whatever T says
                       output_classifier(codebook_T[id_t]) = ?
```

The discrete ID is the **identity** of a token; the codebook entry at any
time is its **meaning**. ROSA records identities; meanings drift; the
output module needs both to be temporally aligned. The Ship of Theseus
problem: same identifier, different actual content over time.

This is a much more general phenomenon than ROSA. Anywhere you have a
**learned discretization** with a **continual data structure indexed by
discrete IDs**, you have this issue. Examples:

- Cluster IDs in online k-means → stored counts/labels indexed by ID
- Tokenizer expansion (BPE merges) in continual pretraining
- VQ-VAE latents fed to a downstream model with their own embeddings
- Memory networks with discrete address slots

In all these, the "identity" of a slot is preserved, but its "meaning"
drifts. Naive accumulation creates inconsistencies that show up at eval
time as the (current-encoder, accumulated-store) joint state being
incoherent.

The deeper question this surfaces: **continual ROSA's value proposition is
predicated on token stability**. ROSA gives you O(1) suffix-match memory
of every sequence ever seen — but only if the sequences are over a stable
alphabet. The moment you introduce a learned tokenizer, every commit
"means" something slightly different from earlier commits, and the SAM
becomes an incoherent mixture rather than an accumulating database.

---

## 9. Mitigations (in order of how much they touch the design)

### 9a. Rebuild the SAM at eval time (one-shot, free)

```
rosa.reset()
for sample in train_set:
    rosa.commit_burst(encode_current(sample))
test_acc = eval(rosa, params, test_set)
```

This is what we did to recover exp20b to 81.79%. Cost: one pass through
training data (~3 seconds in our setup). The SAM now uses only the current
codebook's geometry.

**Tradeoff:** loses the multi-epoch-repetition bonus (each sample
committed once instead of N_EPOCHS×). For exp20c this cost ~5pp.

### 9b. Rebuild periodically (sliding window of recent commits)

Keep a window of last-K commits and let older ones age out. Approximates
a "current-codebook only" SAM without losing all the repetition signal.
Requires the SAM to support deletion, which the basic implementation does
not.

### 9c. Re-encode old text whenever the codebook moves significantly

Track the codebook's drift and trigger a full re-tokenization when it
exceeds a threshold. This is essentially what dynamic vocabularies in
neural LMs do (sometimes called "embedding refresh"). Cost: periodic
expensive rebuild instead of cheap incremental commits.

### 9d. Anchor the codebook with a slow EMA

Prevent late-epoch drift from compounding by using EMA-updated codebook
entries (à la VQ-VAE) instead of letting ES freely move them. Would
suppress the geometric drift directly, at the cost of slower adaptation.

### 9e. Co-evolve the SAM via a slot-versioning scheme

Tag each SAM transition with a `codebook_version` integer; at eval, only
walk transitions whose version matches current params' codebook version
(within some tolerance). This is the "principled" fix but requires
substantial implementation work.

### 9f. Give up on continual ROSA, use a snapshot

Train the encoder to convergence, freeze it, then build ROSA over the
frozen tokenization. This is the Phase A "decoupled" architecture; it
sidesteps the problem by construction but loses the joint optimization
that's the appeal of the ES-trained version.

---

## 10. What this teaches us beyond the immediate result

1. **Discrete intermediate representations in continual learning are
   inherently fragile.** They look beneficial — symbolic, sparse, lookup-able —
   but they require both the upstream encoder and the downstream interpreter
   to stay temporally aligned with the symbol stream. Any drift in either
   side compounds.

2. **The "stronger optimizer → ROSA hurts more" finding from Phase B has
   a clean re-interpretation.** Stronger optimizers cause more codebook
   drift per unit time. More drift → more SAM pollution → more (params, SAM)
   incoherence at eval. ROSA "hurting" isn't because the signal is bad;
   it's because the signal is *moving*.

3. **ROSA's "perfect memory" property is conditional.** It's truly perfect
   on the stream-as-symbols-from-a-fixed-alphabet. With learned tokens, it's
   perfect on the symbol-identity layer but lossy on the
   symbol-meaning layer, and the latter is what matters downstream.

4. **There's a clean information-theoretic framing of this:** the SAM
   stores `H(stream)` bits of information about its commit history, but
   only the bits consistent with the *current* codebook geometry are
   usable at eval. The drift effectively *deletes* bits from the SAM
   without removing entries from `sam.text`. The fresh-SAM rebuild
   recovers those bits by re-tokenizing.

5. **Class-asymmetric failure is a feature of the diagnostic, not a bug.**
   The fact that digits 0, 1, 6, 7 are *helped* by the polluted SAM while
   3, 4, 8 are *catastrophically hurt* tells us this isn't a generic
   degradation. It's a sharp signature of codebook neighborhood reassignment
   for the visually-similar classes — exactly where one would expect
   small encoder shifts to push tokens across class boundaries.

This is, in the end, the same fundamental tension Phase B surfaced: **ROSA
is a database, classification needs a function approximator, and the
glue between them is fragile.** The Ship-of-Theseus diagnosis is what
the gluing looks like up close.

---

## Artifacts

- Script: [`projects/rosa/scripts/tmp/sam_diagnosis_full.py`](.)
- Raw measurements: `projects/rosa/scripts/tmp/sam_diagnosis_full.json`
- Plots: codebook drift, KL curves, rightmost-match positions (linked above).
- Saved (params, ROSA) artifacts: `params_exp20{b,c}_mnist.pkl`, `rosa_exp20{b,c}_mnist.json`.
