# Universal AR — Concepts

**Continual, in-context memory completion.** A model consumes a *stream* of
records — many partial samples (and some labels), interleaved in any order —
builds an associative memory online, and **completes** missing fields on demand.
It is *not* a generative model: we never model a whole sample jointly, we fill in
missing values field-by-field. Classification is just completion of a label field.

Retrieval is **generic / auto-associative**: any subset of fields cues the rest.
There is no fixed "key" vs "value" — the query mask decides direction per query.

Status: **design / idea capture**. No experiments run yet. Start dataset: MNIST.

---

## 1. The core reframing — dataset as a stream of records

Classical supervised learning treats a dataset as a matrix `X ∈ R^{N×D}`
(samples × features) plus labels. We **dissolve the atomic "sample" row.**

A data point is a **record** — a small set of `(role, filler)` fields. For MNIST
a single pixel is:

```
record = { value: v,  pos: p,  ref: s }
```

- `value` — the (binned) content, e.g. a pixel intensity or a class label.
- `pos`   — where it sits, e.g. a pixel position, or the special `pos_label` slot.
- `ref`   — a **reference / variable name** for which sample this record belongs
  to. **Not** a learned per-sample embedding — a binding tag (see §4).

`value / pos / ref` are **roles** (attributes); `v / p / s` are their **fillers**
(the actual contents). The set of roles is fixed and small; fillers vary.

An **episode** is a stream of such records drawn from several samples, each tagged
by a `ref`, each sample possibly **partial** (only some records present), the
label present for some samples and absent for others, all **interleaved in
arbitrary order**.

## 2. The model — auto-associative memory with generic completion

### 2a. Record representation — role ⊗ filler

Bind each field's role to its filler and superpose into one record vector:

```
p = r_value ⊛ ψ(v)  +  r_pos ⊛ φ(p)  +  r_ref ⊛ χ(s)
```

- `r_value, r_pos, r_ref` — **learnable role vectors** (one per attribute).
- `ψ, φ, χ` — filler embeddings (value token, position, ref name).
- `⊛` — a **conjunctive bind**: HRR circular convolution (d-dim) or tensor
  product / TPR (d²-dim). Conjunctive (not additive) so `r_pos⊛φ(p)` for one
  record does not partially match a different field of another record — this is
  what prevents cross-field / cross-sample **ghost** retrievals (the core trick
  of variable binding).

### 2b. Store the record *with itself* — symmetric memory

The one-line change from a hetero-associative memory to a generic one:

```
hetero-associative (fixed key→value):   M = Σ_i  k_i ⊗ v_i        # direction baked in
auto-associative   (any field → rest):  M = Σ_i  p_i ⊗ p_i        # symmetric, self-association
```

`M = Σ p⊗p` is the classic Hopfield weight — symmetric, so **any field can cue
any other**. The role⊗filler binding is what makes individual fields addressable
inside the superposition.

### 2c. Completion — the mask picks key vs value

`?` is the **universal guess / unknown signal**: a special filler token marking a
field we want to read.

```
query (?, p, s)   cue q = r_pos⊛φ(p) + r_ref⊛χ(s) + r_value⊛UNK      → read value
query (v, ?, s)   cue q = r_value⊛ψ(v) + r_ref⊛χ(s) + r_pos⊛UNK      → read pos
query (v, p, ?)   cue q = r_value⊛ψ(v) + r_pos⊛φ(p) + r_ref⊛UNK      → read ref
```

1. **Retrieve the full record pattern** from memory (both reads kept — §5a):
   ```
   linear (Hebbian):    p̂ = M q                                   # fixed-size state, continual
   attention/Hopfield:  p̂ = Σ_i softmax_i(β · p_i·q) p_i          # keep {p_i}, exact recall
   ```
2. **Unbind each queried role** to recover its filler, then decode:
   `f̂ = r_role⁻¹ ⊛ p̂` → nearest token / learned decoder → value.

Because the store is symmetric, this handles **every mask with one mechanism**:
`(?,p,s)→v`, but also the reverse `(v,s→p)`, `(v,p→s)`, and multi-field masks
(unbind several roles). To answer a query about `ref = s`, some records bound to
`s` must already be in memory; a fresh unbound ref has no content — correct.

The memory is built incrementally over the stream (the "AR" / online part) and can
be queried after any prefix — **retrieve anytime, any order.**

### Existing structure in this repo

[`projects/rnn/mnist-rnn.py`](../rnn/mnist-rnn.py) implements an outer-product
**linear** RNN for MNIST: `emb = pos_emb[t] + val_emb[bin(x)]`,
`M_t = M_{t-1} + k_t ⊗ v_t`, order-independent closed form.
**Reuse:** the pos/value embeddings and the outer-product write.
**Differ:** mnist-rnn is *hetero*-associative (fixed key→value) and feeds one
sample's `M` to an MLP classifier. Universal-AR is *auto*-associative (`p⊗p`),
holds many samples in one `M` distinguished only by `ref`, and *reads by query*.

## 3. Evaluation — the label is just another field

The class label is a record at the special `pos_label` position:

```
record = { value: label, pos: pos_label, ref: s }
```

Some samples appear **with** their label record (support / in context) and some
**without**. Classification = completing the value of a label-free sample:

```
query (?, pos_label, s)  →  predicted label
```

Pixel inpainting, denoising, super-resolution are the *same* completion at pixel
positions. One mechanism, many tasks, selected by which field you leave blank.

---

## 4. `ref` — variable binding (the filler for the reference field)

`ref` must be a **pure coreference tag**: it says "these records are the same
sample" without leaking content, so the model generalizes to refs it has never
seen.

### 4a. Random re-assignment each episode  (the anonymization trick)
Every episode, draw ref fillers and bind them to random samples. Because the
mapping ref↔sample is re-randomized, the model **cannot** memorize "ref = this
content" — it must use the ref purely as a binding tag and read content from
context. (Same principle as label-anonymization in in-context learning.)

- **Discrete learned pool (default):** a fixed set of `V` learnable "variable
  name" tokens; sample a random subset per episode. The tokens are trained (safe,
  stable) while the *assignment* is randomized (gives anonymization). Caps
  samples-per-episode at `V`; train them toward near-orthogonality.
- **Continuous random roles (parked — riskier):** a fresh random unit vector per
  sample gives unbounded refs, but the vectors get *no gradient* — all learning
  falls on the surrounding net to handle arbitrary vectors. Revisit only if we
  hit the pool-size ceiling.

### 4b. Fallback — slot-structured memory
If superposition crosstalk is too high, give each active `ref` its own memory
block (a register file / soft slots via attention over refs). Cleaner retrieval,
less elegant, less "continual." A fallback, not the default.

---

## 5. Other design choices

### 5a. Readout — keep both linear and attention as proposals
Both retrieve the full pattern `p̂` from the same stored records; they differ only
in normalization (see §2c). **Keep both.**

- **Linear (Hebbian) `p̂ = M q`:** fixed-size state `M = Σ p⊗p`, O(1) per read,
  never revisits points — truly streaming / continual, but lossy (unnormalized
  crosstalk). The continual-learning-faithful option.
- **Attention / modern-Hopfield `p̂ = Σ softmax(β·p_i·q) p_i`:** keeps every `p_i`,
  O(#records) per read, a growing cache (KV-cache-like) — exact recall, sharpens
  onto the matching record regardless of episode size. The accuracy ceiling.
- They are two ends of one spectrum: the linear memory *is* linear attention with
  feature map `φ = bind`; softmax attention is the exact-but-growing version.
  Keep the per-record `p_i` during an episode so linear vs attention is a swap on
  the *same* stored patterns.
- **Pseudo-inverse (Kohonen OLAM):** least-squares recall, better than raw Hebbian
  when patterns correlate; closed-form, differentiable. Middle rung.
- **Hypernetwork readout (the original primitive):** the memory generates the
  weights of the **unbind + decode** step (a small MLP), giving a learned,
  higher-capacity cleanup than exact HRR unbinding. Start with explicit
  bind/unbind; promote to this once the plumbing works.

### 5b. Value vocabulary — everything is a token
Bin pixels into `K` value tokens (mnist-rnn uses K=32) and add 10 label tokens to
the *same* vocab. Pixel completion and classification become one softmax — the
cleanest expression of "universal."

### 5c. Position encoding — keep it prior-free (decision)
`pos` is a plain **learned embedding** over the 784 positions (plus `pos_label`).
**No** Fourier / coordinate / 2D-locality prior for now — let the model discover
structure from data. (Coordinate / conditional-INR views parked for later.)

### 5d. Objective
Mask a random subset of each sample's records (sometimes the label) and complete
them from the rest, as pointwise cross-entropy over the unified value vocab.
Order is free, not engineered: a summed linear memory is already order-independent,
so held-out completions can be scored in parallel (masked prediction).

### 5e. Interference / capacity
Auto-associative crosstalk grows with superposed writes *and* fields per record.
Ablate: conjunctive vs additive binding, HRR vs tensor product, role/embedding
dimensionality, LayerNorm on the read, linear vs attention vs pseudo-inverse read,
records per episode.

### 5f. Theory anchor (parked — optional)
Value-free retrieval is kernel / Nadaraya–Watson regression (a GP posterior mean);
the embeddings are the learned kernel. Not needed to build; revisit for a
principled loss or uncertainty.

---

## 6. Candidate first experiments (not yet run)

1. **exp1 — in-context completion on MNIST.** Episodes of several partial MNIST
   samples, each bound to a random `ref` from a learned discrete pool; labels
   present for some samples and queried for others. Role⊗filler records,
   conjunctive binding, **symmetric `p⊗p` memory**, unified value+label vocab,
   **both linear and attention reads** computed from the same stored patterns.
   Metric: label-completion accuracy on label-free samples; pixel-completion aux.
   Reuse mnist-rnn embedding code.
2. **exp2 — binding ablation.** Additive vs tensor-product vs HRR binding;
   discrete pool vs continuous random roles; records per episode (crosstalk).
3. **exp3 — readout ablation.** Linear vs attention/Hopfield vs pseudo-inverse vs
   hypernetwork readout, same memory.
4. **exp4 — generic-retrieval probe.** Verify reverse/any-mask queries work:
   `(v,p→s)`, `(v,s→p)`, multi-field masks — not just `(?,pos,ref)→value`.
