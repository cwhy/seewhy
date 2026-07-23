# ROSA on MNIST — Methodology

This is a focused write-up of *how* ROSA is wired into our image-classification
pipeline. It covers the vocabulary, the stream construction, indicator tokens,
boundary marks, the two prediction modes, and the training protocol. The
experiment results live in the
[Phase A](https://media.tanh.xyz/seewhy/26-05-08/rosa_report_phaseA.html) and
[Phase B](https://media.tanh.xyz/seewhy/26-05-14/rosa_report_phaseB.html)
reports.

---

## 1. ROSA in one sentence

**ROSA is a pure-statistics next-integer-token predictor.** Given any prefix
of a stream of integers, it returns the most likely next integer (or a
distribution over them). It has zero trainable parameters; it just stores
every contiguous substring of its training stream in a suffix automaton (SAM).

```
Input  to ROSA: a sequence of integers, e.g.  [12, 47, 89, 1, 5]
Output from ROSA: a single integer (argmax) OR a distribution P(t | prefix) over V
                  e.g.  predicted_next = 8   or   {2:0.71, 5:0.15, 8:0.07, ...}
```

This is the entire interface. Everything else in the project is a wrapper
that decides *what integers to put in* and *what to do with the integers
ROSA returns*.

The implementation is a port of bcml-labs/rosa-plus's `rosaplus.py` to
integer tokens, at `projects/rosa/lib/rosa_core.py`.

---

## 2. Vocabulary layout

Fixed alphabet of size `V = 1024`. The IDs are sliced into three zones:

```
ID 0       :  <SAMPLE>     ─┐
ID 1       :  <LABEL>      ─┤  reserved indicator slots — never produced by the
                              input MLP; inserted manually into the stream
ID 2..11   :  digit-token slots — reserved for the decoupled pipeline (digit y
              maps to token ID y + 2)
ID 12..1023:  content tokens — the input MLP's DiVeQ argmin must land here
              (the argmin is masked to exclude IDs < N_RESERVED, see
              `diveq_quantize` in `experiments1.py`)
```

`<SAMPLE>` and `<LABEL>` are reserved so they can act as **unambiguous
position markers** in the stream — the model knows that "the next token after
`<LABEL>` is structurally the prediction slot," regardless of any specific
content-token IDs.

Note: **ROSA itself has no semantic awareness of indicators.** They are
ordinary integers to the SAM. They create states and transitions like any
content token. The "special meaning" lives entirely in *how the wrapper
builds the stream around them*.

---

## 3. Stream construction (per training sample)

A single Fashion-MNIST sample `(image x, digit y)` becomes a 22-integer burst:

```
                     INPUT ACT-MLP                DiVeQ (argmin over codebook)
   image x  ──────►  16 continuous emb's z  ──►  s_1, s_2, ..., s_16    (each in 12..1023)
   one-hot y ─────►   4 continuous emb's z' ──►  l_1, l_2, l_3, l_4     (each in 12..1023)

then concat with indicators:

   position:    0    1    2    3    4   ...   15   16    17   18   19   20   21
             ┌────┬────┬────┬────┬────┬─ ─ ─┬────┬────┬─────┬────┬────┬────┬────┐
   stream =  │ 0  │ s_1│ s_2│ s_3│ s_4│ ... │s_15│s_16│  1  │ l_1│ l_2│ l_3│ l_4│
             └────┴────┴────┴────┴────┴─ ─ ─┴────┴────┴─────┴────┴────┴────┴────┘
              <S>  ───────── 16 image tokens ─────────  <L>  ── 4 label tokens ──
```

The exact code is in `experiments10.py::build_int_streams`:

```python
def build_int_streams(ids_sample_np, ids_label_np):
    streams = []
    for i in range(ids_sample_np.shape[0]):
        s = [ID_SAMPLE]                              # ← 0 goes in
        s.extend(int(t) for t in ids_sample_np[i])
        s.append(ID_LABEL)                           # ← 1 goes in
        s.extend(int(t) for t in ids_label_np[i])
        streams.append(s)
    return streams
```

**Variants we tested.** The `USE_INDICATORS=False` ablation (exp1n_fmnist)
removes the indicators from the stream entirely — the digit-prediction slot
becomes a *position* the model has to infer from token content alone. It
barely moved the headline accuracy (89.13% vs 89.00% with indicators), so
indicators are not load-bearing for classification accuracy — but they ARE
load-bearing for the *output module's localization*, which is a separate
diagnostic (see Phase B report).

---

## 4. Indicator tokens vs boundary marks

This is the easiest part of the methodology to confuse. ROSA has **two
distinct mechanisms** for marking structure in the stream:

### Indicator tokens (in the stream)

IDs 0 and 1 are committed to the SAM as ordinary integers. They occupy
positions in `sam.text`, create transitions, and contribute to Witten-Bell
counts. They have no special semantic treatment inside the SAM.

### Boundary marks (NOT in the stream)

After each sample's burst is committed, `rosa.sam.mark_boundary()` sets a
flag on the *position* of the last token of the burst. This flag is stored
in a separate `boundary_after: List[bool]` array, parallel to `sam.text`.

The flag is read by `_predict_deterministic` and prevents the rightmost-match
rule from returning a token that would cross an example boundary. The
Witten-Bell counts builder also skips pairs whose first member is at a
boundary position.

```
Example: after committing two bursts to ROSA, the internal state is:

   sam.text         = [0, 12, 47, ..., 99, 1, 5, 8, 0, 17, 22, ..., 88, 1, 3]
                       ─┬──────────────┬──────────  ─┬──────────────┬─────
                        burst 1, 22 tokens            burst 2, 22 tokens

   sam.boundary_after = [F, F,  F, ..., F,  F, F, F, T, F,  F, ..., F,  F, F, T]
                                                    ▲                          ▲
                                              boundary at last position
                                              of burst 1 (l_4 of sample 1)
                                                                          boundary at
                                                                          last position
                                                                          of burst 2
```

The `<SAMPLE>=0` indicator at the start of burst 2 is *visible to the SAM*
(it's an integer in `sam.text`), but a deterministic prediction that would
read across the boundary mark (burst 1's `l_4` → burst 2's `<SAMPLE>`) is
forbidden — the suffix walk falls back to a shorter context.

This split design comes directly from the upstream rosaplus.py "boundary-aware
SAM" implementation; we kept it because it keeps ROSA's per-sample structure
clean while still letting the indicator IDs participate in transition counts.

---

## 5. What ROSA accumulates (training time)

ROSA's "training" is just appending. Every sample's burst is concatenated
onto a master stream, with `mark_boundary()` at the end of each burst:

```
   sample 1 burst:   [0, s¹_1..s¹_16, 1, l¹_1..l¹_4]│   ← │ = boundary mark
   sample 2 burst:   [0, s²_1..s²_16, 1, l²_1..l²_4]│
   sample 3 burst:   [0, s³_1..s³_16, 1, l³_1..l³_4]│
                                              ...
   sample N burst:   [0, sᴺ_1..sᴺ_16, 1, lᴺ_1..lᴺ_4]│

   master stream length:  N × 22 = 60_000 × 22 = 1_320_000 integers
   SAM size (states):     ~ same order of magnitude (1.3M – 40M depending
                            on how many epochs we've recorded — ROSA
                            never forgets, it grows monotonically)
```

The SAM construction is online (Ukkonen-style) so each `feed()` call is
O(1) amortized. Witten-Bell counts are filled with a single sweep over
the examples after all bursts are committed, then propagated up the
suffix-link tree in one pass.

---

## 6. ROSA at query time (the read interface)

Given any prefix of a stream, ROSA walks its SAM and returns a prediction.
We use it at *every* position of every burst:

```
   query prefix    = stream[:p+1]    (everything observed up to and including position p)
   walk SAM        : start at root → traverse with each token → land in state v
   predict_at(v)   : two modes
                     ├─ SAM-deterministic   (the "rightmost-match" rule)
                     │     "what token came after the most recent occurrence
                     │      of the longest suffix of the prefix in training?"
                     │     → returns a single integer ID
                     │
                     └─ Witten-Bell LM      (smoothed n-gram fallback)
                           P(t | state v) = Σ chain   λ · count(t | ancestor)
                           → returns a dict {token_id: probability}
```

### Example walk-through

Suppose sample i's prefix has been streamed up to position 16:

```
   prefix = [0, s_1, s_2, ..., s_16]      (17 integers)
```

Query ROSA at position 16:

- ROSA walks SAM with these 17 ints → state v
- SAM-deterministic returns `predicted_next = 1` (the `<LABEL>` indicator),
  because every training burst has `<LABEL>` after the last sample-token —
  the rightmost match is *always* followed by ID 1.

Query at position 17 (after `<LABEL>`):

- ROSA's job: predict the next token, which should be l_1
- SAM-deterministic returns a content-token from `{12..1023}`,
  specifically whichever label-token followed `<LABEL>` in the training
  burst whose suffix-match was rightmost.
- This is a class signal if the codebook is class-discriminative; noise if
  not. (See Phase A/B reports — for Fashion-MNIST with our learned codebook
  it's mostly noise.)

---

## 7. How ROSA's output is consumed downstream

Two settings explored in this project:

### A. Joint pipeline (exp1, exp6, exp10–exp11d)

At every position p of every batch sample, ROSA's predicted-token-ID is
re-embedded via the codebook and concatenated into the output module's input:

```
                       ┌─────────────────────────────────┐
                       │           OUTPUT MODULE          │
                       │  (small MLP → 11-class logits)   │
                       └────────────────┬────────────────┘
                                        │ feats[p] of dim (W+1)·D = 9·D = 576
                           ┌────────────┴────────────┐
                           │                         │
               ┌───────────────────────┐   ┌────────────────────────┐
               │  WINDOW: last W=8     │   │  ROSA's prediction at  │
               │  token embeddings of  │   │  position p, then re-  │
               │  the current burst    │   │  embedded:             │
               │  (padded if p < 8)    │   │  rosa_pred_emb[p] =    │
               │                       │   │   codebook[ROSA(p)]    │
               │  shape: (W, D)        │   │  (stop_grad in exp1,   │
               └───────────────────────┘   │   no stopgrad in exp6, │
                                           │   N/A for ES variants) │
                                           │  shape: (D,)           │
                                           └────────────────────────┘

  Supervision target at each position p:
    p == 17 (right after <LABEL>):  target = digit class y    (0..9)
    p ≠  17:                         target = NO_OUTPUT       (class 10)

  Loss:  CE_no_output_positions.mean()
       + λ_digit_boost · CE_digit_position.mean()    (λ_digit_boost = 20)
```

### B. Decoupled pipeline (Phase A)

In the decoupled experiments, ROSA's output **is** the prediction — no
separate output module. The label is stored in ROSA as a single
reserved-ID token at the end of each burst (no ACT-encoded label tokens),
and at test time we read off ROSA's distribution over digit IDs:

```
  Stage 2 (build): per training sample, commit the burst
     [<SAMPLE>=0, s_1..s_16, <LABEL>=1, y_token = y+2]    ← y_token ∈ {2..11}

  Stage 3 (test, zero parameters):
     test image x' ──► frozen input MLP ──► s'_1, ..., s'_16
     query prefix = [0, s'_1, ..., s'_16, 1]
     walk SAM with these 18 ints → state v'
     dist = ROSA.lm.probs_for_state(v')

     digit_probs = [dist[2], dist[3], dist[4], dist[5], dist[6],
                    dist[7], dist[8], dist[9], dist[10], dist[11]]
                     digit 0  digit 1   ...                  digit 9

     predicted_digit = argmax(digit_probs)
```

---

## 8. Training protocol summary

| Property | Value |
|---|---|
| ROSA training | continual; SAM grows monotonically across all training epochs |
| ROSA at eval | frozen for the test-set evaluation (no test tokens committed) |
| ROSA per minibatch | queried with the *current* SAM state for all batch samples (frozen per-batch); after the optimizer step, the batch's tokens are committed |
| Witten-Bell LM | built once after all bursts are committed (decoupled Phase A); not used in the joint experiments where SAM-deterministic + unigram fallback is sufficient |
| `max_order` | 1,048,576 (effectively unbounded — the SAM never truncates) |

---

## 9. Asymmetry to notice

**ROSA's input is structured** (we control what bytes go into the stream),
**but ROSA's output is whatever-it-can-extract from suffix statistics.**

In the joint pipeline, that output is *one of many features* the neural net
can use or ignore. In the decoupled pipeline, it IS the prediction. The
experimental matrix in Phases A and B covers this whole spectrum; the
findings are summarized in the
[Phase B report](https://media.tanh.xyz/seewhy/26-05-14/rosa_report_phaseB.html).
