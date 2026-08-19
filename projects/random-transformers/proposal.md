# Random Transformers — Proposal (replication, v1)

Replication of **"Algorithmic Capabilities of Random Transformers"** (Ziqian
Zhong, Jacob Andreas; MIT, NeurIPS 2024, arXiv:2410.04368).

Paper: <https://arxiv.org/abs/2410.04368> · Authors' code:
<https://github.com/fjzzq2002/random_transformers>

Reimplemented from the paper text in JAX; the authors' PyTorch/HuggingFace code
was not consulted, so agreement is evidence about the paper rather than about a
shared codebase.

## Premise

A trained transformer that does modular arithmetic contains a circuit that does
modular arithmetic. The usual assumption is that gradient descent *built* that
circuit out of the training signal. The paper asks a sharper question: how much
of it was there before training started?

The probe is to freeze everything except the way tokens enter and leave the
network. Take a randomly initialised transformer, hold every attention and
feed-forward weight at its initialisation forever, and train only the token
embedding, the positional embedding, and the unembedding. Nothing inside the
model can change. If such a model still reaches perfect accuracy on a task,
then the function was already computable by the random stack — all training did
was find an input encoding that reaches it, and an output decoding that reads
it off.

The paper's answer is that this works, on seven tasks, and that it works only
when the model is wide.

## Claims under test

| ID | Claim | Measured by |
|----|-------|-------------|
| **H1** | Random transformers (embedding-only training, width 1024, 2 layers) reach ~100% on modular addition, needle-in-a-haystack, decimal addition and parenthesis balancing | test accuracy per task, median over seeds, against fully trained models of the same width |
| **H2** | The capability depends on **width**, not on training the interior: width 16 random models collapse (1.3–87.3%) while width 16 *fully trained* models do far better on three of four tasks | width sweep 16 → 1024, random vs full |
| **H3** | All three embedding matrices must be trained; no proper subset suffices across tasks | ablation: `U` only, `E` only, `E_token`+`U` only |
| **H4** | Random transformers store far less arbitrary information than trained ones — ~0.4 vs ~2.9 bits per trainable parameter | memorization task, width 128 |
| **H5** | Random transformers do language modeling badly but not trivially: width 512 random ≈ width 32 fully trained, and output stays grammatical | TinyStories cross-entropy scaling curves + sampled completions |
| **H6** | The mechanism is **subspace selection**: hidden representations concentrate in a low-dimensional subspace that is not neuron-aligned, so it is not sparsification | explained variance from the top 10 principal components vs the top 10 neurons, per layer, per task |
| **H7** | Random transformers can only imitate target circuits that themselves fit in a low-dimensional subspace, with a sharp break between 12 → 32 target dimensions | circuit imitation: KL to a random target transformer, sweeping target width |

H1–H3 are the paper's Section 4; H4–H5 Section 5; H6–H7 Section 6. H6 is the
one that would explain the rest, and H7 is the paper's own falsification test
for it — which is why both are in scope rather than being treated as optional
analysis.

## Deviations from the source, declared up front

The paper's own budgets are large (10⁴ steps × batch 1000 per synthetic run,
ten seeds, several widths, plus a full TinyStories sweep). Two 4090s are not
the authors' cluster, so:

- **Seeds.** 5 per cell for headline tables and 3 for sweeps, against the
  paper's 10. Medians over 5 are noisier; every table reports the spread.
- **Budget.** Step counts are set from measured convergence, not copied. Where
  a run is cut short of the paper's budget the row records both.
- **Language modeling.** A subset of TinyStories rather than 5 epochs over all
  of it, with the token budget held fixed across every architecture in the
  sweep so the comparison inside the figure stays honest.
- **Parenthesis data.** The paper's generate-then-mutate sampler is recursive
  and does not vectorise under `jit`, so a fixed pool of 500k sequences is
  drawn once in NumPy and sampled from, rather than streamed. With a vocabulary
  of 4 and 80 positional embeddings there are ~84 trainable parameters' worth
  of token identity to memorise with, so a finite pool cannot be a shortcut.
- **LSTM baseline.** Included, since "beats a fully trained LSTM" is one of the
  paper's headline comparisons, but with a single-layer textbook LSTM rather
  than a tuned one.
- **Learning rate and warmup.** 3e-4 with a 500-step linear warmup, against the
  paper's 1e-3 and no warmup. Forced: at 1e-3 without warmup our fully trained
  width-1024 models never leave a plateau on needle-in-a-haystack, even at the
  paper's own budget. Applied identically in every condition. See
  `reports/exp1-budget.md`.

## An inconsistency in the source

Table 6 of the paper lists `E_token` for modular addition at 99,328
parameters. At width 1024 that implies a vocabulary of 97, but the task text
fixes the modulus at p = 199, which needs 199 token embeddings (203,776
parameters). The other three tasks' token-embedding counts divide by 1024
exactly into their stated vocabularies (256, 31, 4), so the modular-addition
row is the odd one out — consistent with that row having been produced at
p = 97, itself also prime.

We follow the **main text** (p = 199) and note the discrepancy. If the
published numbers came from p = 97, the modular-addition task we run is
strictly harder than theirs.

## Experiment stages

Each stage's outcome sets the next stage's configuration.

| Exp | Question | Claims |
|-----|----------|--------|
| exp1 | Main table: 4 tasks × {random, full} × {1024, 16} + LSTM | H1, H2 |
| exp2 | Which embedding matrices are needed | H3 |
| exp3 | Width sweep 16 → 1024, random vs full, all 4 tasks | H2 |
| exp4 | Memorization: bits per trainable parameter | H4 |
| exp5 | Subspace selection: PCA vs neuron-basis explained variance | H6 |
| exp6 | Circuit imitation across target widths | H7 |
| exp7 | TinyStories language modeling, 2 and 4 layers | H5 |

exp5 reads parameters saved by exp1 and exp4 rather than training anything, so
it cannot disagree with the models the tables describe.

## What would count as a failed replication

Not "the numbers differ". The paper's claim is qualitative and large: a network
whose entire interior is noise, that nonetheless computes. It fails if width
1024 random models do **not** separate from chance on the algorithmic tasks, or
if they succeed just as well at width 16 — the latter would mean the effect was
in the training procedure rather than in the random stack's width.
