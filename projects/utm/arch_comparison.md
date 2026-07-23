# Architectural Comparison: UTM (arxiv 2604.21999) vs TRM (arxiv 2510.04871)

Both models attack hard reasoning tasks (Sudoku-Extreme) with small recurrent transformers
and shared weights across steps. Beyond that, almost every design choice differs.

---

## 1. Core Recurrence Style

| | UTM | TRM |
|---|---|---|
| **Mechanism** | Single shared block applied K=18 times to a flat sequence | Single 2-layer network applied N_sup=16 times, updating a latent state z |
| **What recurs** | Full sequence (grid cells + memory tokens) — the whole hidden state evolves | Separate latent z_l (updated n=2× per step) and supervisor z_H (updated 1×) |
| **State representation** | Sequence of D-dim vectors: 81 grid tokens + 8 memory tokens = 89 tokens | Two scalar-shaped vectors: z_l (latent) and z_H (supervisor output) |
| **Weight sharing** | Yes — same block weights at every ponder step | Yes — same f_l weights at every recursion step |
| **Depth** | K=18 shared-weight passes | N_sup=16 supervision steps × n=2 latent updates = 32 f_l calls |

UTM is closer to a looped standard Transformer: sequence in, sequence out, loop.
TRM is closer to iterative refinement of a compressed state vector, with a two-level hierarchy (fast latent z_l, slow supervisor z_H).

---

## 2. Halting / Adaptive Computation

| | UTM | TRM |
|---|---|---|
| **Mechanism** | ACT: learned halt probability h_k at each step; soft weighting via survival × h_k | Q-learning halting head (f_continue) during training; full N_sup=16 steps at test time |
| **When it halts** | Soft — all K steps always run, output is a weighted sum of all grid_out_k | Hard — training uses early stopping; inference runs all steps regardless |
| **Halt signal** | Sigmoid over mean of memory tokens → scalar h_k per step | Binary head predicts "continue or stop"; not used at inference |
| **Ponder penalty** | λ × E[ponder depth] added to loss (λ=0 in best experiments) | No ponder penalty; instead deep supervision provides gradient signal at each step |
| **Effective depth** | ponder ≈ 18.00 (never halts early without λ > 0) | N_sup=16 at inference (fixed) |

Key insight: both models in practice use their full compute budget.
UTM's ACT is structurally soft (always runs all K steps, weights them).
TRM's ACT is only a training signal; inference is fixed-depth.

---

## 3. Memory Mechanism

| | UTM | TRM |
|---|---|---|
| **Explicit memory** | 8 learnable memory tokens appended to sequence; attend with all grid cells at every step | z_H: a single D-dim vector (previous supervisor output), used to condition z_l updates |
| **Memory update** | Implicit — memory tokens evolve through attention at every ponder step | Explicit — z_H ← f_H(z_l) once per supervision step |
| **Role** | Halt signal pooled from memory tokens; otherwise serves as global working memory | z_H acts as a "compressed summary" passed top-down to fast latent updates |
| **Analogous to** | Global registers / scratchpad in sequence | Hidden state in an RNN / hierarchical state machine |

UTM's memory is in-sequence (participates in attention).
TRM's memory is out-of-sequence (a conditioning signal fed into the next latent update).

---

## 4. Attention Architecture

| | UTM | TRM |
|---|---|---|
| **Layers per pass** | 1 block (1× MHA + 1× FFN) per ponder step | 2 layers with 4× self-attention blocks total per forward pass |
| **Attention type** | Full self-attention over all 89 tokens | Self-attention over fixed context L; replaced with MLP for L ≤ 19 (Sudoku) |
| **QK-norm** | Yes — L2-normalize Q and K per head | Not mentioned |
| **RoPE** | Yes — applied to Q and K | Not explicitly mentioned (standard learned embeddings likely) |
| **FFN** | SwiGLU | SwiGLU |
| **Normalization** | RMSNorm (pre-norm) | RMSNorm (Add & Norm) |

TRM's key finding: for Sudoku (L=81, fixed), replacing self-attention with an MLP (a single linear layer over the context) improves accuracy from 74.7% → 87.4%. Self-attention is overkill when context is short and fixed.

---

## 5. Training Supervision Strategy

| | UTM | TRM |
|---|---|---|
| **Supervision** | Single loss at final (ACT-weighted) output | Deep supervision at every supervision step — loss at each of N_sup=16 steps |
| **Gradient flow** | Full BPTT through all K=18 ponder steps (with gradient checkpointing at D=768) | Detached between supervision steps — no BPTT across steps, only within each step |
| **EMA** | Yes, decay=0.999 | Yes, decay=0.999 |
| **Optimizer** | AdamW + cosine decay + gradient clipping | AdamW, β₁=0.9, β₂=0.95 + warmup |
| **Batch size** | 64 (D=768) | 768 |
| **Loss** | XE + λ × ponder_cost | XE + optional halting CE loss (ACT training only) |

TRM's detachment of z between supervision steps is a critical engineering choice:
it avoids BPTT through 16 steps (expensive), instead treating each step as an
independent supervised example. UTM does the opposite — backpropagates through
all 18 ponder steps, relying on remat to fit in memory.

---

## 6. Training Data

| | UTM | TRM |
|---|---|---|
| **Training examples** | ~3.8M (full Sudoku-Extreme dataset) | ~1,000 (!!) |
| **Test set** | 20,000 | Not specified (standard split) |
| **Data regime** | Large-data supervised learning | Extreme few-shot generalization |

This is perhaps the biggest difference. TRM is designed for the **few-shot generalization**
regime — learning to reason from 1K examples. UTM is trained on millions of examples.
They are answering different questions: UTM asks "how well can a model fit with sufficient data?"
TRM asks "can a model generalize from almost nothing?"

---

## 7. Parameter Count & Results

| Model | Params | Training examples | Sudoku-Extreme acc |
|-------|--------|------------------|-------------------|
| UTM paper (2604.21999) | 3.2M (D=512) | 3.8M | 57.4% |
| **Our UTM (exp17, D=768)** | **7.1M** | **3.8M** | **56.44%** |
| TRM-Attention | 7M | 1,000 | 74.7% |
| TRM-MLP | 5M | 1,000 | **87.4%** |

TRM achieves 87.4% with **3,800× less training data** and similar parameter count.
This is not a fair comparison (different training regimes) but it highlights that
UTM-style models may be bottlenecked by the training objective or architecture,
not data quantity.

---

## 8. Summary of Key Differences

| Dimension | UTM | TRM |
|-----------|-----|-----|
| Recurrence target | Whole sequence (89 tokens) | Compressed latent vector z |
| State hierarchy | Flat (one shared hidden state) | Two-level: fast z_l + slow z_H |
| Memory | 8 in-sequence memory tokens | Single supervisor vector z_H |
| Attention role | Essential (full self-attention every step) | Optional — replaced by MLP for short fixed contexts |
| Halting | Soft ACT (weighted sum of all step outputs) | Hard ACT training only; fixed depth at inference |
| Ponder penalty | Optional λ term in loss | No ponder penalty |
| Gradient through steps | Full BPTT (with remat) | Detached between steps |
| Supervision | Single final loss | Deep supervision at every step |
| Training scale | Millions of examples | ~1K examples |
| Best Sudoku-Extreme | 56.44% (our run) / 57.4% (paper) | 87.4% |

---

## 9. Architectural Takeaways

**What TRM does that UTM doesn't:**

1. **Deep supervision** — each recurrence step is directly supervised, giving dense gradient signal without backpropagating through all steps. This is likely why TRM generalizes so much better from fewer examples.
2. **State compression** — reasoning over a latent vector z rather than a full sequence reduces the search space the recurrence must traverse.
3. **Two-speed updates** — z_l updates fast (n=2× per step), z_H updates slow (1× per step). This hierarchical rhythm mirrors how iterative algorithms work: fast local refinement + occasional global summary.
4. **MLP over self-attention** — for fixed-length problems, a simple linear layer over the full context outperforms attention. Attention's inductive bias is not helpful when context structure is fixed.

**What UTM does that TRM doesn't:**

1. **Soft ACT** — outputs a weighted mix of all computation depths. In principle this allows graceful degradation at any depth, though in practice ponder saturates to K=18.
2. **Positional encoding (RoPE)** — explicit position information across the sequence.
3. **In-sequence memory** — memory tokens can attend to any grid cell and vice versa, potentially richer information routing than a single z_H vector.
4. **Full BPTT** — gradients flow through all 18 steps, theoretically allowing the model to plan reasoning chains end-to-end.

**The central lesson:** TRM's deep supervision + state compression appears to be the dominant factor explaining the 30+ point accuracy gap. The UTM framework (looped sequence, soft ACT) may be fundamentally harder to train without dense per-step supervision.
