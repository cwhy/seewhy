# small_lm — Setup Progress Report

**Date:** 2026-04-07

## What was built

JAX port of [GuppyLM](https://github.com/arman-bd/guppylm), a minimal 8.7M-parameter
vanilla transformer trained on synthetic fish-persona conversations.

### Data pipeline

- `lib/dataset.py` — faithful port of GuppyLM's `generate_data.py`. 60 conversation
  topics (greeting, food, temperature, water quality, philosophy, ...) with template
  composition. Generates ~60K samples with 19% unique outputs.
- `lib/data_processing.py` — BPE tokenizer training (`tokenizers` library) and
  `load_tokens()` returning padded `(N, 128)` int32 arrays.
- Generated: 57K train / 3K eval samples. Tokenizer converged at **2375 tokens**
  (vs 4096 target — our corpus is less character-diverse than the original).
  Functionally fine: unused embedding rows receive no gradients.

### Model (`exp_baseline.py`)

Direct port of GuppyLM's vanilla transformer:

| Hyperparameter | Value |
|---|---|
| Parameters | **8,726,016** (8.73M) |
| Layers | 6 |
| d_model | 384 |
| Heads | 6 (head_dim=64) |
| FFN hidden | 768 (ReLU) |
| Context window | 128 tokens |
| Positional encoding | Learned |
| LM head | Weight-tied with token embeddings |
| Dropout | None |

Implementation notes:
- Flat dict params (`block_0_wq`, `block_1_wq`, ...) — consistent with project style
- Pre-norm architecture (LayerNorm before attention/FFN)
- Causal mask: module-level `jnp.tril` constant, sliced per sequence length
- Loss: CE with PAD_ID=0 masking on targets
- Optimizer: AdamW + gradient clip + cosine LR with 200-step linear warmup

### Forward pass validation (`check_forward.py`)

All 13 checks passed:

```
[✓] param count in [8M, 10M]     (8,726,016)
[✓] output shape                  (B, T, 4096)
[✓] no NaN in logits
[✓] no Inf in logits
[✓] logit mean ≈ 0                (0.0079)
[✓] CE ≈ log(VOCAB_SIZE)          (8.05 vs 8.32, 3.3% off)
[✓] token_embed / pos_embed present
[✓] logit last dim == VOCAB_SIZE
[✓] causal mask shape correct
[✓] lower triangle True
[✓] upper triangle False
[✓] diagonal True
[✓] loss_fn returns finite scalar (8.40)
```

## Training results — exp_baseline

| Metric | Value |
|---|---|
| Final eval loss | 0.3794 |
| Final eval PPL | **1.46** |
| Training time | 128s (RTX 4090) |
| Steps | 10,000 |

PPL of 1.46 indicates near-memorisation. Expected: the corpus has only 2,375 unique
tokens and 19% unique outputs — it's a template-based dataset with very limited
surface diversity. The model (8.7M params) is substantially overparameterised for it.

### Generation samples (temp=0.7, top_k=50)

```
User:  hi guppy
Guppy: hello. are you the big shape that feeds me. the bubbles feels nice.

User:  how are you feeling
Guppy: i'm ok. fish don't feel much i think. but i feel happy.

User:  are you hungry
Guppy: i was just thinking about food. i've been racing the current waiting for this moment. i promise to eat all of it.

User:  it's hot today
Guppy: too warm is not good. warm water has less oxygen. can you do something about it.

User:  what's the meaning of life
Guppy: food. the answer is always food.

User:  what do you think about cryptocurrency
Guppy: is cryptocurrency something that lives in water. can we talk about daphnia instead.
```

Character is consistent and coherent. The low PPL reflects dataset simplicity, not
model failure.

## What's next

- [x] Launch `exp_baseline` training — done (128s, PPL=1.46)
- [x] Verify generation is coherent — done
- [ ] Sweep: smaller models (2L/4L), longer context (256), higher LR
- [ ] `gen_report_baseline.py` — learning curves + generation samples report
