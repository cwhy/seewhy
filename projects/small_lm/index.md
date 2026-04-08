# small_lm

JAX port of [GuppyLM](https://github.com/arman-bd/guppylm) — a tiny 8.7M-parameter
vanilla transformer trained on synthetic fish-persona conversations.

The goal is to study small-LM training dynamics in a fast, sweepable environment:
architecture depth, context length, learning rate, and data diversity. All experiments
run in under 3 minutes on a single GPU.

---

## Reports

| Report | Description |
|--------|-------------|
| [exp_baseline](https://media.tanh.xyz/seewhy/26-04-07/small_lm_report_baseline.html) | Direct GuppyLM port — 6L, d_model=384, 10K steps. Baseline result: eval_ppl=1.46 |
| [optimizations](https://media.tanh.xyz/seewhy/26-04-07/small_lm_report_optimizations.html) | Muon, RoPE, separate LM head vs baseline. Pareto-front analysis. |
| [ai-eval](https://media.tanh.xyz/seewhy/26-04-07/small_lm_report_ai_eval.html) | Conversational eval: LLM-judged fish-persona quality across 5 rounds, 60 questions. |
| [k-Dyck pretraining](https://media.tanh.xyz/seewhy/26-04-07/small_lm_report_kdyck.html) | Procedural warm-up on balanced brackets before Guppy fine-tuning. |

---

## Project scope

**What this is:**
- A minimal, readable JAX transformer — no frameworks, flat parameter dicts, ~200 lines of model code
- A sweepable testbed: one file per experiment, results append to `results.jsonl`
- A faithful port of GuppyLM's architecture (vanilla transformer, no RoPE/GQA/SwiGLU)

**What this is not:**
- A production LM — context window is 128, vocab 2375 tokens, dataset is synthetic
- A benchmarking effort — eval metric is perplexity on held-out templates, not real-world tasks

**Planned experiments:**
- `exp_2l`, `exp_4l` — shallower models; find the Pareto point for this dataset
- `exp_ctx256` — longer context window (256 tokens)
- `exp_lr_sweep` — learning rate sensitivity
- `exp_data_diverse` — richer template variation to raise effective dataset difficulty

---

## Setup

### 1. Generate data and train tokenizer (once)

```bash
uv run python projects/small_lm/scripts/tmp/setup_data.py
```

Writes `data/train.jsonl` (57K), `data/eval.jsonl` (3K), `data/tokenizer.json`.

### 2. Verify forward pass

```bash
uv run python projects/small_lm/scripts/tmp/check_forward.py
```

Runs 13 numerical checks: shape, NaN/Inf, initial CE ≈ log(4096), causal mask, loss_fn.
All should print `[✓]`.

### 3. Run an experiment

```bash
# Background (recommended)
uv run python projects/small_lm/scripts/run_experiments.py --bg exp_baseline

# Check progress
tail -f projects/small_lm/logs/exp_baseline.log

# Poll for result
uv run python projects/small_lm/scripts/poll_result.py exp_baseline
```

### 4. Generate the report

```bash
uv run python projects/small_lm/scripts/gen_report_baseline.py
uv run python projects/small_lm/scripts/tmp/upload_report_baseline.py
```

---

## Sample generation

Generate text from any trained checkpoint:

```bash
uv run python projects/small_lm/scripts/tmp/generate_samples.py
uv run python projects/small_lm/scripts/tmp/generate_samples.py --exp exp_baseline --temp 0.7 --topk 50
```

Sample output (exp_baseline, temp=0.7, top_k=50):

```
User:  hi guppy
Guppy: hello. are you the big shape that feeds me. the bubbles feels nice.

User:  are you hungry
Guppy: i was just thinking about food. i've been racing the current waiting
       for this moment. i promise to eat all of it.

User:  what's the meaning of life
Guppy: food. the answer is always food.

User:  what do you think about cryptocurrency
Guppy: is cryptocurrency something that lives in water. can we talk about
       daphnia instead.

User:  goodnight guppy
Guppy: nighty night. i'll settle down in my favorite spot.
```

---

## File structure

```
projects/small_lm/
├── index.md                     # This file
├── workflow.md                  # Invariant rules, JAX patterns, results logging
├── results.jsonl                # Append-only experiment results (committed)
│
├── data/
│   ├── tokenizer.json           # BPE tokenizer, 2375 tokens (committed)
│   ├── train.jsonl              # 57K samples (gitignored)
│   └── eval.jsonl               # 3K samples (gitignored)
│
├── pkl/                         # Saved params + history (gitignored)
│
├── lib/
│   ├── dataset.py               # Synthetic conversation generator (60 topics)
│   ├── data_processing.py       # Tokenizer training/loading + load_tokens()
│   └── viz.py                   # Learning curve plots
│
├── exp_baseline.py              # 6L d_model=384 — direct GuppyLM port
│
├── report-baseline.md           # Experiment report (local)
├── report-setup.md              # Setup progress notes
│
└── scripts/
    ├── run_experiments.py
    ├── poll_result.py
    ├── gen_report_baseline.py   # Generates report-baseline.md + uploads plots
    └── tmp/
        ├── setup_data.py        # One-shot data + tokenizer generation
        ├── check_forward.py     # Forward pass sanity checks
        ├── generate_samples.py  # Text generation from trained checkpoint
        └── upload_report_baseline.py
```

---

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Flat dict params (`block_0_wq`, ...) | No pytree registration; easy to inspect |
| Pre-norm (LayerNorm before attn/FFN) | Same as GuppyLM; more stable training |
| ReLU FFN (no SwiGLU) | Direct port; simplicity first |
| Weight-tied LM head | `logits = h @ token_embed.T` — saves ~1.5M params |
| No dropout | Simplicity; add if underfitting observed |
| Step-based training | Matches GuppyLM's cosine+warmup schedule |
| PAD_ID=0 masked in CE loss | Correct loss on variable-length sequences |
