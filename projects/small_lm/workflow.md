# Experiment Workflow — small_lm

JAX port of [GuppyLM](https://github.com/arman-bd/guppylm): a tiny 8.7M-param
vanilla transformer trained on 60K synthetic fish-persona conversations.
Goal: understand small-LM training dynamics; sweep architecture, LR, context length.

---

## Directory Layout

```
projects/small_lm/
├── index.md                     # Project introduction page — always keep up to date
├── workflow.md                  # This file
├── results.jsonl                # Append-only experiment results (committed)
│
├── data/
│   ├── tokenizer.json           # Trained BPE tokenizer, 2375 effective tokens (committed)
│   ├── train.jsonl              # Generated conversations — 57K samples (gitignored)
│   └── eval.jsonl               # 3K samples (gitignored)
│
├── pkl/                         # Saved params + history (gitignored)
│   ├── params_{exp_name}.pkl
│   └── history_{exp_name}.pkl
│
├── lib/
│   ├── __init__.py
│   ├── dataset.py               # Synthetic conversation generator (60 topics, Guppy persona)
│   ├── data_processing.py       # BPE tokenizer training/loading + load_tokens → np.ndarray
│   └── viz.py                   # Learning curve + perplexity plots
│
├── exp_baseline.py              # 6L, d_model=384, ctx=128 — direct GuppyLM port
│
├── report-baseline.md           # Experiment report (local)
├── report-setup.md              # Setup progress notes
│
└── scripts/
    ├── run_experiments.py
    ├── poll_result.py
    ├── gen_report_baseline.py   # Generates report-baseline.md + uploads plot to R2
    └── tmp/
        ├── setup_data.py            # One-shot: generate data + train tokenizer
        ├── check_forward.py         # Numerical forward-pass sanity check
        ├── generate_samples.py      # Text generation from any trained checkpoint
        ├── upload_report_baseline.py
        ├── upload_index.py          # Upload index.md as HTML
        └── read_guppylm_source.py   # Utility: extract cached GuppyLM source
```

---

## Setup (run once)

```bash
# 1. Generate 60K conversations + train BPE tokenizer
uv run python projects/small_lm/scripts/tmp/setup_data.py

# 2. Verify forward pass
uv run python projects/small_lm/scripts/tmp/check_forward.py
```

---

## Invariant Rules

1. **Always `uv run python`** — never raw `python` or `python3`.
2. **Always use `run_experiments.py`** to launch experiments.
3. **Throwaway scripts go in `scripts/tmp/`** — never `python -c "..."` inline.
4. **No bash polling loops** — write a Python script instead.
5. **After every new report: update `index.md` and re-upload it.**
   Add a row to the Reports table in `index.md`, then run:
   ```bash
   uv run python projects/small_lm/scripts/tmp/upload_index.py
   ```
   The index page is the canonical entry point — it must always reflect the current
   state of the project.

---

## Running Experiments

```bash
# Background (returns immediately, logs → logs/{exp_name}.log)
uv run python projects/small_lm/scripts/run_experiments.py --bg exp_baseline

# Sequential blocking
uv run python projects/small_lm/scripts/run_experiments.py exp_baseline

# Check progress
tail -20 projects/small_lm/logs/exp_baseline.log

# Poll until result appears
uv run python projects/small_lm/scripts/poll_result.py exp_baseline
```

---

## Adding a New Experiment

1. Copy an existing `exp_*.py` as starting point.
2. Change `EXP_NAME` and the hyperparameters you want to vary.
3. Update the module docstring (hypothesis, what changed).
4. Run it via `run_experiments.py`.
5. After it finishes, generate a report and update `index.md`.

---

## Report Workflow

For each experiment `exp_foo`:

```bash
# 1. Generate local .md report (uploads plot to R2 automatically)
uv run python projects/small_lm/scripts/gen_report_foo.py

# 2. Upload .md as HTML
uv run python projects/small_lm/scripts/tmp/upload_report_foo.py

# 3. Add report link to index.md, then re-upload the index
uv run python projects/small_lm/scripts/tmp/upload_index.py
```

**Always write the local `.md` first and verify it looks right before uploading.**
Scripts follow the naming pattern:
- `scripts/gen_report_{exp_name}.py` — generates the `.md`
- `scripts/tmp/upload_report_{exp_name}.py` — converts and uploads

---

## Sample Generation

```bash
# Default: loads exp_baseline, temp=0.7, top_k=50
uv run python projects/small_lm/scripts/tmp/generate_samples.py

# Specify experiment and sampling params
uv run python projects/small_lm/scripts/tmp/generate_samples.py --exp exp_foo --temp 1.0 --topk 20
```

Loads `pkl/params_{exp}.pkl`. Data dir and tokenizer path are resolved automatically.

---

## Experiment File Structure

```
module docstring    — hypothesis, what changed, arch summary, usage line
hyperparameters     — UPPER_SNAKE_CASE constants at module top
precomputed consts  — CAUSAL_MASK
utilities           — n_params, append_result
init_params()       — flat dict, one key per parameter tensor
layer_norm()        — pre-norm LayerNorm helper
attention()         — causal MHA (no dropout)
transformer_block() — pre-norm block (attn + ReLU FFN)
forward()           — token_ids → logits, weight-tied LM head
loss_fn()           — CE with pad masking
make_train_step_fn()— JIT-compiled single step (step-based training)
eval_metrics()      — chunked eval loss + perplexity
train()             — step loop with eval checkpoints
__main__            — skip-if-done → load data → init → train → eval → append_result
```

---

## Key Design Decisions

| Choice | Reason |
|--------|--------|
| Flat dict params (`block_0_wq`, etc.) | Consistent with project style; no JAX pytree registration needed |
| Separate W_q, W_k, W_v | Clarity; same capacity as fused QKV |
| No dropout | Simplicity; add later if underfitting |
| Pre-norm (norm before attn/FFN) | Same as GuppyLM |
| ReLU FFN | Same as GuppyLM (no SwiGLU) |
| Weight-tied LM head | `logits = h @ params["token_embed"].T` — saves ~1.5M params |
| PAD_ID=0 masked in loss | CE loss only on non-pad target tokens |
| Step-based training | Matches GuppyLM's 10K-step schedule; warmup + cosine decay |
| Per-step JIT (`make_train_step_fn`) | Natural for step-based loops; recompiles only on shape change |
| pkl/ subdir for weights | Keeps project root clean; all pkl gitignored in one rule |

---

## Metrics

| Metric | Description |
|--------|-------------|
| `eval_loss` | Mean CE on eval set (pad-masked) |
| `eval_ppl` | `exp(eval_loss)` — lower is better; random init ≈ 3100 |
| `final_eval_ppl` | PPL at end of training |

Baseline result: eval_ppl=**1.46** after 10K steps (~5.6 epochs).
This reflects dataset simplicity (2375 effective tokens, 19% unique outputs),
not model quality. Future sweeps should address dataset diversity.

---

## Data Format

Conversation format (chat-ml style):
```
<|im_start|>user
hi guppy<|im_end|>
<|im_start|>assistant
hello. the water is clear today.<|im_end|>
```

Token IDs: `(N, 128)` int32, right-padded with `PAD_ID=0`.
Training pairs: `input=tokens[:, :-1]`, `target=tokens[:, 1:]`.
Effective vocab: **2375 tokens** (BPE converged below the 4096 target due to
limited character diversity in the template corpus; functionally fine).

---

## Results

| experiment | n_params | steps | eval_ppl | time_s | notes |
|------------|----------|-------|----------|--------|-------|
| exp_baseline | 8,726,016 | 10,000 | 1.46 | 128 | direct GuppyLM port |

---

## Template version

Set up with template **v1**.
