# Experiment Workflow — Recall-Gen

Does a model trained **only to retrieve from context** learn anything that
generalises? Each MNIST image is one token; a KDA linear RNN carries the context
in a matrix-valued state; the model completes a masked query image.

For algorithmic details (task, model, loss, metric definitions) see
[concepts.md](concepts.md).

---

## Directory Layout

```
projects/recall-gen/
├── workflow.md             # This file
├── concepts.md             # Task, model, metrics, findings
├── experimentsN.py         # One file per experiment — sets a `Run`, calls `run()`
├── results.jsonl           # Append-only results log
├── logs/                   # Experiment and runner logs (gitignored)
├── reports/*.md            # Intermediate findings
├── paper/                  # The deliverable
├── lib/
│   ├── core.py             # KDA model, episode assembly, metrics
│   ├── evalsets.py         # The fixed 2x2 eval episodes
│   ├── train.py            # `Run` dataclass + the shared train/eval driver
│   ├── figures.py          # Figure specs for both report tiers
│   └── viz.py              # Matplotlib pixel figures
└── scripts/
    ├── run_experiments.py  # Launch and manage runs  ← always use this
    ├── baselines.py        # Model-independent reference points (own JSONL row)
    ├── poll_result.py      # Wait for / display results
    ├── gen_report.py       # Regenerate paper figures
    └── tmp/                # Throwaway scripts (not committed)
```

## Project-specific notes

**Experiment files are thin.** All machinery is in `lib/`; an experiment file
sets a `Run` and calls `run()`. Two runs then differ only by what their `Run`
says they differ by, which is the point — nearly every experiment here is a
controlled pair.

**`sys.path` order matters.** The repo root must be **appended**, not inserted.
The GPU box has an untracked `datasets.py` sitting at its repo root (it is not in
git and not on the Mac). Inserting the root at position 0 makes that file shadow
the HuggingFace `datasets` package `shared_lib.datasets` imports, and every run
dies on a circular import. Appending puts site-packages first and the problem
disappears.

**Baselines live in their own row.** They do not depend on a trained model, so
`scripts/baselines.py` writes `baselines_M{M}_r{rows}` once per task shape
rather than recomputing per experiment.

**The scan is kernel-launch bound.** Sequence length is ~20 tokens, so step time
is nearly flat in batch size: batch 256 costs ~1.2x batch 64 and does 4x the
work. Use large batches. A 12 000-step run takes about five minutes.

---

## Running Experiments

```bash
uv run python projects/recall-gen/scripts/run_experiments.py --bg --parallel exp1 exp2
uv run python projects/recall-gen/scripts/run_experiments.py --bg exp3
uv run python projects/recall-gen/scripts/baselines.py --M 16 --rows 14
```

Smoke-test any experiment first with `SMOKE=1` (200 steps, 64 eval episodes).

Logs go to `projects/recall-gen/logs/{exp_name}.log`.

---

## Invariant Rules

1. **Always `uv run --no-sync python`** on the GPU box — a bare `uv run` can
   silently upgrade packages and break CUDA. `uv` is not on the
   non-interactive-ssh PATH; call `~/.local/bin/uv`.
2. **Always use `run_experiments.py`** to launch experiments.
3. **Throwaway scripts go in `scripts/tmp/`** — never `python -c "..."` inline.
4. **No bash polling loops** — read the log once, or write a Python script.
5. **Never rsync `results.jsonl` up to the box.** Results flow box -> Mac only.

```bash
# push code
rsync -az --exclude 'logs/' --exclude '__pycache__/' --exclude '*.pkl' \
  --exclude 'results.jsonl' \
  projects/recall-gen/ 195.133.135.186:Projects/seewhy/projects/recall-gen/
# pull results
rsync -az 195.133.135.186:Projects/seewhy/projects/recall-gen/results.jsonl \
  projects/recall-gen/
```

---

## Reports and the paper

Two tiers, different contracts — see
[`../TEMPLATES/paper_checklist.md`](../TEMPLATES/paper_checklist.md).

| | `reports/*.md` | `paper/` |
|---|---|---|
| audience | you, next week | someone who has never heard of the project |
| when | as results land | republished throughout |

```bash
uv run python -m shared_lib.publish projects/recall-gen/paper --check
uv run python -m shared_lib.publish projects/recall-gen/paper --stable
```

Plots go to R2 as URLs, never base64.
