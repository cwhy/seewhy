# dust-history — Workflow

Reuses `../dust` (model, data, DUST's estimator pieces): `history/__init__.py` puts that project first on the path, which is
why this project's package is not called `lib`. Results go to this project's own `results.jsonl`.

```
projects/dust-history/
├── proposal.md, concepts.md, workflow.md
├── history/estimator.py    # one noisy pass, a nudge at every site; DUST's local rule for q, k, v, gate, ve
├── history/train.py        # window of the last N estimates; consensus rules mean | agree; results.jsonl
├── experiments1.py         # 1M tokens, N in {1,4,16} x learning rate
├── scripts/check_estimator.py   # cosine to backprop of the N-average, at fixed weights
├── scripts/gpu_queue.py    # runs experiments only on an idle GPU (the box is shared)
└── results.jsonl           # one row per run (gitignored data stays on the box)
```

Run on the box, never `uv sync`:

```bash
rsync -a --exclude results.jsonl --exclude logs projects/dust-history/ 195.133.135.186:Projects/seewhy/projects/dust-history/
ssh 195.133.135.186 'cd Projects/seewhy && setsid -f ~/.local/bin/uv run --no-sync python projects/dust-history/scripts/gpu_queue.py exp1'
```

`SMOKE=1 CUDA_VISIBLE_DEVICES= ... experiments1.py` runs a 2-layer model for 4 updates on CPU to check the plumbing.
