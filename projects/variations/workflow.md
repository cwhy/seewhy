# Experiment Workflow — variations

Controllable autoencoder where 8 binary bits steer the decoder toward nearby
variations of the input — fully self-supervised via IMLE-style stop-gradient
nearest-neighbour matching in the pre-computed kNN graph.

---

## Directory Layout

```
projects/variations/
├── workflow.md             # This file
├── experiments1.py         # exp1: MLP encoder + gram decoder, MSE loss
├── results.jsonl           # Append-only results log
├── params_expN.pkl         # Saved model params (gitignored)
├── history_expN.pkl        # Per-epoch training history (gitignored)
├── logs/                   # Experiment and runner logs (gitignored)
├── lib/
│   └── __init__.py
└── scripts/
    ├── run_experiments.py
    ├── poll_result.py
    └── tmp/
```

---

## Algorithm

1. **Encoder**: MLP(784 → 512 → LATENT_DIM)
2. **Decoder**: gram-based MLP, input = [z, b] where b ∈ {0,1}^8
3. **Reconstruction loss**: decode([z, 0⁸]) vs x/255  (MSE)
4. **Variation loss**: b ~ Bernoulli(0.5), find j = argmin||decode([z,b]) − kNN_j||, MSE vs stop_grad(kNN_j)
5. Total loss: L_recon + α * L_var  (α=1.0)

kNN pre-computed once at startup (64-NN over all 60k training images).

---

## Running Experiments

```bash
# Fire and forget
uv run python projects/variations/scripts/run_experiments.py --bg exp1

# Sequential blocking
uv run python projects/variations/scripts/run_experiments.py exp1

# Check progress
tail -20 projects/variations/logs/exp1.log

# Poll for result
uv run python projects/variations/scripts/poll_result.py exp1
uv run python projects/variations/scripts/poll_result.py --all
```

---

## Invariant Rules

1. Always `uv run python` — never raw `python` or `python3`.
2. Always use `run_experiments.py` to launch experiments.
3. Throwaway scripts go in `scripts/tmp/` — never `python -c "..."`.
4. No bash polling loops — use `poll_result.py` or read logs once.
