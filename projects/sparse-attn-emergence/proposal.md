# Sparse-Attention Emergence — Proposal (small-scale replication, v1)

Small-scale replication of **"Emergent Capabilities Arise Randomly from Learning
Sparse Attention Patterns"** (Baherwani, Chen, Qiu, Wilson, Izmailov; NYU,
arXiv:2606.25010). Synthetic half only — the Pythia / IOI half is out of scope.

Paper: <https://arxiv.org/abs/2606.25010>

## Premise

Downstream capabilities in transformers appear **abruptly and at random times**
across seeds, even while pretraining loss falls smoothly. The paper's mechanistic
claim: each capability needs one **sparse, task-relevant attention pattern**, and
finding that pattern by SGD is a hard search — the model sits on a plateau at the
marginal-entropy loss until a head snaps from near-uniform attention onto the
correct sparse support, at which point loss drops in a few hundred steps.

Synthetic tasks isolate this: the required attention pattern is **known by
construction**, so "did the capability emerge" and "did the head find the right
support" are both directly measurable.

## Claims under test

| ID | Claim | Measured by |
|----|-------|-------------|
| **H1** | Emergence is abrupt and its timing is seed-random, not a smooth function of step | per-seed loss curves; spread of time-to-emergence over ≥16 seeds |
| **H2** | Difficulty is non-monotone in sparsity `s` and grows with context `S` — a "hard window" at intermediate `s` that is *unlearnable* in budget | solve-rate heatmap over (S, s) |
| **H3** | The loss jump **is** the attention pattern being found | attention entropy + support-IoU vs the ground-truth row of `A`, aligned to the jump; ablation of the winning head |
| **H4** | More heads help; head dimension saturates past a minimum capacity | H-sweep at fixed `D`; separate head-dim sweep at fixed `H` |
| **H5** | A non-attention mixer learns the linear map *faster* than attention | causal MLP-Mixer vs transformer, matched params/data |

## Tasks

### Linear map (exps 1–4, 6)

`A ∈ {0,1}^{S×S}` sampled once per run with **exactly `s` nonzeros per row**;
transition `f(x) = Ax mod 2`. Per sequence: `x₀ ~ U{0,1}^S`, `x₁ = f(x₀)`, tokens
are `concat(x₀, x₁)`, length `S·T` with `T = 2`, vocab `C = 2`. Autoregressive
next-token prediction.

- `A` is **fixed per run** → learned in-weights. (In-context variant, where `A`
  changes per sequence, is a possible exp7; not planned.)
- The first half is i.i.d. uniform, so its CE is exactly `ln 2` and carries no
  signal. **All metrics use second-half tokens only** (positions `S…2S−1`);
  full-sequence loss is logged too, for comparability with the paper's figures.
- Predicting position `S+i` requires attending to exactly the `s` positions in
  `row i` of `A` — the ground-truth attention support, known in advance.
- Batch size scales as `1/S` so tokens-per-step `S·T·B` is constant across `S`
  (paper's protocol).

### Cellular automata (exp5)

Lookup table `R : {0,…,C−1}^W → {0,…,C−1}` sampled per run, `W = 3`, `C = 4`.
Next state: `x_{t+1}[i] = R^{(k)}(x_t[i−1], x_t[i], x_t[i+1])` — `R` composed `k`
times per transition, so the required attention span is `2k+1` wide. Trajectory
of `T = 16` states, `S` cells, flattened to `S·T` tokens, AR next-token
prediction. Plateau value is `ln 4 ≈ 1.386`.

*Ambiguity resolved (2026-08-04).* Appendix B.1: **"N: Number of rules; one rule is
sampled per training example"**. So `N = 256` tables are drawn once per run and each
example uses one of them, iterating `x_{t+1} = r^k(x)` from a random initial state.

This matters more than a config detail: the linear map has one `A` per run, learned
into the weights, while the CA task hands the model a *different* rule per sequence.
exp5 therefore tests emergence of an **in-context** sparse-attention circuit — the
model must identify the active rule from the sequence — where exp1–exp4 test an
in-weights one. State size is unstated in the paper; use `S = 16` and record it as ours.

## Models

Paper defaults, kept: **linear map** — 1 layer, `D = 128`, MLP 512, `H = 8`
heads; **CA** — 4 layers, `D = 128`, `H = 8`. Learned positional embeddings,
causal mask, AdamW, 10,000 steps. LR/warmup are unspecified in the paper — pin
`3e-4` + 200-step warmup and record the choice as ours.

**exp6 mixer:** token-mixing MLP replacing attention, with the mixing matrix
**masked lower-triangular** so the model stays causal (a vanilla Mixer would leak
the future and the comparison would be meaningless). Params matched to the
transformer within ~10%.

## Metrics

- `loss2` — second-half CE in nats. Plateau `ln 2 = 0.693` (`ln 4` for CA) = total
  failure; `→0` = solved.
- `acc2` — second-half exact-token accuracy. **Solved** ⇔ `acc2 > 0.99`.
- `t*` — time-to-emergence: first step with `acc2 > 0.95`. Runs that never reach
  it are **censored**, not dropped — reported as solve-rate + survival curve.
- `solve_rate` — fraction of seeds solved within 10k steps. This is the H1/H2
  observable; a mean loss curve hides it.
- `attn_entropy` — `−Σ_j s_ij log s_ij` per head per query row, over training.
- `support_IoU` — IoU between the top-`s` attended positions at query `S+i` and
  the true support of `row i` of `A`. This is the paper's qualitative
  before/after attention map, made scalar and trackable per step.

## Experiment stages

Run one at a time; each stage's result decides the next stage's configs.

**exp1 — baseline & stochasticity (H1).** Paper defaults `S=16, s=3`, 16 seeds.
Expect: a plateau at `ln 2` broken at widely scattered steps, some seeds
uncensored only late. Deliverable: per-seed curves, `t*` histogram.

**exp2 — sparsity × context sweep (H2).** `S ∈ {8,16,32}` × `s ∈ {1,2,3,4,6,8,12,16,24,32}`
(`s ≤ S`) × 16 seeds. Expect: `S=8` solvable at every `s`; a hard window at
intermediate `s` for `S ∈ {16,32}` (both `s=1` and `s=S` are easy — one position,
or attend-to-everything). Deliverable: solve-rate and median-`t*` heatmaps.

**exp3 — heads vs head dimension (H4).** On a *hard but not impossible* config
from exp2. (a) `D=128` fixed, `H ∈ {1,2,4,8,16,32,64,128}`, head dim `128/H`.
(b) `H=8` fixed, head dim `∈ {1,2,4,8,16,32,64}` (width varies) — this second
sweep is what separates "more search attempts" from "more capacity"; the paper
conflates them under fixed `D`. Expect: solve-rate rising monotonically in `H`,
including `H=128, d_h=1`; head dim flat past a small threshold.

**exp4 — mechanism (H3).** Instrument exp1's seeds: log `attn_entropy` and
`support_IoU` every 25 steps, dump attention maps at `t*−Δ` and `t*+Δ`. Then
zero-ablate the head with the highest `support_IoU` at the end of training and
check `loss2` returns to the plateau. This is the causal step that turns a
correlation into the paper's claim.

**exp5 — cellular automata.** 4 layers, `C=4, T=16, W=3`, `k ∈ {1,2,3}`, 8 seeds.
Checks the effect is not linear-map-specific and that difficulty tracks the
`2k+1` span. Most expensive stage (longer sequences, 4 layers).

**exp6 — architecture comparison (H5).** Causal MLP-Mixer vs transformer on
identical linear-map data, on an exp2 config where the transformer *fails*.
Expect the mixer to solve what attention cannot. Optional stretch: a linear-RNN
baseline; skip Mamba/RWKV/xLSTM/GatedDeltaNet (paper has 6 — not small-scale).

## Efficiency: vmap over seeds

Models are ~200k params and sequences are ≤32 tokens, so a single seed does not
fill a 4090. Train **all seeds of a config simultaneously** by `jax.vmap`-ing the
init + scanned epoch function over a leading seed axis (configs with identical
shapes can be stacked too). 16 seeds should cost close to one. Because H1/H2 are
*seed-distribution* claims, this is what makes 16-seed statistics affordable
where the paper reports 3.

Precondition on this design: `jax.lax.scan` over steps per `workflow.md`, and no
large arrays closed over inside `jit`.

## Compute budget

Per config (16 vmapped seeds, 10k steps): minutes, not hours. exp1 ≈ 5 min;
exp2 ≈ 30 configs ≈ 1–2 h split across the two GPUs; exp3 ≈ 15 configs; exp4
reuses exp1 with denser logging; exp5 is the long one (~4× cost/step, longer
sequences) — budget an evening; exp6 ≈ exp1 × 3. Whole project fits in a day of
wall-clock on the 4090 pair, run stage by stage.

## Infrastructure

Set up per `projects/TEMPLATES/project_start.md` (template **v1**): `lib/`
(`tasks.py` for both generators, `models.py` for transformer + mixer, `viz.py`),
`scripts/run_experiments.py`, `scripts/poll_result.py`, append-only
`results.jsonl` as the only committed output, `logs/` and `*.pkl` gitignored.

Every `results.jsonl` row: full hyperparameters, `n_params`, `time_s`, and the
per-seed curves (`loss2`, `acc2`, `attn_entropy`, `support_IoU`) so every plot in
the report can be regenerated without re-running.

### Remote execution — verified

Host `195.133.135.186` (`owner-CROV`): **2× RTX 4090 24 GB**, checkout at
`/home/newuser/Projects/seewhy` on `main` at `15cffae` (in sync with local),
`.venv` healthy — JAX 0.8.1 sees both CUDA devices — and `.env` with R2 creds
present, so figure upload works from the remote.

Two gotchas, both confirmed by hand:
- `uv` is **not** on the non-interactive-ssh `PATH` — call `~/.local/bin/uv`.
- Per the GPU warning in `CLAUDE.md`, use **`uv run --no-sync`** so a run can
  never silently upgrade packages and break CUDA. Verified working.

Loop: edit locally → `rsync` the project dir up → launch detached via
`run_experiments.py --bg` over ssh → read `logs/` and `results.jsonl` back.

```bash
rsync -az --exclude 'logs/' --exclude '__pycache__/' --exclude '*.pkl' \
  projects/sparse-attn-emergence/ \
  195.133.135.186:Projects/seewhy/projects/sparse-attn-emergence/
ssh 195.133.135.186 'cd Projects/seewhy && ~/.local/bin/uv run --no-sync python \
  projects/sparse-attn-emergence/scripts/run_experiments.py --bg exp1'
```

`results.jsonl` comes back by `rsync` and is committed from the Mac.

## Deliverables

1. This proposal + `workflow.md` + `concepts.md` (task math, metric definitions).
2. exp1–exp6 with committed `results.jsonl`.
3. One report per milestone (after exp2, after exp4) and a **final report**
   published to R2 via `shared_lib.report` — a claim-by-claim verdict table
   (H1–H5: replicated / partially / not), plots as R2 URLs, and an explicit
   deviations section (16 seeds vs 3, pinned LR, single-`R` CA reading, 2
   architectures instead of 7).

## Open questions

- **Emergence threshold.** `acc2 > 0.95` is a judgement call; the paper uses
  argmax-correctness of the full continuation. If curves turn out threshold-
  sensitive, report `t*` at three thresholds instead of arguing about one.
- **`N = 256` in the CA config** — see the ambiguity note above.
- **Unlearnable vs slow.** "Unlearnable" is budget-relative. If the hard window
  in exp2 is sharp, spend one extra run at 100k steps on its center to check
  whether it is a wall or just a longer wait.
