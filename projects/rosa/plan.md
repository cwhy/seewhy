# ROSA — Experiment Plan

This is the multi-experiment roadmap. Each row has a hypothesis, a concrete
change from the previous experiment, and an expected metric. See
[concepts.md](concepts.md) for the architecture and
[workflow.md](workflow.md) for how to run.

---

## Phase 0 — Build & sanity (before any "ROSA learns" claims)

| # | Goal | Action |
|---|---|---|
| **0.1** | Port ROSA core to integer tokens | `lib/rosa_core.py`: clone the SAM + Witten–Bell from `rosaplus.py`, swap `dict[str, int]` → `dict[int, int]` for transitions. Add `predict_batch`, `commit_batch`. Write a 50-line test in `scripts/tmp/test_rosa_core.py` (feed `[1,2,3,1,2,3]`, query after `[1,2]` → expect `3`). |
| **0.2** | Standalone ROSA-only baseline | `scripts/tmp/rosa_only_baseline.py`: tokenize MNIST with **frozen, untrained** input module (random codebook), feed all training tokens to ROSA, then on test set predict the digit by reading off the most-likely **label-token** ID right after `<LABEL>` and mapping label-token IDs back to digit by majority vote. **Expected: 11–25%** (slightly above chance). The "ROSA without learning" floor. |
| **0.3** | Vanilla VQ-MLP classifier (no ROSA, no NO_OUTPUT) | `scripts/tmp/vq_only_baseline.py`: same input module + STE codebook, but the output module is a single 10-way classifier fired only at `<LABEL>` (no NO_OUTPUT class, no per-position firing). **Expected: 95–98%.** The "neural without ROSA" ceiling. |
| **0.4** | NO_OUTPUT-only baseline (no ROSA, dense fire) | `scripts/tmp/dense_baseline.py`: same as 0.3 but with the headline 11-class output module (10 digits + NO_OUTPUT) firing at every position. **Expected: should match 0.3 within ±0.5pp** if `λ_digit_boost` is tuned right. Validates that the dense-fire trick on its own is not lossy. |

If 0.2 ≫ chance, ROSA carries real information from random tokens. If 0.3
hits the ceiling fast, the hard part is making ROSA *help on top of* a
working neural classifier. If 0.4 is far below 0.3, the dense-fire setup
itself is broken and the headline pipeline can't possibly work — fix
before exp1.

---

## Phase 1 — End-to-end STE training

The headline approach: jointly train input module + codebook + output module,
with ROSA as a stop-grad feature provider. Each experiment sets the next
hyperparameter knob.

| Exp | Name | Change | Hypothesis | Target |
|-----|------|--------|------------|--------|
| **exp1** | baseline-fixedK | Headline from concepts.md, with ACT *off*: fixed K=16 sample tokens, K=4 label tokens. `V=1024 D=64 W=8 SIGMA=0.01 λ_codebook=1.0 λ_ponder=0 λ_digit_boost=20.0`, DiVeQ. | Trains and beats dense baseline (0.4) by ≥0.5pp. Confirms ROSA contributes BEFORE we add ACT complexity. | ≥95.5% |
| **exp1b** | act-on | exp1 + UTM-style ACT halt head on the recurrent input MLP (variable K per sample, halt bias init = -3, no ponder cost yet). Drops "padding" tokens from ROSA stream so SAM gets variable-length bursts. | Variable token count per image gives ROSA more structure (short bursts for clean digits, long for ambiguous), should match-or-beat exp1 | ≥ exp1 |
| **exp1c** | vq-ste | **Contingent.** exp1 with VQ-STE (`SIGMA=0`, identity backward, `λ_commit=0.25`) instead of DiVeQ. Triggered only if DiVeQ misbehaves (collapse, NaNs, halt-drift). | Discretization fallback path. | within ±1pp of exp1 |
| **exp2** | tight-V256 | `V=256` (everything else as exp1) | Smaller codebook → denser SAM counts → higher `rosa_hit_rate`; output module relies on ROSA more | 94–96%, hit-rate ↑ |
| **exp3** | wide-V4096 | `V=4096`, `D=128` | More plasticity; tests "more vocab → late-stage stuff is better." May hurt early epochs (sparse counts), late-epoch curve cleaner | ≥96% |
| **exp3b** | digit-boost-sweep | `λ_digit_boost ∈ {1, 5, 20, 100}` (4 runs) | Class-imbalance correction is load-bearing; finds the right tradeoff between "always predicts NO_OUTPUT" and "fires randomly" | best within ±0.5pp of exp1 |
| **exp4** | gumbel-st | Replace DiVeQ with Gumbel-softmax-ST. Add `τ` schedule (1.0 → 0.1 cosine over training); separate `V × D` embedding table. `λ_entropy = 0.01` to fight collapse | Stochasticity helps escape codebook minima → ≥ exp1, OR DiVeQ's bias dominates and exp1 wins. Either result is informative. | within ±1pp of exp1 |

**Stop conditions for Phase 1.**
- If exp1 fails to beat 0.3 (vanilla VQ-MLP) by any margin: ROSA's contribution
  is being ignored by the output module. Investigate by lesioning ROSA at eval
  (set `rosa_feat = 0`) and measuring delta. If delta ≈ 0, the stop_grad
  pathway is dead; jump to Phase 3.
- If exp1 *exceeds* 0.3 by ≥1pp: ROSA's stop-grad feature is genuinely useful.
  Run exp1b (ACT) next to see whether variable burst length helps further.
- If DiVeQ specifically misbehaves (codebook collapse, training instability,
  ROSA-token-drift): trigger **exp1c** (VQ-STE swap). If exp1c is healthier,
  the project switches default to VQ-STE and DiVeQ becomes an ablation.

---

## Phase 2 — Ablations to understand WHY (or why not)

Run after exp1 lands. Cheap variants that isolate which ingredient matters.

| Exp | Name | Change | Question answered |
|-----|------|--------|-------------------|
| **exp5** | no-rosa-ablation | Same as exp1 but `rosa_feat` always = 0 (zero vector at every position) | How much accuracy did ROSA actually contribute? exp1 − exp5 = ROSA's contribution. |
| **exp6** | no-stopgrad | Don't `stop_grad` ROSA's predicted-token embedding (let gradient flow into the codebook entry ROSA picked) | Does the codebook learn faster when "the entry ROSA chose should help classify the digit"? Risk: ROSA's choice depends on past tokens → gradient is biased. |
| **exp7** | continual-eval | Test-time ROSA continues to grow on test stream | Closer to "true online deployment." Expect slight improvement on later test samples. |
| **exp8** | reset-rosa-each-epoch | Wipe SAM at start of every epoch; rebuild from current input module's tokens | Tests staleness: does removing stale tokens help, or do we need the cumulative count? |
| **exp9** | ponder-on | exp1b + `λ_ponder = 0.001` with 20k-step warmup (depends on exp1b succeeding) | Does ACT cost shorten K without hurting accuracy? |
| **exp9b** | label-act-vs-fixed | Replace label ACT-encoder with single-token labels (10 reserved IDs); rest as exp1 | Does the multi-token label burst actually help ROSA, or is single-token sufficient? Pins down whether the ACT-label is load-bearing. |

Each Phase 2 experiment should change exactly one knob from exp1.

---

## Phase 3 — EGGROLL fallback (only if Phase 1 stalls)

If end-to-end STE doesn't push past the vanilla VQ-MLP baseline, the
hypothesis is that the **stop_grad on ROSA's prediction** decouples the input
module from ROSA's feature, so the output module learns to ignore it.
EGGROLL/ES applies a black-box gradient on `(input + codebook + output)`
parameters using classification accuracy as the fitness signal — ROSA in the
loop, no STE required.

| Exp | Name | Change | Notes |
|-----|------|--------|-------|
| **exp10** | eggroll-baseline | Replace optimizer with antithetic ES (rank-1, σ=1e-3, n_pop=128, cosine LR; copied from `projects/es/exp3`). Same architecture as exp1. | Slow (~5–10× exp1); first sanity check is whether ES converges *at all* with ROSA in the loop. |
| **exp11** | eggroll-no-rosa-feat | exp10 but `rosa_feat = 0` | Confirms ES converges on the architecture without ROSA. |
| **exp12** | eggroll-rosa-feat | exp10 fully on | The real test. If exp12 > exp11 by ≥1pp, ROSA contributes through ES even when it cannot through STE. |

If exp12 still fails to use ROSA: the architecture is wrong, not the
optimizer. Revisit Phase 4.

---

## Phase 4 — Decoupled / staged training (clean fallback)

If neither STE nor EGGROLL gets ROSA to contribute:

| Exp | Name | Stage | Notes |
|-----|------|-------|-------|
| **exp20** | decoupled-vqae | Phase A: train input module as a VQ-AE on MNIST (reconstruction loss) → frozen codebook & input module | Standard VQ-VAE; gives a clean discrete tokenization unaware of ROSA. |
| **exp21** | decoupled-rosa | Phase B: feed all frozen tokens to ROSA → frozen SAM | One-shot pass, fast. |
| **exp22** | decoupled-head | Phase C: train output module supervised on (frozen sample tokens + ROSA's frozen prediction) | If this beats `exp22 with rosa_feat=0`, the issue was joint optimization, not the architecture. |

This breaks the "single end-to-end" story but is the cleanest baseline for
"does ROSA help at all once everything else is fixed."

---

## Metrics & comparison

Every row in `results.jsonl` must include:

```json
{
  "experiment": "exp1",
  "name": "baseline-V1024",
  "test_acc": 0.962,                // primary: digit-only argmax at <LABEL> position
  "train_acc": 0.985,
  "no_output_acc": 0.991,           // 11-class accuracy across all positions
  "fire_rate": 0.046,                // fraction of positions where output predicts a digit (not NO_OUTPUT) — should approximate 1/22 ≈ 0.045
  "fire_at_label_pos_rate": 0.94,   // of positions tagged DIGIT, fraction at <LABEL> indicator (timing accuracy)
  "rosa_hit_rate": 0.31,            // fraction of <LABEL>-position queries answered by SAM (vs fallback)
  "sam_state_count": 4_120_000,
  "codebook_used": 891,
  "mean_K_sample": 9.4,
  "mean_K_label":  2.1,
  "epochs": 50,
  "time_s": 1820.0,
  "history": {
    "loss": [...],
    "test_acc": [...],
    "rosa_hit_rate": [...],
    "fire_rate": [...],
    "sam_states": [...]
  },
  ... all hyperparameters ...
}
```

The five diagnostic curves to always watch:
1. **`test_acc`** vs epoch — primary metric (digit-only argmax at `<LABEL>`).
2. **`fire_rate`** vs epoch — is the output module learning to fire at all,
   or collapsing to "always NO_OUTPUT"? Should approach `1/22 ≈ 0.045`.
3. **`rosa_hit_rate`** vs epoch — is ROSA actually answering, or always
   in fallback? Should rise from 0 to a stable plateau.
4. **`codebook_used`** vs epoch — codebook collapse warning. If <10% of `V`,
   turn on the entropy regularizer.
5. **`mean_K_sample`** vs epoch — ACT halting stability. Runaway K means
   ponder cost needed; collapsed K=1 means halt-bias init too aggressive.

---

## Schedule (rough)

- Day 1: Phase 0.1, 0.2, 0.3, 0.4. Confirm pieces work in isolation.
- Day 2: exp1. First end-to-end run.
- Day 3: exp2, exp3 in parallel on two GPUs. exp3b (digit-boost sweep).
- Day 4: Phase 2 ablations (exp5 lesion is the highest-priority).
- Day 5+: Phase 3 / Phase 4 only if needed.

---

## Open questions deliberately deferred

These are real but not foundational; revisit after seeing exp1's behavior.

1. **ROSA's prediction sampling** — argmax (default) vs. temperature-sampled
   from fallback distribution. Sampling adds noise at training time which
   might help exploration but probably hurts at eval. Defer.
2. **Boundary handling** — should `<SAMPLE>` and `<LABEL>` indicators block
   ROSA's suffix walk, or be transparent? Default: block at `<SAMPLE>` (so
   ROSA never crosses example boundaries during prediction); transparent at
   `<LABEL>` (so ROSA can leverage `(image_tokens, <LABEL>)` → label_token
   patterns). Defer to exp2 / exp7.
3. **Multi-step ROSA prediction** — option (iii) from the design discussion:
   ROSA produces several next tokens in a row, output module sees the whole
   continuation. Interesting but only worth trying after the single-token
   baseline works.
4. **Per-class SAM** — separate SAM per digit class (10 SAMs), only the
   relevant one is queried at eval. Sharper statistics but breaks the
   "one symbolic system" story. Worth a one-off ablation.
