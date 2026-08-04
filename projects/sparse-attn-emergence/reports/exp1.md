# exp1 — is emergence abrupt, and is its timing seed-random?

**H1.** Testing the shape of learning, not whether learning happens: a plateau at the
marginal-entropy loss, broken abruptly, at a step that varies across seeds.

**Setup.** Linear map `S = 16`, `s = 3`. One layer, `D = 128`, MLP 512, `H = 8`, 10,000
steps, batch 256. See [Methods](sparse_attn_emergence_methods.html) for the task and metric
definitions.

**`A` is fixed across all 16 seeds.** Only initialisation and data order differ, so any
spread in timing is the search itself, not one seed drawing an easier matrix. 167 seconds
for all 16 seeds together.

## Result

![per-seed loss and accuracy](https://media.tanh.xyz/seewhy/26-08-04/sparse_attn_emergence_exp1_seed_curves.svg)

| | |
|---|---|
| plateau (pre-jump `loss2`) | **0.6894** — `ln 2` is 0.6931 ✓ |
| `t*` across seeds | **469 … 2521** steps, median 885 → **5.4× spread** |
| jump width (`loss2` 0.6 → 0.05) | median **354** steps ≈ 0.42 × `t*`, range 78 … 2173 |
| solved (`acc2 > 0.95`) | **16 / 16**, final `loss2` ≤ 3e-5 |

Every seed sits at `ln 2` — predicting a coin flip for a deterministic function — for
hundreds of steps, then falls to numerically zero loss. Nothing about the plateau
distinguishes a seed that is 400 steps from solving the task from one that is 2500 away.

![time-to-emergence histogram](https://media.tanh.xyz/seewhy/26-08-04/sparse_attn_emergence_exp1_tstar.svg)

### Per-seed anatomy

| seed | `t*` | jump width | seed | `t*` | jump width |
|---:|---:|---:|---:|---:|---:|
| 2 | 469 | 78 | 9 | 894 | 295 |
| 5 | 563 | 100 | 12 | 927 | 255 |
| 10 | 566 | 236 | 1 | 983 | 529 |
| 0 | 712 | 385 | 15 | 1086 | 570 |
| 4 | 772 | 113 | 11 | 1107 | 535 |
| 8 | 783 | 291 | 7 | 1196 | 352 |
| 13 | 856 | 357 | 3 | 2187 | 1871 |
| 6 | 876 | 502 | 14 | 2521 | 2173 |

## Verdict: H1 partially replicated

**The seed-randomness claim holds, clearly.** A 5.4× spread at identical difficulty is the
paper's central point, and it is the reason a scaling curve can look smooth while
individual capabilities snap into place. It also argues against the paper's own sample
size: `[469, 563, 566]` and `[1196, 2187, 2521]` are both plausible 3-seed draws from this
run, and they support opposite conclusions about when the capability "should" appear.

**The abruptness claim is weaker than the paper's figures suggest.** The drop is fast
*relative to when it starts* — median 354 steps against a median `t*` of 885 — but it is
not a cliff. Two seeds took over 1800 steps to complete the fall, longer than four other
seeds needed to finish learning entirely. "Abrupt" is a fair description on a log axis
across a full pretraining run; at this resolution it is a fast sigmoid with a
seed-dependent slope, not a step function.

**This configuration is inside the learnable regime**: 16/16 solved. The paper's
*unlearnable* medium-sparsity window is not visible at `S=16, s=3` — that is exp2's
question.

## A caveat about our own instrumentation

![attention alignment and entropy](https://media.tanh.xyz/seewhy/26-08-04/sparse_attn_emergence_exp1_mechanism.svg)

The mechanism panel shows attention alignment rising and entropy dropping around each
seed's jump, which is the expected story — but the alignment metric here is flawed. It
picks a **single head** and averages its overlap with the true support across rows, and
final values span only 0.49–0.97 while loss is already ~0.

The reason is head specialisation: different heads take different rows of `A`, so
best-single-head aggregation systematically understates a model that has solved the task
completely. exp1's stored `diag_iou_max` should be read with that in mind. The corrected
per-row-best-head metric (`iou_row`) is in place from exp2 onward, and exp4 will use it
densely, together with a causal ablation — knocking out the aligned head to check the loss
returns to the plateau. Without that ablation, the alignment/jump coincidence stays
correlational.
