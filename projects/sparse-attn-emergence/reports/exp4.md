# exp4 — is the loss jump the attention pattern being found?

**H3.** exp1 showed alignment rising as loss fell, which is a correlation. This adds the two
things that make it an argument: the **correct alignment metric**, logged densely, and a
**causal ablation**.

**Setup.** Identical to exp1 — `S=16`, `s=3`, fixed `A`, 16 seeds, 10,000 steps — with
diagnostics every 50 steps. Because the diagnostic cadence changes the data order, exp4's
runs are an *independent* 16-seed sample of the same configuration, not a re-run of exp1's.

## The jump

![mechanism](https://media.tanh.xyz/seewhy/26-08-04/sparse_attn_emergence_exp4_mechanism.svg)

Alignment climbs and entropy collapses at each seed's own jump, at whatever step that
happens to be. Final alignment:

| metric | value |
|---|---|
| `iou_row` (per row, best head, then averaged) | **0.843** mean, 0.731 – 0.919 |
| `iou_head` (exp1's single-best-head) | **0.727** mean |
| final `loss2` | ≤ 7.4 × 10⁻⁶, all 16 seeds |

The corrected aggregation is higher, as exp1's caveat predicted — but only by 0.12, and the
reason is not what I guessed there. Per-head alignment for a representative seed:

```
head:  0     1     2     3     4     5     6     7
IoU:   0.27  0.33  0.53  0.82  0.11  0.15  0.38  0.21
```

That is **one dominant head plus partial helpers**, not a clean division of rows between
heads. exp1's "heads specialise per row" reading was wrong; the gap between the two metrics
comes from helper heads covering rows the dominant head handles less well.

![attention maps](https://media.tanh.xyz/seewhy/26-08-04/sparse_attn_emergence_exp4_attention.png)

Pre-jump attention is diffuse; by the end the dominant head's map visibly reproduces the
support of `A`, row by row.

## The ablation

Zero the output-projection block of one head, leave every other weight untouched, re-measure:

![ablation](https://media.tanh.xyz/seewhy/26-08-04/sparse_attn_emergence_exp4_ablation.svg)

| condition | `loss2` |
|---|---|
| intact | **0.0000** |
| best-aligned head removed | **4.2264** |
| worst-aligned head removed | **0.0803** |
| `ln 2` (knowing nothing) | 0.6931 |

Removing the aligned head costs five orders of magnitude; removing the least-aligned head
costs almost nothing. The capability lives in the head whose attention matches `A`.

**Note what 4.23 nats means.** It is *six times* the `ln 2` plateau, so the model does not
fall back to ignorance — it becomes confidently wrong, assigning about 1.4% probability to
the correct token. Ablation does not reset the computation, it corrupts it: the downstream
MLP still expects that head's contribution and produces systematically inverted parity
without it. I had predicted a return *to* the plateau; the truth is more emphatic and worth
stating precisely rather than rounding to "the loss went back up".

## H1, independently

Same configuration, fresh data order, so exp4's timings are a second sample:

| | median `t*` | range | spread |
|---|---|---|---|
| exp1 | 885 | 469 – 2521 | 5.4× |
| **exp4** | **923** | **500 – 1984** | **4.0×** |

exp4's sorted `t*`: `500, 505, 590, 717, 800, 814, 822, 875, 971, 1013, 1080, 1135, 1169,
1240, 1297, 1984`. Two independent 16-seed samples agree on the median within 5% and both
show a 4–5× spread. H1's stochasticity is not an artifact of one draw.

## Verdict: H3 supported

The pattern search and the capability are the same event. Alignment rises exactly when loss
falls, and the aligned head is causally necessary — its removal is catastrophic while an
unaligned head's removal is nearly free.

**One honest limit.** Alignment saturates around 0.84, not 1.0, while loss reaches 7×10⁻⁶.
Soft attention does not need to match the support exactly to support an exact computation:
leftover mass on wrong positions is tolerable if the MLP can still separate the parities. So
"the model finds the pattern" is true in the causal sense but approximate in the geometric
one — the found circuit is *good enough*, not a clean indicator function.
