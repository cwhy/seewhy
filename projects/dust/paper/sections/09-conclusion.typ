#import "/template.typ": *

// OBLIGATIONS
//  - What is now known that was not known before this work. Two paragraphs.
//  - No new evidence here; if it needs support, it belongs in 6 or 7.
//  - Next steps only where they follow from a specific finding above.
//
// The reproduction appendix is not optional: a paper claiming a number that
// nobody can regenerate is a claim about a number, not a result.

= Conclusion

DUST's central claim survives an independent baseline. With EGGROLL implemented from its published description
and set up as in an earlier, audited project, DUST at 64 draws beats EGGROLL at 1,024 forward passes by 0.82 nats at
1M training tokens, at about a fifteenth of the measured cost, and, under SGD, DUST at 1,024 draws reaches the loss of backprop. The
reason is a gradient estimate that is one to two orders of magnitude better aligned with backprop's at the same
population. EGGROLL's alignment is what random-direction arithmetic says it must be, and DUST's per-token rewards
are the way out of that arithmetic. A JAX port that matches the authors' code on identical weights, and reaches
their published losses within seed noise, shows the result does not depend on one codebase.

The efficiency headline reproduces in order of magnitude, and we nearly reported that it did not. Extrapolating our EGGROLL's
ladder to the population that would match DUST's 64 draws gives thousands to tens of thousands of times 64 by two line fits,
as the paper does, and never by a power law whose floor lies above DUST's loss. With three points the same line said hundreds;
the fourth, at 4,096 passes, showed the ladder bending. The ordering is robust, and the multiplier is a line through few points,
here and in the paper. The practical consequence for this repository is that DUST's activation-space estimator is worth
carrying forward as a serious alternative to backprop-free baselines, and that any efficiency factor quoted for it should be
read with the number of points behind it. A better optimiser changes DUST's standing against backprop and not against EGGROLL. Under Adam and Muon backprop gains 0.57 to 0.58 nats and DUST up to 0.47,
so DUST at 1,024 draws, level with backprop under SGD, is 0.095 behind it under Muon on one seed, and 0.036 behind at 4,096 draws. EGGROLL is not helped by Muon, and DUST stays ahead of it by 1.30 to
1.95 nats within each optimiser. The next measurements that follow are a baseline closer to the paper's, which needs
its code, a per-cell search under Adam and Muon, and populations above 4,096, to test the claim that DUST passes backprop, which one seed at 4,096 draws only hints at.

= Appendix: reproduction <appendix-repro>

#kv(
  ("source tree", "`projects/dust` in this repository, not yet committed when this was written"),
  ("hardware", "one RTX 4090 per run, two GPUs, shared with another user's jobs for part of the Adam and Muon runs; JAX 0.8.1; bfloat16 matmuls, float32 weights"),
  ("data", "`python projects/dust/scripts/prepare_data.py`, then `prepare_extra.py` for the 100M-token stream"),
  ("run", "`uv run --no-sync python projects/dust/scripts/run_experiments.py --gpu N expK`, K as in section 5"),
  ("figures", "`scripts/gen_report.py`, `scripts/gen_diagrams.py`; numbers: `scripts/summary.py`, `extrapolate.py`, `eggroll_theory.py`"),
)

*Which rows back what.* Backprop: exp1 (1M and 10M). DUST ladder: exp3 (256, 1,024 draws), exp4 and exp9 (64 draws),
exp13 (100K), exp11 (10M), exp17 (4,096). EGGROLL ladder: exp10 (1M), exp13 (100K), exp11 (10M), exp17 (4,096); its tuning grid:
exp5, exp8, exp14, exp16. The rates control: exp18. Adam and Muon at 1M tokens: grids exp19 to exp23, three-seed cells exp24 and exp26, single-seed cells exp25 and exp27, and the 1,024-draw rate check exp28. Alignment: exp6 (and exp15 at 65,536 and 131,072 passes), with the gradient shares in the
`theory` rows. The parity check is the row of kind `parity`. The extrapolation is the row of kind `extrapolation`.
Seeds are 42, 43 and 44; tuning rows are seed 42. Every number in the prose is checked against `results.jsonl` by
`python -m shared_lib.publish projects/dust/paper --check`, and those that are aggregates or the DUST paper's own numbers are
listed with their provenance in `paper/.lint-allow`.
