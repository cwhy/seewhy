#import "/template.typ": *

= Limitations and negative results <sec-limitations>

== Three cells we did not reproduce

Seventeen of twenty cells in the main table fall within 0.15 of the paper. Three
do not. All three are baselines rather than the condition under test, which is
the less damaging direction but still worth stating plainly.

#table(
  columns: 5,
  stroke: 0.5pt + luma(200), inset: 5pt, align: (left, right, right, right, left),
  table.header([*Cell*], [*Ours*], [*Paper*], [*Chance*], [*Our reading*]),
  [decimal, trained, width 16], [0.381], [0.675], [~0], [under-trained],
  [needle, LSTM, width 1024],   [0.357], [0.995], [0.008], [under-trained],
  [parens, trained, width 1024],[0.735], [0.923], [0.681], [the paper flags this cell too],
)

The third is not really a disagreement. The paper says of that exact cell: the
width-1024 fully trained transformer had trouble reaching perfect accuracy on
parenthesis balancing, likely due to imperfect hyperparameter choices. We see the
same anomaly in the same place, more severely.

The first two are ours. Both are baselines whose losses were still descending at
the end of training. Our budgets were set by the fully trained width-1024
condition, which is not the slowest condition everywhere.

An under-trained baseline inflates our headline, since the paper's claim is that
random transformers match or beat fully trained models and LSTMs. A learning-rate
search for these cells is running and is not included in the numbers above.

== The optimiser details the paper omits

This cost us more time than anything else, and it is the most useful thing we can
report to someone attempting the same replication.

Appendix D.3 of the paper specifies AdamW, learning rate $10^(-3)$, weight decay
$10^(-3)$, gradient clipping at 1.0. Following it exactly, we could not train the
fully trained baselines at all.

#table(
  columns: 3,
  stroke: 0.5pt + luma(200), inset: 5pt, align: (left, left, right),
  table.header([*Learning rate*], [*Schedule*], [*needle, trained, 1024*]),
  [1e-3], [none — as the paper states], [0.14],
  [1e-3], [none, at the paper's full 10,000-step budget], [0.142],
  [3e-4], [none], [~0.40],
  [1e-3], [500-step warmup, constant after], [0.92, then unstable],
  [3e-4], [500-step warmup, constant after], [0.999],
  [1e-3], [500-step warmup, then cosine decay], [0.997],
)

Chance is 0.008. The final row is what the authors' released code does:
`warmup_steps = 500` and `lr_scheduler_type = "cosine"`. Neither appears in the
paper.

Embedding-only training reaches 1.000 under every one of those settings. That is
why the omission leaves no trace in the paper's own results — it only affects the
condition the paper is not making claims about.

== An inconsistency in the paper's Table 6

Table 6 gives $E_"tok"$ for modular addition as 99,328 parameters. At width 1024
that implies a vocabulary of 97. The task text fixes the modulus at 199, which
needs 199 token embeddings and 203,776 parameters.

The other three tasks' entries divide exactly by 1024 into their stated
vocabularies. Only this row does not. The authors' code sets the task to
`modadds_199`, so the code agrees with the text and the table is wrong.

We mention it because a reader checking their own parameter counts against that
table will not match, and will reasonably assume the error is theirs.

== Where our target models differ from the paper's

For circuit imitation the paper states a diagnostic: mean output entropy within
$[2.89, 3.02]$ and mean cross-input divergence within $[0.8, 3.3]$, across all
target widths.

Our targets match the second and not the first. Cross-input divergence lands in
$[1.0, 3.2]$. Entropy is around 4.3 to 4.5, well above their range.

We tried to close the gap and could not do it without breaking the other
diagnostic. Scaling the unembedding by 1.5 brings entropy to $[2.89, 3.28]$ but
pushes divergence to $[2.0, 6.2]$. We kept the literal formula from the code
rather than tuning toward a number, and report both diagnostics per run.

Our targets are therefore less peaked than the paper's. A less peaked target is
easier to imitate, so our divergences are probably optimistic in absolute terms.
The comparison between conditions is unaffected — both students face identical
targets.

== Declared deviations

#kv(
  ("seeds", "5 per cell for the main table and ablation, 3 for sweeps, 1 for language modeling. The paper uses 10."),
  ("budget", "set from measured convergence, not copied. Memorization gets 60,000 steps against the paper's ~168,000."),
  ("language modeling", "one pass over 100M TinyStories tokens, against the paper's 5 epochs over the full corpus."),
  ("parenthesis data", "a fixed pool of 500,000 sequences rather than an endless stream."),
  ("LSTM", "a plain single-layer implementation, not a tuned one."),
)

The memorization budget is the one that visibly bit. Both of our absolute
recall numbers are roughly half the paper's, in the same ratio.

== What we did not do

We did not reproduce the paper's attention-pattern figures, its circular
embedding visualisation for modular addition, or its Clock-versus-Pizza
measurements. Those are qualitative claims about mechanism, and they are the
natural next thing to check.

We did not test any width above 1024, so we cannot say whether the random model's
advantage on needle in a haystack at width 1024 persists or closes.

We ran a single seed per cell for language modeling. Every number in that table
is one run. The differences between adjacent widths are large enough that we
believe the ordering, but the 2-layer against 4-layer comparison at width 512
turns on a gap of 0.136 nats, which one seed cannot establish.
