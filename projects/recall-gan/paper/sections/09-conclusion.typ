#import "/template.typ": *

= Conclusion

A GAN whose real data is a 16-image context, with a discriminator that reads
the context only through a fixed-size delta-rule memory, trains reliably and
turns that memory into an in-context test of belonging. With a generator that
sees the context, the discriminator relies on what it stored (AUROC 0.970 with
its own context against 0.570 with another), rejects unseen real images it was
never trained on (AUROC 0.961), and classifies digits from 16 examples better
than pixel-space nearest-neighbour or RBF density (0.839 against 0.788). With
a blind generator, which here always collapsed, none of this happens.

The memory it learns is coarse: closer to "which kinds of digit were here"
than to "which exact images". Whether a richer generator, a larger context or
more varied data would push it toward item-level membership is the natural
next question.

== Reproduction

All code is in `projects/recall-gan/` of the `seewhy` repository. On one GPU:

```
# the method, 12 seeds
uv run --no-sync python projects/recall-gan/scripts/run_experiments.py exp15
# blind control
uv run --no-sync python projects/recall-gan/scripts/run_experiments.py exp18 exp19
# Adam ablation
uv run --no-sync python projects/recall-gan/scripts/run_experiments.py exp14
# probes and pixel figures
uv run --no-sync python projects/recall-gan/scripts/eval_report.py
# aggregates, then data figures
python3 projects/recall-gan/scripts/summarise_report.py --version=v3
uv run --no-sync python projects/recall-gan/scripts/gen_report.py
```
Every number in this paper is in `results.jsonl`: training runs under
experiments `exp14`, `exp15`, `exp18` and `exp19`; probes under
`eval_report`; medians, ranges and counts under `report_summary_v2`.
Runs are not bit-for-bit repeatable on GPU, so a rerun gives a new sample of
12, not the same 12.
