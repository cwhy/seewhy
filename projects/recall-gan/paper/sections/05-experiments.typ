#import "/template.typ": *

= Experiments

Every run uses the method of the previous section and differs only in the
generator and the seed.

#figure(
  table(
    columns: 4, align: (left, left, left, left), stroke: none,
    table.hline(),
    [*runs*], [*generator*], [*question*], [*source*],
    table.hline(stroke: 0.5pt),
    [12 (seeds 50–61)], [context-aware], [the method], [`experiments15.py`, tag `adam`],
    [12 (seeds 50–61)], [blind], [does the generator need the context?], [`experiments18.py`, `experiments19.py`],
    [12 (seeds 50–61)], [context-aware, Adam $beta_1 = 0.5$, $beta_2 = 0.999$], [is the optimiser setting needed?], [`experiments14.py`, tag `base`],
    [—], [none (pixel baselines)], [what does the classifier probe give without learning?], [`scripts/eval_report.py`],
    table.hline(),
  ),
  caption: [The runs behind every number in this paper. All evaluation is in
  `scripts/eval_report.py` and the training-time logs of each run.],
) <tab:runs>

*Controls.* The blind generator is the main control. It plays the same game
with the same discriminator but cannot see the context, so any difference in
how the discriminator uses its state comes from the generator. The pixel
baselines are the reference for the digit classifier: they see exactly the
same 16 images per digit.

*Repeats.* GPU arithmetic in these runs is not bit-for-bit repeatable, and the
two networks amplify small differences. A run is therefore a sample, not a
fixed outcome, which is why each configuration has 12. Medians and ranges over
the 12 are reported.
