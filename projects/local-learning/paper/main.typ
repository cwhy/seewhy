// Paper entry point. The #include list below IS the structure — add, remove or
// reorder; each section is an independent file under sections/.
//
//   preview:  uv run python -m shared_lib.publish projects/local-learning/paper --preview
//   one part: uv run python -m shared_lib.publish projects/local-learning/paper --section 04-methodology
//   check:    uv run python -m shared_lib.publish projects/local-learning/paper --check
//   publish:  uv run python -m shared_lib.publish projects/local-learning/paper
//
// Flip `status` to "final" when the paper is done: every remaining #todo then
// becomes a compile error instead of a red box.

#import "/template.typ": *

#show: paper.with(
  title: "Local Learning vs Backprop",
  subtitle: none,
  date: none,
  status: "final",
  web: sys.inputs.at("web", default: "0") == "1",
  abstract: [
    A widely shared table compared backpropagation, PC-ALM and Forward-Forward
    on MNIST at about 56,000 parameters each, and ranked Forward-Forward first.
    We reproduce all three numbers to within one standard deviation over five
    seeds, then show that the ranking measures two things other than the
    learning rules: the rows use two different architectures — 56,128
    parameters is a 32-layer residual network, 56,232 is a two-layer plain one —
    and one shared learning rate that is optimal for exactly one of them.
    Backpropagation on Forward-Forward's own architecture reaches 91.16% against
    Forward-Forward's 88.69%, Forward-Forward on the deep network falls to
    57.51%, and giving each method its own best rate reverses the order against
    chance level 10%. The source paper's own claim survives both corrections:
    with the dual variables switched off, the same code loses 24 points of
    accuracy at 32 layers and nothing at 2. We add two arms the table lacks — plain
    predictive coding, the ablation that isolates PC-ALM's contribution, and
    EGGROLL, a gradient-free evolution strategy that reaches 90.25% with an
    update almost orthogonal to the true gradient and is the only rule here
    whose accuracy is flat from 2 to 64 layers.
  ],
)

#include "/sections/01-introduction.typ"
#include "/sections/02-background.typ"
#include "/sections/03-task.typ"
#include "/sections/04-methodology.typ"
#include "/sections/05-experiments.typ"
#include "/sections/06-results.typ"
#include "/sections/07-analysis.typ"
#include "/sections/08-limitations.typ"
#include "/sections/09-conclusion.typ"

#bibliography("/refs.bib", title: "References", style: "ieee")
