// Paper entry point. The #include list below IS the structure — add, remove or
// reorder; each section is an independent file under sections/.
//
//   preview:  uv run python -m shared_lib.publish projects/random-transformers/paper --preview
//   one part: uv run python -m shared_lib.publish projects/random-transformers/paper --section 04-methodology
//   check:    uv run python -m shared_lib.publish projects/random-transformers/paper --check
//   publish:  uv run python -m shared_lib.publish projects/random-transformers/paper
//
// Flip `status` to "final" when the paper is done: every remaining #todo then
// becomes a compile error instead of a red box.

#import "/template.typ": *

#show: paper.with(
  title: "Frozen Random Transformers Already Compute",
  subtitle: [a replication of Zhong and Andreas, _Algorithmic Capabilities of
    Random Transformers_ (NeurIPS 2024)],
  date: none,
  status: "final",
  web: sys.inputs.at("web", default: "0") == "1",
  abstract: [
    How much of a transformer's algorithmic ability exists before it is trained?
    We reimplemented Zhong and Andreas (2024) in JAX from its text, freezing every
    attention and feed-forward weight at its random initialisation and training
    only the token embedding, positional embedding and unembedding. Such models
    reach 1.000 on modular arithmetic, associative recall, decimal addition and
    parenthesis balancing, in five seeds out of five, against chance levels of
    0.005, 0.008, ~0 and 0.681; seventeen of twenty cells in the main table fall
    within 0.15 of the published numbers, and the proposed mechanism — computation
    confined to a low-dimensional subspace that is not neuron-aligned — reproduces,
    including the paper's own falsification test. We add two findings: the paper's
    stated optimiser cannot reproduce its own fully trained baselines, since a
    warmup and cosine decay present in the authors' code are missing from the
    text, and the fact that only the baseline condition is sensitive to this is
    itself evidence for the paper's thesis.
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
