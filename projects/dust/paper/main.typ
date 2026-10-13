// Paper entry point. The #include list below IS the structure — add, remove or
// reorder; each section is an independent file under sections/.
//
//   preview:  uv run python -m shared_lib.publish projects/dust/paper --preview
//   one part: uv run python -m shared_lib.publish projects/dust/paper --section 04-methodology
//   check:    uv run python -m shared_lib.publish projects/dust/paper --check
//   publish:  uv run python -m shared_lib.publish projects/dust/paper
//
// Flip `status` to "final" when the paper is done: every remaining #todo then
// becomes a compile error instead of a red box.

#import "/template.typ": *

#show: paper.with(
  title: "DUST vs EGGROLL",
  subtitle: none,
  date: none,
  status: "final",
  web: sys.inputs.at("web", default: "0") == "1",
  abstract: [
    DUST trains a transformer language model without backpropagation by perturbing activations independently at every
    token, and is reported to be a thousand to ten thousand times more efficient than weight-space evolution strategies
    (EGGROLL). We ported it to JAX, checked it against the authors' code on identical weights, and set it against an
    independently implemented EGGROLL on the paper's protocol. At 1M training tokens DUST with 64 draws reaches a test
    loss of 6.315 and EGGROLL with 1,024 forward passes 7.133, at about a fifteenth of the cost; backprop reaches 5.937 and a
    model that knows only token frequencies 6.994. DUST's gradient estimate is one to two orders of magnitude better aligned
    with backprop's than EGGROLL's, which is what random-direction arithmetic predicts for EGGROLL. The ordering reproduces,
    and so does the order of magnitude of the efficiency claim: by line fits through our ladder, EGGROLL would need thousands
    of times DUST's 64-draw population to match it, and by a power law, whose floor lies above DUST's loss, it never would.
    Under Adam and Muon backprop improves by 0.57 to 0.58 nats and DUST by up to 0.47, so DUST's equality with backprop at 1,024 draws holds under
    SGD and not under Muon; EGGROLL is not helped by Muon.
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
