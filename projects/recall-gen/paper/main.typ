// Paper entry point. The #include list below IS the structure — add, remove or
// reorder; each section is an independent file under sections/.
//
//   preview:  uv run python -m shared_lib.publish projects/recall-gen/paper --preview
//   one part: uv run python -m shared_lib.publish projects/recall-gen/paper --section 04-methodology
//   check:    uv run python -m shared_lib.publish projects/recall-gen/paper --check
//   publish:  uv run python -m shared_lib.publish projects/recall-gen/paper
//
// Flip `status` to "final" when the paper is done: every remaining #todo then
// becomes a compile error instead of a red box.

#import "/template.typ": *

#show: paper.with(
  title: "Recall-Gen",
  subtitle: none,
  date: none,
  status: "draft",
  web: sys.inputs.at("web", default: "0") == "1",
  abstract: [
    Models that solve a task from examples in their input may be learning a rule
    or merely retrieving the nearest example, and in language the two are hard to
    separate. We build a task where they are mutually exclusive by construction —
    each token is a whole MNIST image, and a query image's true completion is
    either present in the context or provably absent — and train a linear
    recurrent model on retrieval alone. What it learns is a similarity metric:
    general enough to identify images, and even digit classes, it has never seen
    at accuracy 1.000, but fitted to its training distribution rather than free
    of it, falling to 0.651 on Fashion-MNIST and to 0.116 — against chance 0.063 —
    on MNIST images whose pixels have been permuted, which carry identical
    statistics and identical pairwise distances. It acquires almost no ability to
    complete an image that is absent: 0.852 normalised MSE against 0.645 for a
    linear regression that ignores the context entirely, worsening as retrieval
    sharpens, while the same architecture trained on a 50/50 mixture reaches the
    full completion ceiling. Enlarging the context past what the fixed memory can
    hold does improve completion, but only by destroying retrieval — at 256
    context images the model scores 0.556 with the answer present and 0.561
    without, the same number, where at 16 the two read 0.017 and 0.852; it has
    converged on the same weight-memorised solution a
    completion-trained model finds. That trade reproduces when the memory is
    shrunk with the context held fixed, and it does not exist at inference: a
    model trained at 16 and evaluated at 256 degrades to 0.942 rather than
    improving. Retrieval training buys no knowledge; where a model appears to
    start generalising, it has stopped retrieving.
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
