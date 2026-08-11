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
    recurrent model on retrieval alone. The retrieval it learns is fully general,
    reaching identification accuracy 1.000 on images and even on digit classes it
    has never seen, but it acquires almost no ability to complete an image that
    is absent: 0.852 normalised MSE against 0.645 for a linear regression that
    ignores the context entirely, and getting worse as retrieval sharpens, while
    the same architecture trained on a 50/50 mixture reaches the full completion
    ceiling. Enlarging the context past what the fixed memory can hold does
    improve completion, but only by destroying retrieval: at 256 context images
    the recall-trained model gains nothing from the answer being present (0.004,
    against 0.835 at 16) and converges on the same weight-memorised solution a
    completion-trained model finds — a trade that reproduces when the memory is
    shrunk with the context held fixed. Within this setting, retrieval training
    buys no generalisation; generalisation appears only where retrieval fails.
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
