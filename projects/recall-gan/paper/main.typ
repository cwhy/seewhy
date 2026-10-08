// Paper entry point. The include list below IS the structure — add, remove or
// reorder; each section is an independent file under sections/.
//
//   preview:  uv run python -m shared_lib.publish projects/recall-gan/paper --preview
//   one part: uv run python -m shared_lib.publish projects/recall-gan/paper --section 04-methodology
//   check:    uv run python -m shared_lib.publish projects/recall-gan/paper --check
//   publish:  uv run python -m shared_lib.publish projects/recall-gan/paper --stable
//
// Flip `status` to "final" when the paper is done: every remaining #todo then
// becomes a compile error instead of a red box.

#import "/template.typ": *

#show: paper.with(
  title: "Recall-GAN",
  subtitle: "An in-context GAN whose discriminator learns what is in its context",
  date: datetime.today().display(),
  status: "final",
  web: sys.inputs.at("web", default: "0") == "1",
  abstract: [
    Can a network learn to tell whether an image belongs to a set it has just
    read, while holding the set only in a fixed-size memory? We train such a
    memory as the discriminator of a GAN. In each episode the real images are
    16 context images, written into a 16 384-number delta-rule (KDA) state,
    and the fakes come from a generator. The discriminator is never shown a
    real image outside the context. With a generator that sees the context,
    the discriminator comes to rely on its state (AUROC 0.970 with its own
    context, 0.570 with another episode's). It rejects unseen real images it
    was never trained on (AUROC 0.961), and, given 16 examples per digit, it
    classifies MNIST digits at 0.839, above pixel nearest-neighbour (0.783)
    and pixel RBF density at its best bandwidth (0.788). With a blind
    generator, which collapsed in all 12 runs, it ignores its state and
    classifies at chance. With Adam $beta_1 = 0, beta_2 = 0.99$, none of 12
    runs failed.
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
