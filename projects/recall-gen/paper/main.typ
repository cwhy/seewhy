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
    #todo[
      Write this last. Four sentences: the question, what was done, the
      headline number with its baseline beside it, and the conclusion.
    ]
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
