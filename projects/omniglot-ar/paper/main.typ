// Paper entry point. The #include list below IS the structure — add, remove or
// reorder; each section is an independent file under sections/.
//
// Build:  uv run python projects/omniglot-ar/scripts/publish_paper.py --preview
// Publish: uv run python projects/omniglot-ar/scripts/publish_paper.py

#import "/template.typ": *

#show: paper.with(
  title: "Token-level in-context classification on Omniglot",
  subtitle: "What a class-disjoint substrate does and does not fix",
  date: "6 August 2026",
  web: sys.inputs.at("web", default: "0") == "1",
  abstract: [
    Treating a dataset as a flat bag of #emph[(position, value, ref)] tokens
    makes classification, inpainting and denoising the same operation:
    complete a masked token's value. Prior work tested this on MNIST with
    per-episode label anonymisation and obtained chance accuracy across six
    architectural interventions. We ask whether that was a property of the
    formulation or of MNIST, and re-run it on Omniglot, whose standard split
    has #emph[disjoint] train and test character inventories — so memorisation
    is impossible by construction rather than by an anonymisation trick, and
    twenty drawings per character remove the incentive to memorise at all.
    Across seven runs — including one in which each query image #emph[is] its
    own support image, where nearest neighbour scores 1.000 — accuracy never
    leaves chance. Positive controls localise the failure precisely: leaking the
    answer into the query's own label token, or into its own pixel tokens, both
    drive the loss to zero within 300 steps, so the pipeline and the
    #emph[ref]-keyed attention the premise rests on are both sound. The missing
    capability is content-dependent matching — attending to a token because its
    value resembles one's own. We then show that this capability #emph[is]
    learnable, and appears as an abrupt phase transition after a long plateau,
    once the effective batch and learning rate are large enough to see the
    gradient above minibatch noise. The seven failed runs were under-resourced
    for crossing that plateau rather than under-trained: the run that solves the
    task does so in half their step budget.
  ],
)

#include "/sections/01-introduction.typ"
#include "/sections/02-background.typ"
#include "/sections/03-task.typ"
#include "/sections/04-methodology.typ"
#include "/sections/05-experiments.typ"
#include "/sections/06-results.typ"
#include "/sections/07-analysis.typ"
#include "/sections/08-conclusion.typ"

#bibliography("/refs.bib", title: "References", style: "ieee")
