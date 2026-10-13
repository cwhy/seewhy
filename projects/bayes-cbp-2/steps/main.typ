// Step-by-step companion to paper/. Same formulation, but laid out the way the
// research prompt (source-note.md, P0) asks: one chapter per stage, each built
// only on earlier stages, each with the same six parts. No algorithm before P11.
//
//   preview:  uv run --no-sync python -m shared_lib.publish projects/bayes-cbp-2/steps --preview
//   publish:  uv run --no-sync python -m shared_lib.publish projects/bayes-cbp-2/steps

#import "/template.typ": *

#show: paper.with(
  title: "Plasticity, Stage by Stage",
  subtitle: "Fourteen research stages from Bayesian first principles to a reset algorithm",
  date: "2026-10-09",
  status: "final",
  web: sys.inputs.at("web", default: "0") == "1",
  abstract: [
    This document answers the research prompt in its own order, one stage per chapter.
    Each stage uses only what earlier stages established, and each has the same six parts:
    core view, derivation, when it holds, limits, alternatives, and next step.
    No algorithm is proposed before stage P11.
    The derivations are the same as in the thematic paper on this project; only the order and the discipline differ.
    Early stages define plasticity as expected information gain and prove it cannot rise under a static model (P1).
    Middle stages build one jump-diffusion for all plasticity mechanisms and test its claims by simulation (P5–P7).
    Late stages derive a reset rule from a change-point model, then attack it (P11–P13).
    Nothing here has been tested on a neural network.
  ],
)

// Chapters are stages: number them P0, P1, … and cite them as "Stage P3".
#set heading(
  numbering: (..n) => {
    let a = n.pos()
    "P" + str(a.at(0) - 1) + if a.len() > 1 { "." + a.slice(1).map(str).join(".") }
  },
  supplement: [Stage],
)

#include "/sections/p00-rules.typ"
#include "/sections/p01-definition.typ"
#include "/sections/p02-gaussian.typ"
#include "/sections/p03-dynamics.typ"
#include "/sections/p04-loss.typ"
#include "/sections/p05-unified.typ"
#include "/sections/p06-generator.typ"
#include "/sections/p07-escape.typ"
#include "/sections/p08-switch.typ"
#include "/sections/p09-utility.typ"
#include "/sections/p10-cbp.typ"
#include "/sections/p11-routes.typ"
#include "/sections/p12-algorithm.typ"
#include "/sections/p13-review.typ"
#include "/sections/p14-plan.typ"

#bibliography("/refs.bib", title: "References", style: "ieee")
