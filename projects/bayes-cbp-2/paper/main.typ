// Paper entry point. Theory only — no network experiments; figures are
// closed-form curves plus Monte Carlo and finite-difference checks, built by
// scripts/gen_figures.py on the box.
//
//   preview:  uv run --no-sync python -m shared_lib.publish projects/bayes-cbp-2/paper --preview
//   check:    uv run --no-sync python -m shared_lib.publish projects/bayes-cbp-2/paper --check
//   publish:  uv run --no-sync python -m shared_lib.publish projects/bayes-cbp-2/paper

#import "/template.typ": *

#show: paper.with(
  title: "Resets as Change-Point Inference",
  subtitle: "Plasticity from first principles, and a Bayesian replacement rule for Continual Backprop",
  date: "2026-10-09",
  status: "final",
  web: sys.inputs.at("web", default: "0") == "1",
  abstract: [
    Networks trained on long task sequences learn new tasks ever more slowly.
    We answer a fourteen-stage research prompt on why, from Bayesian first principles, and derive a reset rule.
    Under a model that assumes a fixed world, expected information gain from new data provably cannot rise: losing plasticity is correct inference there.
    In the Gaussian case plasticity is $1/2 log det(I + Sigma F_"new")$, not posterior variance, and a change model sets a floor under it.
    Resets admit two readings that share one equation.
    As sampler moves they must preserve the posterior: on a double well, naive resets put 68% of the mass in a well that should hold 50%, while Metropolis-corrected resets give 50%.
    As a world model they are the prediction step of a change-point prior, and the filter forgets on purpose.
    Resets help through where they land: resetting to the stuck point raises escape time from 17 to 22, resetting to a fresh draw cuts it to 2.
    From the change-point model we derive Change-Point Reset (CPR): reset each unit with its posterior probability of replacement, which weighs the value of a fresh unit against the cost of removing the old one.
    On dead units with zero output initialisation, CPR reduces to Continual Backprop's fixed rate.
    CPR is untested; we give the experiments that would refute it.
  ],
)

#include "/sections/01-introduction.typ"
#include "/sections/02-plasticity.typ"
#include "/sections/03-gaussian.typ"
#include "/sections/04-dynamics.typ"
#include "/sections/05-mechanisms.typ"
#include "/sections/06-jump-diffusion.typ"
#include "/sections/07-escape.typ"
#include "/sections/08-task-switch.typ"
#include "/sections/09-utility.typ"
#include "/sections/10-unified.typ"
#include "/sections/11-routes.typ"
#include "/sections/12-algorithm.typ"
#include "/sections/13-review.typ"
#include "/sections/14-plan.typ"
#include "/sections/15-conclusion.typ"

#bibliography("/refs.bib", title: "References", style: "ieee")
