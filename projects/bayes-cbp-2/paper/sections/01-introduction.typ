#import "/template.typ": *

= Introduction <sec-intro>

== The question

A network trained on a long sequence of tasks gradually learns new tasks more slowly.
This is #gloss[loss of plasticity][the same network and optimiser learn task 50 much worse than task 1, even though each task alone is easy].
#gloss[Continual Backpropagation][CBP: during training, repeatedly find hidden units judged useless, re-draw their input weights from the initial distribution and zero their output weights]
is a standard remedy @dohare2024loss.
It works, but it is a heuristic: nothing in its derivation says when to reset, which units, to what, or how hard.

The user's research prompt (`source-note.md`) asks for a first-principles answer.
It poses fourteen stages, P0 to P14.
They run from defining plasticity in Bayesian terms, through stochastic dynamics and escape times, to a new algorithm and a falsifiable research plan.
This paper answers them in order, one section per stage.
It contains no neural-network experiments.
Every figure is a closed-form curve or a small Monte Carlo check of one.


== The answers in brief

#callout(title: [Main results])[
  + *Bayesian plasticity is the expected information the next batch carries about the parameters.* #rig
    Under a static model with exchangeable data it can only fall (@sec-plasticity).
    So, for exact Bayes, loss of plasticity is not a malfunction.
    It is the correct response to a world assumed never to change.
    Restoring plasticity requires a model in which the world can change.
  + *Plasticity is not posterior variance.* #rig
    In the Gaussian case it is $1/2 log det(I + Sigma F_"new")$:
    variance counts only along the directions the new data can see (@sec-gaussian).
  + *A reset can play two roles that share one equation but not one meaning.* #rig
    As part of a #emph[sampler], a reset must leave the posterior unchanged.
    Naive resets do not; a Metropolis-corrected reset does, and it still speeds up mixing (@fig-dw-occupancy).
    As part of a #emph[world model], a reset is the prediction step of a change-point prior.
    Then it changes the target on purpose: the filter forgets (@sec-jump).
  + *A reset helps through where it lands, not through the reset event.* #rig
    Resetting to the stuck point never helps a memoryless escape; resetting to a fresh draw near the barrier cuts the escape time from 17 to 0.3 (@fig-mfpt).
  + *Proposed algorithm: Change-Point Reset (CPR).* #heur
    Each unit carries a posterior probability that it should be replaced.
    That probability is a ratio of two marginal likelihoods, computed online from quantities a backward pass already produces.
    Units are then reset by sampling, not by soft averaging: averaging a symmetric reset distribution sends a unit to zero, where it is dead (@sec-algorithm).
    Whether CPR beats CBP is untested. @sec-plan gives the experiments that would refute it.
]

== How to read the claims

The prompt asks for every claim to be graded (P0.2, P13).
Each claim carries one of five tags:

- #rig — proved under the model stated beside it.
- #cond — holds under extra assumptions, which the text names.
- #heur — a heuristic or conjecture.
- #open — not supported yet; needs a proof or an experiment.
- #wrong — an error, with the correction beside it.

*Provenance.* The questions are the user's.
Every answer, derivation, and the algorithm are Claude's, and none has been discussed with the user yet.
They are proposals.
Results taken from the literature are cited; reinterpretations of published methods are marked as ours (P10 asks for this split).

== Map of the paper

#table(
  columns: (auto, auto, 1fr),
  [Stage], [Section], [Question],
  [P1], [@sec-plasticity], [What plasticity is, in Bayesian terms],
  [P2], [@sec-gaussian], [Gaussian models, precision, EWC and VCL],
  [P3], [@sec-dynamics], [SGD, Langevin dynamics, and dynamical measures of plasticity],
  [P4], [@sec-mechanisms], [Why plasticity is lost; mechanisms beyond the gradient],
  [P5–P6], [@sec-jump], [One jump-diffusion for all mechanisms; its generator and stationary laws],
  [P7], [@sec-escape], [Escape times, resetting, spectral gaps],
  [P8], [@sec-switch], [A task switch in a local model],
  [P9], [@sec-utility], [What a low-utility unit is],
  [P10], [@sec-unified], [CBP and its relatives in one table],
  [P11], [@sec-routes], [Three routes from a probability model to a reset rule],
  [P12], [@sec-algorithm], [The CPR algorithm, with pseudocode],
  [P13], [@sec-review], [A sceptical review of all of the above],
  [P14], [@sec-plan, @sec-conclusion], [Research plan, falsification tests, summary],
)
