#import "/template.typ": *

= CBP and its relatives in one frame <sec-unified>

_Answers P10._ The first table states each method as published. The second gives our reinterpretation; it is ours, not the authors' #heur.

== As published

#table(
  columns: (0.9fr, 1.4fr, 0.6fr, 1.3fr, 0.9fr),
  [Method], [Trigger], [Strength], [Target], [Role in @sec-jump],
  [CBP @dohare2024loss], [Lowest contribution utility among units older than a maturity threshold; a fixed fraction $rho$ per step], [Full], [Input weights from the initial distribution; output weights zero], [State-dependent jump],
  [ReDo @sokar2023dormant], [Normalised activation score below a threshold, checked every $F$ steps], [Full], [Same as CBP], [Threshold jump],
  [Shrink-and-perturb @ash2020warm], [Fixed schedule (e.g. each new task)], [Partial, all weights], [$c theta + s epsilon$, $epsilon$ from the initial distribution], [Jump with state-dependent kernel],
  [L2-Init @kumar2023maintaining], [Always], [Weight $gamma$], [Initial parameters $theta_0$], [Drift],
  [Weight decay], [Always], [Weight $gamma$], [Zero], [Drift],
  [Noise injection], [Always], [Temperature $T$], [—], [Diffusion],
  [EWC @kirkpatrick2017ewc], [Always], [Fisher-weighted], [Old optimum $theta_A$], [Drift (for stability, not plasticity)],
  [Plasticity injection @nikishin2023deep], [Fixed times], [—], [Adds a new subnetwork whose initial output cancels], [Structural],
)

== Reinterpreted

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  [Method], [Implied world model], [What exact Bayes would do instead], [Effect on old / new task],
  [Shrink-and-perturb], [Each step, $theta_(t+1) = c theta_t + s epsilon$: a linear-Gaussian drift], [Shrink the mean, $m <- c m$, and inflate the covariance, $Sigma <- c^2 Sigma + s^2 I$. For a point-estimate learner, the covariance increase acts like a larger step size. Adding noise to the point is a #emph[sample] from that prediction], [Uniform partial forgetting / uniform plasticity],
  [L2-Init], [$theta$ mean-reverts toward $theta_0$ (the small-step limit of soft reset, @sec-jump)], [The same, plus the diffusion term that the limit also contains], [Steady pull toward init; slow forgetting everywhere],
  [CBP / ReDo], [Units can be replaced outright (change-point)], [A posterior change probability per unit, from a constant prior hazard and the data (@sec-routes)], [Sparse full forgetting / fresh units],
  [Noise injection], [None: a sampler's temperature, not a world model], [Nothing; at $T = 1$ it samples the static posterior], [Forgets in flat directions (@sec-switch)],
  [EWC], [Static world], [Exact recursive posterior; EWC is its diagonal Laplace form], [Stability only],
)

== Each mechanism, by the five questions of P10

#table(
  columns: (auto, auto, auto, auto, auto),
  [], [Trigger], [Strength], [Target], [Dynamics role],
  [Hard reset by utility], [Low utility], [1], [Fixed $p_0$], [Jump],
  [Reset by activation or gradient], [Low activity], [1], [Fixed $p_0$], [Jump],
  [Continuous soft reset], [None], [$gamma$ per unit time], [Centre $mu$], [Drift + diffusion],
  [Changed prior or regulariser], [None], [Prior precision], [Prior mean], [Drift],
  [Change-point filter (ours)], [Posterior change probability], [Probability $r_j$], [$p_0$, optionally learned], [Jump; prediction step of a filter],
)

== Is there a single underlying problem?

We think so #heur. Every method answers two questions, usually without separating them:

+ *Inference.* Under a world model that allows change, which parts of the network have probably changed? The answer is a posterior over change, not a utility.
+ *Decision.* Given that posterior and a learner that keeps only a point estimate, what should be done to the point? Options: nothing, shrink, sample a reset.

The methods differ in the world model they imply (linear drift, mean-reversion, change-point) and in the shortcut they use for inference (a fixed schedule, a utility rank, a threshold).
Shrink-and-perturb and L2-Init make inference trivial: every parameter changes a little.
CBP and ReDo make it a ranking: the least useful units changed.
Neither computes the posterior.

One coincidence deserves note.
Under the change-point model, if a fresh unit's output weights start at zero as in CBP, a dead unit's posterior change probability equals the prior hazard $rho$ exactly.
The data cannot distinguish "stayed" from "replaced": both contribute nothing (@sec-route2).
CBP resets a fixed fraction $rho$ of its least useful units per step.
So on dead units, CBP's fixed rate is what the filter would do #cond.
It differs on two other kinds of unit:
on useful units the filter's probability falls far below $rho$;
on units whose old setting conflicts with new data, it rises far above $rho$.
CBP never resets the second kind, because their contribution utility is high.
