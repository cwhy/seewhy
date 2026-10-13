#import "/template.typ": *

= Plasticity after a task switch <sec-switch>

_Answers P8._

== A local model

Near the old optimum, model both tasks as quadratics:

$ L_A (theta) = 1/2 (theta - theta_A)^top H_A (theta - theta_A), quad L_B (theta) = 1/2 (theta - theta_B)^top H_B (theta - theta_B). $

Training on $B$ starts at $theta_A$.
#emph[Adaptation] is the fall of $L_B$; #emph[forgetting] is the rise of $L_A$.
All results in this subsection are exact for this model #rig. Their relevance to networks is #cond on the quadratic picture holding over the distance travelled.

== Four dynamics

*Gradient flow on $L_B$.*
$theta(t) = theta_B + e^(-H_B t)(theta_A - theta_B)$.
Direction $i$ of $H_B$ adapts in time $1 \/ h_i$.
Directions with $h_i = 0$ never move, so whatever task $A$ stored there is kept.
Forgetting happens only in directions that both tasks constrain.

*Gradient flow with an EWC penalty* $beta \/ 2 (theta - theta_A)^top Lambda_A (theta - theta_A)$.
It converges at the rates of $H_B + beta Lambda_A$, which are faster, to

$ theta^*(beta) = (H_B + beta Lambda_A)^(-1) (H_B theta_B + beta Lambda_A theta_A). $

With $Lambda_A = H_A$ (exact Laplace), sweeping $beta$ traces the #gloss[Pareto front][the set of points where neither loss can fall without the other rising] of $(L_B, L_A)$.
This holds because the front of two convex quadratics is exactly the set of minimisers of their weighted sums.
Early-stopped gradient flow need not lie on it.
In @fig-pareto, at equal new-task loss 0.5, gradient flow forgets 0.40 while the front forgets 0.10 — four times less.

*Langevin on $L_B$ at temperature $T$.* The mean follows gradient flow.
The covariance in eigen-direction $i$ of $H_B$ is $T(1 - e^(-2 h_i t)) \/ h_i$, which becomes $2 T t$ in a flat direction.
Expected forgetting gains $1/2 tr(H_A C(t))$.
So in directions the new task does not constrain, noise erases old knowledge, linearly in time.
Gradient flow kept exactly that knowledge.
This is the cost of mechanism M4.

*A reset.* Re-draw a block $S$ of coordinates (one unit) from $p_0$.
The immediate forgetting is $1/2 bb(E)[(xi - theta_A)_S^top (H_A)_(S S) (xi - theta_A)_S]$, plus cross terms with the other coordinates.
In a convex quadratic, a reset only moves the starting point.
Convergence from the new point runs at the same rates, so a reset helps only if $p_0$ lands closer to $theta_B$ along the slow directions.
#emph[In a convex model, resets have no special power.]
Their value is entirely non-convex: dead sets and barriers (@sec-mechanisms, @sec-escape).

#fig(include "/figures/pareto.typ", caption: [
  Forgetting against adaptation in a two-parameter quadratic model.
  $H_A$ has eigenvalues 1 and 0.05, rotated by 30°; $H_B = "diag"(1, 0.02)$; $theta_A = (0, 0)$, $theta_B = (2, 1)$.
  Pareto front: the EWC solution $theta^*(beta)$ for $beta in [10^(-3), 10^3]$.
  Gradient flow and Langevin ($T = 0.05$) start at $theta_A$ (top-left is late time) and run to $t = 400$.
  All curves closed form.
]) <fig-pareto>

== What the local model shows and hides

*New-task curvature against old precision.*
The rates are the eigenvalues of $H_B$ alone, or of $H_B + beta Lambda_A$ with a penalty.
The fraction of each direction that moves is set by $f_i \/ (a_i + f_i)$ (@sec-gaussian).

*Flat directions.* Flat directions of $H_B$ are where old knowledge survives gradient flow.
They are also where noise destroys it.
They are not the problem; directions flat for $B$ but in which $B$'s loss could fall #emph[non-locally] are.

*Local response against non-local exploration.* A dead unit has $g_B = 0$ and $H_B = 0$ in its coordinates.
The local model assigns it zero value.
Yet reviving it may lower $L_B$ a lot.
No local quantity (gradient, Fisher, Hessian) can see this value.
This is why utility for resetting must be #emph[counterfactual]: what would happen if the unit were re-drawn (@sec-utility).

*Waiting time against convergence time.* For a reset mechanism,
adaptation time = (wait for a useful reset, $approx 1 \/ (r h)$) + (training from the landing point).
Comparing only the first term with SGD's convergence time is a category error #wrong.

== A measurable adaptation-speed indicator

Let $ell_k (s)$ be the new-task training loss $s$ steps after the switch to task $k$.
Let $ell^"fresh" (s)$ be the same for a freshly initialised network of the same architecture and optimiser.
Define the #emph[relative area under the learning curve]:

$ cal(R)_k = frac(sum_(s <= S) ell_k (s), sum_(s <= S) ell^"fresh" (s)) . $

$cal(R)_k > 1$ means the network learns task $k$ more slowly than a fresh one: lost plasticity.
$cal(R)_k < 1$ means it learns faster: transfer.
It needs no threshold, works for any method, and is cheap to log.
A secondary indicator is the step count $s_epsilon$ to reach the loss a fresh network reaches at step $S \/ 2$.
Its ratio to $S \/ 2$ can be read as a time, matching the escape-time language of @sec-escape.
