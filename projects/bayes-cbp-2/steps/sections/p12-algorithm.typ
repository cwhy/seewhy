#import "/template.typ": *

= Turning the theory into a trainable algorithm <p12>

#callout(title: [The question])[
  Pick the most promising candidate from Stage P11 and make it a complete algorithm:
  state variables; online updates; trigger probability or strength; reset distribution; how hyperparameters are set;
  at which level resets act (parameter, unit, feature, structure); handling of input and output weights and optimiser state;
  numerical safeguards; time and memory per step; PyTorch-ready pseudocode.
  Explain the difference between a Bayesian posterior update and an active reset intervention.
  Keep at least two alternatives from different routes.
]

== Core view

*Change-Point Reset* (CPR) is Stage P11's Route 2 made concrete.
It acts on units: a unit's input weights, bias and outgoing weights are replaced together.
Every few steps it computes each unit's posterior probability of having been replaced, then resets by sampling that probability.
Its overhead is about three forward passes of a layer every $K$ steps.
Its status as a whole is #heur: each step follows from the model, but nothing has run on a network.

== Derivation

=== What is inference and what is intervention

CPR does not make the network Bayesian. SGD or Adam still trains the weights.
Bayesian reasoning enters only the reset decision:

- *Inference:* computing $r_j$, the posterior probability that unit $j$ was replaced, under the change-point model of Stage P11. This is a posterior update.
- *Intervention:* drawing $c_j tilde "Bernoulli"(r_j)$ and overwriting the unit's weights. This acts on the point estimate. It implements Thompson sampling of the change indicator. It is not part of Bayes' rule.

The "posterior" of the weights is itself a caricature: a diagonal Laplace approximation built from moving averages.

=== State

Per hidden layer with $n$ units, $d$ inputs and $m$ next-layer pre-activations:

#notation(
  ($W in bb(R)^(n times d), thin phi in bb(R)^n$, [input weights and biases (trained by the optimiser)]),
  ($V in bb(R)^(m times n), thin phi' in bb(R)^m$, [next layer's weights and biases (trained)]),
  ($overline(h) in bb(R)^n$, [moving average of each unit's activation]),
  ($overline(h^2) in bb(R)^n$, [moving average of each squared activation]),
  ($hat(G) in bb(R)^m$, [moving average of the diagonal output Fisher at $z$]),
  ($"age" in bb(N)^n$, [steps since each unit's last reset]),
)

Hyperparameters: prior hazard $rho$ per step, output-weight prior scale $s_v$, initial distribution $p_0$, evaluation interval $K$, maturity $a_"min"$, and a per-evaluation budget $beta_"max"$.

=== Update equations

*Every step*, after the backward pass, with moving-average rate $epsilon$:

$ overline(h) <- (1 - epsilon) overline(h) + epsilon thin "mean"_b h_b, quad overline(h^2) <- (1 - epsilon) overline(h^2) + epsilon thin "mean"_b h_b^2, quad hat(G) <- (1 - epsilon) hat(G) + epsilon thin "mean"_b tilde(g)_b^2 . $

Here $tilde(g)_b$ is the gradient at $z$ for a label #emph[sampled from the model], not the data label. This makes $hat(G)$ a Fisher estimate rather than the empirical Fisher (@p2).

*Every $K$ steps*, on the current batch of $N$ examples:

+ *Prior per evaluation.* $rho_K = 1 - (1 - rho)^K$.
+ *Removal cost* (with the mean moved into the next bias): $delta_(b j) = (h_(b j) - overline(h)_j) V_(:, j)$ and
  $c_j^"rem" = sum_b [ -g_b^top delta_(b j) + 1/2 delta_(b j)^top "diag"(hat(G)) thin delta_(b j) ]$.
+ *Stay term.* The old posterior variance of $V_(k j)$ is $S_(k j) = (s_v^(-2) + N_"mem" thin overline(h^2)_j hat(G)_k)^(-1)$, with memory $N_"mem" = N \/ rho$ examples, the expected lifetime of a unit under the model. With $a^s_(k j) = sum_b h_(b j) g_(b k)$ and $B^s_(k j) = sum_b h_(b j)^2 hat(G)_k$:
  $ O_j^"stay" = sum_k [ frac((a^s_(k j))^2, 2 (B^s_(k j) + S_(k j)^(-1))) - 1/2 log(1 + S_(k j) B^s_(k j)) ]. $
+ *Fresh term.* Draw $w'_j tilde p_0$, compute centred candidate activations $h'_(b j)$, then $a_(k j) = sum_b h'_(b j) g_(b k)$, $B_(k j) = sum_b h'^2_(b j) hat(G)_k$, and
  $ F_j = sum_k [ frac(a_(k j)^2, 2 (B_(k j) + s_v^(-2))) - 1/2 log(1 + s_v^2 B_(k j)) ]. $
+ *Posterior change probability.* $Delta_j = "clip"(-c_j^"rem" + F_j - O_j^"stay", -20, 20)$ and $r_j = sigma("logit" rho_K + Delta_j)$.
+ *Decision.* For mature units ($"age"_j >= a_"min"$), draw $c_j tilde "Bernoulli"(r_j)$. If more than $beta_"max" n$ units are drawn, keep a random subset of that size.

=== Reset distribution

For each unit with $c_j = 1$:

- input weights $W_(j,:) <- w'_j$ (the same draw used for $F_j$), bias $phi_j <- 0$;
- next-layer bias $phi' <- phi' + V_(:, j) overline(h)_j - v'_j overline(h'_j)$, so the mean output is preserved;
- outgoing weights $V_(:, j) <- v'_j$ with $v'_(k j) = -a_(k j) \/ (B_(k j) + s_v^(-2))$, the posterior mean given the batch. Use $v'_j = 0$ to keep the function unchanged at reset;
- statistics: $overline(h)_j$ and $overline(h^2)_j$ from the candidate's batch activations; $"age"_j <- 0$.

=== Optimiser state

Adam keeps moments per weight. For every weight of a reset unit (row $j$ of $W$, entry $phi_j$, column $j$ of $V$):

- first moment $<- 0$;
- second moment $<-$ the median of that tensor's second moments. A zero second moment would make the first steps about $eta$ in size regardless of the gradient (@p4).

PyTorch's Adam keeps one step counter per tensor, so the bias correction cannot be reset per row. Seeding the second moment with the median avoids the problem.

=== Numerical safeguards

- Work in log-odds; clip $Delta_j$ to $[-20, 20]$.
- $B + s_v^(-2) >= s_v^(-2) > 0$, so no division by zero; floor $hat(G)$ at $10^(-8)$.
- Use $1 - (1 - rho)^K$, never $rho K$, so probabilities stay below one.
- If a candidate is dead on the batch ($h' = 0$ everywhere), redraw it, at most three times.
- Clip $||v'_j||$ to the layer's median column norm of $V$, to bound the function change of a reset.
- The budget $beta_"max"$ bounds forgetting per evaluation however wrong the evidence is.

=== Cost

Per evaluation and layer: one candidate forward pass, $O(N d n)$; the removal, stay and fresh terms, $O(N n m)$ each, i.e. matrix products of layer size.
So the overhead is about three forward passes of the layer every $K$ steps, plus one extra backward pass to the layer for the model-sampled $tilde(g)$ every step.
Memory: $O(n + m)$ per layer beyond the weights.

=== Pseudocode

```python
# CPR for one Linear+ReLU hidden layer `fc` followed by `nxt` (also Linear).
# Hooks record h = relu(fc(x)) and g = dloss/dz at z = nxt(h) for the batch.

@torch.no_grad()
def update_stats(st, h, g_sampled, eps):
    st.h_mean.lerp_(h.mean(0), eps)
    st.h_sq.lerp_((h ** 2).mean(0), eps)
    st.G.lerp_((g_sampled ** 2).mean(0), eps)
    st.age += 1

@torch.no_grad()
def cpr_step(fc, nxt, opt, st, x, h, g, hp):
    N = h.shape[0]
    rho_K = 1 - (1 - hp.rho) ** hp.K
    G = st.G.clamp_min(1e-8)                          # (m,)
    V = nxt.weight                                    # (m, n)

    # removal cost, centred
    hc = h - st.h_mean                                # (N, n)
    first = -(hc * (g @ V)).sum(0)                    # -sum_b g_b . delta_bj
    second = 0.5 * (hc ** 2).sum(0) * ((V ** 2) * G[:, None]).sum(0)
    c_rem = first + second                            # (n,)

    # stay term: re-fit v_j from its old posterior
    S_inv = hp.sv ** -2 + (N / hp.rho) * st.h_sq[None, :] * G[:, None]  # (m, n)
    a_s = g.T @ h                                     # (m, n)
    B_s = G[:, None] * (h ** 2).sum(0)[None, :]
    O_stay = (a_s ** 2 / (2 * (B_s + S_inv)) - 0.5 * torch.log1p(B_s / S_inv)).sum(0)

    # fresh term: one candidate per unit
    W_new = sample_init_like(fc.weight)               # (n, d), from p0
    h_new = torch.relu(x @ W_new.T)                   # bias 0
    h_new_mean = h_new.mean(0)
    hn = h_new - h_new_mean
    a = g.T @ hn                                      # (m, n)
    B = G[:, None] * (hn ** 2).sum(0)[None, :]
    prec = B + hp.sv ** -2
    F = (a ** 2 / (2 * prec) - 0.5 * torch.log1p(hp.sv ** 2 * B)).sum(0)

    delta = (-c_rem + F - O_stay).clamp(-20, 20)
    r = torch.sigmoid(torch.logit(torch.tensor(rho_K)) + delta)
    reset = (torch.rand_like(r) < r) & (st.age >= hp.a_min)
    reset = cap_random(reset, int(hp.beta_max * r.numel()))
    j = reset.nonzero().squeeze(1)
    if j.numel() == 0:
        return r

    v_new = -a[:, j] / prec[:, j]                     # Newton fit
    v_new = clip_cols(v_new, V.norm(dim=0).median())
    nxt.bias += V[:, j] @ st.h_mean[j] - v_new @ h_new_mean[j]
    fc.weight[j] = W_new[j]; fc.bias[j] = 0
    V[:, j] = v_new
    st.h_mean[j] = h_new_mean[j]
    st.h_sq[j] = (h_new[:, j] ** 2).mean(0)
    st.age[j] = 0
    reset_adam_rows(opt, fc.weight, rows=j); reset_adam_rows(opt, fc.bias, rows=j)
    reset_adam_cols(opt, V, cols=j)                   # m <- 0, v <- median(v)
    return r
```

== When it holds

- The evidence uses Stage P11's second-order expansion in output space and a diagonal Fisher #cond.
- $N_"mem" = N \/ rho$ assumes the Fisher is roughly constant over a unit's lifetime #heur.
- The Laplace "posterior" of the weights is a caricature built from moving averages.

== Limits

- The model never values retention (Stage P11); CPR alone will trade old knowledge for plasticity at a rate set by $rho$.
- One candidate per evaluation gives a noisy, biased estimate of the fresh evidence.
- $rho$ is not learnable from dead units (with zero output initialisation they return $r_j = rho$ whatever $rho$ is).

== Alternatives


*CPR-D (Route 3).* Replace the posterior probability by the value $V_j = G_j (H) - c_j^(B,"rem") - kappa c_j^(A,"rem")$.
Reset when $V_j > 0$, within the budget.
Draw $M$ candidates and keep the best.
Needs stored old-task statistics ($overline(h)$, $overline(h^2)$, $hat(G)$ frozen at the end of each task) for $c^(A,"rem")$.

*MPR, the Metropolis-corrected reset (@p6).* At rate $lambda_0$, propose a whole new unit from the prior: $w'_j tilde p_0$ and $v'_j tilde cal(N)(0, s_v^2 I)$.
Because the proposal is the prior, the prior and proposal ratios cancel. Accept with probability $min{1, exp(L_"old" - L_"new")}$, the likelihood ratio.
(Proposing $v'_j = 0$ deterministically would break reversibility: the reverse move could never be proposed.)
On a minibatch this test is biased #cond; it is exact only with the full-data likelihood.
MPR tests whether computational plasticity alone is enough.


== Next step

Stage P13 attacks everything so far, including CPR, as a hostile reviewer would, and revises the algorithm.
