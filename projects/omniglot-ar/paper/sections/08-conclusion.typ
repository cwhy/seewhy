#import "/template.typ": *

= Discussion and conclusion

== What was established

The substrate argument was correct and was never the obstacle. Omniglot does
everything it was chosen to do: memorisation is impossible by construction
rather than by an anonymisation trick, twenty drawings per character remove the
incentive to memorise at all, and the information is demonstrably present in the
tokens — nearest neighbour over exactly those pixels reaches 0.431 to 1.000
depending on the run. None of that was true of MNIST.

The obstacle was optimisation, and it took three stages to see that.

+ *Seven runs at chance*, including a positive control where the query image is
  a copy of its support and nearest neighbour scores 1.000. Varying the task —
  fewer classes, more pixels, a label field, binarised values, an ink-biased
  pool, coarse-and-complete observation — moved nothing.
+ *Controls localised the missing capability.* Leaking the answer into the
  query's own label token, or its own pixel tokens, both solve to 1.000 within
  300 steps. So the loss, target, forward pass, head, and `ref`-keyed attention
  all work. What was missing was content-dependent matching.
+ *That capability is learnable after all* — for #emph[exact] matching. At batch
  64 and learning rate $10^(-3)$ it appears as a sharp transition after ~2000
  flat steps and solves completely, in half the step budget the failed runs used.
+ *But exact and approximate matching dissociate.* The same recipe on the real
  task is flat at $ln 2$ after 25 000 steps, at $28 times 28$ (exp8) and at
  $10 times 10$ fully observed where nearest neighbour reaches 0.805 (exp10).
  Coarsening the images — the sharpest available test of "approximate is just
  far-away exact" — changes nothing.

The lesson generalises past this project: seven negative results sharing a batch
size and learning rate are one negative result. The interventions in
§#link(<sec:analysis>)[7.3] were all aimed at the task when the binding
constraint was the gradient's signal-to-noise, which is why none of them
registered.

== The shape of the obstacle

§#link(<sec:analysis>)[7.7] argues the dissociation comes from what the circuit
can accumulate. The label-field circuit pools a per-position agreement vote
additively; the margin between the right support and a wrong one is ≈0.3 when
matching is exact and ≈0.1 when it is approximate, in both cases buried in a sum
dominated by shared background. Nearest neighbour succeeds on the same pixels
because it normalises and takes an argmax across candidates — a global operation
over whole drawings that an additive per-position accumulator cannot express.

If that is right, the fix is not another optimiser sweep, and not another
task-side knob: it is a normalised, global comparison. A pooled per-drawing
representation, or an explicitly normalised similarity in the score function.
Both reintroduce a representation of a *sample* — precisely the structure the
token-level premise set out to dissolve. That tension is the project's result.

== Limitations

The negative half of this is bounded by budget: 25 000 steps at effective batch
64, 1.5 M parameters. Having watched one transition arrive abruptly at step
~2500, we are wary of declaring a second impossible, and a genuinely large-scale
run remains the honest caveat.

Two scope limits remain: positions are learned embeddings with no spatial prior,
so adjacency must itself be learned; and Omniglot's stroke programs are unused,
so compositionality is tested only through rendered bitmaps.

== What follows

The original plan — sweep episode shape, add masked-pixel completion, test
alphabet-level hold-out — presupposes a working matching step, and is now worth
resuming once exp8/exp9 settle whether approximate matching also transitions.
Ahead of that:

/ Report the plateau, not just the endpoint: any future run in this family
  should be judged on whether it crossed a transition, not on its accuracy at a
  fixed step count. A run reported as "chance at 12 000 steps" may be a run
  reported before its transition.

/ Tune batch and learning rate first: they were the binding constraint here and
  are cheap to sweep relative to architectural surgery.

/ Then change what is accumulated: a normalised similarity, or a pooled
  per-drawing summary token that the query compares against, directly targets
  the margin problem in §#link(<sec:analysis>)[7.7]. This is the first
  intervention that would test the premise itself rather than its tuning.
