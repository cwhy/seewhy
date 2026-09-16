#import "/template.typ": *
#import "@preview/cetz:0.4.2" as cetz

#show: deck.with(
  title: "Can cortex run predictive coding?",
  subtitle: [A biological-plausibility ledger for PC and PC-ALM \
    #text(size: 15pt)[from the local-learning replication — MNIST, 56,128 parameters]],
  footer: "PC and PC-ALM: the plausibility ledger",
)

#slide[The objection backprop has to answer][
  Backpropagation runs in three strict phases, in order, for every update:

  #v(4pt)
  #align(center, cetz.canvas(length: 1.05cm, {
    import cetz.draw: *
    let xs = (0, 3.2, 6.4, 9.6)
    for (i, x) in xs.enumerate() { neuron((x, 0), [#(i + 1)], fill: soft, r: 0.46) }
    // 1. forward, above the row
    for i in range(3) {
      flow((xs.at(i) + 0.5, 0.95), (xs.at(i + 1) - 0.5, 0.95), col: accent)
    }
    for (i, x) in xs.enumerate() {
      if i < 3 { line((x, 0.5), (x, 0.85), stroke: 0.7pt + accent) }
      if i > 0 { line((x, 0.5), (x, 0.85), stroke: 0.7pt + accent) }
    }
    content((4.8, 1.55), text(size: 13pt, fill: accent)[*1.* forward pass])
    // 2. backward, below the row, through the transpose
    for i in range(3) {
      flow((xs.at(i + 1) - 0.5, -0.95), (xs.at(i) + 0.5, -0.95), col: warn)
      content(((xs.at(i) + xs.at(i + 1)) / 2, -1.35), text(size: 12pt, fill: warn, $W_#(i + 2)^top$))
    }
    content((4.8, -2.05), text(size: 13pt, fill: warn)[*2.* backward pass — through the #underline[transpose] of the forward weights])
    content((4.8, -2.75), text(size: 13pt, fill: muted)[*3.* only now may any synapse change])
  }))

  #v(6pt)
  #grid(columns: (1fr, 1fr), gutter: 18pt,
    verdict("no")[*Phase locking.* An early synapse must hold its state and wait
      for a signal that has travelled to the output and back. No mechanism is
      known that could enforce that timing across cortex.],
    verdict("no")[*Weight transport.* The backward pass multiplies by $W^top$: a
      synapse would have to know the strength of a different synapse, in another
      neuron, pointing the other way.],
  )
]

#slide[What a synapse is allowed to use][
  #v(2pt)
  #align(center, cetz.canvas(length: 1cm, {
    import cetz.draw: *
    neuron((0, 0), [pre], fill: soft)
    neuron((5, 0), [post], fill: soft)
    line((0.45, 0), (4.55, 0), stroke: 1.4pt)
    circle((2.5, 0), radius: 0.22, fill: accent, stroke: none)
    content((1.45, 0.55), text(size: 12pt, fill: accent)[the synapse])
    flow((0.1, -1.5), (2.3, -0.35), col: muted, label: none)
    content((0.1, -1.8), text(size: 12pt)[its own activity])
    flow((4.9, -1.5), (2.7, -0.35), col: muted)
    content((5.2, -1.8), text(size: 12pt)[its neuron's state])
    flow((2.5, 2.0), (2.5, 0.35), col: good)
    content((2.5, 2.35), text(size: 12pt, fill: good)[a broadcast signal (neuromodulator)])
  }))

  #v(4pt)
  A learning rule is *local* if it needs nothing else. Predictive coding's claim
  is that a network of such synapses, left to settle, computes what backprop
  computes — without the three phases, and without waiting.
]

#slide[Predictive coding: every layer predicts the one below][
  #grid(columns: (1.35fr, 1fr), gutter: 16pt,
    cetz.canvas(length: 1.0cm, {
      import cetz.draw: *
      let xs = (0, 2.7, 5.4)
      content((-1.5, 0), text(size: 12pt, fill: muted)[input])
      line((-1.05, 0), (-0.5, 0), stroke: 0.9pt + muted, mark: (end: ">", scale: 0.5))
      for (i, x) in xs.enumerate() {
        neuron((x, 0), $z_#(i + 1)$, fill: soft)
        neuron((x, 2.1), $r_#(i + 1)$, fill: warn.lighten(82%), stroke: warn)
        line((x, 0.44), (x, 1.66), stroke: (paint: muted, thickness: 0.8pt, dash: "dashed"))
      }
      for i in range(2) {
        flow((xs.at(i) + 0.45, 0), (xs.at(i + 1) - 0.45, 0), col: accent)
      }
      // the error message coming back down one layer — the transport term
      bezier((xs.at(1), 2.54), (xs.at(0) + 0.22, 0.42), (xs.at(0) + 0.6, 2.9),
             stroke: (paint: warn, thickness: 1pt),
             mark: (end: ">", fill: warn, scale: 0.5))
      content((xs.at(0) + 1.35, 3.05), text(size: 12pt, fill: warn, $W_2^top$))
      line((xs.at(2) + 0.45, 0), (xs.at(2) + 1.5, 0), stroke: 0.9pt + muted,
           mark: (end: ">", scale: 0.5))
      content((xs.at(2) + 2.15, 0), text(size: 12pt, fill: muted)[target \ #text(size: 10pt)[(clamped)]])
      content((2.7, -1.15), text(size: 12pt, fill: accent)[predictions travel forward])
      content((2.7, -1.75), text(size: 12pt, fill: warn)[errors travel back one layer])
    }),
    [
      Each layer holds a *value* $z_i$ and an *error* $r_i$ — the part of its
      state the layer below failed to predict.

      #v(6pt)
      Inference is not a pass. The values settle, exchanging messages only with
      their neighbours, for as long as it takes:

      #v(2pt)
      $ z_i <- z_i - eta (underbrace(r_i, "from below") - underbrace(W_(i+1)^top (sigma' ⊙ r_(i+1)), "from above")) $

      #v(4pt)
      Then one weight step, presynaptic activity times postsynaptic error — a
      delta rule.
    ])
]

#slide[What predictive coding fixes, and what it does not][
  #grid(columns: (1fr, 1fr), gutter: 22pt,
    [
      #text(fill: good, weight: "bold")[Fixed]
      #v(4pt)
      #verdict("yes")[*No phase locking.* Layers run continuously; nothing waits
        for a signal to return from the output.]
      #v(6pt)
      #verdict("yes")[*Local computation.* Every update needs only the layer
        above and the layer below.]
      #v(6pt)
      #verdict("yes")[*Hebbian-shaped update.* Postsynaptic error times
        presynaptic activity, which is the form real plasticity rules take.]
    ],
    [
      #text(fill: warn, weight: "bold")[Not fixed]
      #v(4pt)
      #verdict("no")[*Weight transport, again.* $W_(i+1)^top$ sits in the
        settling equation. The objection backprop was charged with is inherited,
        not answered.]
      #v(6pt)
      #verdict("no")[*The target is clamped* onto the output for the whole
        settling phase — a supervisor holding the answer in place.]
      #v(6pt)
      #verdict("~")[*A second population.* One error neuron per value neuron
        doubles the circuit. Plausible, but a real claim about anatomy.]
    ])
]

#slide[Where predictive coding fails — and it is not biology][
  #grid(columns: (1.15fr, 1fr), gutter: 16pt,
    figure(include "/figures/pc_depth.typ"),
    [
      Supervision enters only at the output and has to work down through a chain
      of local compromises. Each layer attenuates it.

      #v(6pt)
      Measured on our 32-unit-wide residual network, one epoch of MNIST:
      predictive coding holds level with backprop to 8 layers, then falls away —
      0.9055 at two layers, 0.6137 at 32.

      #v(6pt)
      #note[The gradient it produces drifts with it: cosine to the true gradient
        0.999 at two layers, 0.806 at 32. The rule is local, but past a certain
        depth it is local to nothing useful.]
    ])
]

#slide[PC-ALM adds one neuron per unit: an integrator][
  #grid(columns: (1.3fr, 1fr), gutter: 16pt,
    cetz.canvas(length: 1.0cm, {
      import cetz.draw: *
      let xs = (0, 2.7, 5.4)
      for (i, x) in xs.enumerate() {
        neuron((x, 0), $z_#(i + 1)$, fill: soft)
        neuron((x, 2.1), $r_#(i + 1)$, fill: warn.lighten(82%), stroke: warn)
        neuron((x, 4.2), $lambda_#(i + 1)$, fill: accent.lighten(82%), stroke: accent)
        line((x, 0.44), (x, 1.66), stroke: (paint: muted, thickness: 0.8pt, dash: "dashed"))
        // error accumulates into the dual
        flow((x, 2.54), (x, 3.76), col: accent)
        // and the dual pushes back on the settling
        bezier((x + 0.42, 4.2), (x + 0.42, 0.1), (x + 2.15, 2.1),
               stroke: (paint: accent, thickness: 0.9pt, dash: "dashed"),
               mark: (end: ">", fill: accent, scale: 0.45))
      }
      for i in range(2) {
        flow((xs.at(i) + 0.45, 0), (xs.at(i + 1) - 0.45, 0), col: accent)
      }
      content((8.6, 4.2), text(size: 12pt, fill: accent)[integrates the error \ its own layer makes])
      content((8.6, 1.6), text(size: 12pt, fill: accent)[and feeds it back \ into the settling])
      content((2.7, -1.2), text(size: 12pt, fill: muted)[the only change: one integrating \ neuron per unit (blue)])
    }),
    [
      $ lambda_i <- lambda_i + alpha r_i $

      #v(4pt)
      The dual neuron integrates the error its own layer keeps making, and feeds
      it back into the settling. Proportional plus integral: each layer becomes a
      small feedback controller.

      #v(8pt)
      LeCun showed in 1988 that these multipliers *are* the backprop credit
      signals at equilibrium. PC-ALM is the observation that local dynamics can
      accumulate them.
    ])
]

#slide[What the integrator buys][
  #grid(columns: (1fr, 1fr), gutter: 20pt,
    [
      *It recovers the gradient exactly.* In a linear network, run to
      convergence, the update PC-ALM produces converges on backprop's:

      #v(4pt)
      #align(center, table(
        columns: 3, align: (right, right, right), stroke: 0.4pt + luma(200),
        inset: 6pt,
        text(size: 14pt)[settling steps], text(size: 14pt)[PC-ALM], text(size: 14pt)[PC],
        text(size: 14pt)[8], text(size: 14pt)[0.766], text(size: 14pt)[0.704],
        text(size: 14pt)[512], text(size: 14pt)[0.994], text(size: 14pt)[0.971],
        text(size: 14pt)[2048], text(size: 14pt)[*0.999982*], text(size: 14pt)[0.968],
      ))
      #v(2pt)
      #text(size: 13pt, fill: muted)[cosine to the backpropagation gradient]
    ],
    [
      *And it survives depth.* Same network, same budget, one scalar different
      ($alpha = 0$ is predictive coding exactly):

      #v(6pt)
      #align(center, table(
        columns: 3, align: (right, right, right), stroke: 0.4pt + luma(200),
        inset: 6pt,
        text(size: 14pt)[layers], text(size: 14pt)[PC], text(size: 14pt)[PC-ALM],
        text(size: 14pt)[2], text(size: 14pt)[0.9055], text(size: 14pt)[0.9049],
        text(size: 14pt)[16], text(size: 14pt)[0.8043], text(size: 14pt)[0.8628],
        text(size: 14pt)[32], text(size: 14pt)[0.6137], text(size: 14pt)[0.8554],
        text(size: 14pt)[64], text(size: 14pt)[0.5547], text(size: 14pt)[0.8207],
      ))
      #v(2pt)
      #text(size: 13pt, fill: muted)[MNIST test accuracy, one epoch, three seeds]
    ])
]

#slide[PC-ALM's ledger: one fixed, one added, one given up][
  #grid(columns: (1fr, 1fr), gutter: 22pt,
    [
      #verdict("yes")[*The integrator is cheap, biologically.* Slow accumulation
        of a local error is what dendritic calcium and slow membrane potentials
        already do. Asking for one is a mild request.]
      #v(8pt)
      #verdict("yes")[*Credit reaches depth.* Whatever the objection, it is no
        longer that the signal dies before it arrives.]
    ],
    [
      #verdict("no")[*Weight transport is untouched.* $W_(i+1)^top$ is still in
        the settling equation. PC-ALM improves the part predictive coding was
        good at and leaves the part it was criticised for.]
      #v(8pt)
      #verdict("~")[*Prospective configuration is given up.* Plain PC's settled
        activations move towards the target before the weights change — the
        property argued to make it a better model of learning than backprop.
        PC-ALM's activations return to the forward pass.]
    ])
  #v(8pt)
  #note[So PC-ALM is *better credit assignment* and, on one axis, *a worse
    biological story* than the method it improves.]
]

#slide[The ledger, side by side][
  #v(4pt)
  #align(center, table(
    columns: (1.7fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center),
    stroke: 0.4pt + luma(200), inset: 8pt,
    text(size: 15pt)[], text(size: 15pt, weight: "bold")[backprop],
    text(size: 15pt, weight: "bold")[PC], text(size: 15pt, weight: "bold")[PC-ALM],

    text(size: 15pt)[No phase locking], text(fill: warn)[✗], text(fill: good)[✓], text(fill: good)[✓],
    text(size: 15pt)[Local computation only], text(fill: warn)[✗], text(fill: good)[✓], text(fill: good)[✓],
    text(size: 15pt)[Hebbian-shaped weight update], text(fill: warn)[✗], text(fill: good)[✓], text(fill: good)[✓],
    text(size: 15pt)[No weight transport], text(fill: warn)[✗], text(fill: warn)[✗], text(fill: warn)[✗],
    text(size: 15pt)[No clamped target], text(fill: good)[✓], text(fill: warn)[✗], text(fill: warn)[✗],
    text(size: 15pt)[No extra neuron population], text(fill: good)[✓], text(fill: warn)[✗], text(fill: warn)[✗✗],
    text(size: 15pt)[Prospective configuration], text(fill: warn)[✗], text(fill: good)[✓], text(fill: warn)[✗],
    text(size: 15pt)[Credit survives 32 layers], text(fill: good)[✓], text(fill: warn)[✗], text(fill: good)[✓],
  ))
]

#slide[The one test that would settle the last row][
  Weight transport is the only objection both methods still fail — and it is
  directly testable, because the fix is known: replace $W_(i+1)^top$ in the
  settling equation with a *fixed random matrix*, as feedback alignment does.

  #v(8pt)
  #grid(columns: (1fr, 1fr), gutter: 20pt,
    note[*What we already know.* On this same network, feedback alignment with
      random backward weights tracks backprop at 2 and 4 layers, is at chance by
      16, and diverges at 64. Random feedback alone does not survive depth.],
    note[*What is not known.* Predictive coding settles for $T$ steps between
      weight updates, and PC-ALM's integrators accumulate over those steps. That
      is exactly the machinery that might repair a feedback signal feedback
      alignment cannot.],
  )

  #v(8pt)
  If PC-ALM survives random feedback, its last biological objection goes away.
  If it collapses, its locality was resting on transport all along.
]
