#import "/template.typ": *

= Background <sec-background>

This section defines everything the rest of the paper relies on. A reader who
has trained transformers can skip to @sec-task.

== What a transformer does

A transformer maps a sequence of symbols to a prediction of what comes next.
Symbols are integers — a *vocabulary* is just the set of distinct symbols a
model can read or write.

Three stages:

+ *Embedding.* Each input symbol $x_t$ is replaced by a vector. Two lookup
  tables contribute: a token embedding $E_"tok"$, indexed by which symbol it is,
  and a positional embedding $E_"pos"$, indexed by where in the sequence it
  sits. Their sum is the starting representation $h_t^((0))$. The positional
  table exists because the next stage has no inherent notion of order.

+ *The stack.* A sequence of identical blocks, each refining the vectors. One
  block does two things. #gloss[Self-attention][each position computes a
  weighted average of the other positions' vectors, choosing the weights from
  the vectors themselves] lets positions exchange information. A
  #gloss[feed-forward network][a small two-layer function applied independently
  at each position] then transforms each position on its own. Both are wrapped
  in residual connections, meaning each writes an update that is *added* to the
  running representation rather than replacing it.

+ *Unembedding.* The final vector at each position is multiplied by a matrix $U$
  to produce one score per vocabulary symbol. A softmax turns those scores into
  a probability distribution over the next symbol.

Attention is *causal* here: position $t$ may look at positions $1 dots t$ but
not beyond, which is what makes next-symbol prediction well posed.

The *width* $d$ is the length of the vectors flowing through the stack. It is
the quantity this paper is really about. A block holds roughly $12 d^2$
parameters, so the stack's size grows quadratically in width, while the two
embedding tables and the unembedding grow only linearly.

== Training, and what it means to freeze

Training minimises a loss by #gloss[gradient descent][repeatedly nudging every
parameter in the direction that most reduces the loss]. The loss here is
*cross-entropy*: the negative log of the probability the model assigned to the
symbol that actually came next, averaged over positions. It is measured in nats;
a model that assigns probability $1$ to the truth scores $0$, and a model that
spreads its probability uniformly over $v$ symbols scores $ln v$.

*Freezing* a parameter means excluding it from that update. It keeps its initial
value forever. Crucially, gradients still flow *through* a frozen weight. The
weights at the edges can only be improved by computing how the loss depends on
them. That computation passes through the frozen interior.
Freezing removes the interior from the search, not from the computation.

#callout(title: [The one distinction the whole paper turns on])[
  A *fully trained* model optimises everything. A *random transformer*, in this
  paper's sense, optimises only $E_"tok"$, $E_"pos"$ and $U$, and holds every
  attention and feed-forward weight at its random initial value. The two are
  otherwise identical — same architecture, same data, same optimiser.
]

== Why this is a meaningful probe

Consider what a random transformer can and cannot express. The function computed
by the frozen stack is fixed before any data arrives; call it $F$. Training
chooses an encoding $E$ of inputs into vectors and a decoding $U$ of vectors
into predictions. The model is $U compose F compose E$, and only the outer two
are learnable.

So if such a model achieves a task, the composition $F$ must *already* map some
encoding of the inputs to some linearly-readable representation of the correct
outputs. Training did not create that correspondence; it searched for a way to
address it. This is why success is evidence about the architecture and its
initialisation rather than about the learning signal.

The inverse also holds, and gives the paper its negative results. Anything the
model must *store* — an arbitrary lookup table with no structure to exploit —
has nowhere to go but $E$ and $U$, because $F$ cannot be written to. Capacity
should therefore be much lower than for a fully trained model, and it is
(@sec-results).

== Related work

The tasks are chosen because each is already understood in fully trained
models. Modular addition is the standard setting for #gloss[grokking][a long
plateau of memorisation followed by an abrupt jump to generalisation]
@power2022grokking. The circuits trained models use for it have been
reverse-engineered @nanda2023progress @zhong2023clock. Associative recall is the
setting where *induction heads* — attention patterns that find an earlier
occurrence of the current token and copy what followed it — were identified
@olsson2022context. Parenthesis balancing connects to formal results on what
attention can express @yao2021self.

Separately, a line of work shows that randomly initialised networks contain
subnetworks that already perform well without weight training — the lottery
ticket hypothesis @frankle2019lottery and its "supermasks" successors
@zhou2019deconstructing @ramanujan2020whats. Those results find sparse
subnetworks by *pruning*. As @sec-analysis shows, the mechanism here is
measurably not that.
