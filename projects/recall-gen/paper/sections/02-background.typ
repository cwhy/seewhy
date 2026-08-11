#import "/template.typ": *

// OBLIGATIONS
//  - Mandatory section. Defines everything §3-§7 rely on.
//  - Related work only insofar as it frames the question.

= Background

This section defines the machinery the rest of the paper uses. A reader who
writes code and reads statistics but has never trained a neural network should
be able to continue from here.

== Sequence models, and what a "token" is

A sequence model reads an ordered list of items and produces an output for each.
The items are called *tokens*. In language models a token is roughly a word; the
choice is a convention, not a requirement. Any object that can be turned into a
fixed-length vector of numbers can be a token, and in this paper each token is a
whole 28 #sym.times 28 image flattened into 784 numbers.

Internally the model represents each token as a vector of $d = 256$ numbers,
called its #gloss[embedding][a learned vector standing in for an input item],
obtained by multiplying the raw input by a matrix that is learned along with
everything else. Layers of the model transform these vectors; a final matrix
turns the last vector back into 784 predicted pixel values.

== Two ways to mix information between tokens

The interesting part of any sequence model is how information moves *between*
tokens. Two families matter here.

*Attention* @vaswani2017attention gives every token a query vector and every
token a key vector, scores each pair by their similarity, and lets each token
read a weighted average of all the others' value vectors. It is exact and
flexible: nothing is ever compressed away, because every token stays available
to be re-read. It is also expensive — with $N$ tokens there are $N^2$ pairs — and
its memory is the input itself.

*Linear recurrent models* instead sweep through the sequence once, carrying a
fixed-size state and updating it at each token. Cost grows linearly with $N$, but
the state cannot grow: everything the model will ever know about the first
thousand tokens has to fit in the same fixed budget. Recent architectures in this
family — DeltaNet @schlag2021linear, Mamba @gu2023mamba, and Kimi Delta Attention
@kda2025, which we use — make that state a *matrix* and update it with rules
borrowed from associative memory, which makes them surprisingly good at look-up
despite the compression.

The fixed budget is why this paper uses a linear recurrent model rather than
attention. With attention, "the context does not fit in memory" is not a
condition one can create; here it is a dial.

== Associative memory and the delta rule

A matrix state $S$ can be read as a lookup table over vectors. Store a key–value
pair by adding their outer product, $S <- S + v k^top$; retrieve by multiplying,
$S q approx v$ when $q$ is close to $k$ and the stored keys are close to
orthogonal. This is the classical linear associative memory, and it degrades
gracefully: as more pairs are written into a state of fixed size, retrievals
become blends of the nearest stored items rather than failing outright.

Writing $S <- S + v k^top$ blindly is wasteful, because whatever is already
stored under $k$ gets added to rather than replaced. The *delta rule* first reads
what is there and writes only the difference, $S <- S + beta(v - S k)k^top$. §4
gives the exact form we use.

The connection worth holding onto: a single such read is a *similarity-weighted
average of stored values*. That is exactly the "soft look-up" reference strategy
of §4 — which is why it, and not a hard nearest-neighbour, is the fair ceiling
for what this architecture can extract from its context.

== In-context learning, retrieval, and why they are hard to separate

A model is said to do in-context learning when its answer improves because of
examples in its input, with no weight update. The mechanism is contested. Some
of it is demonstrably retrieval-like: *induction heads*, circuits that find an
earlier occurrence of the current token and copy what followed it, form early in
training and account for a substantial part of the effect @olsson2022induction.
Other work argues that transformers can implement genuine learning algorithms
in their forward pass @vonoswald2023transformers.

Telling these apart on natural language is difficult, because the training corpus
is enormous and largely unknown: an apparent generalisation may be a retrieval
from something the model saw during training. The design in §3 removes that
problem by controlling, per episode, whether the answer is available at all.

== What "generalisation" means here

Because the task is pixel prediction rather than classification, we are precise
about the word. A model generalises, in this paper, when it produces a good
completion for a query whose true image is *not* in the context and *not* in its
training pool — condition D of §3. Doing well on such a query requires knowledge
about digits in general, held either in the weights or extracted from the other
context images. Doing well when the image *is* in the context requires only
finding it.
