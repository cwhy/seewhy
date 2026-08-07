#import "/template.typ": *

= The question

A dataset is a flat bag of `(pos, value, ref)` tokens. Every task — classifying,
inpainting, denoising — is one operation: complete a masked token's value given
its address. Classification is predicting the value at `pos_label`. That premise
is inherited from `projects/universal-ar`, and it is not what is under test here.

What is under test is narrower:

#callout(title: [Claim])[
  A token-level attention model, trained only on episodes built from Omniglot's
  #emph[background] characters, can classify characters from the
  #emph[evaluation] split — which it has never seen — in context, above chance
  and above a pixel nearest-neighbour baseline computed on the very same pixels.
]

Thirty-nine experiments on MNIST said no. Under per-episode label anonymisation,
4-versus-9 sat at chance across six architectural interventions: an MLP
token-embedding combiner, context-generated weights, FiLM conditioning,
retrieval-only training data, a task-balanced loss, and PCA features.

#table(
  columns: (1fr, auto),
  [*condition (universal-ar, 4v9)*], [*label accuracy*],
  [deterministic labels (exp28)], [0.875],
  [deterministic labels, PCA-32 (exp34)], [0.977],
  [anonymised labels (exp13, exp15)], [~0.50],
  [anon. + MLP-combiner embedding (exp22)], [0.461],
  [anon. + context-generated weights (exp24)], [0.508],
  [anon. + FiLM conditioning (exp25)], [0.422],
  [anon. + retrieval-only training data (exp26)], [0.516],
  [anon. + task-balanced loss (exp30)], [0.445],
  [anon. + PCA-32 features (exp31)], [0.477],
)

With labels held deterministic the same encoder reached 0.88–0.98, so it could
always tell the two digits apart. It simply never learned to bind that knowledge
to a label token whose meaning changed every episode.

Six failed interventions on one dataset is a hint that the dataset, not the
architecture, is the variable worth moving. The next section argues why.
