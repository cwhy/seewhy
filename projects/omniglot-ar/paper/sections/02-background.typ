#import "/template.typ": *

= Background

== The token-level premise

The formulation under study comes from `projects/universal-ar`. A dataset is not
a set of samples but a flat bag of tokens, each a triple $(p, v, r)$:

- $p$ — #m[position], the address within an image (or a reserved address for the
  label);
- $v$ — #m[value], the content at that address, drawn from a single vocabulary
  that unifies pixel-intensity bins with class labels;
- $r$ — #m[ref], a coreference tag shared by every token belonging to the same
  sample and re-drawn at random each episode.

Every task is then one operation: complete a masked token's value given its
address. Inpainting completes a pixel; classification completes the token at the
reserved label address. Attention runs over the bag with no causal mask, because
the bag is a set and $p$ is a field rather than sequence order.

== The prior negative result

Thirty-nine experiments applied this to MNIST. Under per-episode label
anonymisation — where the class-to-label-slot map is re-randomised each episode,
so a memorised map is worthless — a two-class problem sat at chance across six
architectural interventions.

#figure(
  table(
    columns: (1fr, auto),
    [*condition (universal-ar, 4 vs 9)*], [*label accuracy*],
    [deterministic labels (exp28)], [0.875],
    [deterministic labels, PCA-32 (exp34)], [0.977],
    [anonymised labels (exp13, exp15)], [~0.50],
    [anon. + MLP-combiner embedding (exp22)], [0.461],
    [anon. + context-generated weights (exp24)], [0.508],
    [anon. + FiLM conditioning (exp25)], [0.422],
    [anon. + retrieval-only training data (exp26)], [0.516],
    [anon. + task-balanced loss (exp30)], [0.445],
    [anon. + PCA-32 features (exp31)], [0.477],
  ),
  caption: [Prior results on MNIST. Chance is 0.500. With labels held
    deterministic the same encoder reaches 0.88–0.98, so it can separate the two
    digits; it never binds that knowledge to a label whose meaning changes every
    episode.],
)

== Why MNIST cannot settle the question

Three properties of MNIST make that result uninformative about the formulation,
and they compound.

/ Ten classes, ~6000 examples each: Memorising a class prototype in the weights
  is always the fastest descent direction. Anonymisation removes the payoff at
  evaluation but not the pressure during training — across episodes the label
  token is uncorrelated with the image, so the shortest path is to ignore the
  label pathway entirely, and the gradient never acquires a reason to build the
  binding circuit that the claim is about.

/ Held-out samples are not held-out concepts: The split was over samples, so the
  generalisation metric measured within-class interpolation, which a model that
  memorised ten prototypes scores perfectly on. Nothing asked whether the
  mechanism extends to an unseen class.

/ Digits have no part structure: The founding principle — bind seen parts into an
  unseen whole — has no substrate in MNIST.

== Omniglot

Omniglot @lake2015 was constructed as the transfer-learning inverse of MNIST:
1623 characters across 50 alphabets, with only 20 drawings of each, produced by
different writers. It repairs all three properties.

Twenty drawings is far too few to build a usable prototype in the weights, so
reading the support set is the only route to an answer — the pressure that made
MNIST degenerate is removed by the data itself. The standard background /
evaluation split (964 characters from 30 alphabets versus 659 from 20) shares no
characters, so test episodes use classes the model has never seen and
memorisation is impossible *by construction* rather than by an anonymisation
trick. And characters are composed of strokes and grouped into alphabets, giving
the compositional hold-out principle a real substrate.

Two practical properties matter too: strokes are sparse and high-contrast, so
content matching has strong signal; and the images are near-binary, which
collapses the pixel-bin vocabulary and removes the loss imbalance that let pixel
cross-entropy (~16 nats) swamp label cross-entropy (~0.05) in the prior work.
