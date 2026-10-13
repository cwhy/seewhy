#import "/template.typ": *

= Task and data <sec-task>

*Task.* Next-token prediction on web text. The corpus is FineWeb
(`HuggingFaceFW/fineweb`, subset `sample-10BT`), with the first 200,000 documents skipped, as in
the DUST release. Text is tokenised with the authors' byte-level byte-pair encoding, whose
vocabulary has $V = 4096$ entries including one end-of-document marker. Documents are joined with
that marker and cut into rows of 2,049 tokens: 2,048 inputs and the same sequence shifted by one
as targets.

*Training budgets.* One epoch over a fixed prefix of the token stream, in updates of 8 sequences
($8 times 2048 = 16384$ tokens). The 100K-token budget is 6 updates, the 1M budget 61, and the 10M
budget 610. Training sequences are shuffled once per seed. Smaller budgets are prefixes of larger
ones.

*Held-out data.* Validation and test are 544 sequences each, 1,114,112 predicted tokens apiece, cut
from documents that follow the training pool in the stream, so they overlap neither it nor each
other. Every reported loss is the *test* loss of the checkpoint with the lowest *validation* loss
over training. The validation set is therefore used for one decision, which checkpoint, and never
for fitting.

== Chance level

The loss is the mean over tokens of $ell_t = -ln p(y_t | x_(<= t))$. A model that spreads
probability uniformly over the vocabulary scores $ln V = ln 4096 = 8.318$; we measure
8.319 at initialisation, because the output head starts at near-zero weights.

That is chance, but it is not the floor a trained network should be read against. A model that has
learned only *how often each token occurs*, and ignores context entirely, scores lower. Counting
token frequencies on the 1M-token training prefix and scoring the test split with them gives
*6.994* (7.027 from the 100K prefix, 6.978 from the 10M prefix). Every loss in the results section
is read against this number: a network that lands above it has not yet learned anything a lookup
table could not, and one below it is using context.

== What makes the task hard, and the shortcut

The data is real text, so the loss cannot be driven toward zero: some tokens are unpredictable. At
one epoch over at most ten million tokens nothing is memorised, because no token is seen twice.
That is the point of the protocol: it measures how fast a method *learns*, not how well it can
fit a fixed set.

The shortcut to rule out is the unigram floor above: a method can lower its loss from 8.3 to
about 7 by learning token frequencies alone, without using a single context token. Every ladder in
this paper therefore carries that line, and a gain is only a gain in using context once it is below it.
