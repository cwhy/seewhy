# AR-Recall — Concepts

Recall-gen made one image one token and masked half of it. This project takes the
same question apart into a token stream: an item is no longer a token, it is a
run of `(label, position, value)` triples, and recall means binding a label to
content spread across many tokens.

## Task / data

**Labels.** A fixed vocabulary of label tokens. An episode samples `L` of them
and binds each to one image, **shuffled**, so which label owns which image is
episode-specific and cannot be memorised in the weights. This is what makes the
binding an in-context problem rather than a lookup.

**Items.** MNIST first. An image is `P` positions, each carrying a value
quantised to `V` bins. Positions and values are tokens like labels are.

**The stream.** For each label, a revealed subset of its positions is emitted as

    label_a pos_3 val_9   label_a pos_7 val_2   label_b pos_3 val_0   ...

Three tokens per revealed value. Half of each image is revealed, which is the
same fraction recall-gen masked — this is the replication, restructured.

**The query.** `(label_q, pos_q, ?)` and the model predicts the value token.
`label_q` is always a label that appears in the context: the whole point is
recall of an item that is *partly* present, so the label must be bindable.

## The 2x2

Two independent axes, and they are not the same question.

|                        | `(label_q, pos_q)` **in this context** | **not in this context** |
|------------------------|----------------------------------------|--------------------------|
| combination **seen in training** | A — copy something already in the stream | C — infer within a familiar binding |
| combination **held out of training** | B — copy, but this pairing is new | D — infer, and the pairing is new |

**Axis 1, context.** Did the exact triple `(label_q, pos_q, v)` already appear in
this episode? If it did, the answer is in the stream and the task is retrieval.
If it did not, only *other* positions of `label_q` appeared, and the value has to
come from those plus whatever structure the other labels reveal.

**Axis 2, training.** Was the *combination* `label_q x pos_q` ever seen during
training? Labels are shuffled per episode, so over enough episodes almost every
combination occurs — unless a subset is deliberately held out. Holding a subset
out makes axis 2 a compositional-generalisation test: the model has seen this
label with other positions, and this position with other labels, never the two
together.

Axis 1 is about the episode. Axis 2 is about the weights.

## Model & loss

Next-token prediction over a single vocabulary with disjoint label / position /
value ranges, plus a role embedding for slot-in-triple. Cross-entropy on the
value token. The mixer is selectable — softmax attention or the KDA delta rule —
reusing `recall-gen`'s finding that the two differ sharply on transfer.

## Metrics

Value-token accuracy on the queried position, per 2x2 cell, against:

- **chance** — 1/V
- **the marginal** — always predict the most common value bin, which on MNIST is
  background and is a high bar
- **the copy ceiling** — for cell A and B, the exact triple is in the stream, so
  a perfect retriever scores 1.0

Recall-gen's lesson applies: report the reference point beside every number, and
prefer a measure that degrades smoothly over one that steps.

## Findings

(none yet)
