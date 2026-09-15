"""Forward-Forward (Hinton, 2022) on a plain MLP.

There is no backward pass. Every layer has its own objective: make its
*goodness* — the mean square of its activations — large for positive data and
small for negative data. Positive data is an image with the correct label
written into its first ten pixels; negative data is the same image with a wrong
label. Layer i's loss, with threshold theta:

    g(h) = mean(h^2)
    loss_i = softplus(-(g(h_i^pos) - theta)) + softplus(g(h_i^neg) - theta)

Each layer's input is detached, so the gradient of the summed loss is exactly
the collection of per-layer local gradients; the layers happen to be trained
simultaneously rather than greedily, which is what keeps training to one pass
over the data.

Each layer's input is also normalised to unit mean square before the matmul.
That is the load-bearing trick of the method: it strips the magnitude — which
*is* the previous layer's answer — and leaves only the direction, so a layer
cannot win by inheriting its predecessor's goodness. (Hinton divides by the L2
norm; unit mean square is the same direction, rescaled by a constant that the
weights would otherwise have to learn.)

Classification at test time has no output layer to read. Instead every one of
the ten labels is written into the image in turn, and the label whose goodness
summed over layers 2..n is largest wins — ten forward passes per image, which
is why Forward-Forward's inference is the most expensive of the three methods
even though its training is the cheapest.
"""

import jax
import jax.numpy as jnp
from jax import Array

from lib.data import overlay_label
from lib.model import Params

EPS = 1e-4


def normalise(h: Array) -> Array:
    """Unit mean square, per example."""
    return h / (jnp.sqrt(jnp.mean(h * h, axis=-1, keepdims=True)) + EPS)


def layer_activations(params: Params, x: Array, *, detach: bool = False) -> list[Array]:
    """Post-ReLU activations of every hidden layer, inputs normalised from layer 2 on."""
    acts, h = [], x
    for i in range(len(params) // 2):
        inp = h if i == 0 else normalise(h)
        if detach:
            inp = jax.lax.stop_gradient(inp)
        h = jax.nn.relu(inp @ params[2 * i].T + params[2 * i + 1])
        acts.append(h)
    return acts


def goodness(h: Array) -> Array:
    return jnp.mean(h * h, axis=-1)


def ff_loss(params: Params, x_pos: Array, x_neg: Array, theta: float) -> Array:
    """Summed per-layer local loss. Detached inputs make its gradient layer-local."""
    pos = layer_activations(params, x_pos, detach=True)
    neg = layer_activations(params, x_neg, detach=True)
    total = 0.0
    for hp, hn in zip(pos, neg):
        total = total + jnp.mean(jax.nn.softplus(-(goodness(hp) - theta))
                                 + jax.nn.softplus(goodness(hn) - theta))
    return total


def probe_features(params: Params, x: Array) -> Array:
    """Normalised activations of layers 2..n, concatenated — Hinton's linear readout input."""
    acts = layer_activations(params, x)
    return jnp.concatenate([normalise(h) for h in acts[1:]], axis=-1)


def probe_loss(head: Params, params: Params, x_pos_neutral: Array, y: Array) -> Array:
    """Cross-entropy of a linear softmax head on *detached* features."""
    feats = jax.lax.stop_gradient(probe_features(params, x_pos_neutral))
    logits = feats @ head[0].T + head[1]
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(logits, axis=-1), axis=-1))


def neutral_input(x: Array) -> Array:
    """The image with a uniform, uninformative label written in (Hinton's neutral label)."""
    return overlay_label(x, jnp.full(x.shape[:-1] + (10,), 0.1))


def predict_goodness(params: Params, x: Array) -> Array:
    """Ten label sweeps; the label with the highest summed goodness wins.

    Layer 1 is excluded from the sum: it sees the label pixels directly and
    would let the network score a label without using any learned structure.
    """
    eye = jnp.eye(10, dtype=x.dtype)

    def score(onehot):
        acts = layer_activations(params, overlay_label(x, onehot[None, :]))
        return sum(goodness(h) for h in acts[1:])

    return jnp.argmax(jax.vmap(score)(eye), axis=0)


def predict_probe(head: Params, params: Params, x: Array) -> Array:
    logits = probe_features(params, neutral_input(x)) @ head[0].T + head[1]
    return jnp.argmax(logits, axis=-1)


def make_negatives(key: Array, labels: Array) -> Array:
    """A uniformly sampled wrong label for every example."""
    offset = jax.random.randint(key, labels.shape, 1, 10)
    return jnp.eye(10, dtype=jnp.float32)[(labels + offset) % 10]
