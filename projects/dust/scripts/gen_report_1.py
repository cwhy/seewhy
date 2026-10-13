"""Report 1 -- the JAX port reproduces DUST's published numbers.

Builds the figures, assembles the markdown as one string, publishes it, and writes the
same string to reports/01-port.md, so the tracked file and the published page cannot drift.

    ~/.local/bin/uv run --no-sync python projects/dust/scripts/gen_report_1.py
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))          # repo root, appended: see gpu-box-stray-datasets-py

from lib import diagrams, paper_ref as ref, results as R
from lib.figures import parity_cosines
from shared_lib.media import save_media
from shared_lib.report import save_report
from shared_lib.typst_report import save_figure

rows = R.load()
bp1, bp10 = R.cell(rows, method="bp", budget="1M"), R.cell(rows, method="bp", budget="10M")
d256 = R.cell(rows, method="dust", budget="1M", pop=256)
d1k = R.cell(rows, method="dust", budget="1M", pop=1024)
par = next(r for r in rows if r.get("kind") == "parity")


def ms(c):
    return f"{c['mean']:.3f} ± {c['sd']:.3f} (n={c['n']})"


def diff(c, published):
    return f"{c['mean'] - published:+.3f}"


url_pass = save_media("dust_pass_value.svg", diagrams.pass_value().encode(), "image/svg+xml")
url_grad = save_media("dust_reward_to_gradient.svg", diagrams.reward_to_gradient().encode(), "image/svg+xml")
url_par = save_figure(parity_cosines(rows, name="parity_cosines_h"), fmt="svg")

within = all(abs(c["mean"] - pub) <= c["sd"] for c, pub in
             ((bp1, ref.TABLE1["bp"]["1M"]), (bp10, ref.TABLE1["bp"]["10M"]), (d256, ref.TABLE1["dust"][256]["1M"]),
              (d1k, ref.TABLE1["dust"][1024]["1M"])))
within_text = ("Every difference is smaller than the seed-to-seed spread." if within
               else "At least one difference is larger than the seed-to-seed spread; see the table.")
law = {t: float(ref.cosine_law(t, 256)) for t in ("q", "k", "v", "gate", "value_emb")}
cos_ref, cos_jax = par["dust_cos_reference"], par["dust_cos_jax"]

md = f"""# The JAX port reproduces DUST's published numbers

We re-implemented DUST in JAX and checked it three ways. Backprop on our model reaches
**{bp1['mean']:.3f}** at 1M tokens (published {ref.TABLE1['bp']['1M']}) and **{bp10['mean']:.3f}** at 10M
(published {ref.TABLE1['bp']['10M']}). DUST at 256 draws reaches **{d256['mean']:.3f}**
(published {ref.TABLE1['dust'][256]['1M']}) and **{d1k['mean']:.3f}** at 1,024 draws (published {ref.TABLE1['dust'][1024]['1M']}),
which is level with our backprop. On identical weights, our gradient estimate and the authors'
PyTorch one agree layer by layer, and their gradient norms agree to {abs(par['dust_norm_ratio_jax'] / par['dust_norm_ratio_reference'] - 1) * 100:.1f}%.
So differences between methods in the next reports are differences between methods, not between codebases.

## How DUST works, in one page

**The problem.** A language model has about 38 million parameters. Backprop gets the gradient
of the loss with respect to all of them from one backward pass. This project asks what you can
do with forward passes alone.

**The obvious way, and why it fails.** Perturb the weights at random, run a forward pass, and
see whether the loss went down. Do it for a *population* of perturbations and average. That is
evolution strategies, and EGGROLL is a fast version of it. But each forward pass returns *one
number* — the mean loss — and that number has to steer 38 million weights. The cosine between
the estimate and the true gradient grows only as √(P/d). With d = 3.8×10⁷ and P = 1,000 passes
that is 0.005. The estimate is almost all noise.

**DUST's move.** Perturb the *activations* instead, and do it separately at every token. A batch
is 16,384 tokens, and the loss is a sum of one term per token. If each token's loss mostly
responds to its own noise, then one forward pass is not one experiment. It is sixteen thousand.

![One forward pass, two ways]({url_pass})

*Left: EGGROLL. The pass is one population member and returns one number. Right: DUST. Every token
carries its own noise (coloured dots) and returns its own loss.*

**From rewards to a gradient.** For one linear layer, each draw adds noise `σ·aₜ` to the layer's
output at every token. A token's *reward* is how much lower its loss came out than the average
across the draws run together. Noise that helped, weighted by how much it helped and averaged over
draws, points *against* the gradient at that layer's output. That is an estimate of ∂L/∂y, the
quantity backprop gets from the chain rule. Multiply by the layer's cached input and sum over tokens,
and you have the weight gradient. It is the same outer product backprop forms.

![Reward to gradient]({url_grad})

**What makes it work on a real transformer.** Three things. *Interference:* jittering every layer at
once would let one layer's noise leak into another layer's reward, so DUST jitters different layers in
separate passes. *Attention internals:* queries, keys and values act through other tokens, so a token's
own loss barely sees them; DUST scores them against the estimated error at the attention output instead.
*The rest:* the output head is jittered directly on the cached logits, and the 16 residual-mixing scalars
use ordinary weight-space ES.

## 1. The losses land on the published ones

Test loss at the best-validation checkpoint, seeds 42–44. The published numbers are three-seed means.

| run | ours, mean ± sd | published | ours − published |
|---|---|---:|---:|
| backprop, 1M tokens | {ms(bp1)} | {ref.TABLE1['bp']['1M']} | {diff(bp1, ref.TABLE1['bp']['1M'])} |
| backprop, 10M tokens | {ms(bp10)} | {ref.TABLE1['bp']['10M']} | {diff(bp10, ref.TABLE1['bp']['10M'])} |
| DUST, 256 draws, 1M tokens | {ms(d256)} | {ref.TABLE1['dust'][256]['1M']} | {diff(d256, ref.TABLE1['dust'][256]['1M'])} |
| DUST, 1,024 draws, 1M tokens | {ms(d1k)} | {ref.TABLE1['dust'][1024]['1M']} | {diff(d1k, ref.TABLE1['dust'][1024]['1M'])} |

{within_text} A model that knew only token frequencies
would score 6.994 on this test set, so the DUST and backprop numbers are well inside the range where
the network is using context.

## 2. The estimator matches the authors' code

Same random weights, same two-sequence batch, both implementations, cosine to the PyTorch autograd gradient.
First the control: the two frameworks' *autograd* gradients agree to a cosine of 1.00 on every layer
type, with equal norm. The models are the same function. Then the DUST estimates at 256 draws:

![Cosine to backprop, authors' code vs ours]({url_par})

| layer type | authors' PyTorch | this port |
|---|---:|---:|
| LM head | {cos_ref['head']:.3f} | {cos_jax['head']:.3f} |
| token embedding | {cos_ref['token_emb']:.3f} | {cos_jax['token_emb']:.3f} |
| attention output | {cos_ref['attn_out']:.3f} | {cos_jax['attn_out']:.3f} |
| MLP output | {cos_ref['mlp_out']:.3f} | {cos_jax['mlp_out']:.3f} |
| MLP input | {cos_ref['mlp_in']:.3f} | {cos_jax['mlp_in']:.3f} |
| total gradient norm ÷ backprop's | {par['dust_norm_ratio_reference']:.3f} | {par['dust_norm_ratio_jax']:.3f} |

The five large types agree to within 0.03. The remaining five (Q, K, V, value gate, value embedding) sit
within ±0.1 of zero in both. That is where the paper's own fitted law puts them at this population:
{', '.join(f'{k} {v:.2f}' for k, v in law.items())}. One reference run is {par['reference_seconds']:.0f} s on a 24-thread CPU.

## What this changes

- The port is faithful enough to carry the EGGROLL comparison. The next report puts the two methods
  on one ladder at 1M tokens.
- Backprop is reproduced at two budgets, so "DUST reaches backprop at 1k draws" is tested against our
  own baseline, not only the paper's. Here DUST ({d1k['mean']:.3f} ± {d1k['sd']:.3f}) and backprop
  ({bp1['mean']:.3f} ± {bp1['sd']:.3f}) are level. The paper has DUST slightly *below*
  ({ref.TABLE1['dust'][1024]['1M']} vs {ref.TABLE1['bp']['1M']}); our backprop is a little stronger than theirs.
- Not yet checked: DUST above 1,024 draws, and anything at 10M tokens. Seeds and data order
  differ from the release (a numpy permutation, not torch's), and weights are kept in f32 where the
  reference stores embeddings in bf16. Neither moved a number outside the seed spread.

Code: `lib/model.py`, `lib/dust.py`. Parity: `scripts/tmp/parity_export.py`, `parity_torch.py`.
"""

out = PROJECT / "reports" / "01-port.md"
out.write_text(md)
url = save_report("dust_report_1_port", md)
print(url)
