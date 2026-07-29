"""
Living report: leaving the k-NN memory behind — context-generated functions.
Renders the framing now; fills in results as exp24 / exp25 land.

Usage (server): uv run python projects/universal-ar/scripts/gen_report_hyper.py
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.report import save_report

ROOT = Path(__file__).parent.parent
LOGS = ROOT / "logs"
RE = re.compile(r"step\s+(\d+).*?LAB\s+([\d.]+)\].*?RETRIEVAL lab\s+([\d.]+)\s+pix\s+([\d.]+)"
                r"\s+\|\s+GENERALISE lab\s+([\d.]+)\s+pix\s+([\d.]+)")


def curve(name):
    out = []
    f = LOGS / f"{name}.log"
    if not f.exists():
        return out
    for line in open(f, errors="ignore"):
        m = RE.search(line)
        if m:
            out.append(dict(step=int(m[1]), lab_loss=float(m[2]), retr_lab=float(m[3]),
                            retr_pix=float(m[4]), gen_lab=float(m[5]), gen_pix=float(m[6])))
    return out


runs = {n: curve(n) for n in ("exp24", "exp25", "exp26", "exp27")}
done = {n: (c[-1] if c else None) for n, c in runs.items()}

LN2 = 0.6931
def row(name, label):
    d = done.get(name)
    if not d:
        return f"| {label} | _running_ | _running_ | _running_ |\n"
    return (f"| {label} | {d['gen_lab']:.3f} | {d['lab_loss']:.4f} | "
            f"{d['retr_lab']:.3f} / {d['retr_pix']:.3f} |\n")

results = ("| variant | generalise label | label loss | retrieval lab / pix |\n|---|---|---|---|\n"
           + row("exp24", "**exp24** — generated weights (hypernetwork)")
           + row("exp25", "**exp25** — FiLM modulation")
           + "| _reference_: six earlier interventions | ~0.50 | 0.6931 | 1.000 / 1.000 |\n"
           + "| _reference_: naive k-NN on this task | 0.58 (1-shot) | — | — |\n")

kda = ""
if done.get("exp26") and done.get("exp27"):
    a, c = done["exp26"], done["exp27"]
    kda = f"""
## Postscript — a third memory structure (KDA / gated delta net)

Replacing the readout was not enough, so the next attempt replaced the *memory*: a
Kimi-Delta-Attention layer in place of attention. Its state is a `d_v x d_k` matrix
written by the delta rule — read what is already stored at this key, write only the
correction — which is online least squares, and precisely the fix for the crosstalk
that killed this project's original `M = sum k(x)v` memory. Tokens were shuffled within
each episode so the decay falls evenly on every sample instead of starving whichever
was written first.

| | retrieval label | retrieval pixel | generalise label | label loss |
|---|---|---|---|---|
| **KDA, 0v1 (control)** | {c['retr_lab']:.3f} | {c['retr_pix']:.3f} | **{c['gen_lab']:.3f}** | {c['lab_loss']:.4f} |
| **KDA, 4v9** | {a['retr_lab']:.3f} | {a['retr_pix']:.3f} | {a['gen_lab']:.3f} | {a['lab_loss']:.4f} |

The control solves 0-vs-1 outright, so the memory works — and 4-vs-9 still sits at
chance. That makes **seven** distinct architectures failing at the same point, each
with a passing control.

**The first attempt at this was junk and the control caught it.** The decay bias was
initialised at 3.0, giving a per-token retention of 0.953 and a memory horizon of
**21 tokens in a 3674-token episode** — the state was multiplied down to e^-179 before
anything could read it. Retrieval fell to 0.40 on the *easy* pair, which is not a
result, it is a broken run. Deriving the init from the token count instead
(alpha = 0.99973, horizon = 3674) restored retrieval to ~1.0. Shuffling controls *who*
gets forgotten; it cannot control *how much*, and those are separate knobs.

One genuinely new cost: KDA's pixel retrieval is {a['retr_pix']:.2f} against 1.000 for
attention. Attention keeps all 6810 tokens and can address any one exactly; a 64x64
state per head must compress ~3700 writes into 4096 numbers. Labels survive, per-pixel
recall does not. Removing the pooling bottleneck introduced a compression one.
"""

probe_sec = """
## Interlude — where the information actually goes

Before trying a third memory, we froze the encoder and fitted a plain **supervised**
classifier (true labels, no in-context anything) on the pooled per-sample summaries:

| classifier | test accuracy |
|---|---|
| on pooled summaries — linear | 0.670 |
| on pooled summaries — MLP | 0.702 |
| on raw pixels — linear | 0.962 |
| on raw pixels — MLP | 0.982 |

Raw pixels are 96-98% separable for 4-vs-9. After mean-pooling ~400 tokens into one
256-d vector that collapses to **70%**. So two thirds of the discriminative signal is
destroyed *before any readout sees it*, and every readout experiment was competing for
a ceiling of 0.70 that had already been set upstream — none of them reached even that.
"""

verdict = ""
if all(done.values()):
    a, b = done["exp24"], done["exp25"]
    best = max(a["gen_lab"], b["gen_lab"])
    broke = min(a["lab_loss"], b["lab_loss"]) < LN2 - 0.01
    verdict = (
        "## Verdict\n\n"
        + (f"**The barrier moved.** Best generalise-label accuracy {best:.3f} with label "
           f"cross-entropy below ln 2 — the first architecture in this project to extract "
           f"any information on 4-vs-9.\n" if broke else
           f"**The barrier held.** Best generalise-label accuracy {best:.3f}, label cross-entropy "
           f"still at ln 2. Replacing the weighted-average readout with a context-generated "
           f"function did not, on its own, produce the missing comparison.\n")
        + f"\nRetrieval stayed at {a['retr_lab']:.3f} / {b['retr_lab']:.3f}, so the change did not "
          "break anything that previously worked.\n")

md = f"""# Leaving the k-NN memory: can a context-generated function do what averaging cannot?

*Status: framing written before the runs; results filled in as they land.*

## Where this comes from

Every architecture in this project so far — the Hebbian associative memory, and then
the token-level transformer that replaced it — computes its answer the same way:

```
prediction  =  Σ_j  (similarity between query and stored item j) × (item j's value)
```

That is kernel regression: a weighted average of things already in the context, with
the weights coming from a learned similarity. The original design document flagged
this as a "theory anchor" in passing. It turned out to be the literal description of
what got built.

Measuring the trained weights made the consequence exact. The similarity the model
learned is an **address match**: position contributes +2.38, sample-tag +4.17, and
content only +0.83. It behaves as a content-addressable lookup table. So:

- **Retrieval** — the query's exact key is in the context — is perfect, 1.000.
- **Generalisation** — find the nearest item *by content*, then copy its label — sits
  at ln 2 = 0.6931, which is the cross-entropy of a model emitting the uniform prior.
  Zero information.

Six interventions failed to move that: shared positions, doubled depth, 20-shot
support, retrieval-only auxiliary training, and a conjunctive token embedding (which
also regressed the working case). A naive nearest-neighbour baseline on the same task
reaches 0.711 with 20-shot support, so the signal exists — the architecture cannot
exploit it.

## The move

Stop averaging stored values. Compile the context into **parameters**, then run the
query through the function those parameters define:

```
θ       =  g(support set)        compile the context into weights
logits  =  f_θ(query)            process the query with them
```

Nothing is compared to anything. The query is not matched against stored items — it is
*processed* by a function the context wrote. This is the direction the project
originally proposed (a memory state emitting the weights of an MLP) and which was lost
when the token-level rewrite replaced the readout with a plain softmax head.

It also targets the failing step directly. "Summarise the support, derive a decision
rule, apply it to the query" is exactly the computation that six attempts failed to
coax out of attention — here it is an explicit architectural component rather than
something hoped for.

## Two variants

Both keep the token transformer for what it demonstrably does well — building
per-sample representations and completing pixels — and replace only the label path.
Per `ref`, tokens are pooled into a summary; a permutation-invariant encoder turns the
labelled support summaries into `h`; and `h` generates the parameters.

**exp24 — generated weights.** A two-layer MLP whose first layer is generated
low-rank (`W₁ = U·diag(g(h))·V`, so the context supplies a few dozen numbers rather
than 16k) and whose output layer is generated directly.

**exp25 — FiLM modulation.** The first layer is fixed and learned; the context only
supplies per-feature scale and shift. Cheaper and far more stable — the standard way
to make weight generation trainable.

Two design constraints worth stating, because getting either wrong makes the
experiment meaningless:

- **The output layer must be context-generated in both variants.** Labels are
  anonymised — which slot means which digit is re-drawn every episode — so a fixed
  output layer cannot possibly work.
- **The generated function must be nonlinear.** If `f_θ` were a single linear layer,
  `logits = z·w_c` is a prototype classifier, which is k-NN again wearing a different
  hat. Both variants put a GELU in the middle.

## What is held fixed

Same episodes, same anonymised labels, same 4-vs-9 pair, same 8000 steps, same
effective batch, same retrieval/generalisation split. Only the label path changes, so
the numbers read directly against the established baselines. Retrieval stays in the
loss as a canary: if it degrades, the change broke something basic rather than fixing
anything.

## Results

{results}
{verdict}
{probe_sec}
{kda}"""

url = save_report("universal-ar_report_hypernet", md)
print("REPORT:", url)
