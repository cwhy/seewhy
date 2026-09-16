"""Regenerates the Forward-Forward audit report.

Numbers here come from the diagnostic scripts in scripts/tmp/ (check_ff.py,
check_ff2.py, check_ff3.py) and from exp24/exp25 in results.jsonl. The
diagnostics are one-seed probes by design — they are asking whether the
implementation is right, not measuring an effect — and are labelled as such.

Run on the GPU box: `uv run --no-sync python projects/local-learning/scripts/gen_md_report_ff_audit.py`
"""
import statistics as st
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))

from shared_lib.report import save_report                    # noqa: E402
from shared_lib.results import load_results                  # noqa: E402

REPORT = PROJECT / "reports" / "03-forward-forward-audit.md"
ROWS = load_results(PROJECT / "results.jsonl")


def cells(exp, **m):
    return [r for r in ROWS.values() if r["exp"] == exp
            and all(r.get(k) == v for k, v in m.items())]


def pm(rows, field="test_acc"):
    v = [r[field] for r in rows if r.get(field) is not None]
    if not v:
        return "—"
    return f"{st.mean(v):.4f} ± {(st.stdev(v) if len(v) > 1 else 0.0):.4f}"


def mean(rows, field="test_acc"):
    v = [r[field] for r in rows if r.get(field) is not None]
    return st.mean(v) if v else float("nan")


def main():
    hard = cells("exp25", tag="hard")
    hard_deep = cells("exp25", tag="hard_deep")
    hard5 = cells("exp25", tag="hard_5ep")
    hard_deep5 = cells("exp25", tag="hard_deep_5ep")
    unif = cells("exp2", method="ff", tag="")
    unif_deep = cells("exp2", method="ff", tag="deep")
    bp_plain = cells("exp2", method="bp_plain", tag="")

    md = f"""# Auditing the Forward-Forward implementation

The [replication](https://media.tanh.xyz/seewhy/26-09-16/local-learning_report_replication.html)
rests on a Forward-Forward arm we wrote ourselves from Hinton's description, so
the obvious question is whether that arm is Forward-Forward. This is the audit.

**Short version: the mechanics are right, one documented shortcut is expensive,
and two of the paper's claims have to be withdrawn because of it.**

## What is verified

| check | result |
|---|---|
| the gradient is layer-local — d(layer i loss)/d(W_j) for j ≠ i | exactly 0.000e+00 |
| classification uses the label, not the image | shuffle image against label → 0.1135 (chance 0.10) |
| both layers separate positive from negative data | P(goodness⁺ > goodness⁻) = 0.981, 0.974 |
| goodness straddles the threshold rather than saturating softplus | layer 1 g⁺ ∈ [0.23, 6.06], g⁻ ∈ [0.03, 5.60], θ = 2 |
| the loss is Hinton's | softplus(−(g⁺−θ)) + softplus(g⁻−θ), minimised |
| the readout is Hinton's | ten label sweeps, goodness summed over layers 2..n |

## What did not look right

Hinton reports about 1.36% error on MNIST with four hidden layers of 2000 units.
Ours, at his scale:

| network | epochs | test accuracy |
|---|---:|---:|
| 2×2000 | 20 | 0.9530 |
| 4×2000 | 20 | 0.9376 |
| 2×2000 | 60 | 0.9705 |

Two things wrong with that. We are three points short of the paper at 20 epochs
and 1.6 short at 60. And **more layers made it worse** — the opposite of what
the method claims. Per-layer accuracy on the four-layer net says why: layers 1
and 2 reach 0.971 and 0.953, layers 3 and 4 reach 0.654 and 0.484, and summing
their goodness into the answer drags it down.

A hypothesis that turned out to be wrong, recorded because it cost time: that
ReLU activations, being non-negative, leave a large constant component that
dominates the direction handed to the next layer. Measured, the cosine between
two different examples' normalised activity is 0.120 — not dominant — and
mean-subtracting before normalising makes every case *worse* (4×2000 falls to
0.5799).

## What was wrong: the negatives

We sample the negative label uniformly among the nine wrong ones. Hinton samples
the wrong label the network currently finds most plausible. That sounds like a
refinement and is not: a uniformly random wrong label is rejected by layer 1 on
the label pixels alone, so every layer above it is handed a task that is already
solved, and learns nothing.

| network | epochs | uniform negatives | hard negatives |
|---|---:|---:|---:|
| 2×2000 | 10 | 0.9356 | 0.9628 |
| 4×2000 | 10 | 0.9100 | 0.9621 |
| 784-66-66 | 1 | 0.8875 | 0.9089 |

The per-layer profile is the tell. With uniform negatives the four-layer net runs
0.959, 0.936, 0.590, 0.418 across its layers; with hard negatives, 0.968, 0.969,
0.809, 0.670. The deeper layers were starved, not broken.

## What this costs the paper's claims

Re-running the project's own Forward-Forward arms with hard negatives, at the
rate exp24's grid picks for each ({len(hard)} seeds, one epoch):

| arm | uniform negatives (published) | hard negatives | change |
|---|---|---|---:|
| Forward-Forward 2×66 | {pm(unif)} | {pm(hard)} | {mean(hard) - mean(unif):+.4f} |
| Forward-Forward 31×32 | {pm(unif_deep)} | {pm(hard_deep)} | {mean(hard_deep) - mean(unif_deep):+.4f} |

Two claims in the paper do not survive this.

**1. "Backprop beats Forward-Forward on Forward-Forward's own architecture by
2.5 points."** Backprop on 784-66-66 gets {pm(bp_plain)}. Forward-Forward with
hard negatives gets {pm(hard)}. The gap is {100*(mean(bp_plain) - mean(hard)):.1f} points, not 2.5, and that is
inside the seed spread. The architecture argument still stands on its other leg —
Forward-Forward needs a short wide network, and the deep-narrow one is the
PC-ALM benchmark — but the head-to-head on the shared architecture is now a tie,
not a win.

**2. "Forward-Forward collapses on the deep-narrow stack, and the collapse is a
readout failure."** It does not collapse: {pm(hard_deep)} against the published
{pm(unif_deep)}. The collapse was our negatives. The probe evidence that the
features were fine was correct; the conclusion drawn from it — that goodness is
a poor readout at depth — was measuring the wrong thing.

At five epochs the corrected arms give {pm(hard5)} on 784-66-66 and
{pm(hard_deep5)} on the deep stack.

## What remains short

Even with hard negatives we reach 0.9621 at 4×2000 against Hinton's 0.9864. The
remaining gap is presumably the parts of his recipe we still do not implement:
peer normalisation, his layer-wise rate schedule, and 60 epochs rather than 10.
We did not chase it, because this project needs an arm that is a fair
Forward-Forward at 56K parameters and one epoch, not a reproduction of his
headline number.

Still-open deviations, all now measured or bounded: unit mean-square
normalisation rather than unit L2 (tested, makes no difference at this scale);
simultaneous rather than greedy layer training; θ fixed at 2.0 and never swept;
no peer normalisation.

---

Diagnostics: `scripts/tmp/check_ff.py`, `check_ff2.py`, `check_ff3.py`.
Cells: exp24 (16), exp25 ({len(hard) + len(hard_deep) + len(hard5) + len(hard_deep5)}).
"""
    REPORT.write_text(md)
    print(f"wrote {REPORT}")
    print(f"published {save_report('local-learning_report_ff_audit', md)}")


if __name__ == "__main__":
    main()
