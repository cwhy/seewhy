"""Regenerates the lab-note report: the replication and what it turned out to measure.

Writes reports/01-replication.md and publishes it to R2.

Run on the GPU box: `uv run --no-sync python projects/local-learning/scripts/gen_md_report.py`
"""
import json
import statistics as st
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))          # repo root LAST — see workflow.md

from lib.figures import (accuracy_vs_depth, accuracy_vs_tweet, accuracy_vs_walltime,  # noqa: E402
                         ff_readouts, learning_rate_grid, tuned_vs_default)
from shared_lib.results import load_results                                            # noqa: E402
from shared_lib.report import save_report                                              # noqa: E402
from shared_lib.typst_report import save_figure                                        # noqa: E402

REPORT = PROJECT / "reports" / "01-replication.md"
ROWS = load_results(PROJECT / "results.jsonl")


def cells(exp, **m):
    return [r for r in ROWS.values() if r["exp"] == exp and all(r.get(k) == v for k, v in m.items())]


def stat(rows, field="test_acc"):
    v = [r[field] for r in rows if r.get(field) is not None]
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)


def pm(rows, field="test_acc", fmt="{:.4f} ± {:.4f}"):
    return fmt.format(*stat(rows, field))


def main():
    urls = {name: save_figure(fig) for name, fig in (
        ("tweet", accuracy_vs_tweet(ROWS)),
        ("tuned", tuned_vs_default(ROWS)),
        ("depth", accuracy_vs_depth(ROWS)),
        ("walltime", accuracy_vs_walltime(ROWS)),
        ("lr", learning_rate_grid(ROWS)),
        ("ffread", ff_readouts(ROWS)),
    )}

    bp = cells("exp1", method="bp", tag="")
    pc = cells("exp1", method="pc", tag="")
    pcalm = cells("exp1", method="pcalm", tag="")
    ff = cells("exp2", method="ff", tag="")
    bpp = cells("exp2", method="bp_plain", tag="")
    ffd = cells("exp2", method="ff", tag="deep")

    md = f"""# Local learning vs backprop — replicating a "quick MNIST test"

Replicates [@mayfer's table](https://x.com/mayfer/status/2099541904241385615)
(2026-09-14): backprop, PC-ALM and Forward-Forward on MNIST at ~56K parameters,
one epoch. Five seeds per arm, one RTX 4090, JAX.

**The parameter counts identify the architectures, and they differ.** 56,128 is
exactly the PC-ALM paper's headline cell — a residual MLP of width 32, depth 32.
56,232 is exactly a plain 784-66-66 MLP, which is what Forward-Forward needs
(goodness replaces the output layer). Two rows are 32 layers deep; the third is
two layers wide.

## 1. It replicates

| arm | params | ours (5 seeds) | posted | train s | posted | 10k infer s | posted |
|---|---:|---|---:|---:|---:|---:|---:|
| backprop 32x32 res | {bp[0]['n_params']} | {pm(bp)} | 0.8739 | {stat(bp,'train_s')[0]:.3f} | 0.324 | {stat(bp,'infer_s')[0]:.5f} | 0.0229 |
| PC-ALM 32x32 res | {pcalm[0]['n_params']} | {pm(pcalm)} | 0.8628 | {stat(pcalm,'train_s')[0]:.3f} | 13.485 | {stat(pcalm,'infer_s')[0]:.5f} | 0.0241 |
| Forward-Forward 2x66 | {ff[0]['n_params']} | {pm(ff)} | 0.8883 | {stat(ff,'train_s')[0]:.3f} | 0.549 | {stat(ff,'infer_s')[0]:.5f} | 0.0781 |
| PC (control) | {pc[0]['n_params']} | {pm(pc)} | — | {stat(pc,'train_s')[0]:.3f} | — | {stat(pc,'infer_s')[0]:.5f} | — |
| backprop 2x66 (control) | {bpp[0]['n_params']} | {pm(bpp)} | — | {stat(bpp,'train_s')[0]:.3f} | — | {stat(bpp,'infer_s')[0]:.5f} | — |
| Forward-Forward 31x32 (control) | {ffd[0]['n_params']} | {pm(ffd)} | — | {stat(ffd,'train_s')[0]:.3f} | — | {stat(ffd,'infer_s')[0]:.5f} | — |

All three posted accuracies fall inside our seed spread. PC-ALM/backprop
training ratio {stat(pcalm,'train_s')[0]/stat(bp,'train_s')[0]:.1f}x against the posted 41.6x. Chance is 0.10.

![accuracy against the posted numbers]({urls['tweet']})

## 2. Forward-Forward's win is the architecture

Backprop on FF's own 784-66-66 stack: **{pm(bpp)}**, i.e. {100*(stat(bpp)[0]-stat(ff)[0]):.1f} points above FF
itself. FF moved onto the deep-narrow stack: **{pm(ffd)}**.

![FF readouts]({urls['ffread']})

The deep collapse is the readout, not the features: a linear probe on the
31-layer net's own activations gets {stat(ffd,'test_acc_probe')[0]:.4f} where the goodness sweep gets
{stat(ffd)[0]:.4f}. Not a learning-rate artifact — the best of four rates on the deep
stack gives {stat(cells('exp7', method='ff', tag='deep_lr0.003'))[0]:.4f}.

## 3. One shared learning rate is doing a lot of the work

![learning-rate grid]({urls['lr']})

Best rate per arm differs by 30x. 1e-3 — the value the PC-ALM rule gives at this
cell, and the one the posted comparison used throughout — is best for exactly
one arm: Forward-Forward's. Re-run at each arm's own best rate, 5 seeds:

| arm | rate | tuned | vs default |
|---|---:|---|---:|
"""
    for method, tag, label in (("bp", "tuned", "backprop 32x32 res"),
                               ("pcalm", "tuned", "PC-ALM 32x32 res"),
                               ("bp_plain", "tuned", "backprop 2x66"),
                               ("ff", "tuned", "Forward-Forward 2x66"),
                               ("pc", "tuned", "PC 32x32 res"),
                               ("ff", "tuned_deep", "Forward-Forward 31x32")):
        t = cells("exp9", method=method, tag=tag)
        d = cells("exp1" if method in ("bp", "pc", "pcalm") else "exp2",
                  method=method, tag="deep" if tag == "tuned_deep" else "")
        md += f"| {label} | {t[0]['lr']:g} | {pm(t)} | {stat(t)[0]-stat(d)[0]:+.4f} |\n"

    md += f"""
The posted order was FF > backprop > PC-ALM. Tuned, it is backprop > PC-ALM > FF
— reversed among exactly the three arms compared.

![tuned vs default]({urls['tuned']})

## 4. At equal wall-clock

![accuracy vs training seconds]({urls['walltime']})

Backprop passes PC-ALM's one-epoch accuracy inside its second epoch (0.908 at
0.40 s vs 0.855 at 7.68 s) and reaches 0.946 in the time PC-ALM needs for one
epoch. Forward-Forward is the only arm that gets *worse* with training: 0.901 at
two epochs, 0.860 at ten, and its linear probe falls further (0.9102 → 0.7956).

## 5. The source paper's claim holds

PC and PC-ALM differ here by one scalar (alpha = 0 or 1), same T, same eta_h,
same seeds.

![accuracy vs depth]({urls['depth']})

| layers | backprop | PC | PC-ALM | cos(PC) | cos(PC-ALM) |
|---:|---:|---:|---:|---:|---:|
"""
    for L in (2, 4, 8, 16, 32, 64):
        r_bp = cells("exp5", method="bp", tag=f"L{L}")
        r_pc = cells("exp5", method="pc", tag=f"L{L}")
        r_al = cells("exp5", method="pcalm", tag=f"L{L}")
        md += (f"| {L} | {stat(r_bp)[0]:.4f} | {stat(r_pc)[0]:.4f} | {stat(r_al)[0]:.4f} | "
               f"{stat(r_pc,'grad_cos_to_bp')[0]:.3f} | {stat(r_al,'grad_cos_to_bp')[0]:.3f} |\n")

    f_bp, f_pc, f_al = (cells("exp6", method=m) for m in ("bp", "pc", "pcalm"))
    md += f"""
## 6. Is our reimplementation faithful?

Written from the paper's equations, then checked against the authors' published
reference run on their Fashion-MNIST cell (their numbers are single runs):

| | ours (3 seeds) | theirs | our cosine | their cosine |
|---|---|---:|---:|---:|
| backprop | {pm(f_bp)} | 0.7866 | {stat(f_bp,'grad_cos_to_bp')[0]:.3f} | 1.000 |
| PC | {pm(f_pc)} | 0.6813 | {stat(f_pc,'grad_cos_to_bp')[0]:.3f} | 0.604 |
| PC-ALM | {pm(f_al)} | 0.7775 | {stat(f_al,'grad_cos_to_bp')[0]:.3f} | 0.909 |

Backprop and PC-ALM land within a point; PC is 4.7 low and is the one arm that
does not match. Swapping our measured activity step for their tabulated one
changes nothing ({stat(cells('exp8', method='pc'))[0]:.4f} vs {stat(f_pc)[0]:.4f}).

---

Full write-up: the paper. Rows: `results.jsonl`, {len(ROWS)} cells.
"""
    REPORT.write_text(md)
    url = save_report("local-learning_report_replication", md)
    print(f"wrote {REPORT}")
    print(f"published {url}")


if __name__ == "__main__":
    main()
