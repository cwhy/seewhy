"""
Universal-AR — final report on the in-context-learning investigation (exp10-exp23).

Genuine ICL demonstrated; the capability boundary mapped; six interventions ruled
out against measured baselines. Parses the logs (authoritative — some results.jsonl
rows were lost to a git checkout) and publishes.

Usage (server): uv run python projects/universal-ar/scripts/gen_report_boundary.py
"""
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure
from shared_lib.report import save_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS = Path(__file__).parent.parent / "logs"
LN2 = float(np.log(2))

# three log generations
RE_A = re.compile(r"step\s+(\d+)\s+loss\s+([\d.]+)\s+label_te\(balanced\)\s+([\d.]+)\s+ink_te\s+([\d.]+)")
RE_B = re.compile(r"step\s+(\d+)\s+loss\s+([\d.]+)\s+\[pix\s+([\d.]+)\s+\|\s+LAB\s+([\d.]+)\]\s+label tr/te\s+([\d.]+)/([\d.]+)")
RE_C = re.compile(r"step\s+(\d+)\s+loss\s+([\d.]+)\s+\[pix\s+([\d.]+)\s+\|\s+LAB\s+([\d.]+)\]\s+RETRIEVAL lab\s+([\d.]+)\s+pix\s+([\d.]+)\s+\|\s+GENERALISE lab\s+([\d.]+)\s+pix\s+([\d.]+)")


def parse(name):
    """Return dict of lists; keys present depend on the log generation."""
    out = {k: [] for k in ("step", "loss", "lab_loss", "label", "label_tr", "retr_lab", "retr_pix", "ink")}
    for line in open(LOGS / f"{name}.log", errors="ignore"):
        if (m := RE_C.search(line)):
            out["step"].append(int(m[1])); out["loss"].append(float(m[2]))
            out["lab_loss"].append(float(m[4])); out["retr_lab"].append(float(m[5]))
            out["retr_pix"].append(float(m[6])); out["label"].append(float(m[7])); out["ink"].append(float(m[8]))
        elif (m := RE_B.search(line)):
            out["step"].append(int(m[1])); out["loss"].append(float(m[2]))
            out["lab_loss"].append(float(m[4])); out["label_tr"].append(float(m[5])); out["label"].append(float(m[6]))
        elif (m := RE_A.search(line)):
            out["step"].append(int(m[1])); out["loss"].append(float(m[2]))
            out["label"].append(float(m[3])); out["ink"].append(float(m[4]))
    return {k: v for k, v in out.items() if v}


E = {n: parse(n) for n in ("exp10", "exp11", "exp12", "exp13", "exp14",
                           "exp15", "exp16", "exp17", "exp18", "exp19",
                           "exp20", "exp21", "exp22", "exp23")}
f = lambda n, k="label": E[n][k][-1] if k in E[n] else float("nan")
for n in E:
    print(n, "label", round(f(n), 3), "lab_loss", round(f(n, "lab_loss"), 4) if "lab_loss" in E[n] else "-")

# ── fig 1: the capability boundary ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 4.6))
bars = [("exp10\n10-way\nFIXED labels", f("exp10"), 0.10, "#7f8c8d"),
        ("exp11\n10-way\npermuted", f("exp11"), 0.10, "#c0392b"),
        ("exp12\n2-way 0v1\npermuted", f("exp12"), 0.50, "#1e8449"),
        ("exp13\n2-way 4v9\npermuted", f("exp13"), 0.50, "#c0392b"),
        ("exp14\n5-way\npermuted", f("exp14"), 0.20, "#c0392b")]
x = np.arange(len(bars))
ax.bar(x, [b[1] for b in bars], color=[b[3] for b in bars], width=.6)
for i, b in enumerate(bars):
    ax.plot([i - .3, i + .3], [b[2], b[2]], "k-", lw=2)
    ax.text(i, b[1] + .02, f"{b[1]:.3f}", ha="center", fontsize=10, weight="bold")
ax.text(len(bars) - .5, 0.55, "black bar = chance", fontsize=8, ha="right", color="#333")
ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
ax.set_ylabel("label accuracy (test query)"); ax.set_ylim(0, 1.1)
ax.set_title("In-context learning under permuted labels: one success, three failures")
ax.grid(alpha=.25, axis="y")
url_map = save_matplotlib_figure("universal-ar_boundary_map", fig, format="svg"); plt.close(fig)

# ── fig 2: six interventions on 4v9, all at ln(2) ────────────────────────────
iv = [("baseline\n(exp15)", f("exp15", "lab_loss")),
      ("shared\npositions\n(exp17)", f("exp17", "lab_loss")),
      ("8 layers\n(exp18)", f("exp18", "lab_loss")),
      ("20-shot\n(exp19)", f("exp19", "lab_loss")),
      ("retrieval\ndata (exp20)", f("exp20", "lab_loss")),
      ("MLP combiner\n(exp22)", f("exp22", "lab_loss"))]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
x = np.arange(len(iv))
ax.bar(x, [v for _, v in iv], color="#c0392b", width=.6)
ax.axhline(LN2, color="k", ls="--", lw=1.6, label=f"ln(2) = {LN2:.4f}  (zero information)")
ax.axhline(f("exp16", "lab_loss"), color="#1e8449", ls=":", lw=1.8,
           label=f"0v1 control solved it: {f('exp16','lab_loss'):.4f}")
for i, (_, v) in enumerate(iv):
    ax.text(i, v + .012, f"{v:.3f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([n for n, _ in iv], fontsize=8.5)
ax.set_ylabel("label cross-entropy (4 vs 9)"); ax.set_ylim(0, 0.85)
ax.set_title("Six interventions on the hard pair — every one pinned at zero information")
ax.legend(fontsize=8.5, loc="lower left"); ax.grid(alpha=.25, axis="y")
url_iv = save_matplotlib_figure("universal-ar_boundary_interventions", fig, format="svg"); plt.close(fig)

# ── fig 3: retrieval vs generalisation, and the combiner regression ──────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
groups = [("4 vs 9\n(exp20)", f("exp20", "retr_lab"), f("exp20", "retr_pix"), f("exp20")),
          ("0 vs 1\n(exp21)", f("exp21", "retr_lab"), f("exp21", "retr_pix"), f("exp21"))]
w = 0.25; x = np.arange(len(groups))
a1.bar(x - w, [g[1] for g in groups], w, label="RETRIEVAL label", color="#1e8449")
a1.bar(x, [g[2] for g in groups], w, label="RETRIEVAL pixel", color="#7fbf7f")
a1.bar(x + w, [g[3] for g in groups], w, label="GENERALISE label", color="#c0392b")
for i, g in enumerate(groups):
    for dx, v in ((-w, g[1]), (0, g[2]), (w, g[3])):
        a1.text(i + dx, v + .02, f"{v:.2f}", ha="center", fontsize=8)
a1.set_xticks(x); a1.set_xticklabels([g[0] for g in groups]); a1.set_ylim(0, 1.15)
a1.set_ylabel("accuracy"); a1.legend(fontsize=8); a1.grid(alpha=.25, axis="y")
a1.set_title("Addressing is perfect; only comparison fails")

pairs = [("0 vs 1", f("exp21"), f("exp23")), ("4 vs 9", f("exp20"), f("exp22"))]
x = np.arange(len(pairs)); w = 0.32
a2.bar(x - w/2, [p[1] for p in pairs], w, label="additive embedding", color="#2874a6")
a2.bar(x + w/2, [p[2] for p in pairs], w, label="MLP combiner", color="#e67e22")
a2.axhline(0.50, color="k", ls="--", lw=1.2, label="chance")
for i, p in enumerate(pairs):
    a2.text(i - w/2, p[1] + .02, f"{p[1]:.2f}", ha="center", fontsize=9)
    a2.text(i + w/2, p[2] + .02, f"{p[2]:.2f}", ha="center", fontsize=9)
a2.annotate("control BROKE", xy=(0 + w/2, f("exp23")), xytext=(0.35, 0.80), fontsize=9,
            color="#c0392b", weight="bold", arrowprops=dict(arrowstyle="->", color="#c0392b"))
a2.set_xticks(x); a2.set_xticklabels([p[0] for p in pairs]); a2.set_ylim(0, 1.15)
a2.set_ylabel("generalise label accuracy"); a2.legend(fontsize=8); a2.grid(alpha=.25, axis="y")
a2.set_title("The MLP combiner regressed the working case")
fig.tight_layout()
url_split = save_matplotlib_figure("universal-ar_boundary_split", fig, format="svg"); plt.close(fig)

md = f"""# Universal-AR — in-context learning: what works, what doesn't, and what we ruled out

*Token-level model: every datum is a `(pos, value, ref)` token, the label is a token
at `pos_label`, and a 4-layer transformer (d=256, 3.4M params) does masked-token
completion over a flat bag of tokens. Support drawn from TRAIN, query from TEST,
balanced evaluation, labels **anonymised** (a fresh class→token permutation each
episode) so memorisation is worth exactly chance.*

## Summary

1. **Genuine in-context learning is real here** — perfect (1.000) on 2-way 0-vs-1
   under permuted labels, where a memoriser scores exactly 0.500.
2. **It has a sharp boundary.** The model matches on *marginal statistics* only. Any
   task needing shape comparison (4 vs 9, and hence 5-way and 10-way) sits at
   exactly zero information.
3. **Addressing and copying are perfect** (retrieval 1.000). The failure is
   localised to one step: deciding *which* support sample a query resembles.
4. **Six interventions failed to move it**, each against a measured baseline.
5. **Our best mechanistic explanation was falsified** by its own control.

## 1 · The test, and the one success

![capability map]({url_map})

| exp | task | labels | accuracy | chance |
|---|---|---|---|---|
| exp10 | 10-way | fixed | {f('exp10'):.3f} | 0.10 |
| exp11 | 10-way | permuted | {f('exp11'):.3f} | 0.10 |
| **exp12** | **2-way 0v1** | **permuted** | **{f('exp12'):.3f}** | **0.50** |
| exp13 | 2-way 4v9 | permuted | {f('exp13'):.3f} | 0.50 |
| exp14 | 5-way | permuted | {f('exp14'):.3f} | 0.20 |

exp10's {f('exp10'):.3f} looked like in-context learning and was **not** — with fixed
label semantics a query can classify itself against memorised prototypes. exp11
proves it: permute the labels and it collapses to chance. exp12 then shows the
capability genuinely exists: a perfect score is unreachable by memorisation when the
mapping is swapped in half the episodes.

**The boundary is discriminability, not class count.** exp13 holds class count at 2
and only swaps in a confusable pair — and lands exactly at chance. 5-way and 10-way
are not separate failures; every such set contains hard pairs.

## 2 · What is actually achievable (the baselines that reframed everything)

Nearest-neighbour on binned pixels — the naive ceiling:

| pair | 1-shot, half-observed | 1-shot, **full image** | 20-shot |
|---|---|---|---|
| 0 vs 1 | 0.860 | 0.887 | 0.987 |
| **4 vs 9** | **0.565** | **0.584** | **0.787** |

1-shot 4-vs-9 is barely above chance *even with the entire image*. Several early
"failures" were therefore chasing signal that did not exist — a lesson in measuring
the ceiling first. It also exposed a design flaw of ours: training used ~5 support
examples per class while evaluation was 1-shot, i.e. testing on a harder setting
than training. exp19 fixed that with matched 20-shot support where naive NN reaches
**0.711** — and the model still scored chance. That is the version that matters.

## 3 · Where the failure lives

![retrieval vs generalisation]({url_split})

Adding **retrieval-only** tasks (query a token whose `(pos, value, ref)` triple *is*
in the context, so the answer is copyable) split the pipeline cleanly:

| | retrieval label | retrieval pixel | generalise label |
|---|---|---|---|
| 4 vs 9 (exp20) | **{f('exp20','retr_lab'):.3f}** | **{f('exp20','retr_pix'):.3f}** | {f('exp20'):.3f} |
| 0 vs 1 (exp21) | **{f('exp21','retr_lab'):.3f}** | **{f('exp21','retr_pix'):.3f}** | {f('exp21'):.3f} |

**Addressing and copying are flawless.** Nothing basic is broken. The single failing
step is cross-sample comparison — and it fails selectively, exactly when marginal
statistics do not separate the classes.

## 4 · Six interventions, all pinned at zero information

![interventions]({url_iv})

| intervention | label loss (4v9) | verdict |
|---|---|---|
| baseline (exp15) | {f('exp15','lab_loss'):.4f} | ln(2) |
| shared positions (exp17) | {f('exp17','lab_loss'):.4f} | no effect |
| 8 layers (exp18) | {f('exp18','lab_loss'):.4f} | no effect |
| 20-shot support (exp19) | {f('exp19','lab_loss'):.4f} | no effect (NN ceiling 0.711) |
| retrieval-only data (exp20) | {f('exp20','lab_loss'):.4f} | no effect on labels |
| MLP combiner (exp22) | {f('exp22','lab_loss'):.4f} | no effect **and broke the control** |
| *0v1 control (exp16)* | *{f('exp16','lab_loss'):.4f}* | *solved* |

ln(2) = {LN2:.4f} is the cross-entropy of a model emitting the uniform prior — not
"learning slowly", but extracting **exactly zero** information. Train and test
accuracy are equal throughout, so this is a failure to *fit*, not to generalise.
Pixel completion trains normally in every one of these runs.

## 5 · A hypothesis, and its falsification

We argued: to compare two samples the model must aggregate each into a summary, and
attention aggregates by **summing** token embeddings. With additive embeddings
`Σ pos_emb + Σ val_emb`, positions and values sum *separately*, so a summary encodes
"which positions were observed" and "which values occurred" but never "which value
at which position". 0-vs-1 differ in the value marginal (a 1 has far less ink);
4-vs-9 differ only in the conjunction. Retrieval is unaffected because it is a
direct address match needing no summary.

It explained every observation, and predicted that a **conjunctive** token embedding
`MLP(concat[pos, val, ref])` would fix 4v9 while leaving 0v1 perfect.

**It did not.** 4v9 stayed at chance ({f('exp22'):.3f}) — and 0v1 **regressed from
{f('exp21'):.3f} to {f('exp23'):.3f}**. Two of three predictions wrong; the
hypothesis is rejected.

The regression is nevertheless the most informative single number here: destroying
the additive path destroyed the working case, which **confirms the model was relying
on a linear value-marginal readout** for 0v1 — while showing that a per-token
nonlinearity supplies nothing usable in its place.

## 6 · Where this leaves it

Established: the architecture does genuine in-context learning, its addressing and
copying are exact, and its matching is restricted to linearly-decodable marginal
statistics. Shape-based comparison is not merely hard for it — it extracts zero
information, and is unmoved by depth, context alignment, support size, auxiliary
retrieval training, or conjunctive embeddings.

The remaining suspect is the **comparison operation** itself. Attention returns a
weighted sum; it has no primitive for "compare my value at position p against yours
at position p, then pool the agreement over positions, grouped by ref". That needs a
multiplicative interaction followed by ref-grouped pooling — expressible in
principle, evidently not discovered by gradient descent here. Testing that means a
structural change (an explicit pairwise-comparison stage, or per-ref pooled
summaries the query can score against), not another embedding variant.

*Caveat on provenance: a `git checkout` of the tracked `results.jsonl` during a merge
discarded the exp17/exp18 rows. All numbers in this report are parsed from the
training logs, which are complete and authoritative. `results.jsonl` should be
untracked to prevent a recurrence.*
"""

url = save_report("universal-ar_report_boundary", md)
print("REPORT URL:", url)
print("figs:", url_map, url_iv, url_split, sep="\n  ")
