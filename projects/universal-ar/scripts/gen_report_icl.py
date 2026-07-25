"""
Universal-AR — report: the in-context-learning test (exp10 / exp11 / exp12).

exp10 fixed labels 0.828 → exp11 permuted labels 0.109 (memorisation exposed)
→ exp12 2-way permuted 1.000 (match-and-copy proven).

Parses logs + results.jsonl on the server, builds figures, publishes the report.

Usage (server): uv run python projects/universal-ar/scripts/gen_report_icl.py
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure
from shared_lib.report import save_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
LOGS = ROOT / "logs"
STEP_RE = re.compile(r"step\s+(\d+)\s+loss\s+([\d.]+)\s+label_te\(balanced\)\s+([\d.]+)\s+ink_te\s+([\d.]+)")


def curve(name):
    step, loss, lab, ink = [], [], [], []
    for line in open(LOGS / f"{name}.log", errors="ignore"):
        m = STEP_RE.search(line)
        if m:
            step.append(int(m.group(1))); loss.append(float(m.group(2)))
            lab.append(float(m.group(3))); ink.append(float(m.group(4)))
    return dict(step=step, loss=loss, label=lab, ink=ink)


rows = {}
for line in open(ROOT / "results.jsonl"):
    d = json.loads(line); rows[d["experiment"]] = d

c10, c11, c12 = curve("exp10"), curve("exp11"), curve("exp12")
r10, r11, r12 = rows["exp10"], rows["exp11"], rows["exp12"]
print("exp10", r10["label_te"], "exp11", r11["label_te"], "exp12", r12["label_te"])

# ── figure 1: the three curves, each against its own chance line ──────────────
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), sharey=True)
specs = [(axes[0], c10, 0.10, "exp10 — 10-way, FIXED labels", "#c0392b",
          "looks great,\nbut memorised"),
         (axes[1], c11, 0.10, "exp11 — 10-way, PERMUTED labels", "#7f8c8d",
          "collapses to chance:\nexp10 was memorisation"),
         (axes[2], c12, 0.50, "exp12 — 2-way (0 vs 1), PERMUTED", "#1e8449",
          "perfect ⇒ genuine\nmatch-and-copy")]
for ax, c, ch, title, col, note in specs:
    ax.plot(c["step"], c["label"], "-o", ms=4, color=col, lw=2)
    ax.axhline(ch, color="gray", ls="--", lw=1.2)
    ax.text(c["step"][-1], ch + 0.02, f"chance {ch:.2f}", ha="right", fontsize=8, color="gray")
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("training step"); ax.set_ylim(0, 1.05); ax.grid(alpha=.25)
    ax.text(0.5, 0.06, note, transform=ax.transAxes, ha="center", fontsize=9,
            color=col, weight="bold")
    ax.annotate(f"{c['label'][-1]:.3f}", xy=(c["step"][-1], c["label"][-1]),
                xytext=(-8, 8 if c["label"][-1] < 0.9 else -18), textcoords="offset points",
                fontsize=11, color=col, weight="bold", ha="right")
axes[0].set_ylabel("label accuracy (balanced, test query)")
fig.suptitle("Anonymised-label test: is it in-context learning, or memorisation?", fontsize=12)
fig.tight_layout()
url_curves = save_matplotlib_figure("universal-ar_icl_curves", fig, format="svg")
print("fig curves:", url_curves)

# ── figure 2: accuracy above chance ──────────────────────────────────────────
names = ["exp10\n10-way\nfixed", "exp11\n10-way\npermuted", "exp12\n2-way\npermuted"]
acc = [r10["label_te"], r11["label_te"], r12["label_te"]]
chance = [0.10, 0.10, 0.50]
lift = [a - c for a, c in zip(acc, chance)]
fig2, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4))
x = range(3); cols = ["#c0392b", "#7f8c8d", "#1e8449"]
axa.bar(x, acc, color=cols, width=.55)
axa.plot(x, chance, "k_", ms=40, mew=2, label="chance")
for i, (a, c) in enumerate(zip(acc, chance)):
    axa.text(i, a + .02, f"{a:.3f}", ha="center", fontsize=10, weight="bold")
axa.set_xticks(list(x)); axa.set_xticklabels(names, fontsize=9)
axa.set_ylabel("label accuracy"); axa.set_ylim(0, 1.15); axa.legend(fontsize=8); axa.grid(alpha=.25, axis="y")
axa.set_title("Raw accuracy (different chance levels!)")

axb.bar(x, lift, color=cols, width=.55)
axb.axhline(0, color="k", lw=1)
for i, l in enumerate(lift):
    axb.text(i, l + (.02 if l >= 0 else -.05), f"{l:+.3f}", ha="center", fontsize=10, weight="bold")
axb.set_xticks(list(x)); axb.set_xticklabels(names, fontsize=9)
axb.set_ylabel("accuracy − chance"); axb.set_ylim(-0.1, 0.85); axb.grid(alpha=.25, axis="y")
axb.set_title("Lift over chance — the honest comparison")
fig2.tight_layout()
url_bars = save_matplotlib_figure("universal-ar_icl_lift", fig2, format="svg")
print("fig bars:", url_bars)

md = f"""# Universal-AR — in-context learning: exposed, then proven

*Token-level model: every datum is a `(pos, value, ref)` token, the label is a token
at `pos_label`, a 4-layer transformer (d=256, 3.4M params) does masked-token
completion over a flat token bag. Half-observed images (OBS_FRAC=0.5, ~6.5k-token
context), effective batch 8, support from TRAIN, query from TEST, balanced
1-shot-per-class evaluation.*

**Three runs, one question: was any of this actually in-context learning?**

| exp | task | label semantics | label acc | chance | verdict |
|---|---|---|---|---|---|
| exp10 | 10-way | **fixed** | {r10['label_te']:.3f} | 0.10 | memorisation |
| exp11 | 10-way | **permuted** | {r11['label_te']:.3f} | 0.10 | nothing |
| **exp12** | **2-way (0 vs 1)** | **permuted** | **{r12['label_te']:.3f}** | **0.50** | **genuine ICL** |

![curves]({url_curves})

## The test

With **fixed** label semantics, class 3 always maps to the same label token in every
episode. A query that sees half of its own image can therefore classify itself
against **memorised class prototypes** without ever reading the support set. Any
score under that regime is ambiguous.

**Anonymisation removes the ambiguity.** Each episode draws a fresh random
permutation of classes onto label tokens:

```
label_token(sample) = K + perm[true_class],   perm ~ random permutation, per episode
```

Now "which token means this digit" changes every episode. Memorised prototypes are
worth **exactly chance**, so the only way to score above chance is to match the
query's pixels against a support sample and copy *that episode's* token.

## What happened

**exp11 — the 10-way anonymised run collapsed to chance ({r11['label_te']:.3f}).**
Flat from step 1000 through 8000 (0.10, 0.13, 0.13, 0.10, 0.08, 0.09, 0.09, 0.11);
never a hint of a trend. This retroactively **invalidates exp10's {r10['label_te']:.3f}
as evidence of in-context learning** — it was memorisation, exactly as the standing
caveat warned.

A revealing detail: content completion *improved* under anonymisation
({r10['ink_te']:.3f} → {r11['ink_te']:.3f} ink). The label task became unlearnable, so
capacity moved to pixel completion. The model was training fine; it simply could not
do the label task.

**exp12 — the same test, made easy, is perfect ({r12['label_te']:.3f}).** Two classes
(digits 0 vs 1), labels still permuted, everything else untouched. Chance is 0.50
because the mapping is swapped in half the episodes — a memoriser scores 0.50 by
construction. The model reaches 0.93 by step 2000 and a flat **1.000** from step
3000 onward.

![lift over chance]({url_bars})

## Conclusion

**The match-and-copy circuit exists.** A perfect score under permuted labels cannot
be produced by memorisation; the model is genuinely reading the support set,
matching a half-observed query against a half-observed support example, and copying
that example's label token. This is the first unambiguous in-context learning result
in the project.

**exp11's failure was task difficulty, not architecture.** That is the important
correction — a 10-way collapse alone looked like it might be a dead end for this
design. It isn't.

Both facts follow from the *same* architecture and the *same* training recipe. Only
the number of classes changed.

## Open question — where does it break?

The gap between 2-way-perfect and 10-way-chance is enormous and unexplored. Two
hypotheses, each cheap to test:

1. **Class count.** Matching may degrade as the support set grows.
   → sweep N-way ∈ {{2, 3, 5, 10}} and find the cliff. *(exp14: 5-way, running)*
2. **Visual difficulty.** 0 vs 1 is trivially separable; MNIST's confusable pairs
   (4/9, 3/8, 5/6) may not be under half-observation.
   → 2-way with a **hard** pair holds class count fixed and isolates
   discriminability. *(exp13: 4 vs 9, running)*

If 4-vs-9 also approaches 1.0, the limit is class count. If it collapses toward
0.50, the limit is visual matching under partial observation — and more classes
simply compounds it.

## Caveat that remains

The content (ink ≈ {r12['ink_te']:.3f}) result is **not** evidence of in-context
learning. Completing a sample's held-out pixels can be done from that sample's own
observed half plus learned MNIST statistics — it needs no cross-sample reading. Only
the anonymised **label** metric probes genuine ICL.
"""

url = save_report("universal-ar_report_icl", md)
print("REPORT URL:", url)
