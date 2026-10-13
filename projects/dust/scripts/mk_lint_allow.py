"""Explain every prose number the paper's numeric check could not match, or say that it cannot.

`--check` compares each number in the paper's prose with the individual values in results.jsonl. Three kinds of
number legitimately fail that: aggregates the paper quotes (a mean over seeds, a difference between two means, a mean of
cosines over layers), numbers that are the DUST paper's own, and constants of the code. This script finds which kind each
unmatched literal is by recomputing the candidates, and writes `.lint-allow` with the reason beside each. A literal it
cannot trace is printed as UNEXPLAINED and not written: that is a typo or a number with no stated provenance, and it
needs fixing, not silencing.

    CUDA_VISIBLE_DEVICES= ~/.local/bin/uv run --no-sync python projects/dust/scripts/mk_lint_allow.py
"""

import re
import sys
from itertools import product
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))

from lib import paper_ref as ref, results as R
from shared_lib.paper_lint import check_paper

PAPER = PROJECT / "paper"
rows = R.load()
cands: list[tuple[int, float, str]] = []


def _priority(why: str) -> int:
    """When two sources round to the same literal, the more specific one is the explanation: a constant of the code
    or a chunk of a named integer, then the DUST paper's own number, then an aggregate of the ladder, then the rest."""
    if why.startswith(("chunk of", "hardware", "population of", "draws per update")):
        return 0
    if why.startswith("published"):
        return 1
    if why.startswith(("mean cosine", "random-direction", "head share")):
        return 3
    return 2


def add(x, why):
    if x is not None and np.isfinite(x):
        cands.append((_priority(why), float(x), why))


# 1. the DUST paper's own numbers
for m in ("dust", "eggroll"):
    for p, d in ref.TABLE1[m].items():
        for b, v in d.items():
            add(v, f"published: DUST paper Table 1, {m}, {p} population, {b} tokens")
for b, v in ref.TABLE1["bp"].items():
    add(v, f"published: DUST paper Table 1, backprop, {b} tokens")
for k, (cmax, c, rmse) in ref.COSINE_FIT.items():
    for K in (64, 256, 1024, 4096, 16384):
        add(float(ref.cosine_law(k, K)), f"published law: DUST paper Table 8 fit for {k} evaluated at K={K}")
pub_slope = (ref.TABLE1["eggroll"][64]["1M"] - ref.TABLE1["eggroll"][1024]["1M"]) / 4
add(pub_slope, "published: DUST paper EGGROLL ladder, loss gained per doubling of population, 64 to 1,024 passes at 1M tokens")
for mult in (6000, 8200, 23000):
    add(mult * 64, f"published: DUST paper Table 2 multiple {mult} x 64 forward passes")

# 2. aggregates of logged runs
cells = {}
for m, b, p in product(("bp", "dust", "eggroll"), ("100K", "1M", "10M"), (0, 64, 256, 1024, 4096)):
    c = R.cell(rows, method=m, budget=b, pop=p)
    if c:
        cells[(m, b, p)] = c
        tag = f"{m} {b} {p}"
        add(c["mean"], f"mean over seeds of test loss, {tag}")
        add(c["sd"], f"sd over seeds of test loss, {tag}")
        add(c["step_s"], f"mean seconds per update, {tag}")
        pubv = ref.TABLE1.get(m, {}).get(p, {}).get(b) if m != "bp" else ref.TABLE1["bp"].get(b)
        if pubv is not None:
            add(c["mean"] - pubv, f"difference, ours minus published, {tag}")
for (m, b, p), c in cells.items():
    bp = cells.get(("bp", b, 0))
    if m == "dust" and bp:
        add(c["mean"] - bp["mean"], f"DUST minus backprop, {b} tokens, {p} draws")
    e = cells.get(("eggroll", b, p))
    if m == "dust" and e:
        add(e["mean"] - c["mean"], f"EGGROLL minus DUST, {b} tokens, population {p}")
        add(e["step_s"] / c["step_s"], f"EGGROLL / DUST seconds per update, population {p}")
        if p in ref.TABLE1["dust"]:
            add(ref.TABLE1["eggroll"][p][b] - ref.TABLE1["dust"][p][b], f"published EGGROLL minus DUST, {b} tokens, population {p}")
    if m == "eggroll" and p in ref.TABLE1["eggroll"]:
        add(c["mean"] - ref.TABLE1["eggroll"][p][b], f"our EGGROLL minus the paper's, {b} tokens, {p} passes")
d64, e1k = cells.get(("dust", "1M", 64)), cells.get(("eggroll", "1M", 1024))
if d64 and e1k:
    add(e1k["mean"] - d64["mean"], "EGGROLL at 1,024 passes minus DUST at 64 draws, 1M tokens")
    add(e1k["step_s"] / d64["step_s"], "cost ratio, EGGROLL 1,024 passes over DUST 64 draws")
if all(k in cells for k in (("eggroll", "1M", 256), ("dust", "1M", 256))):
    e, d = cells[("eggroll", "1M", 256)]["mean"], cells[("dust", "1M", 256)]["mean"]
    add((e - d) / (e - ref.TABLE1["eggroll"][256]["1M"]), "ladder gap over our-EGGROLL-vs-paper gap, 256 population, 1M tokens")

# 2b. Adam and Muon (section 6.8): cells by (method, optimiser, population), and the differences the prose quotes
from lib.figures import opt_cell
OPTS = ("sgd", "adam", "muon")
ocell = {}
for m, p in product(("bp", "dust", "eggroll"), (0, 64, 256, 1024, 4096)):
    for o in OPTS:
        c = opt_cell(rows, m, p, o)
        if c:
            ocell[(m, p, o)] = c
            tag = f"{m} {p} {o}"
            add(c["mean"], f"mean over seeds of test loss, {tag} (optimiser comparison)")
            add(c["sd"], f"sd over seeds of test loss, {tag} (optimiser comparison)")
            add(c["step_s"], f"mean seconds per update, {tag} (optimiser comparison)")
for (m, p, o), c in ocell.items():
    bpc = ocell.get(("bp", 0, o))
    if m == "dust" and bpc:
        add(c["mean"] - bpc["mean"], f"DUST minus backprop under {o}, {p} draws")
    sgd = ocell.get((m, p, "sgd"))
    if sgd and o != "sgd":
        add(sgd["mean"] - c["mean"], f"gain of {o} over SGD, {m} {p}")
    if m == "eggroll" and o == "muon" and ocell.get((m, p, "adam")):
        add(c["mean"] - ocell[(m, p, "adam")]["mean"], f"EGGROLL under Muon minus under Adam, {p} passes")
    if m == "dust" and o == "muon" and ocell.get((m, p, "adam")):
        add(ocell[(m, p, "adam")]["mean"] - c["mean"], f"DUST under Adam minus under Muon, {p} draws")
    e = ocell.get(("eggroll", p, o))
    if m == "dust" and e:
        add(e["mean"] - c["mean"], f"EGGROLL minus DUST under {o}, population {p}")
for o in ("adam", "muon"):
    b = ocell.get(("bp", 0, o))
    if b and ocell.get(("bp", 0, "sgd")):
        add(ocell[("bp", 0, "sgd")]["mean"] - b["mean"], f"gain of {o} over SGD, backprop")
if ocell.get(("bp", 0, "adam")):
    add(ocell[("bp", 0, "adam")]["mean"] - ref.ADAM_PAPER["bp_1M"], "our backprop under Adam minus the DUST paper's")
for k, v in ref.ADAM_PAPER.items():
    add(v, f"published: DUST paper Section 3 (Adam), {k}")
ex18 = R.cell(rows, exps=("exp18",), method="bp", budget="1M", pop=0)
if ex18:
    add(ex18["mean"], "mean over seeds, exp18 (backprop with DUST's group rates)")
    add(ex18["sd"], "sd over seeds, exp18")
rep = [(a, b) for a in R.train_rows(rows, exps=("exp21",), seed=42, lr=0.0015)
       for b in R.train_rows(rows, exps=("exp24",), method="dust", pop=256, opt="adam", seed=42)]
if rep:
    add(abs(rep[0][0]["test_at_best_val"] - rep[0][1]["test_at_best_val"]), "same cell run twice: difference of the two test losses (GPU nondeterminism)")

# 3. alignment: mean of cosines over noise draws, and over layer groups
cos = [r for r in rows if r.get("kind") == "cosine"]
types = ("head", "attn_out", "mlp_in", "mlp_out", "token_emb", "v", "q", "k", "gate", "value_emb")
for ck, m, p in {(r["ckpt"], r["method"], r["pop"]) for r in cos}:
    rs = [r for r in cos if (r["ckpt"], r["method"], r["pop"]) == (ck, m, p)]
    for t in types:
        add(np.mean([r["cos"][t] for r in rs]), f"mean cosine to backprop, {t}, {m}, {p}, {ck}")
    add(np.mean([(r["cos"]["mlp_in"] + r["cos"]["mlp_out"]) / 2 for r in rs]), f"mean cosine, MLP (in and out averaged), {m}, {p}, {ck}")
for th in (r for r in rows if r.get("kind") == "theory"):
    for t in types:
        if t not in th["share"]:
            continue
        for n in (32, 128, 512, 2048, 8192, 32768, 65536):
            vals = [s / np.sqrt(s * s + th["dim"][t][l] / n) for l, s in th["share"][t].items()]
            add(np.mean(vals), f"random-direction prediction s/sqrt(s^2+d/n), {t}, n={n} pairs, {th['ckpt']}")
    add(th["share"]["head"]["0"] ** 2, f"head share of squared gradient norm, {th['ckpt']}")

# 4. the extrapolation row
ex = next((r for r in rows if r.get("kind") == "extrapolation"), None)
if ex:
    for k, f in ex["fits"].items():
        if f.get("multiple_of_64"):
            add(f["multiple_of_64"], f"extrapolation row: {k} multiple of 64")
            add(f["multiple_of_64"] * 64, f"extrapolation row: {k} forward passes")
        if "slope_per_doubling" in f:
            add(-f["slope_per_doubling"], f"extrapolation row: {k} loss gained per doubling")
    for a, b in product([f.get("multiple_of_64") for f in ex["fits"].values() if f.get("multiple_of_64")], (6000, 8200)):
        add(b / a, "published multiple over ours")
        add(b * 64 / a / 64, "published multiple over ours")

# 5. constants of the code and the data
CONSTANTS = {
    "1000": "population of 1,000 forward passes in the worked example of the Background; also the token embedding's SGD rate (released recipe)",
    "4090": "hardware: RTX 4090",
    "110": "draws per update for the residual writers at K=256 (released recipe, Appendix E of the DUST paper)",
    "1024": "population 1,024 (a rung of the ladder) and the attention window",
    "3.4445": "Muon's quintic Newton-Schulz coefficient a (modded-nanoGPT)",
    "4.7750": "Muon's quintic Newton-Schulz coefficient -b (modded-nanoGPT)",
    "2.0315": "Muon's quintic Newton-Schulz coefficient c (modded-nanoGPT)",
    "0.95": "Muon momentum, Adam beta2 (DUST paper Table 7), Nesterov",
    "0.7": "Adam beta1 for DUST (DUST paper Table 7)",
    "0.8": "Adam beta1 for backprop (DUST paper Table 7)",
    "0.5": "Adam beta1 for EGGROLL (DUST paper Table 7); also the token embedding's share of the Adam draws",
    "0.2": "residual-scalar rate for backprop under Adam (Table 7); Muon's per-entry RMS factor; noise scale of the SGD recipe",
    "0.3": "embedding rate for backprop under Adam (Table 7); value-embedding rate of the released SGD recipes",
    "0.4": "noise scale of the token embedding under Adam; attention outputs in blocks 5-8",
    "0.003": "EGGROLL rate and noise scale; Adam rate",
    "0.03": "residual-scalar noise scale and SGD rate (released recipes)",
    "0.25": "share of K for residual writers and the MLP first layer under Adam (DUST paper Appendix E)",
    "0.375": "share of K for the attention outputs under Adam",
    "1.375": "0.25 + 0.25 + 0.375 + 0.5: draws per update under Adam as a multiple of the nominal population",
    "0.43": "110 / 256: share of K for residual writers and the MLP first layer under the SGD recipe",
    "0.10": "26 / 256: share of K for the attention outputs under the SGD recipe",
    "0.04": "10 / 256: share of K for the token embedding under the SGD recipe",
    "300": "embedding rate multiple under Adam for DUST (Table 7)",
    "100": "residual-scalar rate multiple under Adam for DUST (Table 7)",
}
for name, why in CONSTANTS.items():
    add(float(name), why)
# a thousands-separated integer is split by the check at the commas: explain each chunk by the integer it came from
integers = {int(v): "" for r in rows for v in [r.get("pop"), r.get("heldout_tokens"), r.get("n_params")] if isinstance(v, (int, float)) and v}
integers.update({16384: "population 16,384", 65536: "population 65,536", 131072: "population 131,072", 1114112: "544 x 2048 held-out tokens",
                 380000: "rounded from 6,000 x 64 = 384,000", 520000: "rounded from 8,200 x 64 = 524,800"})
if ex:
    for k, f in ex["fits"].items():
        if f.get("multiple_of_64"):
            integers[int(round(f["multiple_of_64"]))] = f"extrapolation row: {k} multiple of 64, rounded"
            integers[int(round(f["multiple_of_64"] * 64))] = f"extrapolation row: {k} forward passes, rounded"
for v, why in integers.items():
    s = f"{v:,}"
    for chunk in s.split(","):
        if chunk.isdigit():
            add(float(chunk), f"chunk of the thousands-separated {s}" + (f" ({why})" if why else ""))


def explain(literal: str) -> str | None:
    if "," in literal:
        return None
    v = float(literal)
    p = len(literal.split(".")[1]) if "." in literal else 0
    hits = [(prio, why) for prio, x, why in cands if abs(round(x, p) - v) < 1e-9 or (p == 0 and abs(round(x) - v) < 1e-9)]
    return min(hits)[1] if hits else None


(PAPER / ".lint-allow").unlink(missing_ok=True)      # lint against an empty allow-list, or nothing would be left to explain
report = check_paper(PAPER, results=PROJECT / "results.jsonl")
unmatched = [f.message.split()[0] for f in report.findings if f.kind == "unmatched-number"]
lines, unexplained = [], []
for lit in sorted(set(unmatched), key=lambda s: (len(s), s)):
    why = explain(lit)
    (lines if why else unexplained).append((lit, why))

header = ("# Prose numbers that are not single values in results.jsonl.\n"
          "# Generated by scripts/mk_lint_allow.py, which refuses to list a number it cannot trace to an aggregate\n"
          "# of logged values, to the DUST paper's own tables, or to a constant of the code. Every line's reason is that provenance.\n\n")
(PAPER / ".lint-allow").write_text(header + "".join(f"{lit:<12} # {why}\n" for lit, why in lines))
print(f"{len(lines)} explained, {len(unexplained)} unexplained")
for lit, _ in unexplained:
    print("UNEXPLAINED:", lit)
