"""Report 3 -- gradient alignment with backprop against population (paper Figure 5, Table 8).

    CUDA_VISIBLE_DEVICES= ~/.local/bin/uv run --no-sync python projects/dust/scripts/gen_report_3.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))

from lib import diagrams, paper_ref as ref, results as R
from lib.figures import CKPT_NAME, LAYER_NAME, cosine_dust, cosine_vs_eggroll
from shared_lib.media import save_media
from shared_lib.report import save_report
from shared_lib.typst_report import save_figure

rows = R.load()
cos_rows = [r for r in rows if r.get("kind") == "cosine"]
theory = {r["ckpt"]: r for r in rows if r.get("kind") == "theory"}
CK = "bp100M_step6100"
POPS = (64, 256, 1024, 4096, 16384)
TYPES = ["head", "attn_out", "mlp", "token_emb", "v", "q", "k", "gate", "value_emb"]
NAME = dict(LAYER_NAME, mlp="MLP (in, out mean)")


def val(ckpt, method, pop, t):
    rs = [r for r in cos_rows if r["ckpt"] == ckpt and r["method"] == method and r["pop"] == pop]
    if not rs:
        return None
    keys = ("mlp_in", "mlp_out") if t == "mlp" else (t,)
    return float(np.mean([np.mean([r["cos"][k] for k in keys]) for r in rs]))


def f(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def predicted(ckpt, t, n_pairs):
    """Mean over layers of s_l / sqrt(s_l^2 + d_l / n): random-direction ES at n antithetic pairs."""
    th = theory.get(ckpt)
    if th is None:
        return None
    keys = ("mlp_in", "mlp_out") if t == "mlp" else (t,)
    out = []
    for k in keys:
        for l, s in th["share"][k].items():
            d = th["dim"][k][l]
            out.append(s / np.sqrt(s * s + d / n_pairs))
    return float(np.mean(out))


# ── figures ──────────────────────────────────────────────────────────────────
figs = {}
for ck in ("init", "bp100M_step610", CK):
    figs[ck] = (save_figure(cosine_dust(rows, ck, name=f"cosine_dust_{ck}_d"), fmt="svg") if cosine_dust(rows, ck) else None,
                save_figure(cosine_vs_eggroll(rows, ck, name=f"cosine_vs_eggroll_{ck}_d"), fmt="svg") if cosine_vs_eggroll(rows, ck) else None)
url_egg = save_media("dust_eggroll_update.svg", diagrams.eggroll_update().encode(), "image/svg+xml")

# ── tables ───────────────────────────────────────────────────────────────────
def dust_table(ck):
    head = "| layer type | " + " | ".join(f"K = {p:,}" for p in POPS) + " | paper's law, K = 16,384 |\n|---|" + "---:|" * (len(POPS) + 1) + "\n"
    body = ""
    for t in TYPES:
        law = ""
        if ck == CK:
            laws = [ref.cosine_law(k, 16384) for k in (("mlp_in", "mlp_out") if t == "mlp" else (t,)) if k in ref.COSINE_FIT]
            law = f(float(np.mean(laws))) if laws else "–"
        body += f"| {NAME[t]} | " + " | ".join(f(val(ck, "dust", p, t)) for p in POPS) + f" | {law} |\n"
    return head + body


def egg_table(ck):
    head = ("| layer type | " + " | ".join(f"{p:,} passes" for p in POPS)
            + " |\n|---|" + "---:|" * len(POPS) + "\n")
    body = ""
    for t in ("head", "attn_out", "mlp", "token_emb", "v", "q", "k"):
        cells = []
        for p in POPS:
            m, pr = val(ck, "eggroll", p, t), predicted(ck, t, p // 2)
            cells.append(f"{f(m)} (pred. {f(pr)})" if m is not None else "–")
        body += f"| {NAME[t]} | " + " | ".join(cells) + " |\n"
    return head + body


def fitted(ck, t):
    ks = [p for p in POPS if val(ck, "dust", p, t) is not None]
    if len(ks) < 4:
        return None
    return ref.fit_cosine_law(ks, [val(ck, "dust", p, t) for p in ks])


fit_rows = ""
for t in ("head", "attn_out", "mlp", "token_emb", "v", "q", "k"):
    fr = fitted(CK, t)
    keys = ("mlp_in", "mlp_out") if t == "mlp" else (t,)
    pap = [ref.COSINE_FIT[k] for k in keys if k in ref.COSINE_FIT]
    pc = f"{np.mean([p[0] for p in pap]):.2f}, {np.mean([p[1] for p in pap]):,.0f}" if pap else "–"
    fit_rows += f"| {NAME[t]} | {f'{fr[0]:.2f}, {fr[1]:,.0f}' if fr else '–'} | {pc} |\n"

d16, e16 = {t: val(CK, "dust", 16384, t) for t in TYPES}, {t: val(CK, "eggroll", 16384, t) for t in TYPES}
e_max = max(v for t, v in e16.items() if v is not None and t != "head")
egg_big = {p: val(CK, "eggroll", p, "head") for p in (65536, 131072)}
CMP = ("head", "attn_out", "mlp", "token_emb", "v", "q", "k")
n_fall = sum(1 for t in CMP if val(CK, "dust", 16384, t) < val("init", "dust", 16384, t))
ratios = [d16[t] / e16[t] for t in ("head", "attn_out", "mlp", "v", "q", "k") if e16[t] and e16[t] > 0]
ratio_lo, ratio_hi = min(ratios), max(ratios)
big_rows = [r for r in cos_rows if r["method"] == "eggroll" and r["pop"] >= 65536 and r["ckpt"] == CK]
big_txt = ""
if big_rows:
    big_txt = ("\n**At the paper's own population.** The paper quotes EGGROLL's cosine at 128k passes as under 0.05 for every type but the head. "
               "One measurement each at 65,536 and 131,072 passes on this checkpoint:\n\n| layer type | 65,536 passes | 131,072 passes |\n|---|---:|---:|\n"
               + "".join(f"| {NAME[t]} | {f(val(CK, 'eggroll', 65536, t))} | {f(val(CK, 'eggroll', 131072, t))} |\n"
                         for t in ("head", "attn_out", "mlp", "token_emb", "v", "q", "k")))
    others = [t for t in ("attn_out", "mlp", "token_emb", "v", "q", "k") if val(CK, "eggroll", 131072, t) is not None]
    under = all(val(CK, "eggroll", 131072, t) < 0.05 for t in others)
    growth = [val(CK, "eggroll", 131072, t) / val(CK, "eggroll", 65536, t) for t in ("head", "attn_out", "mlp", "v", "q", "k")]
    big_txt += (f"\nAt 131,072 passes {'every type but the head is under 0.05, as the paper reports' if under else 'not every type but the head is under 0.05'}; "
                f"the head is {f(val(CK, 'eggroll', 131072, 'head'))}. Doubling the passes multiplies the cosines by {min(growth):.2f} to {max(growth):.2f}, "
                "where a square-root law says 1.41. Even at this population EGGROLL's gradient is aligned to a few percent.\n")

md = f"""# DUST's gradient aligns with backprop's; EGGROLL's stays near zero

On the 100M-token checkpoint, DUST's gradient estimate at 16,384 draws has cosine **{f(d16['head'])}** with
backprop's on the head, **{f(d16['attn_out'])}** on attention outputs, **{f(d16['mlp'])}** on the MLPs and
**{f(d16['v'])}** on values. EGGROLL at 16,384 forward passes: **{f(e16['head'])}** on the head and at most
**{f(e_max)}** everywhere else. A random-direction estimate on this network should land there, and does. This is
the mechanism behind the ladder in the next report: the two methods feed the optimiser gradients of very different quality.

## What is measured

Take one batch of eight held-out sequences and one set of weights. Backprop gives the gradient. Each zeroth-order method
gives an estimate of it from forward passes. The **cosine** between the two, layer by layer, says how much of the estimate
points along the truth: 1.0 is backprop itself, 0 is no information. Population is draws (DUST) or forward passes (EGGROLL).

**What EGGROLL's cosine must be.** One population member is one number, and that number has to steer every parameter.
Average `n` antithetic pairs along random directions and the error in layer `l` is isotropic noise of energy
`d_l·|g|²/n`, where `d_l` is the layer's size and `|g|` the norm of the *whole* gradient, because every pair's score is
a sum over all layers. With `s_l = |g_l|/|g|` the layer's share of the gradient norm,

    cos_l(n) = s_l / sqrt( s_l² + d_l / n )  ≈  s_l · sqrt(n / d_l)

![EGGROLL on one weight matrix]({url_egg})

The shares are measured, not assumed (`scripts/eggroll_theory.py`). At initialisation the head holds
{theory['init']['share']['head']['0'] ** 2 * 100:.1f}% of the squared gradient norm, because the head's weights start near zero; at this checkpoint it holds
{theory[CK]['share']['head']['0'] ** 2 * 100:.0f}%. Every other layer is a small share of a gradient that EGGROLL must find
by random search, so its cosine there is tiny whatever the optimiser does with it.

## DUST: the cosine climbs with population

![DUST cosine to backprop, 100M-token checkpoint]({figs[CK][0]})

Cosine to backprop, 100M-token checkpoint, mean over noise draws. Last column: the paper's fitted law
`c_max / sqrt(1 + c/K)` (their Table 8, same checkpoint size) at K = 16,384.

{dust_table(CK)}
The curves have the shape of the paper's law. The head is near its ceiling by 4,096 draws ({f(val(CK, 'dust', 4096, 'head'))}).
Attention outputs ({f(val(CK, 'dust', 16384, 'attn_out'))}), MLPs ({f(val(CK, 'dust', 16384, 'mlp'))}) and queries
({f(val(CK, 'dust', 16384, 'q'))}) are still climbing at 16,384. Keys ({f(val(CK, 'dust', 16384, 'k'))}) and value embeddings
({f(val(CK, 'dust', 16384, 'value_emb'))}) are lowest. At K = 16,384 our cosines sit above the paper's law for every type, including
the head, by {f(min(val(CK, 'dust', 16384, t) - float(np.mean([ref.cosine_law(k, 16384) for k in (('mlp_in', 'mlp_out') if t == 'mlp' else (t,))])) for t in ('head', 'attn_out', 'mlp', 'token_emb', 'v', 'q', 'k', 'value_emb')), 2)} to
{f(max(val(CK, 'dust', 16384, t) - float(np.mean([ref.cosine_law(k, 16384) for k in (('mlp_in', 'mlp_out') if t == 'mlp' else (t,))])) for t in ('head', 'attn_out', 'mlp', 'token_emb', 'v', 'q', 'k', 'value_emb')), 2)}.
The law is fitted to their checkpoint and their per-population estimator constants, so the offset is not a defect.
Fitting `c_max / sqrt(1 + c/K)` to our five populations and setting it beside the paper's (`c_max, c`):

| layer type | ours | paper's Table 8 |
|---|---|---|
{fit_rows}
Five points cannot pin the ceiling of a curve that has not flattened, so only the half-saturation scale `c` is
informative for the slow types.

## EGGROLL: where random directions put it

![DUST and EGGROLL, 100M-token checkpoint]({figs[CK][1]})

Measured cosine at each population, with the prediction `cos_l(n)` above in brackets (n = passes ÷ 2 pairs):

{egg_table(CK)}{big_txt}
## Alignment at a fixed population falls as the network trains

Cosine at K = 16,384 draws, DUST, on the initial weights and on the two checkpoints. It is lower after training than at
initialisation for {n_fall} of these {len(CMP)} layer types.

| layer type | initialisation | 10M tokens | 100M tokens |
|---|---:|---:|---:|
""" + "".join(f"| {NAME[t]} | {f(val('init', 'dust', 16384, t))} | {f(val('bp100M_step610', 'dust', 16384, t))} | {f(val(CK, 'dust', 16384, t))} |\n"
              for t in ("head", "attn_out", "mlp", "token_emb", "v", "q", "k")) + f"""
## What this changes

- The ladder gap has a cause we can measure. At 16,384 the two methods hand the optimiser gradients whose cosine differs
  by a factor of {ratio_lo:.0f} to {ratio_hi:.0f}, depending on the layer.
- EGGROLL needs no explanation beyond arithmetic: its cosine is `s·sqrt(n/d)`, and `d` is 3.8×10⁷.
- DUST's weakest layers (keys and value embeddings) are credited through the *estimated* attention-output error rather than
  the loss, so their error compounds the attention-output estimate's own. They improve slowest with population. If the
  ladder flattens before backprop, look there first.
- One batch of eight sequences, a few noise draws per cell, and our own 100M-token checkpoint (trained with the paper's
  10M-token optimiser settings): the shapes are firm, the third decimal is not.

Code: `lib/align.py`, `scripts/eggroll_theory.py`, `experiments6.py`{", `experiments15.py`" if big_rows else ""}.
"""

out = PROJECT / "reports" / "03-alignment.md"
out.write_text(md)
print(save_report("dust_report_3_alignment", md))
