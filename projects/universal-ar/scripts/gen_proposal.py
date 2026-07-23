"""
Build the token-level Universal-AR proposal: two diagrams + the proposal text,
publish to R2, and write projects/universal-ar/proposal.md.

Usage (server): uv run python projects/universal-ar/scripts/gen_proposal.py
"""
import io, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_media
from shared_lib.report import save_report

C = dict(pos="#bcd7f0", pos_s="#5a8fc0", val="#c6e6c9", val_s="#5aa564",
         ref1="#f2d4a8", ref1s="#c9924a", ref2="#d9c7f0", ref2s="#8f79c8", ref3="#bfe6de", ref3s="#4fa895",
         mask="#f6c3c0", mask_s="#d0453f", box="#f4f7fb", box_s="#c3ccd6", txt="#1c2733", mut="#67727e", hl="#d0453f")


def _svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'font-family="ui-sans-serif,system-ui,sans-serif">'
            f'<rect width="{w}" height="{h}" fill="white"/>'
            '<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
            f'<path d="M0,0 L7,3 L0,6 Z" fill="{C["mut"]}"/></marker></defs>' + body + '</svg>')


def T(x, y, s, sz=13, col=C["txt"], w="normal", anc="start"):
    return f'<text x="{x}" y="{y}" font-size="{sz}" fill="{col}" font-weight="{w}" text-anchor="{anc}">{s}</text>'
def R(x, y, w, h, fill, stroke, rx=6):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'
def L(x1, y1, x2, y2, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C["mut"]}"{d} marker-end="url(#ar)"/>'


def token(x, y, pv, ref_fill, ref_s, mask=False, held=False):
    """Draw a token: 3 stacked chips pos/value/ref (compact)."""
    b = [R(x, y, 96, 52, C["box"], (C["hl"] if held else C["box_s"]), 7)]
    if held: b.append(f'<rect x="{x-3}" y="{y-3}" width="102" height="58" rx="9" fill="none" stroke="{C["hl"]}" stroke-width="2.5" stroke-dasharray="5 3"/>')
    b.append(R(x+6, y+6, 40, 15, C["pos"], C["pos_s"], 3) + T(x+26, y+17, pv[0], 9.5, C["txt"], "normal", "middle"))
    vf, vs = (C["mask"], C["mask_s"]) if mask else (C["val"], C["val_s"])
    b.append(R(x+50, y+6, 40, 15, vf, vs, 3) + T(x+70, y+17, ("_" if mask else pv[1]), 10.5, C["txt"], "bold" if mask else "normal", "middle"))
    b.append(R(x+6, y+25, 84, 15, ref_fill, ref_s, 3) + T(x+48, y+36, pv[2], 9.5, C["txt"], "normal", "middle"))
    return "".join(b)


def diagram_anatomy():
    b = [T(26, 30, "Anatomy: a token, the bag, and what is held out", 18, C["txt"], "bold")]
    # 1 · token
    b.append(T(26, 62, "1 · Every datum is a token = (pos, value, ref)", 14, C["txt"], "bold"))
    b.append(token(40, 74, ("pos p", "value v", "ref s"), C["ref1"], C["ref1s"]))
    b.append(T(150, 92, "pos and ref are the address; value is the content.", 12.5, C["mut"]))
    b.append(T(150, 110, "A LABEL is just a token at a special position:", 12.5, C["txt"]))
    b.append(token(150, 118, ("pos_label", "class", "ref s"), C["ref1"], C["ref1s"]))
    b.append(T(260, 136, "value vocab = pixel-bins ∪ classes. Classifying a", 12.5, C["mut"]))
    b.append(T(260, 154, "sample = predict value at (pos_label, ref) — same op as any pixel.", 12.5, C["mut"]))

    # 2 · the bag
    b.append(T(26, 202, "2 · The context is one flat BAG of tokens (no sample rows); ref groups a sample's tokens", 14, C["txt"], "bold"))
    bag = [(60, 224, ("p12", "v", "s=🟠"), C["ref1"], C["ref1s"]), (176, 232, ("p03", "v", "s=🟣"), C["ref2"], C["ref2s"]),
           (300, 220, ("p45", "v", "s=🟢"), C["ref3"], C["ref3s"]), (420, 236, ("p71", "v", "s=🟠"), C["ref1"], C["ref1s"]),
           (540, 222, ("pL", "class", "s=🟣"), C["ref2"], C["ref2s"]), (660, 234, ("p22", "v", "s=🟢"), C["ref3"], C["ref3s"]),
           (60, 292, ("pL", "class", "s=🟠"), C["ref1"], C["ref1s"]), (300, 292, ("p61", "v", "s=🟣"), C["ref2"], C["ref2s"]),
           (540, 292, ("p08", "v", "s=🟢"), C["ref3"], C["ref3s"])]
    for x, y, pv, f, s in bag:
        b.append(token(x, y, pv, f, s))
    b.append(T(800, 250, "• order-free (a set)", 12, C["mut"]))
    b.append(T(800, 268, "• ref = random tag per", 12, C["mut"])); b.append(T(812, 284, "episode, but the SAME", 12, C["mut"]))
    b.append(T(812, 300, "tag is on all of a sample's", 12, C["mut"])); b.append(T(812, 316, "tokens (exact-match key).", 12, C["mut"]))

    # 3 · hold-out
    b.append(T(26, 366, "3 · Hold-out principle — the joint is absent, but each part appears individually", 14, C["txt"], "bold"))
    b.append(token(40, 380, ("pos*", "value*", "ref*"), C["ref1"], C["ref1s"], mask=True, held=True))
    b.append(T(150, 398, "HELD-OUT token: query (pos*, _, ref*) → predict value*.", 12.5, C["txt"]))
    b.append(T(150, 418, "In the context, individually present (never jointly):", 12.5, C["mut"]))
    parts = [(150, 430, ("pos*", "v'", "ref other"), C["ref2"], C["ref2s"]), (270, 430, ("pos other", "value*", "ref other"), C["ref3"], C["ref3s"]),
             (390, 430, ("pos other", "v''", "ref*"), C["ref1"], C["ref1s"])]
    for x, y, pv, f, s in parts:
        b.append(token(x, y, pv, f, s))
    b.append(T(520, 452, "pos* seen (other refs) · value* seen (other tokens) · ref* seen (its", 12, C["mut"]))
    b.append(T(520, 468, "other pixels). The model must BIND seen parts into an unseen whole.", 12, C["mut"]))
    return _svg(1040, 500, "".join(b))


def diagram_mechanism():
    b = [T(26, 30, "Mechanism: token-level attention completes a masked token", 18, C["txt"], "bold")]
    b.append(T(26, 58, "Context tokens (bag)", 12.5, C["mut"]))
    xs = [(40, C["ref1"], C["ref1s"]), (140, C["ref2"], C["ref2s"]), (240, C["ref3"], C["ref3s"]),
          (340, C["ref1"], C["ref1s"]), (440, C["ref2"], C["ref2s"])]
    for x, f, s in xs:
        b.append(token(x, 68, ("pos", "val", "ref"), f, s))
    b.append(T(560, 58, "Query token", 12.5, C["hl"]))
    b.append(token(560, 68, ("pos*", "_", "ref*"), C["ref1"], C["ref1s"], mask=True, held=True))

    # attention block
    b.append(R(40, 156, 620, 44, C["box"], C["box_s"], 8))
    b.append(T(350, 183, "L layers of multi-head self-attention over the token set  (permutation-invariant; pos is a field, not order)", 12.5, C["txt"], "normal", "middle"))
    for x, *_ in xs + [(560,)]:
        b.append(L(x+48, 120, x+48, 156))
    # two hops
    b.append(T(40, 234, "What the layers learn to do:", 13, C["txt"], "bold"))
    b.append(R(40, 244, 300, 62, "#fbf6ee", C["ref1s"], 8))
    b.append(T(52, 264, "hop 1 · gather same-ref* tokens", 12.5, C["txt"], "bold"))
    b.append(T(52, 282, "(exact ref match) → build ref*'s", 12, C["mut"]))
    b.append(T(52, 298, "content representation", 12, C["mut"]))
    b.append(R(360, 244, 300, 62, "#eef7f4", C["ref3s"], 8))
    b.append(T(372, 264, "hop 2 · match cross-sample tokens", 12.5, C["txt"], "bold"))
    b.append(T(372, 282, "at pos* from similar samples →", 12, C["mut"]))
    b.append(T(372, 298, "read their value  ⇒  predict value*", 12, C["mut"]))
    b.append(L(340, 275, 360, 275))
    # output
    b.append(L(660, 275, 700, 275))
    b.append(R(700, 250, 150, 50, C["val"], C["val_s"], 8))
    b.append(T(775, 272, "predicted value*", 12.5, C["txt"], "bold", "middle"))
    b.append(T(775, 290, "(softmax over vocab)", 11, C["mut"], "normal", "middle"))
    b.append(T(40, 340, "One random ref can't drive similarity, so hop 2 needs learned content matching — which is why it is MULTI-layer, not one associative read.", 12, C["mut"]))
    return _svg(1040, 370, "".join(b))


def main():
    a_url = save_media("universal-ar_proposal_anatomy.svg", io.BytesIO(diagram_anatomy().encode()), "image/svg+xml")
    m_url = save_media("universal-ar_proposal_mechanism.svg", io.BytesIO(diagram_mechanism().encode()), "image/svg+xml")

    md = f"""# Universal AR — Proposal (token-level, v1)

*A clean restart. The archived exp1–7 (per-sample HRR associative memory) is kept
under `archive/` for its results and lessons; this proposal adopts the token-level
consensus.*

## Premise
Dissolve the "sample". A dataset is a **flat bag of tokens**, each a
`(pos, value, ref)` triple. Every prediction — classification, inpainting,
denoising — is the **same operation**: complete a masked token's value given its
`(pos, ref)`. In-context, continual, compositional.

## Principles

![anatomy]({a_url})

1. **Token = `(pos, value, ref)`** — the atomic unit; `pos`+`ref` are the address,
   `value` the content.
2. **The label is just a token at `pos_label`** (its value is the class); the value
   vocab is **pixel-bins ∪ classes**. Classification = predict value at
   `(pos_label, ref)` — identical to completing any pixel.
3. **`ref` is a coreference tag**, randomly drawn per episode from a pool: **the
   same tag sits on all of a sample's tokens** (an exact-match key that groups
   them), re-randomized across episodes (blocks memorization). The correct `ref`
   is always present in the context.
4. **Attention is over TOKENS, not samples** — no per-sample aggregation. A small
   **multi-layer** attention over the flat bag (a set; permutation-invariant;
   position is a *field*, not sequence order).
5. **Completion = masked-token prediction**: a query token is `(pos*, _, ref*)`
   with the value replaced by a guess/MASK token; attention predicts the value.
6. **Hold-out principle**: a held-out joint `(pos, value, ref)` has each of `pos`,
   `value`, `ref` present *individually* in the context — only the joint is
   absent. The task is to **bind seen parts into an unseen whole**.
7. **One mechanism, many tasks.**

## Mechanism

![mechanism]({m_url})

A query `(pos*, _, ref*)` attends over the bag. The layers learn to (1) gather
`ref*`'s own tokens (exact ref match) into a content representation, then
(2) match cross-sample tokens at `pos*` from *similar* samples and read their
value. A single random `ref` can't drive step (2), so it is genuinely
**multi-layer**, not one associative read (that was the archived approach's limit).

## Proposed exp1 architecture
- **Token embedding**: `e = pos_emb[pos] + value_emb[value|MASK] + ref_emb[ref]`
  (learned tables; `ref` from a pool of `V`). *(HRR role⊗filler is an option.)*
- **L = 2–4** transformer layers (multi-head self-attention + MLP), no causal mask.
- **Head**: masked/query tokens → softmax over the unified value vocab; CE loss.

## Decisions kept from the archived arc
- Unified value vocab (pixel-bins ∪ classes); **label as `pos_label`**.
- Position = plain **learned** embedding, **no spatial prior**.
- `ref` = random pool assignment (anonymization) — now per-token / coreference.
- **Attention (softmax) over linear** — attention was needed for sparse retrieval.
- **Efficiency lessons**: host-gather images (avoid the O(N) in-jit gather);
  **one-hot matmul for high-contention embeddings** (MNIST is ~81% background →
  a gather's backward scatter serializes, 345× slower); device-sample
  positions/refs; jit the whole step.
- **Metrics**: retrieval (recall observed), label generalization, content
  generalization with **ink-only accuracy + a background baseline** (the raw
  pixel-accuracy is ~0.81 background and misleading).
- **Eval**: support from **TRAIN**, held-out query from **TEST**; *supervised*
  (full query) vs *supervised+* (more context); watch the train≈test gap.
- Models **converge fast** — modest step budgets; **context breadth** matters more
  than training length.
- The **hold-out principle** (parts present, joint absent) is now formalized.

## What changes vs archived
- Per-sample HRR aggregation → **token-level multi-layer attention**.
- Separate `r_label` field → **label as `pos_label`** (unified).
- Single associative read → **≥2 attention layers** (gather-then-match).

## Experiment plan
1. **exp1** — core token-level attention on MNIST: masked-token completion (pixels
   + label); establish label + content completion above the background/marginal
   baselines; train-support / test-query; ink metric.
2. **exp2** — hold-out & context-size sweep (does more context help; supervised vs
   supervised+).
3. **exp3** — **anonymized/permuted labels** → true in-context learning (class
   semantics inferred from context, not memorized).
4. **exp4** — ablations: # layers, `ref` on/off, position embedding, unified vocab.
5. **exp5+** — Fashion-MNIST / EMNIST; cross-dataset generalization.

## Open questions to confirm
- Token embedding: **additive learned role embeddings** (my default) vs a learned
  MLP combiner vs HRR role⊗filler?
- **# attention layers** (propose 2–4) and model width.
- Episode size / ref-pool size defaults.
"""
    url = save_report("universal-ar_proposal", md)
    (Path(__file__).parent.parent / "proposal.md").write_text(md, encoding="utf-8")
    print("ANATOMY:", a_url); print("MECHANISM:", m_url); print("PROPOSAL:", url)


if __name__ == "__main__":
    main()
