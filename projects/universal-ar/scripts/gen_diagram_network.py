"""
Detailed architecture diagram: the actual retrieval network — real tensor shapes,
the transformer stack, and the attention computation that performs a retrieval.

Usage (server): uv run python projects/universal-ar/scripts/gen_diagram_network.py
"""
import io, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_media
from shared_lib.report import save_report

C = dict(pos="#bcd7f0", pos_s="#5a8fc0", val="#c6e6c9", val_s="#5aa564",
         ref="#f2d4a8", ref_s="#c9924a", mask="#f6c3c0", mask_s="#d0453f",
         box="#f4f7fb", box_s="#c3ccd6", txt="#1c2733", mut="#67727e",
         ok="#1e8449", bad="#c0392b", acc="#6b4fa8", hi="#fff6d5", tens="#eef3f9")
P = []
def T(x, y, s, sz=12, col=None, w="normal", anc="start", fam=None):
    ff = ' font-family="ui-monospace,SFMono-Regular,Menlo,monospace"' if fam else ""
    P.append(f'<text x="{x}" y="{y}" font-size="{sz}" fill="{col or C["txt"]}" font-weight="{w}" text-anchor="{anc}"{ff}>{s}</text>')
def R(x, y, w, h, fill, stroke, rx=5, sw=1, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    P.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def A(x1, y1, x2, y2, col=None, sw=1.4, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    P.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col or C["mut"]}" stroke-width="{sw}"{d} marker-end="url(#ar)"/>')

W, H = 1280, 1560
P.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="ui-sans-serif,system-ui,sans-serif">')
P.append(f'<rect width="{W}" height="{H}" fill="white"/>')
P.append('<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
         f'<path d="M0,0 L7,3 L0,6 Z" fill="{C["mut"]}"/></marker>'
         '<marker id="arg" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
         f'<path d="M0,0 L7,3 L0,6 Z" fill="{C["ok"]}"/></marker></defs>')

T(28, 36, "The retrieval network — actual architecture and tensor shapes", 21, w="bold")
T(28, 60, "3.4M params · 4 layers · d_model 256 · 8 heads × 32 · episode = 6810 tokens · effective batch 8 (micro 4 × accum 2)", 12.5, C["mut"])

# ═══ 1 · EPISODE / TOKENISATION ═══
y = 88
R(20, y, W-40, 168, C["box"], C["box_s"], 9)
T(38, y+24, "1 · EPISODE  →  FLAT TOKEN SEQUENCE", 14.5, w="bold")
T(38, y+44, "16 samples (10 support with labels given + 6 query) × 425 tokens each + 10 extra label-retrieval tokens  =  6810 tokens", 12, C["mut"])
per = [("392", "CONTEXT pixels", "value GIVEN", C["val"], C["val_s"]),
       ("16", "RETRIEVAL queries", "address IS in context", C["hi"], C["ok"]),
       ("16", "GENERALISE queries", "held-out positions", C["mask"], C["mask_s"]),
       ("1", "label token", "given, or masked", C["ref"], C["ref_s"])]
x = 40
for n, name, note, f_, s_ in per:
    R(x, y+62, 268, 54, f_, s_, 6)
    T(x+12, y+82, f"{n} ×", 13, w="bold"); T(x+46, y+82, name, 12.5, w="bold")
    T(x+12, y+100, note, 11, C["mut"])
    x += 280
T(38, y+140, "Every token carries three integer fields.  The sequence is a SET — position is a field, never sequence order; no causal mask.", 12)

# ═══ 2 · EMBEDDING ═══
y = 276
R(20, y, W-40, 150, C["box"], C["box_s"], 9)
T(38, y+24, "2 · EMBEDDING  —  three lookups, summed", 14.5, w="bold")
emb = [("pos", "pos_emb", "(785, 256)", "784 pixels + p_label", C["pos"], C["pos_s"]),
       ("value", "val_emb", "(43, 256)", "32 ink bins + 10 labels + MASK", C["val"], C["val_s"]),
       ("ref", "ref_emb", "(64, 256)", "random tag pool, re-drawn per episode", C["ref"], C["ref_s"])]
x = 40
for fld, nm, shp, note, f_, s_ in emb:
    R(x, y+44, 300, 62, f_, s_, 6)
    T(x+12, y+64, f"{fld} id  (B, 6810)", 11.5, C["mut"], fam=1)
    T(x+12, y+82, f"{nm}  {shp}", 12.5, w="bold", fam=1)
    T(x+12, y+98, note, 10.5, C["mut"])
    if x < 640: T(x+312, y+80, "+", 17, w="bold")
    x += 320
A(1000, y+75, 1050, y+75)
R(1058, y+52, 182, 46, C["tens"], C["box_s"], 6)
T(1149, y+72, "x  (B, 6810, 256)", 12.5, w="bold", anc="middle", fam=1)
T(1149, y+88, "token states", 10.5, C["mut"], anc="middle")
T(38, y+130, "Additive: the ADDRESS (pos, ref) and the CONTENT (value) occupy the same 256-d vector. This is what makes one-hop address matching possible.", 12, C["mut"])

# ═══ 3 · TRANSFORMER STACK ═══
y = 446
R(20, y, W-40, 250, C["box"], C["box_s"], 9)
T(38, y+24, "3 · TRANSFORMER STACK  —  4 identical layers (gradient-checkpointed)", 14.5, w="bold")
# stack blocks
for i in range(4):
    xx = 40 + i*104
    R(xx, y+44, 92, 40, C["tens"], C["box_s"], 6)
    T(xx+46, y+62, f"layer {i+1}", 12, anc="middle", w="bold")
    T(xx+46, y+77, "pre-LN", 10, C["mut"], anc="middle")
    if i < 3: A(xx+92, y+64, xx+104, y+64)
A(456, y+64, 486, y+64)
T(496, y+68, "→ final LN → head", 12, C["mut"])
# expanded layer
T(38, y+108, "one layer, expanded:", 12.5, w="bold")
blocks = [(40, "LayerNorm", "(256)"), (160, "Multi-Head Attn", "Wqkv (256, 768)\nWo (256, 256)  ·  8 heads × 32"),
          (400, "+ residual", ""), (520, "LayerNorm", "(256)"),
          (640, "MLP  GELU", "W1 (256, 1024)\nW2 (1024, 256)"), (880, "+ residual", "")]
for x, nm, shp in blocks:
    h = 58 if shp else 38
    R(x, y+124, 108 if not shp else 232 if "\n" in shp else 108, h, "white", C["acc"] if "Attn" in nm else C["box_s"], 6,
      2 if "Attn" in nm else 1)
    T(x+10, y+144, nm, 12, C["acc"] if "Attn" in nm else C["txt"], "bold" if "Attn" in nm else "normal")
    for j, ln in enumerate(shp.split("\n")):
        if ln: T(x+10, y+160+j*14, ln, 10, C["mut"], fam=1)
for x in (148, 392, 508, 628, 872):
    A(x, y+146, x+12, y+146)
T(38, y+212, "Attention is over ALL 6810 tokens — every query token sees every context token of every sample. This is where retrieval happens.", 12, C["mut"])
T(38, y+232, "O(N²) at N=6810 is the memory driver: hence gradient checkpointing and micro-batch 4.", 11.5, C["mut"])

# ═══ 4 · ATTENTION: THE RETRIEVAL OPERATION ═══
y = 716
R(20, y, W-40, 430, "#f7fbf8", C["ok"], 9, 2)
T(38, y+26, "4 · INSIDE ATTENTION  —  how a RETRIEVAL query is answered", 15, C["ok"], "bold")
T(38, y+48, "query token  (p₇, MASK, 🟠)   must recover the value stored at that address", 12.5)

T(38, y+78, "per head h (32 dims):", 12, w="bold")
eqs = [("q  =  Wq · x_query", "the query's 32-d probe — built from pos_emb[p₇] + val_emb[MASK] + ref_emb[🟠]"),
       ("k_j  =  Wk · x_j", "a key for every one of the 6810 context tokens"),
       ("v_j  =  Wv · x_j", "the payload that will be copied out"),
       ("α_j  =  softmax_j( q · k_j / √32 )", "attention weights — a similarity over ADDRESSES"),
       ("out  =  Σ_j α_j v_j", "weighted sum of payloads")]
for i, (e, note) in enumerate(eqs):
    T(56, y+102+i*26, e, 12.5, C["txt"], "bold", fam=1)
    T(340, y+102+i*26, note, 11.5, C["mut"])

# score table
T(38, y+246, "because the address fields are additive, q·k_j decomposes — and only ONE token scores on both:", 12.5)
hdrs = ["context token j", "pos term", "ref term", "q·k_j", "α_j"]
cols = [56, 210, 300, 390, 470]
for c, h in zip(cols, hdrs): T(c, y+270, h, 11, C["mut"], "bold")
rows = [("(p₃, v=12, 🟠)", "low", "HIGH", "medium", "0.01", C["mut"]),
        ("(p₇, v=27, 🟠)", "HIGH", "HIGH", "HIGHEST", "0.97", C["ok"]),
        ("(p₇, v=04, 🟣)", "HIGH", "low", "medium", "0.01", C["mut"]),
        ("(p₉, v=27, 🟠)", "low", "HIGH", "medium", "0.01", C["mut"])]
for i, r in enumerate(rows):
    yy = y+292+i*22
    if r[5] == C["ok"]: R(50, yy-14, 480, 20, C["hi"], C["ok"], 3)
    for c, v in zip(cols, r[:5]):
        T(c, yy, v, 11.5, r[5], "bold" if r[5] == C["ok"] else "normal", fam=(c > 200))
T(56, y+390, "⇒ out ≈ v of the matching token — its VALUE field is carried through the residual stream to the head.", 12, C["ok"], "bold")

# right side: head
R(580, y+80, 300, 120, "white", C["box_s"], 7)
T(596, y+102, "final LN  →  head", 13, w="bold")
T(596, y+124, "head_W (256, 42)", 11.5, C["mut"], fam=1)
T(596, y+142, "logits (B, 6810, 42)", 11.5, C["mut"], fam=1)
T(596, y+162, "42 = 32 ink bins + 10 label slots", 10.5, C["mut"])
T(596, y+182, "one shared vocabulary for pixels AND labels", 10.5, C["mut"])
A(880, y+140, 916, y+140, C["ok"])
R(924, y+112, 316, 76, "#e8f6ed", C["ok"], 7, 2)
T(1082, y+136, "argmax → value 27", 14, C["ok"], "bold", anc="middle")
T(1082, y+156, "ONE attention hop, no comparison needed", 11, C["mut"], anc="middle")
T(1082, y+174, "measured accuracy 1.000", 12, C["ok"], "bold", anc="middle")

R(580, y+216, 660, 190, "white", C["box_s"], 7)
T(596, y+238, "loss — only masked tokens are scored", 13, w="bold")
T(596, y+260, "CE = softmax_xent(logits, target) · is_query", 11.5, C["mut"], fam=1)
T(596, y+280, "scored per episode:  2112 of 6810 tokens", 11.5, C["mut"], fam=1)
T(596, y+302, "1064  retrieval pixels", 11.5, C["ok"])
T(596, y+320, "  40  retrieval labels", 11.5, C["ok"])
T(596, y+338, " 256  generalise pixels", 11.5, C["mut"])
T(596, y+356, "  24  generalise labels   ← only ~1% of scored tokens", 11.5, C["bad"], "bold")
T(596, y+382, "That imbalance is why the combined loss hides the label task entirely —", 11, C["mut"])
T(596, y+396, "the pixel and label losses must be reported separately.", 11, C["mut"])

# ═══ 5 · WHY THE SAME NETWORK CANNOT GENERALISE ═══
y = 1166
R(20, y, W-40, 370, "#fdf7f7", C["bad"], 9, 2)
T(38, y+26, "5 · THE SAME NETWORK ON A GENERALISATION QUERY  —  the address no longer identifies anything", 15, C["bad"], "bold")
T(38, y+50, "query token  (p_label, MASK, 🔴)  —  sample 🔴's label is never in the context", 12.5)

T(38, y+80, "the q·k decomposition now degenerates:", 12.5, w="bold")
cols2 = [56, 260, 360, 470]
for c, h in zip(cols2, ["context label token", "pos term", "ref term", "α_j"]): T(c, y+104, h, 11, C["mut"], "bold")
rows2 = [("(p_label, lab=A, 🟠)", "HIGH", "low", "0.25"), ("(p_label, lab=B, 🟣)", "HIGH", "low", "0.25"),
         ("(p_label, lab=A, 🟢)", "HIGH", "low", "0.25"), ("(p_label, lab=B, 🔵)", "HIGH", "low", "0.25")]
for i, r in enumerate(rows2):
    yy = y+126+i*21
    for c, v in zip(cols2, r): T(c, yy, v, 11.5, C["mut"], fam=(c > 250))
T(56, y+232, "every support label token scores identically → α is uniform → the output is the MEAN label. Zero information.", 12, C["bad"], "bold")

T(38, y+264, "to break the tie the network would have to compute, inside these same layers:", 12.5, w="bold")
steps = [(56, "① aggregate", "pool 🔴's 392 pixel tokens, and each support's,\ninto per-sample summaries", C["mut"]),
         (440, "② compare", "score 🔴's summary against each support summary\n— multiplicative, then pooled per ref", C["bad"]),
         (824, "③ copy", "re-use the panel-4 retrieval to copy\nthe winner's label token", C["ok"])]
for x, t, b, col in steps:
    R(x, y+278, 360, 72, "white", col, 7, 1.6)
    T(x+12, y+298, t, 12.5, col, "bold")
    for j, ln in enumerate(b.split("\n")):
        T(x+12, y+316+j*15, ln, 11, C["mut"])
A(404, y+314, 436, y+314); A(788, y+314, 820, y+314)
P.append("</svg>")

svg = "".join(P)
url = save_media("universal-ar_network_detail.svg", io.BytesIO(svg.encode()), "image/svg+xml")

md = f"""# Universal-AR — the retrieval network in detail

![network]({url})

## Architecture (exactly as run)

| component | shape | note |
|---|---|---|
| `pos_emb` | (785, 256) | 784 pixel positions + `p_label` |
| `val_emb` | (43, 256) | 32 ink bins + 10 label slots + MASK |
| `ref_emb` | (64, 256) | random tag pool, re-drawn each episode |
| per layer | `Wqkv` (256, 768), `Wo` (256, 256) | 8 heads × 32 |
| | `W1` (256, 1024), `W2` (1024, 256) | GELU MLP |
| head | `head_W` (256, 42) | one shared vocabulary for pixels **and** labels |

4 layers, 3.4M parameters, **6810 tokens per episode**, effective batch 8
(micro-batch 4 × accumulation 2), gradient-checkpointed because attention is O(N²)
at N=6810. The sequence is a **set** — position is a field, never sequence order,
and there is no causal mask.

## How a retrieval is actually computed

Token states are `x = pos_emb[p] + val_emb[v] + ref_emb[r]`. Per head:

```
q   = Wq · x_query                     32-d probe
k_j = Wk · x_j        v_j = Wv · x_j   for all 6810 context tokens
α_j = softmax_j( q · k_j / √32 )
out = Σ_j α_j v_j
```

Because the fields are **additive**, `q · k_j` decomposes into a position term and a
ref term. For the query `(p₇, MASK, 🟠)`:

| context token | pos term | ref term | α |
|---|---|---|---|
| (p₃, v=12, 🟠) | low | HIGH | 0.01 |
| **(p₇, v=27, 🟠)** | **HIGH** | **HIGH** | **0.97** |
| (p₇, v=04, 🟣) | HIGH | low | 0.01 |
| (p₉, v=27, 🟠) | low | HIGH | 0.01 |

Exactly one token scores on **both** address fields, so attention collapses onto it
and its value rides the residual stream to the head. **One hop, no comparison —
measured accuracy 1.000.**

## The loss imbalance worth knowing about

Of 6810 tokens per episode, 2112 are scored:

| | count | share of scored |
|---|---|---|
| retrieval pixels | 1064 | 50% |
| retrieval labels | 40 | 2% |
| generalise pixels | 256 | 12% |
| **generalise labels** | **24** | **~1%** |

The task we actually care about is ~1% of the scored tokens. That is why the
combined loss hides it completely and pixel/label losses must be reported
separately — a lesson that cost us several misread runs.

## Why the identical network fails to generalise

For `(p_label, MASK, 🔴)` the decomposition degenerates: every support label token
has the *same* position `p_label` and *none* has ref 🔴, so all score identically,
α is uniform, and the output is the mean label — **zero information**, which is
exactly the ln(2) we measure.

Breaking that tie requires ① pooling each sample into a summary, ② a multiplicative
comparison pooled per ref, ③ then the panel-4 copy. ① and ③ demonstrably work.
**② is the failing step**, and it is unmoved by depth, shared positions, 20-shot
support, retrieval-only data, or conjunctive embeddings.
"""
rep = save_report("universal-ar_network_detail", md)
print("DIAGRAM:", url); print("REPORT:", rep)
