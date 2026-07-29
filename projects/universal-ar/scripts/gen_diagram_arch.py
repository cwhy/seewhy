"""
Proper architecture dataflow diagram: tensors, operators, shapes on every edge,
the attention block expanded as a real graph, and the attention matrix drawn for a
retrieval query vs a generalisation query.

Usage (server): uv run python projects/universal-ar/scripts/gen_diagram_arch.py
"""
import io, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_media
from shared_lib.report import save_report

C = dict(pos="#bcd7f0", pos_s="#5a8fc0", val="#c6e6c9", val_s="#5aa564",
         ref="#f2d4a8", ref_s="#c9924a", tens="#e8eef6", tens_s="#8fa8c4",
         wt="#efe6f7", wt_s="#8f79c8", op="#fff3e0", op_s="#e0a44c",
         txt="#16202b", mut="#6b7785", ok="#1e8449", bad="#c0392b", hi="#fff6d5")
P = []
def T(x, y, s, sz=11.5, col=None, w="normal", anc="start", mono=False):
    ff = ' font-family="ui-monospace,Menlo,monospace"' if mono else ""
    P.append(f'<text x="{x}" y="{y}" font-size="{sz}" fill="{col or C["txt"]}" font-weight="{w}" text-anchor="{anc}"{ff}>{s}</text>')
def box(x, y, w, h, fill, stroke, rx=4, sw=1.2, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    P.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def tensor(x, y, w, h, label, shape, fill=None, stroke=None):
    """A tensor block with its shape underneath."""
    box(x, y, w, h, fill or C["tens"], stroke or C["tens_s"], 4)
    T(x + w/2, y + h/2 + 1, label, 12, w="bold", anc="middle")
    T(x + w/2, y + h + 13, shape, 10, C["mut"], anc="middle", mono=True)
def op(x, y, w, h, label, sub=""):
    box(x, y, w, h, C["op"], C["op_s"], 14)
    T(x + w/2, y + (h/2 + 1 if not sub else h/2 - 4), label, 11.5, w="bold", anc="middle")
    if sub: T(x + w/2, y + h/2 + 11, sub, 9.5, C["mut"], anc="middle", mono=True)
def arrow(x1, y1, x2, y2, lbl="", col=None, dash="", sw=1.5, lblside="above"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    P.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col or C["mut"]}" stroke-width="{sw}"{d} marker-end="url(#a)"/>')
    if lbl:
        mx, my = (x1+x2)/2, (y1+y2)/2
        T(mx, my - 6 if lblside == "above" else my + 14, lbl, 9.5, C["mut"], anc="middle", mono=True)
def plus(cx, cy, r=11):
    P.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="white" stroke="{C["mut"]}" stroke-width="1.4"/>')
    T(cx, cy + 4.5, "+", 15, C["txt"], "bold", "middle")

W, H = 1360, 1700
P.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="ui-sans-serif,system-ui,sans-serif">')
P.append(f'<rect width="{W}" height="{H}" fill="white"/>')
P.append('<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6.5" refY="2.8" orient="auto">'
         f'<path d="M0,0 L6.5,2.8 L0,5.6 Z" fill="{C["mut"]}"/></marker></defs>')
T(28, 34, "Universal-AR — network architecture", 20, w="bold")
T(28, 56, "every edge carries its tensor shape · B = micro-batch 4 · N = 6810 tokens · d = 256 · 4 layers · 8 heads × 32 · 3.4M params", 11.5, C["mut"])

# ══════ SECTION 1: tokenisation → embedding ══════
y0 = 78
box(20, y0, W-40, 250, "#fbfcfe", C["tens_s"], 8, 1)
T(38, y0+22, "1 · INPUT  →  TOKEN STATES", 13.5, w="bold")

ids = [("pos ids", "(B, N) int32", C["pos"], C["pos_s"], "0…784"),
       ("value ids", "(B, N) int32", C["val"], C["val_s"], "0…42  (42 = MASK)"),
       ("ref ids", "(B, N) int32", C["ref"], C["ref_s"], "0…63")]
tbl = [("pos_emb", "(785, 256)"), ("val_emb", "(43, 256)"), ("ref_emb", "(64, 256)")]
for i, ((nm, shp, f_, s_, rng), (tn, tsh)) in enumerate(zip(ids, tbl)):
    yy = y0 + 48 + i*62
    tensor(44, yy, 118, 34, nm, shp, f_, s_)
    T(170, yy+21, rng, 9.5, C["mut"], mono=True)
    arrow(262, yy+17, 300, yy+17, "gather")
    tensor(302, yy, 140, 34, tn, tsh, C["wt"], C["wt_s"])
    arrow(444, yy+17, 492, yy+17, "(B,N,256)")
plus(516, y0+48+62+17)
for i in range(3):
    yy = y0 + 48 + i*62 + 17
    P.append(f'<path d="M494,{yy} L505,{yy} L505,{y0+48+62+17} L{505},{y0+48+62+17}" fill="none" stroke="{C["mut"]}" stroke-width="1.3"/>')
arrow(530, y0+48+62+17, 578, y0+48+62+17)
tensor(580, y0+48+62, 150, 34, "x", "(B, N, 256)")
T(580, y0+48+62+66, "the residual stream", 10, C["mut"])
T(760, y0+96, "Additive fields: the ADDRESS (pos, ref) and the", 11.5)
T(760, y0+114, "CONTENT (value) share one 256-d vector, so a", 11.5)
T(760, y0+132, "single dot product can match on address alone.", 11.5)
T(760, y0+162, "The sequence is a SET — position is a field,", 11.5, C["mut"])
T(760, y0+180, "not sequence order. No causal mask.", 11.5, C["mut"])
T(760, y0+210, "N = 6810 = 16 samples × 425 tokens + 10", 11, C["mut"], mono=True)

# ══════ SECTION 2: the layer ══════
y1 = 344
box(20, y1, W-40, 580, "#fbfcfe", C["tens_s"], 8, 1)
T(38, y1+22, "2 · ONE TRANSFORMER LAYER  ×4   (pre-LN, gradient-checkpointed)", 13.5, w="bold")

# residual stream (horizontal spine)
spine = y1 + 78
P.append(f'<line x1="60" y1="{spine}" x2="1300" y2="{spine}" stroke="{C["tens_s"]}" stroke-width="3"/>')
T(60, spine-14, "residual stream   x : (B, N, 256)", 11, C["mut"], mono=True)

# attention branch
bx = 260
P.append(f'<path d="M{bx},{spine} L{bx},{spine+52}" fill="none" stroke="{C["mut"]}" stroke-width="1.4" marker-end="url(#a)"/>')
op(bx-45, spine+56, 90, 30, "LayerNorm")
arrow(bx, spine+86, bx, spine+112, "", sw=1.4)
op(bx-58, spine+116, 116, 34, "x @ Wqkv", "(256, 768)")
# split into q k v
for i, (nm, dx) in enumerate((("q", -130), ("k", 0), ("v", 130))):
    P.append(f'<path d="M{bx},{spine+150} L{bx},{spine+168} L{bx+dx},{spine+168} L{bx+dx},{spine+186}" fill="none" stroke="{C["mut"]}" stroke-width="1.3" marker-end="url(#a)"/>')
    tensor(bx+dx-52, spine+190, 104, 30, nm, "(B,8,N,32)")
# attention core
arrow(bx-130, spine+220, bx-130, spine+248, "", sw=1.3)
arrow(bx, spine+220, bx, spine+248, "", sw=1.3)
op(bx-186, spine+252, 152, 34, "q · kᵀ / √32", "(B, 8, N, N)")
arrow(bx-110, spine+286, bx-110, spine+312, "", sw=1.3)
op(bx-166, spine+316, 112, 30, "softmax", "over keys")
arrow(bx-110, spine+346, bx-110, spine+372, "α", sw=1.3)
# weighted sum with v
P.append(f'<path d="M{bx+130},{spine+220} L{bx+130},{spine+356} L{bx-30},{spine+356} L{bx-30},{spine+376}" fill="none" stroke="{C["mut"]}" stroke-width="1.3" marker-end="url(#a)"/>')
op(bx-166, spine+376, 190, 32, "α @ v  →  merge heads", "(B, N, 256)")
arrow(bx-70, spine+408, bx-70, spine+424, "", sw=1.3)
op(bx-126, spine+428, 112, 30, "@ Wo", "(256, 256)")
# back to spine
P.append(f'<path d="M{bx-70},{spine+458} L{bx-70},{spine+498} L530,{spine+498} L530,{spine+14}" fill="none" stroke="{C["mut"]}" stroke-width="1.3"/>')
plus(530, spine)
T(548, spine+4, "residual add", 10.5, C["mut"])
box(bx-200, spine+46, 420, 420, "none", C["wt_s"], 10, 1.6, "6 4")
T(bx-196, spine+40, "MULTI-HEAD ATTENTION", 11, C["wt_s"], "bold")

# MLP branch
mx = 760
P.append(f'<path d="M{mx},{spine} L{mx},{spine+52}" fill="none" stroke="{C["mut"]}" stroke-width="1.4" marker-end="url(#a)"/>')
op(mx-45, spine+56, 90, 30, "LayerNorm")
arrow(mx, spine+86, mx, spine+112, "", sw=1.4)
op(mx-62, spine+116, 124, 34, "@ W1 + b1", "(256, 1024)")
arrow(mx, spine+150, mx, spine+176, "", sw=1.4)
op(mx-45, spine+180, 90, 28, "GELU")
arrow(mx, spine+208, mx, spine+234, "", sw=1.4)
op(mx-62, spine+238, 124, 34, "@ W2 + b2", "(1024, 256)")
P.append(f'<path d="M{mx},{spine+272} L{mx},{spine+320} L{mx+190},{spine+320} L{mx+190},{spine+14}" fill="none" stroke="{C["mut"]}" stroke-width="1.3"/>')
plus(mx+190, spine)
box(mx-90, spine+46, 300, 260, "none", C["op_s"], 10, 1.6, "6 4")
T(mx-86, spine+40, "MLP  (4× expansion)", 11, C["op_s"], "bold")

# head
hx = 1160
P.append(f'<path d="M{hx},{spine} L{hx},{spine+52}" fill="none" stroke="{C["mut"]}" stroke-width="1.4" marker-end="url(#a)"/>')
op(hx-52, spine+56, 104, 30, "final LN")
arrow(hx, spine+86, hx, spine+112, "", sw=1.4)
op(hx-66, spine+116, 132, 34, "@ head_W", "(256, 42)")
arrow(hx, spine+150, hx, spine+176, "", sw=1.4)
tensor(hx-84, spine+180, 168, 34, "logits", "(B, N, 42)")
T(hx-84, spine+232, "42 = 32 ink bins + 10 label slots", 10.5, C["mut"])
T(hx-84, spine+250, "ONE shared vocabulary for", 10.5, C["mut"])
T(hx-84, spine+266, "pixels AND labels", 10.5, C["mut"])
box(hx-110, spine+46, 240, 240, "none", C["ok"], 10, 1.6, "6 4")
T(hx-106, spine+40, "OUTPUT HEAD", 11, C["ok"], "bold")

# ══════ SECTION 3: the attention matrix ══════
y2 = 950
box(20, y2, W-40, 700, "#fbfcfe", C["tens_s"], 8, 1)
T(38, y2+24, "3 · THE ATTENTION MATRIX  α  —  what the query row actually looks like", 13.5, w="bold")
T(38, y2+44, "one row of α (B,8,N,N): a single query token against context tokens. This is the entire difference between the two tasks.", 11.5, C["mut"])

def grid(ox, oy, title, cols, weights, note, col, verdict, vcol):
    T(ox, oy - 10, title, 12.5, col, "bold")
    cw, ch = 96, 40
    T(ox - 8, oy + ch/2 + 4, "query", 10.5, C["mut"], anc="end")
    for i, (lab, wgt) in enumerate(zip(cols, weights)):
        x = ox + i * (cw + 6)
        inten = min(1.0, wgt / 0.5)
        fill = f"rgb({int(255-inten*(255-30))},{int(255-inten*(255-132))},{int(255-inten*(255-73))})" if wgt > 0.4 else \
               f"rgb({int(255-wgt*180)},{int(255-wgt*100)},{int(255-wgt*60)})"
        box(x, oy, cw, ch, fill, col if wgt > 0.4 else C["tens_s"], 4, 2 if wgt > 0.4 else 1)
        T(x + cw/2, oy + 25, f"{wgt:.2f}", 13, "white" if wgt > 0.4 else C["txt"], "bold", "middle")
        for j, ln in enumerate(lab.split("\n")):
            T(x + cw/2, oy + ch + 15 + j*13, ln, 9.5, C["mut"], anc="middle", mono=(j == 0))
    T(ox, oy + ch + 62, note, 11.5)
    T(ox, oy + ch + 84, verdict, 12.5, vcol, "bold")

grid(120, y2 + 96, "RETRIEVAL query  (p₇, MASK, 🟠)  —  its answer IS in context",
     ["(p₃,🟠)\npos ✗  ref ✓", "(p₇,🟠)\npos ✓  ref ✓", "(p₇,🟣)\npos ✓  ref ✗", "(p₉,🟠)\npos ✗  ref ✓", "…6806 more\n—"],
     [0.01, 0.97, 0.01, 0.01, 0.00],
     "Exactly one token scores on BOTH address fields → α collapses onto it → its value is copied out.",
     C["ok"], "→ measured accuracy 1.000  (pixels and labels, easy and hard pairs alike)", C["ok"])

grid(120, y2 + 300, "GENERALISE query  (p_label, MASK, 🔴)  —  its answer is NOWHERE in context",
     ["(p_lab,🟠)\npos ✓  ref ✗", "(p_lab,🟣)\npos ✓  ref ✗", "(p_lab,🟢)\npos ✓  ref ✗", "(p_lab,🔵)\npos ✓  ref ✗", "…pixels\n≈0"],
     [0.25, 0.25, 0.25, 0.25, 0.00],
     "All support label tokens share p_label and none has ref 🔴 → identical scores → α is uniform → output = mean label.",
     C["bad"], "→ zero information: cross-entropy pinned at ln(2) = 0.6931", C["bad"])

T(120, y2 + 520, "To break that tie the same 4 layers would have to compute:", 12.5, w="bold")
for x, t, b, cc in ((120, "① aggregate", "pool each sample's 392 pixel tokens\ninto a per-sample summary", C["mut"]),
                    (500, "② compare", "multiplicative similarity, pooled per ref\n— unmoved by 6 interventions", C["bad"]),
                    (880, "③ copy", "re-use the retrieval above\n(already perfect)", C["ok"])):
    box(x, y2 + 534, 356, 66, "white", cc, 7, 1.6)
    T(x + 12, y2 + 554, t, 12, cc, "bold")
    for j, ln in enumerate(b.split("\n")):
        T(x + 12, y2 + 572 + j*15, ln, 10.5, C["mut"])
arrow(480, y2 + 567, 496, y2 + 567); arrow(860, y2 + 567, 876, y2 + 567)
T(120, y2 + 628, "Loss composition per episode: 2112 of 6810 tokens scored — 1064 retrieval pixels, 40 retrieval labels, 256 generalise pixels,", 11, C["mut"])
T(120, y2 + 646, "and only 24 generalise labels (~1% of scored tokens). The task under study is 1% of the training signal.", 11, C["bad"])
P.append("</svg>")

svg = "".join(P)
url = save_media("universal-ar_architecture.svg", io.BytesIO(svg.encode()), "image/svg+xml")
md = f"""# Universal-AR — network architecture (dataflow)

![architecture]({url})

A proper dataflow view: every edge is annotated with its tensor shape, the attention
block is expanded as a graph rather than prose, and the attention matrix `α` is drawn
for both a retrieval query and a generalisation query — which is where the entire
difference between "works perfectly" and "zero information" lives.

**Section 1 — token states.** Three integer id streams `(B, N)` gather from
`pos_emb (785,256)`, `val_emb (43,256)`, `ref_emb (64,256)` and are **summed** into
the residual stream `x (B, N, 256)`. Because the fields are additive, the address
(pos, ref) and the content (value) share one vector — which is exactly what lets a
single dot product match on address alone.

**Section 2 — one layer, ×4.** The residual spine runs left to right; attention and
MLP hang off it. Attention is drawn in full: `x @ Wqkv (256,768)` → split into
q/k/v `(B,8,N,32)` → `q·kᵀ/√32` giving `(B,8,N,N)` → softmax → `α @ v` → merge heads
→ `@ Wo`. Then the head: final LN → `@ head_W (256,42)` → `logits (B,N,42)`, one
shared vocabulary for pixels and labels.

**Section 3 — the attention row.** For the retrieval query `(p₇, MASK, 🟠)` exactly
one context token scores on both address fields, α collapses to 0.97 on it, and its
value is copied → **1.000**. For the generalisation query `(p_label, MASK, 🔴)` every
support label token shares `p_label` and none has ref 🔴, so all score identically,
α is uniform at 0.25, and the output is the mean label → **ln(2), zero information**.

The three hops needed to break that tie are shown beneath: ① aggregate and ③ copy
demonstrably work; **② compare is the failing step**, unmoved by depth, shared
positions, 20-shot support, retrieval-only data, or conjunctive embeddings.

Also visible in the numbers: of 6810 tokens per episode only 2112 are scored, and
just **24 of them are generalise-labels — about 1% of the training signal.**
"""
rep = save_report("universal-ar_architecture", md)
print("DIAGRAM:", url); print("REPORT:", rep)
